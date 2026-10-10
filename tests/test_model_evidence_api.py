"""Verified chart contracts; explicit unit artifacts, plus real sampling in test_api."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from decisionguard.api.app import create_app
from decisionguard.api.database import Dataset, ModelRun
from decisionguard.config import Settings, artifact_path
from decisionguard.data.artifacts import file_hash, write_json
from tests.test_api import database as database
from tests.test_api import queued_experiment


@pytest.fixture
def evidence_run(database: Settings) -> tuple[UUID, Path]:
    """Publish tiny deterministic chart fixtures, not claimed scientific results."""
    with TestClient(create_app(database)) as client:
        _, _, identifier = queued_experiment(client)
    engine = create_engine(database.database_url)
    try:
        with Session(engine) as session, session.begin():
            run = session.get(ModelRun, UUID(identifier))
            assert run is not None
            dataset = session.get(Dataset, run.dataset_id)
            assert dataset is not None
            output = artifact_path(database.artifact_root, run.artifact_uri)
            output.mkdir(parents=True)
            weeks = pd.date_range(dataset.date_start, periods=78, freq="W-MON")
            pd.DataFrame(
                {
                    "week": weeks,
                    "split": ["training"] * 65 + ["holdout"] * 13,
                    "actual": np.arange(78, dtype=float) + 1000,
                    "predicted": np.arange(78, dtype=float) + 900,
                    "lower_90": np.arange(78, dtype=float) + 800,
                    "upper_90": np.arange(78, dtype=float) + 1200,
                }
            ).to_parquet(output / "predictions.parquet", index=False)
            grid = np.array([[0.0], [10.0], [20.0]])
            np.savez_compressed(
                output / "channel_draws.npz",
                channels=np.array(["search_spend"]),
                adstocked_spend_grid=grid,
                saturation_response=np.arange(1, 9).reshape(1, 8, 1, 1) * grid,
            )
            (output / "posterior.nc").write_bytes(
                b"Unit chart fixture, not a posterior"
            )
            record: dict[str, Any] = {
                "model_family": run.model_family,
                "model_status": "CANDIDATE_UNEVALUATED",
                "config": run.configuration,
                "dataset": {"hash": dataset.hash},
                "training_window": {
                    "start": str(weeks[0].date()),
                    "end": str(weeks[64].date()),
                    "rows": 65,
                },
                "prediction_context": "conditional_on_realized_covariates",
                "limitations": ["Unit fixture, not a scientific result"],
                "diagnostics": {"diagnostic_status": "INCOMPLETE"},
                "response_curve_axis": (
                    "adstocked_spend_AUD; conditional saturation response"
                ),
                "channels": {
                    "search_spend": {
                        "roi_mean": 4.5,
                        "roi_interval_90": [1.35, 7.65],
                        "contribution_mean": 4500.0,
                        "contribution_interval_90": [1350.0, 7650.0],
                    }
                },
                "artifacts": {
                    name: file_hash(output / name)
                    for name in (
                        "predictions.parquet",
                        "channel_draws.npz",
                        "posterior.nc",
                    )
                },
            }
            write_json(output / "model.json", record)
            run.status = "SUCCEEDED"
            run.model_status = record["model_status"]
            run.record_hash = file_hash(output / "model.json")
            run.training_window = record["training_window"]
    finally:
        engine.dispose()
    return UUID(identifier), output


def test_chart_evidence_preserves_units_uncertainty_and_split_without_release(
    database: Settings, evidence_run: tuple[UUID, Path]
) -> None:
    identifier, path = evidence_run
    prefix = f"/model-runs/{identifier}"
    with TestClient(create_app(database)) as client:
        response = client.get(prefix + "/predictions", params={"limit": 520})
        assert response.status_code == 200, response.text
        predictions = response.json()
        assert predictions["total"] == 78
        assert predictions["units"] == "AUD/week"
        assert predictions["interval_probability"] == 0.9
        assert all(point["residual"] == 100 for point in predictions["rows"])
        assert [point["split"] for point in predictions["rows"]] == (
            ["training"] * 65 + ["holdout"] * 13
        )
        provenance = predictions["provenance"]
        assert provenance["model_record_hash"] == file_hash(path / "model.json")
        assert provenance["prediction_context"] == "conditional_on_realized_covariates"
        assert (
            provenance["decision_use"]
            == "requires_current_health_and_scenario_validation"
        )
        assert provenance["training_window"]["end"] < predictions["rows"][65]["week"]
        subset = client.get(
            prefix + "/predictions",
            params={"split": "holdout", "offset": 3, "limit": 2},
        ).json()
        assert subset["total"] == 13
        assert subset["rows"] == predictions["rows"][68:70]
        assert client.get(prefix + "/predictions?offset=78").json()["rows"] == []

        response = client.get(prefix + "/channels/search_spend")
        assert response.status_code == 200, response.text
        channel = response.json()
        assert channel["provenance"] == provenance
        assert channel["summary"]["roi_interval_90"] == [1.35, 7.65]
        assert channel["contribution_units"] == "AUD over training window"
        assert channel["roi_definition"] == "training_contribution / training_spend"
        assert channel["response_curve_axis"].startswith("adstocked_spend_AUD;")
        assert channel["response_curve"][1] == pytest.approx(
            {"adstocked_spend": 10, "lower_90": 13.5, "median": 45, "upper_90": 76.5}
        )
        assert client.get(provenance["health_url"]).json()["decision_status"] == "BLOCK"
        assert (
            client.post(
                prefix + "/scenarios",
                json={"total": 100, "current": {"search_spend": 100}},
            ).status_code
            == 409
        )
        assert str(database.artifact_root) not in json.dumps([predictions, channel])
    with TestClient(create_app(database)) as restarted:
        assert restarted.get(prefix + "/channels/search_spend").json() == channel
        assert restarted.get(prefix + "/predictions?limit=520").json() == predictions


def test_chart_routes_validate_requests_and_reject_unfinished_runs(
    database: Settings,
) -> None:
    with TestClient(create_app(database)) as client:
        _, _, identifier = queued_experiment(client)
        prefix = f"/model-runs/{identifier}"
        for suffix in ("predictions", "channels/search_spend"):
            assert client.get(f"/model-runs/{uuid4()}/{suffix}").status_code == 404
            assert client.get(prefix + "/" + suffix).status_code == 409
        assert client.get(prefix + "/channels/unknown").status_code == 404
        for query in ("split=test", "offset=-1", "limit=0", "limit=521"):
            assert client.get(prefix + "/predictions?" + query).status_code == 422


@pytest.mark.parametrize(
    "target", ["predictions.parquet", "channel_draws.npz", "posterior.nc", "model.json"]
)
def test_chart_routes_reject_changed_components(
    database: Settings, evidence_run: tuple[UUID, Path], target: str
) -> None:
    identifier, path = evidence_run
    with (path / target).open("ab") as stream:
        stream.write(b"tampered")
    with TestClient(create_app(database)) as client:
        for suffix in ("predictions", "channels/search_spend"):
            response = client.get(f"/model-runs/{identifier}/{suffix}")
            assert response.status_code == 409
            assert str(path) not in response.text


@pytest.mark.parametrize("target", ["configuration", "training_window", "dataset"])
def test_chart_routes_reject_changed_registration(
    database: Settings, evidence_run: tuple[UUID, Path], target: str
) -> None:
    identifier, _ = evidence_run
    engine = create_engine(database.database_url)
    try:
        with Session(engine) as session, session.begin():
            run = session.get(ModelRun, identifier)
            assert run is not None
            if target == "configuration":
                run.configuration = {**run.configuration, "seed": 999}
            elif target == "training_window":
                run.training_window = {**(run.training_window or {}), "rows": 64}
            else:
                dataset = session.get(Dataset, run.dataset_id)
                assert dataset is not None
                dataset.hash = "changed"
        with TestClient(create_app(database)) as client:
            for suffix in ("predictions", "channels/search_spend"):
                assert (
                    client.get(f"/model-runs/{identifier}/{suffix}").status_code == 409
                )
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "defect",
    [
        "split",
        "date",
        "nonfinite",
        "interval",
        "curve_channels",
        "curve_shape",
        "curve_nonfinite",
        "curve_empty",
        "curve_truncated",
    ],
)
def test_chart_routes_reject_malformed_registered_artifacts(
    database: Settings, evidence_run: tuple[UUID, Path], defect: str
) -> None:
    identifier, path = evidence_run
    # Simulate a producer publishing a malformed but checksum-consistent artifact.
    if defect.startswith("curve"):
        with np.load(path / "channel_draws.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if defect == "curve_channels":
            arrays["channels"] = np.array(["wrong_channel"])
        elif defect == "curve_shape":
            arrays["saturation_response"] = arrays["saturation_response"][:, :, :2, :]
        elif defect == "curve_nonfinite":
            arrays["saturation_response"][0, 0, -1, 0] = np.nan
        name = "channel_draws.npz"
        np.savez_compressed(path / name, **arrays)
        if defect in {"curve_empty", "curve_truncated"}:
            (path / name).write_bytes(b"" if defect == "curve_empty" else b"PK\x03\x04")
        suffix = "channels/search_spend"
    else:
        name = "predictions.parquet"
        frame = pd.read_parquet(path / name)
        if defect == "split":
            frame.loc[65, "split"] = "training"
        elif defect == "date":
            frame.loc[65, "week"] = frame.loc[64, "week"]
        elif defect == "nonfinite":
            frame.loc[77, "predicted"] = np.nan
        else:
            frame.loc[77, "lower_90"] = 1e6
        frame.to_parquet(path / name, index=False)
        # Even a page before the defect must reject the complete malformed evidence.
        suffix = "predictions?split=training&limit=1"
    record = json.loads((path / "model.json").read_text())
    record["artifacts"][name] = file_hash(path / name)
    write_json(path / "model.json", record)
    engine = create_engine(database.database_url)
    try:
        with Session(engine) as session, session.begin():
            run = session.get(ModelRun, identifier)
            assert run is not None
            run.record_hash = file_hash(path / "model.json")
        with TestClient(create_app(database)) as client:
            assert client.get(f"/model-runs/{identifier}/{suffix}").status_code == 409
    finally:
        engine.dispose()
