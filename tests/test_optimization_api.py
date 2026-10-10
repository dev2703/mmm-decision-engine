"""PostgreSQL lifecycle with real release gates and real numerical optimization."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from decisionguard.api.app import create_app
from decisionguard.api.database import Dataset, ModelRun, OptimizationRun, Scenario
from decisionguard.api.jobs import evaluate_model, optimize_scenario
from decisionguard.config import Settings, artifact_path
from decisionguard.optimization import artifacts
from decisionguard.optimization.allocation import PosteriorResponse
from tests.test_api import database as database
from tests.test_api import registered_evaluation_models, stub_evaluator


def ready_scenario(
    client: TestClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[UUID, UUID, UUID]:
    model, alternative, template = registered_evaluation_models(client, settings)
    response = client.post(
        f"/model-runs/{model}/evaluate", json={"sensitivity_run_id": str(alternative)}
    )
    assert response.status_code == 202
    stub_evaluator(monkeypatch, template)
    evaluate_model(model, settings)
    response = client.post(
        f"/model-runs/{model}/scenarios",
        json={
            "total": 200,
            "current": {"search_spend": 100, "meta_spend": 100},
            "max_movement": {"search_spend": 0.1, "meta_spend": 0.1},
            "risk_policy": "conservative",
            "horizon_weeks": 4,
        },
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["id"]), model, alternative


def response_fixture(model: Path, **kwargs: Any) -> tuple[PosteriorResponse, list[int]]:
    """Replace posterior loading only; these inputs are not scientific fit evidence."""
    assert model.is_dir()
    assert kwargs["dataset"].is_dir()
    values = np.array([[1000.0, 1100.0], [1100.0, 1000.0], [1050.0, 1050.0]])
    return PosteriorResponse(
        ("search_spend", "meta_spend"),
        np.zeros_like(values),
        values,
        np.full_like(values, 100.0),
        np.array([[0.0, 0.0], [200.0, 200.0]]),
        horizon=kwargs["horizon"],
        lags=1,
    ), [0, 1, 2]


@pytest.mark.parametrize("supported", [True, False])
def test_registered_optimization_keeps_results_and_revokes_stale_authority(
    database: Settings,
    monkeypatch: pytest.MonkeyPatch,
    supported: bool,
) -> None:
    def response(model: Path, **kwargs: Any) -> tuple[PosteriorResponse, list[int]]:
        posterior, selected = response_fixture(model, **kwargs)
        if not supported:
            posterior = replace(posterior, history=np.array([[0.0, 0.0], [90.0, 90.0]]))
        return posterior, selected

    with TestClient(create_app(database)) as client:
        scenario, model, alternative = ready_scenario(client, database, monkeypatch)
        queued = client.post(
            f"/scenarios/{scenario}/optimize", json={"draws": 3, "seed": 7}
        )
        assert queued.status_code == 202, queued.text
        identifier = queued.json()["id"]
        assert queued.json()["summary"] is None
        assert not queued.json()["recommendation_allowed"]
        assert (
            client.post(f"/scenarios/{scenario}/optimize", json={}).status_code == 409
        )
        monkeypatch.setattr(artifacts, "load_response", response)
        result = optimize_scenario(scenario, database)
        with pytest.raises(ValueError, match="no queued"):
            optimize_scenario(scenario, database)
    with TestClient(create_app(database)) as client:
        completed = client.get(f"/optimization-runs/{identifier}")
        assert completed.status_code == 200, completed.text
        evidence = completed.json()
        assert evidence["status"] == "SUCCEEDED"
        assert evidence["evidence_current"]
        assert evidence["recommendation_allowed"] is supported
        summary = evidence["summary"]
        assert summary["risk_policy"] == "conservative"
        assert summary["selected"] == result["alternatives"]["conservative"]
        assert "allocation_draws" not in summary
        allocation = summary["selected"]["allocation"]
        assert sum(allocation.values()) == pytest.approx(200)
        assert all(90 <= value <= 110 for value in allocation.values())
        assert summary["selected"]["media_revenue_interval_90"][0] > 0
        if not supported:
            assert summary["selected"]["decision_state"] == "RESTRICT"
            assert summary["selected"]["extrapolated_channels"]
        client.post(
            f"/model-runs/{model}/evaluate",
            json={"sensitivity_run_id": str(alternative)},
        )
        stale = client.get(f"/optimization-runs/{identifier}").json()
        assert stale["summary"] == summary
        assert not stale["evidence_current"] and not stale["recommendation_allowed"]
        assert (
            client.post(f"/scenarios/{scenario}/optimize", json={}).status_code == 409
        )


@pytest.mark.parametrize("change", ["pending", "model", "dataset", "scenario"])
def test_changed_inputs_cannot_reach_solver(
    database: Settings,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    with TestClient(create_app(database)) as client:
        scenario, model, alternative = ready_scenario(client, database, monkeypatch)
        queued = client.post(f"/scenarios/{scenario}/optimize", json={}).json()

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("changed input reached numerical optimization")

        monkeypatch.setattr(artifacts, "optimize_run", forbidden)
        engine = create_engine(database.database_url)
        try:
            with Session(engine) as session, session.begin():
                if change == "scenario":
                    stored = session.get(Scenario, scenario)
                    assert stored is not None
                    stored.risk_policy = "expected"
                elif change == "dataset":
                    model_row = session.get(ModelRun, model)
                    assert model_row is not None
                    dataset = session.get(Dataset, model_row.dataset_id)
                    assert dataset is not None
                    (
                        artifact_path(database.artifact_root, dataset.artifact_uri)
                        / "raw.parquet"
                    ).write_bytes(b"changed")
            if change == "pending":
                client.post(
                    f"/model-runs/{model}/evaluate",
                    json={"sensitivity_run_id": str(alternative)},
                )
            elif change == "model":
                (database.artifact_root / "model" / "posterior.nc").write_bytes(
                    b"changed"
                )
            with pytest.raises(ValueError):
                optimize_scenario(scenario, database)
            failed = client.get(f"/optimization-runs/{queued['id']}").json()
            assert failed["status"] == "FAILED" and failed["summary"] is None
            assert not failed["recommendation_allowed"]
            with Session(engine) as session:
                job = session.get(OptimizationRun, UUID(queued["id"]))
                assert job is not None and job.record_hash is None
                assert not artifact_path(
                    database.artifact_root, job.artifact_uri
                ).exists()
        finally:
            engine.dispose()


@pytest.mark.parametrize("failure", ["new_evaluation", "solver"])
def test_single_worker_and_no_publication_after_midflight_failure(
    database: Settings,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    entered, release = Event(), Event()

    def blocked_response(
        model: Path, **kwargs: Any
    ) -> tuple[PosteriorResponse, list[int]]:
        entered.set()
        assert release.wait(15)
        if failure == "solver":
            raise ValueError("deliberate numerical failure")
        return response_fixture(model, **kwargs)

    with (
        TestClient(create_app(database)) as client,
        ThreadPoolExecutor(max_workers=1) as workers,
    ):
        scenario, model, alternative = ready_scenario(client, database, monkeypatch)
        identifier = client.post(f"/scenarios/{scenario}/optimize", json={}).json()[
            "id"
        ]
        monkeypatch.setattr(artifacts, "load_response", blocked_response)
        future = workers.submit(optimize_scenario, scenario, database)
        try:
            assert entered.wait(15)
            assert (
                client.get(f"/optimization-runs/{identifier}").json()["status"]
                == "RUNNING"
            )
            with pytest.raises(ValueError, match="no queued"):
                optimize_scenario(scenario, database)
            assert (
                client.post(f"/scenarios/{scenario}/optimize", json={}).status_code
                == 409
            )
            if failure == "new_evaluation":
                # This completes while computation is paused: the model lock is free.
                assert (
                    client.post(
                        f"/model-runs/{model}/evaluate",
                        json={"sensitivity_run_id": str(alternative)},
                    ).status_code
                    == 202
                )
        finally:
            release.set()
        with pytest.raises(ValueError):
            future.result(timeout=15)
        failed = client.get(f"/optimization-runs/{identifier}").json()
        assert failed["status"] == "FAILED" and failed["summary"] is None
        assert failed["finished_at"] and failed["error_type"] == "ValueError"


@pytest.mark.parametrize("target", ["artifact", "summary"])
def test_registered_optimization_rejects_tampering(
    database: Settings,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    with TestClient(create_app(database)) as client:
        scenario, _, _ = ready_scenario(client, database, monkeypatch)
        identifier = client.post(f"/scenarios/{scenario}/optimize", json={}).json()[
            "id"
        ]
        monkeypatch.setattr(artifacts, "load_response", response_fixture)
        optimize_scenario(scenario, database)
        engine = create_engine(database.database_url)
        try:
            with Session(engine) as session, session.begin():
                job = session.get(OptimizationRun, UUID(identifier))
                assert job is not None
                if target == "summary":
                    job.summary = {"fabricated": "result"}
                else:
                    path = (
                        artifact_path(database.artifact_root, job.artifact_uri)
                        / "optimization.json"
                    )
                    value = json.loads(path.read_text())
                    value["alternatives"]["conservative"]["expected_improvement"] = 1e9
                    path.write_text(json.dumps(value))
            assert client.get(f"/optimization-runs/{identifier}").status_code == 409
        finally:
            engine.dispose()


def test_optimization_request_limits_and_missing_objects(database: Settings) -> None:
    with TestClient(create_app(database)) as client:
        for body in (
            {"draws": 0},
            {"draws": 2001},
            {"draws": True},
            {"seed": -1},
            {"seed": "2"},
            {"risk_policy": "expected"},
        ):
            assert (
                client.post(f"/scenarios/{uuid4()}/optimize", json=body).status_code
                == 422
            )
        assert client.post(f"/scenarios/{uuid4()}/optimize", json={}).status_code == 404
        assert client.get(f"/optimization-runs/{uuid4()}").status_code == 404
