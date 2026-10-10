"""Real PostgreSQL migrations, transaction behavior and HTTP data contracts."""

import json
import os
import shutil
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema

from decisionguard.api.app import artifact_path, create_app
from decisionguard.api.database import (
    Dataset,
    EvaluationRun,
    ModelRun,
    Project,
    Scenario,
)
from decisionguard.config import Settings
from decisionguard.data.artifacts import (
    dataset_evidence,
    file_hash,
    load_model_ready,
    write_json,
)
from decisionguard.models.config import MMMConfig


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Settings]:
    value = os.environ.get("DECISIONGUARD_TEST_DATABASE_URL")
    if not value:
        pytest.skip("real PostgreSQL requires DECISIONGUARD_TEST_DATABASE_URL")
    url = make_url(value)
    if not url.database or not url.database.endswith("_test"):
        pytest.fail("integration tests require a dedicated database ending in _test")
    engine = create_engine(url)
    schema = "test_" + uuid4().hex
    with engine.begin() as connection:
        connection.execute(CreateSchema(schema))
    scoped = url.update_query_dict({"options": "-csearch_path=" + schema})
    settings = Settings(
        scoped.render_as_string(hide_password=False), tmp_path / "artifacts"
    )
    configuration = Config("alembic.ini")
    configuration.attributes["database_url"] = settings.database_url
    command.upgrade(configuration, "head")
    try:
        yield settings
    finally:
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        engine.dispose()


def test_migrations_round_trip_on_postgresql(database: Settings) -> None:
    engine = create_engine(database.database_url)
    assert {
        "projects",
        "datasets",
        "evaluation_runs",
        "scenarios",
        "optimization_runs",
        "alembic_version",
    } <= set(inspect(engine).get_table_names())
    configuration = Config("alembic.ini")
    configuration.attributes["database_url"] = database.database_url
    command.downgrade(configuration, "base")
    assert "projects" not in inspect(engine).get_table_names()
    command.upgrade(configuration, "head")
    command.check(configuration)
    assert "datasets" in inspect(engine).get_table_names()
    engine.dispose()


def test_project_dataset_quality_flow_survives_new_app(database: Settings) -> None:
    with TestClient(create_app(database)) as client:
        project = client.post("/projects", json={"name": "Marketing"})
        assert project.status_code == 201
        project_id = project.json()["id"]
        assert UUID(project.headers["X-Correlation-ID"])
        response = client.post(
            f"/projects/{project_id}/datasets/generate", json={"weeks": 156, "seed": 42}
        )
        assert response.status_code == 201, response.text
        dataset = response.json()
        assert dataset["version"] == 1
        assert dataset["source"] == "SYNTHETIC"
        quality = client.get(f"/datasets/{dataset['id']}/quality")
        assert quality.status_code == 200
        assert quality.json()["clean_hash"] == dataset["hash"]
        second = client.post(
            f"/projects/{project_id}/datasets/generate", json={"seed": 42}
        )
        assert second.status_code == 201
        assert second.json()["version"] == 2
        assert second.json()["hash"] == dataset["hash"]
    with TestClient(create_app(database)) as restarted:
        fetched = restarted.get(f"/projects/{project_id}")
        assert fetched.json()["current_dataset_id"] == second.json()["id"]
        assert restarted.get(f"/datasets/{dataset['id']}/quality").status_code == 200


def test_failed_transaction_does_not_leave_partial_project(database: Settings) -> None:
    engine = create_engine(database.database_url)
    identifier = uuid4()
    with pytest.raises(IntegrityError), Session(engine) as session, session.begin():
        session.add(Project(id=identifier, name="Transient"))
        session.flush()
        session.add(Project(id=identifier, name="Duplicate"))
        session.flush()
    with Session(engine) as session:
        assert session.get(Project, identifier) is None
    engine.dispose()


def test_api_rejects_invalid_missing_and_changed_evidence(database: Settings) -> None:
    with TestClient(create_app(database)) as client:
        assert client.post("/projects", json={"name": " "}).status_code == 422
        assert client.get("/projects/not-a-uuid").status_code == 422
        assert client.get(f"/projects/{uuid4()}").status_code == 404
        assert (
            client.post(f"/projects/{uuid4()}/datasets/generate", json={}).status_code
            == 404
        )
        project_id = client.post("/projects", json={"name": "Quality"}).json()["id"]
        for body in (
            {"weeks": 10000},
            {"weeks": "156"},
            {"seed": -1},
            {"unexpected": True},
        ):
            assert (
                client.post(
                    f"/projects/{project_id}/datasets/generate", json=body
                ).status_code
                == 422
            )
        dataset_id = client.post(
            f"/projects/{project_id}/datasets/generate", json={}
        ).json()["id"]
        engine = create_engine(database.database_url)
        with Session(engine) as session:
            dataset = session.scalar(
                select(Dataset).where(Dataset.id == UUID(dataset_id))
            )
            assert dataset is not None
            path = artifact_path(database.artifact_root, dataset.artifact_uri)
        (path / "clean.parquet").write_bytes(b"changed")
        assert client.get(f"/datasets/{dataset_id}/quality").status_code == 409
        engine.dispose()


def test_artifact_path_rejects_escape_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    for uri in ("../outside", str(outside), "link/data"):
        with pytest.raises(ValueError, match="escapes"):
            artifact_path(root, uri)
    assert artifact_path(root, "projects/data") == root / "projects/data"


def queued_experiment(client: TestClient, *, weeks: int = 78) -> tuple[str, str, str]:
    project_id = client.post("/projects", json={"name": "MMM experiment"}).json()["id"]
    dataset = client.post(
        f"/projects/{project_id}/datasets/generate", json={"weeks": weeks}
    ).json()
    experiment = client.post(
        f"/projects/{project_id}/experiments",
        json={
            "hypothesis": "Fast serialization smoke; cannot justify release",
            "configuration": {
                "draws": 8,
                "tune": 16,
                "chains": 1,
                "channels": ["search_spend"],
            },
        },
    )
    assert experiment.status_code == 201, experiment.text
    experiment_id = experiment.json()["id"]
    run = client.post(f"/experiments/{experiment_id}/run")
    assert run.status_code == 202, run.text
    assert run.json()["status"] == "QUEUED"
    assert run.json()["dataset_id"] == dataset["id"]
    return project_id, experiment_id, run.json()["id"]


def test_queued_runs_snapshot_data_and_validate_configuration(
    database: Settings,
) -> None:
    with TestClient(create_app(database)) as client:
        project_id, experiment_id, run_id = queued_experiment(client)
        assert client.post(f"/experiments/{experiment_id}/run").status_code == 409
        old = client.get(f"/model-runs/{run_id}").json()
        client.post(f"/projects/{project_id}/datasets/generate", json={"seed": 123})
        assert (
            client.get(f"/model-runs/{run_id}").json()["dataset_id"]
            == old["dataset_id"]
        )
        assert len(client.get(f"/projects/{project_id}/model-runs").json()) == 1
        for configuration in (
            {"draws": 0},
            {"draws": 1000000},
            {"channels": ["unknown"]},
            {"run_shell": True},
        ):
            assert (
                client.post(
                    f"/projects/{project_id}/experiments",
                    json={"hypothesis": "Invalid", "configuration": configuration},
                ).status_code
                == 422
            )
        other = client.post("/projects", json={"name": "Other"}).json()["id"]
        assert (
            client.post(
                f"/projects/{other}/experiments",
                json={
                    "hypothesis": "No cross-project data",
                    "dataset_id": old["dataset_id"],
                },
            ).status_code
            == 409
        )


@pytest.mark.scientific
def test_real_registered_training_publishes_checked_candidate(
    database: Settings,
) -> None:
    from decisionguard.api.jobs import train_experiment

    with TestClient(create_app(database)) as client:
        _, experiment_id, run_id = queued_experiment(client)
        result = train_experiment(UUID(experiment_id), database)
        assert result["diagnostics"]["diagnostic_status"] == "INCOMPLETE"
        stored = client.get(f"/model-runs/{run_id}").json()
        assert stored["status"] == "SUCCEEDED"
        assert stored["model_status"] == "CANDIDATE_UNEVALUATED"
        assert stored["training_window"]["rows"] == 65
        assert stored["code_version"] == result["code_hash"]
        assert stored["finished_at"] is not None
        health = client.get(f"/model-runs/{run_id}/health")
        assert health.status_code == 200, health.text
        assert health.json()["decision_status"] == "BLOCK"
        assert health.json()["diagnostics"] == result["diagnostics"]
        with pytest.raises(ValueError, match="no queued"):
            train_experiment(UUID(experiment_id), database)


def test_failed_registered_training_cannot_publish_candidate(
    database: Settings,
) -> None:
    from decisionguard.api.jobs import train_experiment

    with TestClient(create_app(database)) as client:
        project_id, experiment_id, run_id = queued_experiment(client)
        dataset_id = client.get(f"/projects/{project_id}").json()["current_dataset_id"]
        engine = create_engine(database.database_url)
        with Session(engine) as session:
            dataset = session.get(Dataset, UUID(dataset_id))
            assert dataset is not None
            (
                artifact_path(database.artifact_root, dataset.artifact_uri)
                / "raw.parquet"
            ).write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="hash mismatch"):
            train_experiment(UUID(experiment_id), database)
        stored = client.get(f"/model-runs/{run_id}").json()
        assert stored["status"] == "FAILED"
        assert stored["model_status"] == "PENDING"
        assert stored["metrics"] is None
        assert stored["error_type"] == "ValueError"
        engine.dispose()


def test_second_worker_cannot_claim_running_job(
    database: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from typing import Any

    from decisionguard.api.jobs import train_experiment
    from decisionguard.models import bayesian

    started, release = Event(), Event()

    def blocked_fit(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(10), "test did not release blocked fit"
        raise ValueError("deliberate unit-test worker failure")

    monkeypatch.setattr(bayesian, "train_mmm", blocked_fit)
    with (
        TestClient(create_app(database)) as client,
        ThreadPoolExecutor(max_workers=1) as workers,
    ):
        _, experiment_id, run_id = queued_experiment(client)
        future = workers.submit(train_experiment, UUID(experiment_id), database)
        try:
            assert started.wait(10)
            assert client.get(f"/model-runs/{run_id}").json()["status"] == "RUNNING"
            with pytest.raises(ValueError, match="no queued"):
                train_experiment(UUID(experiment_id), database)
        finally:
            release.set()
        with pytest.raises(ValueError, match="deliberate unit-test"):
            future.result(timeout=10)
        assert client.get(f"/model-runs/{run_id}").json()["status"] == "FAILED"


def registered_evaluation_models(
    client: TestClient, settings: Settings, *, placebo_passes: bool = True
) -> tuple[UUID, UUID, Path]:
    """Unit artifact fixtures only: these bytes are never scientific training output."""
    import pandas as pd

    from decisionguard.evaluation.artifacts import source_evidence
    from decisionguard.evaluation.policy import assess_release
    from tests.test_release_artifacts import release_fixture

    project_id = client.post("/projects", json={"name": "Evaluation boundary"}).json()[
        "id"
    ]
    data = client.post(f"/projects/{project_id}/datasets/generate", json={}).json()
    template, model_path = release_fixture(
        settings.artifact_root,
        placebo_passes=placebo_passes,
        channels=["search_spend", "meta_spend"],
    )
    identifiers: list[UUID] = []
    engine = create_engine(settings.database_url)
    try:
        with Session(engine) as session, session.begin():
            dataset = session.get(Dataset, UUID(data["id"]))
            assert dataset is not None
            dataset_path = artifact_path(settings.artifact_root, dataset.artifact_uri)
            for path in (model_path, settings.artifact_root / "alternative"):
                config = asdict(
                    MMMConfig(
                        draws=8,
                        tune=16,
                        channels=("search_spend", "meta_spend"),
                        media_prior_mean=0.15 if path == model_path else 0.25,
                    )
                )
                response = client.post(
                    f"/projects/{project_id}/experiments",
                    json={
                        "hypothesis": "Unit lifecycle fixture",
                        "configuration": config,
                    },
                )
                assert response.status_code == 201, response.text
                record = json.loads((path / "model.json").read_text())
                record["config"] = {**config, "sampler": "nutpie", "cores": 1}
                record["dataset"] = dataset_evidence(
                    dataset_path, load_model_ready(dataset_path)
                )
                write_json(path / "model.json", record)
                run = ModelRun(
                    id=uuid4(),
                    experiment_id=UUID(response.json()["id"]),
                    dataset_id=dataset.id,
                    model_family=record["model_family"],
                    configuration=config,
                    artifact_uri=path.name,
                    status="SUCCEEDED",
                    model_status="CANDIDATE_UNEVALUATED",
                    record_hash=file_hash(path / "model.json"),
                    training_window=record["training_window"],
                )
                session.add(run)
                identifiers.append(run.id)
    finally:
        engine.dispose()
    evidence = json.loads((template / "evaluation.json").read_text())
    evidence["model_record_hash"] = file_hash(model_path / "model.json")
    evidence["progress_identity"]["sensitivity_record_hash"] = file_hash(
        settings.artifact_root / "alternative" / "model.json"
    )
    evidence["policy"] = asdict(
        assess_release(
            source_evidence(
                pd.read_parquet(template / "mmm_eval_raw.parquet"),
                record["config"]["channels"],
            ),
            record["diagnostics"],
            record["dataset"]["quality_status"],
            record["config"]["channels"],
            prior_sensitivity=dict.fromkeys(record["config"]["channels"], 0.0),
        )
    )
    write_json(template / "evaluation.json", evidence)
    return identifiers[0], identifiers[1], template


def stub_evaluator(monkeypatch: pytest.MonkeyPatch, template: Path) -> None:
    """Substitute only the expensive optional runtime, keeping real integrity/policy."""

    def evaluate(
        model: Path, alternative: Path, output: Path, **kwargs: Any
    ) -> dict[str, Any]:
        assert file_hash(model / "model.json")
        assert file_hash(alternative / "model.json")
        assert kwargs["dataset"].is_dir()
        shutil.copytree(template, output)
        record = json.loads((output / "evaluation.json").read_text())
        record["configuration"] = {
            "fit": {key: kwargs[key] for key in ("draws", "tune")}
        }
        write_json(output / "evaluation.json", record)
        return record

    module = ModuleType("decisionguard.evaluation.mmm_eval")
    monkeypatch.setattr(module, "evaluate_mmm", evaluate, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize("placebo_passes", [True, False])
def test_registered_evaluation_publishes_verified_policy_and_survives_restart(
    database: Settings, monkeypatch: pytest.MonkeyPatch, placebo_passes: bool
) -> None:
    from decisionguard.api.jobs import evaluate_model

    with TestClient(create_app(database)) as client:
        model_id, alternative_id, template = registered_evaluation_models(
            client, database, placebo_passes=placebo_passes
        )
        health = client.get(f"/model-runs/{model_id}/health").json()
        assert health["decision_status"] == "BLOCK" and health["policy"] is None
        body = {"sensitivity_run_id": str(alternative_id), "draws": 10, "tune": 20}
        queued = client.post(f"/model-runs/{model_id}/evaluate", json=body)
        assert queued.status_code == 202, queued.text
        assert (
            client.post(f"/model-runs/{model_id}/evaluate", json=body).status_code
            == 409
        )
        assert (
            client.get(f"/model-runs/{model_id}/health").json()["evaluation"]["status"]
            == "QUEUED"
        )
        stub_evaluator(monkeypatch, template)
        result = evaluate_model(model_id, database)
        expected = "WARN" if placebo_passes else "BLOCK"
        assert result["policy"]["state"] == expected
        with pytest.raises(ValueError, match="no queued"):
            evaluate_model(model_id, database)
    with TestClient(create_app(database)) as restarted:
        health = restarted.get(f"/model-runs/{model_id}/health").json()
        assert health["decision_status"] == expected
        assert health["policy"] == result["policy"]
        assert health["evaluation"]["status"] == "SUCCEEDED"
        assert health["model_status"] == "CANDIDATE_UNEVALUATED"
        # A newer incomplete evaluation cannot silently fall back to an old PASS.
        assert (
            restarted.post(f"/model-runs/{model_id}/evaluate", json=body).status_code
            == 202
        )
        assert (
            restarted.get(f"/model-runs/{model_id}/health").json()["decision_status"]
            == "BLOCK"
        )


@pytest.mark.parametrize("target", ["raw", "model", "evaluation"])
def test_registered_health_rejects_changed_evidence(
    database: Settings, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from decisionguard.api.jobs import evaluate_model

    with TestClient(create_app(database)) as client:
        model_id, alternative_id, template = registered_evaluation_models(
            client, database
        )
        client.post(
            f"/model-runs/{model_id}/evaluate",
            json={"sensitivity_run_id": str(alternative_id)},
        )
        stub_evaluator(monkeypatch, template)
        evaluate_model(model_id, database)
        engine = create_engine(database.database_url)
        try:
            with Session(engine) as session:
                evaluation = session.scalar(
                    select(EvaluationRun).where(EvaluationRun.model_run_id == model_id)
                )
                assert evaluation is not None
                path = artifact_path(database.artifact_root, evaluation.artifact_uri)
            if target == "raw":
                (path / "mmm_eval_raw.parquet").write_bytes(b"changed")
            elif target == "evaluation":
                record = json.loads((path / "evaluation.json").read_text())
                record["refits"] = []
                write_json(path / "evaluation.json", record)
            else:
                record_path = database.artifact_root / "model" / "model.json"
                record = json.loads(record_path.read_text())
                record["channels"]["search_spend"]["roi_mean"] = 5
                write_json(record_path, record)
            assert client.get(f"/model-runs/{model_id}/health").status_code == 409
        finally:
            engine.dispose()


def test_budget_scenario_persists_constraints_and_evidence(
    database: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from decisionguard.api.jobs import evaluate_model

    with TestClient(create_app(database)) as client:
        model_id, alternative_id, template = registered_evaluation_models(
            client, database
        )
        client.post(
            f"/model-runs/{model_id}/evaluate",
            json={"sensitivity_run_id": str(alternative_id)},
        )
        stub_evaluator(monkeypatch, template)
        evaluate_model(model_id, database)
        body = {
            "total": 200,
            "current": {"search_spend": 100, "meta_spend": 100},
            "floors": {"meta_spend": 80},
            "caps": {"search_spend": 150, "meta_spend": 130},
            "max_movement": {"search_spend": 0.1, "meta_spend": 0.2},
            "protected_spend": {"meta_spend": 95},
            "horizon_weeks": 26,
            "risk_policy": "expected",
        }
        response = client.post(f"/model-runs/{model_id}/scenarios", json=body)
        assert response.status_code == 201, response.text
        scenario = response.json()
        assert scenario["budget"] == 200
        assert scenario["constraints"] == {
            key: value
            for key, value in body.items()
            if key not in {"horizon_weeks", "risk_policy"}
        }
        assert scenario["effective_bounds"]["search_spend"] == pytest.approx([90, 110])
        assert scenario["effective_bounds"]["meta_spend"] == pytest.approx([95, 120])
        assert scenario["release"]["state"] == "WARN"
        assert len(scenario["evaluation_record_hash"]) == 64
        assert len(scenario["code_version"]) == 64
    with TestClient(create_app(database)) as restarted:
        assert restarted.get(f"/scenarios/{scenario['id']}").json() == scenario
        # Creation evidence remains inspectable when a newer evaluation is pending.
        restarted.post(
            f"/model-runs/{model_id}/evaluate",
            json={"sensitivity_run_id": str(alternative_id)},
        )
        assert (
            restarted.post(f"/model-runs/{model_id}/scenarios", json=body).status_code
            == 409
        )
        assert restarted.get(f"/scenarios/{scenario['id']}").json() == scenario


def test_scenarios_reject_blocked_and_changed_evidence(
    database: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from decisionguard.api.jobs import evaluate_model

    with TestClient(create_app(database)) as client:
        model_id, alternative_id, template = registered_evaluation_models(
            client, database, placebo_passes=False
        )
        body = {"total": 200, "current": {"search_spend": 100, "meta_spend": 100}}
        assert (
            client.post(f"/model-runs/{model_id}/scenarios", json=body).status_code
            == 409
        )
        client.post(
            f"/model-runs/{model_id}/evaluate",
            json={"sensitivity_run_id": str(alternative_id)},
        )
        stub_evaluator(monkeypatch, template)
        evaluate_model(model_id, database)
        blocked = client.post(f"/model-runs/{model_id}/scenarios", json=body)
        assert blocked.status_code == 409 and "BLOCK" in blocked.text
        (database.artifact_root / "model" / "model.json").write_text("{}")
        changed = client.post(f"/model-runs/{model_id}/scenarios", json=body)
        assert changed.status_code == 409 and "integrity" in changed.text
        engine = create_engine(database.database_url)
        try:
            with Session(engine) as session:
                assert session.scalar(select(Scenario.id)) is None
        finally:
            engine.dispose()


def test_scenario_contract_rejects_invalid_and_infeasible_requests(
    database: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from decisionguard.api.jobs import evaluate_model

    with TestClient(create_app(database)) as client:
        model_id, alternative_id, template = registered_evaluation_models(
            client, database
        )
        client.post(
            f"/model-runs/{model_id}/evaluate",
            json={"sensitivity_run_id": str(alternative_id)},
        )
        stub_evaluator(monkeypatch, template)
        evaluate_model(model_id, database)
        body = {"total": 200, "current": {"search_spend": 100, "meta_spend": 100}}
        for invalid in (
            {"total": 0},
            {"total": True},
            {"total": "200"},
            {"current": {"search_spend": "100", "meta_spend": 100}},
            {"caps": {"search_spend": -1}},
            {"current": {"search_spend": True, "meta_spend": 100}},
            {"current": {"search_spend": 100}},
            {"caps": {"unknown": 100}},
            {"floors": {"search_spend": 201}},
            {"caps": {"search_spend": 10, "meta_spend": 10}},
            {"protected_spend": {"meta_spend": 110}, "caps": {"meta_spend": 100}},
            {"max_movement": {"meta_spend": -0.1}},
            {"horizon_weeks": 53},
            {"horizon_weeks": True},
            {"risk_policy": "unlimited"},
            {"execute": "shell"},
        ):
            response = client.post(
                f"/model-runs/{model_id}/scenarios", json=body | invalid
            )
            assert response.status_code == 422, response.text
        assert (
            client.post(f"/model-runs/{uuid4()}/scenarios", json=body).status_code
            == 404
        )
        assert client.get(f"/scenarios/{uuid4()}").status_code == 404
        # Valid JSON can still overflow the native floating-point representation.
        response = client.post(
            f"/model-runs/{model_id}/scenarios",
            content='{"total":1e400,"current":{"search_spend":100,"meta_spend":100}}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "finite_number"
        assert "input" not in response.json()["detail"][0]


def test_scenario_bounds_include_verified_channel_restrictions(
    database: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    from decisionguard.api.jobs import evaluate_model
    from decisionguard.evaluation.artifacts import source_evidence
    from decisionguard.evaluation.policy import assess_release

    with TestClient(create_app(database)) as client:
        model_id, alternative_id, template = registered_evaluation_models(
            client, database
        )
        raw_path = template / "mmm_eval_raw.parquet"
        raw = pd.read_parquet(raw_path)
        selected = (raw["test_name"] == "perturbation") & raw[
            "specific_metric_name"
        ].str.endswith("_search_spend")
        raw.loc[selected, "metric_pass"] = False
        raw.loc[selected, "metric_value"] = 10.0
        raw.to_parquet(raw_path, index=False)
        model = json.loads(
            (database.artifact_root / "model" / "model.json").read_text()
        )
        record = json.loads((template / "evaluation.json").read_text())
        record["raw_artifact_hash"] = file_hash(raw_path)
        record["policy"] = asdict(
            assess_release(
                source_evidence(raw, model["config"]["channels"]),
                model["diagnostics"],
                model["dataset"]["quality_status"],
                model["config"]["channels"],
                prior_sensitivity=dict.fromkeys(model["config"]["channels"], 0.0),
            )
        )
        write_json(template / "evaluation.json", record)
        client.post(
            f"/model-runs/{model_id}/evaluate",
            json={"sensitivity_run_id": str(alternative_id)},
        )
        stub_evaluator(monkeypatch, template)
        evaluate_model(model_id, database)
        body = {
            "total": 200,
            "current": {"search_spend": 100, "meta_spend": 100},
            "max_movement": {"search_spend": 0.5},
        }
        response = client.post(f"/model-runs/{model_id}/scenarios", json=body)
        assert response.status_code == 201, response.text
        assert response.json()["release"]["state"] == "RESTRICT"
        assert response.json()["effective_bounds"]["search_spend"] == [95, 105]
        infeasible = client.post(
            f"/model-runs/{model_id}/scenarios",
            json=body | {"floors": {"search_spend": 110}},
        )
        assert infeasible.status_code == 422


def test_registered_evaluation_failure_and_exclusive_worker(
    database: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from decisionguard.api.jobs import evaluate_model

    started, release = Event(), Event()

    def blocked_evaluate(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(10)
        raise ValueError("deliberate evaluation boundary failure")

    module = ModuleType("decisionguard.evaluation.mmm_eval")
    monkeypatch.setattr(module, "evaluate_mmm", blocked_evaluate, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with (
        TestClient(create_app(database)) as client,
        ThreadPoolExecutor(max_workers=1) as workers,
    ):
        model_id, alternative_id, _ = registered_evaluation_models(client, database)
        client.post(
            f"/model-runs/{model_id}/evaluate",
            json={"sensitivity_run_id": str(alternative_id)},
        )
        future = workers.submit(evaluate_model, model_id, database)
        try:
            assert started.wait(10)
            assert (
                client.get(f"/model-runs/{model_id}/health").json()["evaluation"][
                    "status"
                ]
                == "RUNNING"
            )
            with pytest.raises(ValueError, match="no queued"):
                evaluate_model(model_id, database)
        finally:
            release.set()
        with pytest.raises(ValueError, match="boundary failure"):
            future.result(timeout=10)
        health = client.get(f"/model-runs/{model_id}/health").json()
        assert health["decision_status"] == "BLOCK"
        assert health["evaluation"]["status"] == "FAILED"
        assert health["evaluation"]["summary"] is None
        assert health["evaluation"]["error_type"] == "ValueError"


def test_evaluation_request_rejects_invalid_and_incompatible_inputs(
    database: Settings,
) -> None:
    with TestClient(create_app(database)) as client:
        model_id, alternative_id, _ = registered_evaluation_models(client, database)
        body = {"sensitivity_run_id": str(alternative_id)}
        for invalid in (
            {"draws": 0},
            {"draws": True},
            {"tune": "20"},
            {"tune": 20001},
            {"execute": "shell"},
        ):
            assert (
                client.post(
                    f"/model-runs/{model_id}/evaluate", json=body | invalid
                ).status_code
                == 422
            )
        assert (
            client.post(f"/model-runs/{uuid4()}/evaluate", json=body).status_code == 404
        )
        assert (
            client.post(
                f"/model-runs/{model_id}/evaluate",
                json={"sensitivity_run_id": str(model_id)},
            ).status_code
            == 409
        )
        _, _, queued_id = queued_experiment(client)
        assert (
            client.post(f"/model-runs/{queued_id}/evaluate", json=body).status_code
            == 409
        )
        # Rewriting a model and its self-declared checksums cannot replace DB identity.
        record_path = database.artifact_root / "alternative" / "model.json"
        record = json.loads(record_path.read_text())
        record["config"]["media_prior_mean"] = 0.5
        write_json(record_path, record)
        assert (
            client.post(f"/model-runs/{model_id}/evaluate", json=body).status_code
            == 409
        )


@pytest.mark.parametrize("target", ["dataset", "model", "policy"])
def test_evaluation_cannot_publish_changed_inputs_or_fabricated_policy(
    database: Settings, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from decisionguard.api.jobs import evaluate_model

    with TestClient(create_app(database)) as client:
        model_id, alternative_id, template = registered_evaluation_models(
            client, database
        )
        response = client.post(
            f"/model-runs/{model_id}/evaluate",
            json={"sensitivity_run_id": str(alternative_id)},
        )
        assert response.status_code == 202
        stub_evaluator(monkeypatch, template)
        if target == "dataset":
            model = json.loads(
                (database.artifact_root / "model" / "model.json").read_text()
            )
            (Path(model["dataset"]["path"]) / "raw.parquet").write_bytes(b"changed")
        else:
            path = (
                (database.artifact_root / "model" / "model.json")
                if target == "model"
                else template / "evaluation.json"
            )
            record = json.loads(path.read_text())
            if target == "model":
                record["config"]["seed"] = 100
            else:
                record["policy"]["state"] = "PASS"
            write_json(path, record)
        with pytest.raises(ValueError):
            evaluate_model(model_id, database)
        engine = create_engine(database.database_url)
        try:
            with Session(engine) as session:
                evaluation = session.get(EvaluationRun, UUID(response.json()["id"]))
                assert evaluation is not None
                assert evaluation.status == "FAILED"
                assert evaluation.summary is None and evaluation.record_hash is None
                assert evaluation.error_type == "ValueError"
        finally:
            engine.dispose()
