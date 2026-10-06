"""Real PostgreSQL migrations, transaction behavior and HTTP data contracts."""

import os
from collections.abc import Iterator
from pathlib import Path
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
from decisionguard.api.database import Dataset, Project
from decisionguard.config import Settings


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
    assert {"projects", "datasets", "alembic_version"} <= set(
        inspect(engine).get_table_names()
    )
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
