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
