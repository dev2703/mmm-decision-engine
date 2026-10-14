"""Real database evidence and deployment access boundaries for the dashboard."""

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from decisionguard.api.app import create_app
from decisionguard.api.database import Project
from decisionguard.api.jobs import evaluate_model
from decisionguard.config import Settings
from decisionguard.demo import seed_demo
from tests.test_api import database, registered_evaluation_models, stub_evaluator

__all__ = ["database"]


def test_production_requires_access_control(monkeypatch: pytest.MonkeyPatch) -> None:
    for token in (None, "short"):
        with pytest.raises(ValueError, match="API_TOKEN"):
            Settings(
                "postgresql+psycopg:///demo", environment="production", api_token=token
            )
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg:///demo")
    monkeypatch.setenv("READ_ONLY", "tru")
    with pytest.raises(ValueError, match="READ_ONLY"):
        Settings.from_environment()


def test_auth_readonly_and_readiness(database: Settings) -> None:
    settings = replace(
        database, api_token="a" * 32, environment="production", read_only=True
    )
    assert settings.api_token is not None
    assert settings.api_token not in repr(settings)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        settings.artifact_root.mkdir()
        assert client.get("/ready").status_code == 200
        response = client.get("/projects")
        assert response.status_code == 401 and response.headers["X-Correlation-ID"]
        assert client.get("/openapi.json").status_code == 401
        assert (
            client.get(
                "/capabilities", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        client.headers["Authorization"] = "Bearer " + settings.api_token
        assert client.get("/capabilities").json() == {
            "read_only": True,
            "agent_enabled": False,
        }
        assert client.get("/projects").json() == []
        assert (
            client.post("/projects", json={"name": "Cannot write"}).status_code == 403
        )
        assert client.post("/agent/query", json={}).status_code == 403


def test_project_dataset_discovery_and_genuine_seed(database: Settings) -> None:
    seeded = seed_demo(database)
    with TestClient(create_app(database)) as client:
        project_id = seeded["project_id"]
        other = client.post("/projects", json={"name": "Separate workspace"}).json()
        assert len(client.get("/projects").json()) == 2
        assert len(client.get("/projects?offset=1&limit=1").json()) == 1
        datasets = client.get(f"/projects/{project_id}/datasets").json()
        assert len(datasets) == 1 and datasets[0]["id"] == seeded["dataset_id"]
        assert client.get(f"/projects/{other['id']}/datasets").json() == []
        assert client.get(f"/projects/{uuid4()}/datasets").status_code == 404
        quality = client.get(f"/datasets/{seeded['dataset_id']}/quality").json()
        assert quality["clean_hash"] == datasets[0]["hash"]
        assert quality["issues"] and quality["profile"]["row_count"] > 150
        assert seeded["model_run_ids"] == []


def test_comparison_and_evaluation_reject_changed_artifacts(
    database: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(create_app(database)) as client:
        model, alternative, template = registered_evaluation_models(client, database)
        project_id = client.get("/projects").json()[0]["id"]
        comparison = client.get(
            f"/projects/{project_id}/model-comparison?limit=1"
        ).json()
        assert comparison["total"] == 2 and comparison["next_offset"] == 1
        assert comparison["items"][0]["model_record_hash"]
        queued = client.post(
            f"/model-runs/{model}/evaluate",
            json={"sensitivity_run_id": str(alternative)},
        ).json()
        url = f"/evaluation-runs/{queued['id']}/evidence"
        assert client.get(url).status_code == 409
        stub_evaluator(monkeypatch, template)
        evaluate_model(model, database)
        evidence = client.get(url)
        assert evidence.status_code == 200, evidence.text
        assert evidence.json()["metrics"] and evidence.json()["record_hash"]
        assert (
            evidence.json()["policy"]
            == client.get(f"/model-runs/{model}/health").json()["policy"]
        )
        (database.artifact_root / "model" / "model.json").write_text("{}")
        assert client.get(url).status_code == 409
        assert client.get(f"/projects/{project_id}/model-comparison").status_code == 409


def test_demo_import_rolls_back_on_changed_identity(database: Settings) -> None:
    with TestClient(create_app(database)) as client:
        _, _, template = registered_evaluation_models(client, database)
        dataset_id = client.get("/projects").json()[0]["current_dataset_id"]
        project_id = client.get("/projects").json()[0]["id"]
        path = database.artifact_root / f"projects/{project_id}/datasets/{dataset_id}"
        (database.artifact_root / "model" / "predictions.parquet").write_bytes(
            b"changed"
        )
        with pytest.raises(ValueError, match="artifact hash"):
            seed_demo(
                database,
                dataset=path,
                model=database.artifact_root / "model",
                sensitivity=database.artifact_root / "alternative",
                evaluation=template,
            )
    engine = create_engine(database.database_url)
    try:
        with Session(engine) as session:
            assert session.scalar(select(func.count()).select_from(Project)) == 1
    finally:
        engine.dispose()
    with pytest.raises(ValueError, match="inside ARTIFACT_ROOT"):
        seed_demo(database, dataset=Path("/outside"))
