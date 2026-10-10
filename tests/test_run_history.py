"""Real PostgreSQL pagination, project isolation and job lineage contracts."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

from decisionguard.api.app import create_app
from decisionguard.api.database import (
    EvaluationRun,
    ModelRun,
    OptimizationRun,
    Scenario,
)
from decisionguard.config import Settings
from tests.test_api import database as database
from tests.test_api import queued_experiment


@pytest.fixture
def history(database: Settings) -> dict[str, str]:
    """Metadata-only jobs; no fitted model or release approval is claimed."""
    with TestClient(create_app(database)) as client:
        project, experiment, model = queued_experiment(client)
        other, _, foreign = queued_experiment(client)
    engine = create_engine(database.database_url)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    scenario_id = uuid4()
    try:
        with Session(engine) as session, session.begin():
            run = session.get(ModelRun, UUID(model))
            assert run is not None
            dataset_id = str(run.dataset_id)
            run.status = "SUCCEEDED"
            run.created_at = started
            run.started_at = started + timedelta(minutes=1)
            run.finished_at = started + timedelta(minutes=2)
            run.record_hash = "unit-model-record"
            # UUIDs are table-local: equal IDs across job kinds must not collapse.
            evaluation = EvaluationRun(
                id=UUID(model),
                model_run_id=run.id,
                sensitivity_run_id=run.id,
                evaluator="mmm-eval",
                configuration={},
                artifact_uri="unit-evaluation",
                status="SUCCEEDED",
                summary={"state": "BLOCK"},
                record_hash="unit-evaluation-record",
                created_at=started + timedelta(minutes=10),
            )
            session.add(evaluation)
            session.flush()
            session.add(
                Scenario(
                    id=scenario_id,
                    model_run_id=run.id,
                    evaluation_run_id=evaluation.id,
                    evaluation_record_hash=evaluation.record_hash,
                    budget=100,
                    constraints={},
                    effective_bounds={},
                    release={"state": "WARN"},
                    horizon_weeks=13,
                    risk_policy="conservative",
                    code_version="unit-history-fixture",
                )
            )
            session.flush()
            session.add_all(
                [
                    EvaluationRun(
                        id=UUID(int=101),
                        model_run_id=run.id,
                        sensitivity_run_id=run.id,
                        evaluator="mmm-eval",
                        configuration={},
                        artifact_uri="unit-new-evaluation",
                        status="QUEUED",
                        created_at=started + timedelta(minutes=40),
                    ),
                    OptimizationRun(
                        id=UUID(int=102),
                        scenario_id=scenario_id,
                        configuration={},
                        artifact_uri="unit-failed-optimization",
                        status="FAILED",
                        error_type="ValueError",
                        created_at=started + timedelta(minutes=30),
                        started_at=started + timedelta(minutes=31),
                        finished_at=started + timedelta(minutes=32),
                    ),
                    OptimizationRun(
                        id=UUID(int=103),
                        scenario_id=scenario_id,
                        configuration={},
                        artifact_uri="unit-running-optimization",
                        status="RUNNING",
                        created_at=started + timedelta(minutes=20),
                    ),
                ]
            )
    finally:
        engine.dispose()
    return {
        "project": project,
        "experiment": experiment,
        "dataset": dataset_id,
        "model": model,
        "scenario": str(scenario_id),
        "other": other,
        "foreign": foreign,
    }


def test_history_links_all_job_types_and_keeps_scientific_authority_separate(
    database: Settings, history: dict[str, str]
) -> None:
    url = f"/projects/{history['project']}/runs"
    with TestClient(create_app(database)) as client:
        # Moving the project's current dataset must not rewrite historical lineage.
        client.post(f"/projects/{history['project']}/datasets/generate", json={})
        response = client.get(url)
        assert response.status_code == 200, response.text
        result = response.json()
        items = result["items"]
        assert [(item["kind"], item["status"]) for item in items] == [
            ("evaluation", "QUEUED"),
            ("optimization", "FAILED"),
            ("optimization", "RUNNING"),
            ("evaluation", "SUCCEEDED"),
            ("training", "SUCCEEDED"),
        ]
        assert result["next_offset"] is None
        for item in items:
            assert item["model_run_id"] == history["model"]
            assert item["dataset_id"] == history["dataset"]
            assert item["experiment_id"] == history["experiment"]
            assert item["id"] != history["foreign"]
            assert (
                not {"summary", "release", "decision_status", "artifact_uri"}
                & item.keys()
            )
        for item in items[1:3]:
            assert item["scenario_id"] == history["scenario"]
            assert item["evaluation_run_id"] == history["model"]
        assert items[0]["evaluation_run_id"] == items[0]["id"]
        assert items[0]["record_hash"] is None
        assert items[1]["error_type"] == "ValueError"
        assert items[1]["started_at"] < items[1]["finished_at"]
        assert items[-1]["evaluation_run_id"] is None
        assert items[-1]["scenario_id"] is None
        assert items[-1]["record_hash"] == "unit-model-record"
        other = client.get(f"/projects/{history['other']}/runs").json()["items"]
        assert len(other) == 1 and other[0]["id"] == history["foreign"]
    with TestClient(create_app(database)) as restarted:
        assert restarted.get(url).json() == result


def test_pagination_orders_across_job_types_with_timestamp_and_uuid_ties(
    database: Settings, history: dict[str, str]
) -> None:
    engine = create_engine(database.database_url)
    try:
        with engine.begin() as connection:
            for table in (ModelRun, EvaluationRun, OptimizationRun):
                connection.execute(
                    update(table).values(created_at=datetime(2026, 1, 1, tzinfo=UTC))
                )
    finally:
        engine.dispose()
    with TestClient(create_app(database)) as client:
        url = f"/projects/{history['project']}/runs"
        all_items = client.get(url).json()["items"]
        seen: list[tuple[str, str]] = []
        offset: int | None = 0
        while offset is not None:
            page = client.get(url, params={"offset": offset, "limit": 1}).json()
            assert len(page["items"]) == 1
            seen.append((page["items"][0]["kind"], page["items"][0]["id"]))
            offset = page["next_offset"]
            assert len(seen) <= 5
        assert seen == [(item["kind"], item["id"]) for item in all_items]
        assert len(set(seen)) == 5
        assert seen.index(("training", history["model"])) + 1 == seen.index(
            ("evaluation", history["model"])
        )
        empty = client.get(url, params={"offset": 5}).json()
        assert empty["items"] == [] and empty["next_offset"] is None


def test_history_filters_before_pagination_and_reflects_job_transitions(
    database: Settings, history: dict[str, str]
) -> None:
    url = f"/projects/{history['project']}/runs"
    with TestClient(create_app(database)) as client:
        for kind, count in (("training", 1), ("evaluation", 2), ("optimization", 2)):
            page = client.get(url, params={"kind": kind}).json()
            assert len(page["items"]) == count
            assert all(item["kind"] == kind for item in page["items"])
        for status, count in (
            ("QUEUED", 1),
            ("RUNNING", 1),
            ("FAILED", 1),
            ("SUCCEEDED", 2),
        ):
            page = client.get(url, params={"status": status}).json()
            assert len(page["items"]) == count
            assert all(item["status"] == status for item in page["items"])
        page = client.get(
            url, params={"kind": "optimization", "limit": 1, "offset": 1}
        ).json()
        assert page["items"][0]["status"] == "RUNNING"
        assert page["next_offset"] is None
        page = client.get(url, params={"kind": "training", "status": "FAILED"}).json()
        assert page["items"] == [] and page["next_offset"] is None
        engine = create_engine(database.database_url)
        try:
            with engine.begin() as connection:
                connection.execute(
                    update(OptimizationRun)
                    .where(OptimizationRun.id == UUID(int=103))
                    .values(status="FAILED", error_type="RuntimeError")
                )
        finally:
            engine.dispose()
        assert client.get(url, params={"status": "RUNNING"}).json()["items"] == []
        failed = client.get(
            url, params={"kind": "optimization", "status": "FAILED"}
        ).json()
        assert len(failed["items"]) == 2


def test_history_empty_unknown_project_and_invalid_filters(database: Settings) -> None:
    with TestClient(create_app(database)) as client:
        project = client.post("/projects", json={"name": "Empty"}).json()["id"]
        url = f"/projects/{project}/runs"
        empty = client.get(url)
        assert empty.status_code == 200
        assert empty.json()["items"] == [] and empty.json()["next_offset"] is None
        assert client.get(f"/projects/{uuid4()}/runs").status_code == 404
        for query in (
            "kind=unknown",
            "status=PASS",
            "status=BLOCK",
            "offset=-1",
            f"offset={2**63}",
            "limit=0",
            "limit=201",
        ):
            assert client.get(url + "?" + query).status_code == 422
