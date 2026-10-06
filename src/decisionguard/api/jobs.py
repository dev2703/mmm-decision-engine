"""CLI-owned heavy work with short claim and publication transactions."""

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from decisionguard.api.app import artifact_path
from decisionguard.api.database import Dataset, ModelRun
from decisionguard.config import Settings
from decisionguard.data.artifacts import load_model_ready
from decisionguard.data.integrity import dataset_hash
from decisionguard.models.artifacts import load_mmm_record
from decisionguard.models.config import MMMConfig


def train_experiment(experiment_id: UUID, settings: Settings) -> dict[str, Any]:
    """A second worker cannot claim a RUNNING run; failures never publish results."""
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    run_id: UUID | None = None
    try:
        with Session(engine) as session, session.begin():
            run = session.scalar(
                select(ModelRun)
                .where(
                    ModelRun.experiment_id == experiment_id, ModelRun.status == "QUEUED"
                )
                .order_by(ModelRun.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if run is None:
                raise ValueError("no queued model run available for this experiment")
            dataset = session.get(Dataset, run.dataset_id)
            if dataset is None:
                raise ValueError("model run dataset missing")
            run_id = run.id
            configuration = dict(run.configuration)
            dataset_path = artifact_path(settings.artifact_root, dataset.artifact_uri)
            expected_hash = dataset.hash
            output = artifact_path(settings.artifact_root, run.artifact_uri)
            run.status = "RUNNING"
            run.started_at = datetime.now(UTC)
        # No open transaction while sampling, and no scientific imports at API startup.
        if dataset_hash(load_model_ready(dataset_path)) != expected_hash:
            raise ValueError("registered dataset identity changed")
        from decisionguard.models.bayesian import train_mmm

        train_mmm(dataset_path, output, MMMConfig(**configuration))
        result = load_mmm_record(output)
        if result["dataset"]["hash"] != expected_hash:
            raise ValueError("completed model uses a different dataset")
        record_hash = sha256((output / "model.json").read_bytes()).hexdigest()
        with Session(engine) as session, session.begin():
            run = session.get(ModelRun, run_id, with_for_update=True)
            if run is None or run.status != "RUNNING":
                raise ValueError("model run no longer belongs to this worker")
            run.status = "SUCCEEDED"
            run.model_status = result["model_status"]
            run.code_version = result["code_hash"]
            run.record_hash = record_hash
            run.training_window = result["training_window"]
            run.metrics = result["posterior_predictive"]
            run.finished_at = datetime.now(UTC)
        return result
    except Exception as error:
        if run_id is not None:
            with Session(engine) as session, session.begin():
                run = session.get(ModelRun, run_id, with_for_update=True)
                if run is not None and run.status == "RUNNING":
                    run.status = "FAILED"
                    run.error_type = type(error).__name__
                    run.finished_at = datetime.now(UTC)
        raise
    finally:
        engine.dispose()
