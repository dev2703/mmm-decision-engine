"""Shared verified candidate comparison for dashboard and agent reads."""

from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from decisionguard.api.database import Dataset, Experiment, ModelRun
from decisionguard.api.evaluations import load_registered_model


def model_comparison(
    sessions: sessionmaker[Session],
    root: Path,
    project_id: UUID,
    *,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    with sessions() as session:
        selection = (
            select(ModelRun).join(Experiment).where(Experiment.project_id == project_id)
        )
        total = session.scalar(select(func.count()).select_from(selection.subquery()))
        models = session.scalars(
            selection.order_by(ModelRun.created_at.desc(), ModelRun.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()
        items: list[dict[str, Any]] = []
        for model in models:
            record = (
                load_registered_model(model, root)
                if model.status == "SUCCEEDED"
                else None
            )
            if record is not None:
                dataset = session.get(Dataset, model.dataset_id)
                if dataset is None or dataset.hash != record["dataset"]["hash"]:
                    raise ValueError("Dataset differs from registration")
            items.append(
                {
                    "model_run_id": str(model.id),
                    "dataset_id": str(model.dataset_id),
                    "job_status": model.status,
                    "model_record_hash": model.record_hash,
                    "training_window": record["training_window"] if record else None,
                    "dataset_hash": record["dataset"]["hash"] if record else None,
                    "configuration": record["config"]
                    if record
                    else model.configuration,
                    "prediction_context": record.get("prediction_context")
                    if record
                    else None,
                    "posterior_predictive": record.get("posterior_predictive")
                    if record
                    else None,
                    "limitations": record.get("limitations", []) if record else [],
                    "health_url": f"/model-runs/{model.id}/health",
                }
            )
    return {
        "items": items,
        "total": total,
        "offset": offset,
        "next_offset": offset + len(items)
        if total is not None and offset + len(items) < total
        else None,
        "ranking": (
            "not_performed; compare matching data, windows and prediction context"
        ),
        "decision_use": "requires_current_health_and_scenario_validation",
    }
