"""Bind local evaluation evidence to PostgreSQL's registered identities."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from decisionguard.api.database import EvaluationRun, ModelRun
from decisionguard.config import artifact_path
from decisionguard.data.artifacts import file_hash
from decisionguard.evaluation.artifacts import load_release
from decisionguard.models.artifacts import load_mmm_record


def latest_evaluation(session: Session, model_run_id: UUID) -> EvaluationRun | None:
    """Latest request governs decisions, including pending/failed requests."""
    return session.scalar(
        select(EvaluationRun)
        .where(EvaluationRun.model_run_id == model_run_id)
        .order_by(EvaluationRun.created_at.desc(), EvaluationRun.id.desc())
        .limit(1)
    )


def registered_model_path(run: ModelRun, root: Path) -> Path:
    path = artifact_path(root, run.artifact_uri)
    if run.status != "SUCCEEDED" or file_hash(path / "model.json") != run.record_hash:
        raise ValueError("registered model is incomplete or changed")
    return path


def load_registered_model(run: ModelRun, root: Path) -> dict[str, Any]:
    record = load_mmm_record(registered_model_path(run, root))
    if {
        key: value
        for key, value in record["config"].items()
        if key not in {"sampler", "cores"}
    } != run.configuration or record["training_window"] != run.training_window:
        raise ValueError("model configuration or window differs from registration")
    return record


def verified_evaluation(
    evaluation: EvaluationRun, model: ModelRun, alternative: ModelRun, root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = registered_model_path(model, root)
    alternative_path = registered_model_path(alternative, root)
    if (
        model.record_hash != evaluation.configuration["model_record_hash"]
        or alternative.record_hash
        != evaluation.configuration["sensitivity_record_hash"]
    ):
        raise ValueError("registered evaluation inputs changed")
    path = artifact_path(root, evaluation.artifact_uri)
    if evaluation.status == "SUCCEEDED" and (
        evaluation.record_hash is None or evaluation.summary is None
    ):
        raise ValueError("successful evaluation lacks registered evidence")
    if (
        evaluation.record_hash is not None
        and file_hash(path / "evaluation.json") != evaluation.record_hash
    ):
        raise ValueError("registered evaluation record changed")
    record = json.loads((path / "evaluation.json").read_text())
    if (
        record["evaluator"] != evaluation.evaluator
        or record["commit"] != evaluation.configuration["commit"]
        or any(
            record["configuration"]["fit"][key] != evaluation.configuration[key]
            for key in ("draws", "tune")
        )
    ):
        raise ValueError("evaluation configuration differs from registration")
    decision, model_record = load_release(
        path,
        model_path,
        sensitivity_run=alternative_path,
    )
    summary = json.loads(json.dumps(asdict(decision)))
    if evaluation.summary is not None and summary != evaluation.summary:
        raise ValueError("registered policy differs from verified evidence")
    return summary, model_record
