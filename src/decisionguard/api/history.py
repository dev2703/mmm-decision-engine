"""Project job history from existing tables, without reading scientific artifacts."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import Select, Uuid, cast, literal, select, union_all
from sqlalchemy.orm import Session

from decisionguard.api.database import (
    EvaluationRun,
    Experiment,
    ModelRun,
    OptimizationRun,
    Scenario,
)

RunKind = Literal["training", "evaluation", "optimization"]
JobStatus = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]


class RunHistoryItem(BaseModel):
    id: UUID
    kind: RunKind
    status: JobStatus
    experiment_id: UUID
    dataset_id: UUID
    model_run_id: UUID
    evaluation_run_id: UUID | None
    scenario_id: UUID | None
    record_hash: str | None
    error_type: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class RunHistoryPage(BaseModel):
    project_id: UUID
    items: list[RunHistoryItem]
    offset: int
    limit: int
    next_offset: int | None


def project_run_history(
    session: Session,
    project_id: UUID,
    *,
    kind: RunKind | None,
    status: JobStatus | None,
    offset: int,
    limit: int,
) -> RunHistoryPage:
    branches: list[Select[Any]] = []
    for table, run_kind in (
        (ModelRun, "training"),
        (EvaluationRun, "evaluation"),
        (OptimizationRun, "optimization"),
    ):
        if kind is not None and kind != run_kind:
            continue
        query = select(
            table.id,
            literal(run_kind).label("kind"),
            table.status,
            ModelRun.experiment_id,
            ModelRun.dataset_id,
            ModelRun.id.label("model_run_id"),
            (
                EvaluationRun.id
                if table is EvaluationRun
                else Scenario.evaluation_run_id
                if table is OptimizationRun
                else cast(None, Uuid(as_uuid=True))
            ).label("evaluation_run_id"),
            (
                Scenario.id
                if table is OptimizationRun
                else cast(None, Uuid(as_uuid=True))
            ).label("scenario_id"),
            table.record_hash,
            table.error_type,
            table.created_at,
            table.started_at,
            table.finished_at,
        ).select_from(table)
        if table is EvaluationRun:
            query = query.join(ModelRun, EvaluationRun.model_run_id == ModelRun.id)
        elif table is OptimizationRun:
            query = query.join(
                Scenario, OptimizationRun.scenario_id == Scenario.id
            ).join(ModelRun, Scenario.model_run_id == ModelRun.id)
        query = query.join(Experiment, ModelRun.experiment_id == Experiment.id).where(
            Experiment.project_id == project_id
        )
        if status is not None:
            query = query.where(table.status == status)
        branches.append(query)
    history = union_all(*branches).subquery()
    # Sort and bound the combined stream in PostgreSQL, not separately per job kind.
    page = (
        session.execute(
            select(history)
            .order_by(
                history.c.created_at.desc(), history.c.id.desc(), history.c.kind.desc()
            )
            .offset(offset)
            .limit(limit + 1)
        )
        .mappings()
        .all()
    )
    return RunHistoryPage(
        project_id=project_id,
        items=[RunHistoryItem.model_validate(row) for row in page[:limit]],
        offset=offset,
        limit=limit,
        next_offset=offset + limit if len(page) > limit else None,
    )
