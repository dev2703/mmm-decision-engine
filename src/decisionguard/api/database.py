"""PostgreSQL metadata; large scientific arrays remain immutable artifacts."""

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    current_dataset_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL", use_alter=True)
    )


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("project_id", "version"),
        CheckConstraint("version > 0"),
        CheckConstraint("quality_status IN ('RESOLVED', 'WARNING', 'BLOCKER')"),
        CheckConstraint("date_end >= date_start"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="RESTRICT"), index=True
    )
    version: Mapped[int]
    source: Mapped[str]
    artifact_uri: Mapped[str] = mapped_column(unique=True)
    schema_version: Mapped[str]
    date_start: Mapped[date]
    date_end: Mapped[date]
    hash: Mapped[str]
    quality_status: Mapped[str]
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB)
    quality: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="RESTRICT"), index=True
    )
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT")
    )
    hypothesis: Mapped[str]
    model_family: Mapped[str]
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelRun(Base):
    __tablename__ = "model_runs"
    __table_args__ = (
        CheckConstraint("status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    experiment_id: Mapped[UUID] = mapped_column(
        ForeignKey("experiments.id", ondelete="RESTRICT"), index=True
    )
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT")
    )
    model_family: Mapped[str]
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB)
    artifact_uri: Mapped[str] = mapped_column(unique=True)
    status: Mapped[str]
    model_status: Mapped[str]
    code_version: Mapped[str | None]
    record_hash: Mapped[str | None]
    training_window: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_type: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        CheckConstraint("status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    model_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_runs.id", ondelete="RESTRICT"), index=True
    )
    sensitivity_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_runs.id", ondelete="RESTRICT")
    )
    evaluator: Mapped[str]
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB)
    artifact_uri: Mapped[str] = mapped_column(unique=True)
    status: Mapped[str]
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    record_hash: Mapped[str | None]
    error_type: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Scenario(Base):
    __tablename__ = "scenarios"
    __table_args__ = (
        CheckConstraint("budget > 0"),
        CheckConstraint("horizon_weeks BETWEEN 1 AND 52"),
        CheckConstraint("risk_policy IN ('expected', 'conservative')"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    model_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_runs.id", ondelete="RESTRICT"), index=True
    )
    evaluation_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="RESTRICT")
    )
    evaluation_record_hash: Mapped[str]
    budget: Mapped[float]
    constraints: Mapped[dict[str, Any]] = mapped_column(JSONB)
    effective_bounds: Mapped[dict[str, Any]] = mapped_column(JSONB)
    release: Mapped[dict[str, Any]] = mapped_column(JSONB)
    horizon_weeks: Mapped[int]
    risk_policy: Mapped[str]
    code_version: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
