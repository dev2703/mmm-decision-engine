"""Synchronous metadata/data endpoints; Bayesian jobs stay outside HTTP workers."""

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime
from time import perf_counter
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from starlette.responses import Response

from decisionguard.api.database import (
    Dataset,
    EvaluationRun,
    Experiment,
    ModelRun,
    Project,
)
from decisionguard.config import Settings, artifact_path


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
    ]


class ProjectView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    created_at: datetime
    current_dataset_id: UUID | None


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    weeks: int = Field(default=156, ge=52, le=520)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class DatasetView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    project_id: UUID
    version: int
    source: str
    schema_version: str
    date_start: date
    date_end: date
    hash: str
    quality_status: str
    configuration: dict[str, Any]
    created_at: datetime


class ExperimentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: UUID | None = None
    hypothesis: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    configuration: dict[str, Any] = Field(default_factory=dict)


class ExperimentView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    project_id: UUID
    dataset_id: UUID
    hypothesis: str
    model_family: str
    configuration: dict[str, Any]
    created_at: datetime


class ModelRunView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    experiment_id: UUID
    dataset_id: UUID
    model_family: str
    configuration: dict[str, Any]
    status: str
    model_status: str
    code_version: str | None
    training_window: dict[str, Any] | None
    metrics: dict[str, Any] | None
    error_type: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class EvaluationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sensitivity_run_id: UUID
    draws: int | None = Field(default=None, strict=True, ge=1, le=10000)
    tune: int | None = Field(default=None, strict=True, ge=1, le=20000)


class EvaluationRunView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    model_run_id: UUID
    sensitivity_run_id: UUID
    evaluator: str
    configuration: dict[str, Any]
    status: Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]
    summary: dict[str, Any] | None
    error_type: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ModelHealthView(BaseModel):
    model_run_id: UUID
    model_status: str
    decision_status: Literal["PASS", "WARN", "RESTRICT", "BLOCK"]
    diagnostics: dict[str, Any] | None
    policy: dict[str, Any] | None
    evaluation: EvaluationRunView | None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    sessions = sessionmaker(engine, expire_on_commit=False)
    root = settings.artifact_root

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        yield
        engine.dispose()

    app = FastAPI(title="mmm-decision-engine", lifespan=lifespan)

    @app.middleware("http")
    async def correlation(request: Request, call_next: Any) -> Response:
        # Generate our own IDs: untrusted headers cannot inject logs or traces.
        request.state.correlation_id = str(uuid4())
        started = perf_counter()
        response: Response = await call_next(request)
        response.headers["X-Correlation-ID"] = request.state.correlation_id
        route = request.scope.get("route")
        logging.getLogger("decisionguard.api").info(
            json.dumps(
                {
                    "correlation_id": request.state.correlation_id,
                    "method": request.method,
                    "operation": getattr(route, "path", "unmatched"),
                    "status": response.status_code,
                    "duration_seconds": perf_counter() - started,
                }
            )
        )
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/projects", response_model=ProjectView, status_code=201)
    def create_project(body: ProjectCreate) -> ProjectView:
        with sessions.begin() as session:
            project = Project(name=body.name)
            session.add(project)
            session.flush()
            return ProjectView.model_validate(project)

    @app.get("/projects/{project_id}", response_model=ProjectView)
    def get_project(project_id: UUID) -> ProjectView:
        with sessions() as session:
            project = session.get(Project, project_id)
            if project is None:
                raise HTTPException(404, "Project not found")
            return ProjectView.model_validate(project)

    @app.post(
        "/projects/{project_id}/datasets/generate",
        response_model=DatasetView,
        status_code=201,
    )
    def generate(project_id: UUID, body: GenerationRequest) -> DatasetView:
        from decisionguard.data.artifacts import load_quality, write_simulation
        from decisionguard.data.corruption import CorruptionConfig, corrupt_dataset
        from decisionguard.data.integrity import IntegrityConfig
        from decisionguard.data.synthetic import SyntheticConfig, generate_dataset

        with sessions.begin() as session:
            project = session.scalar(
                select(Project).where(Project.id == project_id).with_for_update()
            )
            if project is None:
                raise HTTPException(404, "Project not found")
            version = (
                session.scalar(
                    select(func.max(Dataset.version)).where(
                        Dataset.project_id == project_id
                    )
                )
                or 0
            ) + 1
            identifier = uuid4()
            uri = f"projects/{project_id}/datasets/{identifier}"
            output = artifact_path(root, uri)
            generating = SyntheticConfig(weeks=body.weeks, seed=body.seed)
            clean = generate_dataset(generating)
            dirty = corrupt_dataset(clean, CorruptionConfig(seed=body.seed))
            start, end = (
                clean.observations["week"].min(),
                clean.observations["week"].max(),
            )
            write_simulation(
                clean,
                dirty,
                output,
                IntegrityConfig(
                    expected_start=str(start)[:10], expected_end=str(end)[:10]
                ),
            )
            quality = load_quality(output)
            dataset = Dataset(
                id=identifier,
                project_id=project_id,
                version=version,
                source="SYNTHETIC",
                artifact_uri=uri,
                schema_version=quality["schema_version"],
                date_start=start.date(),
                date_end=end.date(),
                hash=quality["clean_hash"],
                quality_status=quality["status"],
                configuration=json.loads((output / "generation.json").read_text())[
                    "generation"
                ],
                quality=quality,
            )
            session.add(dataset)
            session.flush()
            project.current_dataset_id = identifier
            return DatasetView.model_validate(dataset)

    @app.get("/datasets/{dataset_id}/quality")
    def get_quality(dataset_id: UUID) -> dict[str, Any]:
        from decisionguard.data.artifacts import load_quality

        with sessions() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                raise HTTPException(404, "Dataset not found")
            try:
                quality = load_quality(artifact_path(root, dataset.artifact_uri))
                if quality != dataset.quality or quality["clean_hash"] != dataset.hash:
                    raise ValueError("dataset evidence differs from persisted metadata")
            except (ValueError, OSError) as error:
                raise HTTPException(
                    409, "Dataset artifact integrity check failed"
                ) from error
            return quality

    @app.post(
        "/projects/{project_id}/experiments",
        response_model=ExperimentView,
        status_code=201,
    )
    def create_experiment(project_id: UUID, body: ExperimentCreate) -> ExperimentView:
        from decisionguard.models.config import MMMConfig

        try:
            config = MMMConfig(**body.configuration)
            if (
                config.draws > 10000
                or config.tune > 20000
                or config.chains > 8
                or config.max_tree_depth > 16
                or config.adstock_lags > 52
            ):
                raise ValueError("sampling request exceeds local job limits")
        except (TypeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
        with sessions.begin() as session:
            project = session.get(Project, project_id)
            if project is None:
                raise HTTPException(404, "Project not found")
            dataset = (
                session.get(Dataset, body.dataset_id or project.current_dataset_id)
                if body.dataset_id or project.current_dataset_id
                else None
            )
            if dataset is None or dataset.project_id != project_id:
                raise HTTPException(409, "Select a dataset belonging to this project")
            if dataset.quality_status not in ("RESOLVED", "WARNING"):
                raise HTTPException(409, "Data quality blocks experimentation")
            if (
                config.holdout
                > (dataset.date_end - dataset.date_start).days // 7 + 1 - 52
            ):
                raise HTTPException(422, "Holdout leaves fewer than 52 training weeks")
            experiment = Experiment(
                project_id=project_id,
                dataset_id=dataset.id,
                hypothesis=body.hypothesis,
                model_family="pymc_marketing_mmm",
                configuration=asdict(config),
            )
            session.add(experiment)
            session.flush()
            return ExperimentView.model_validate(experiment)

    @app.post(
        "/experiments/{experiment_id}/run", response_model=ModelRunView, status_code=202
    )
    def queue_model(experiment_id: UUID) -> ModelRunView:
        with sessions.begin() as session:
            experiment = session.scalar(
                select(Experiment)
                .where(Experiment.id == experiment_id)
                .with_for_update()
            )
            if experiment is None:
                raise HTTPException(404, "Experiment not found")
            active = session.scalar(
                select(ModelRun.id).where(
                    ModelRun.experiment_id == experiment_id,
                    ModelRun.status.in_(["QUEUED", "RUNNING"]),
                )
            )
            if active is not None:
                raise HTTPException(409, "Experiment already has an active model run")
            identifier = uuid4()
            run = ModelRun(
                id=identifier,
                experiment_id=experiment_id,
                dataset_id=experiment.dataset_id,
                model_family=experiment.model_family,
                configuration=dict(experiment.configuration),
                artifact_uri=f"projects/{experiment.project_id}/model-runs/{identifier}",
                status="QUEUED",
                model_status="PENDING",
            )
            session.add(run)
            session.flush()
            return ModelRunView.model_validate(run)

    @app.get("/model-runs/{model_run_id}", response_model=ModelRunView)
    def get_model(model_run_id: UUID) -> ModelRunView:
        with sessions() as session:
            run = session.get(ModelRun, model_run_id)
            if run is None:
                raise HTTPException(404, "Model run not found")
            return ModelRunView.model_validate(run)

    @app.get("/projects/{project_id}/model-runs", response_model=list[ModelRunView])
    def list_models(project_id: UUID) -> list[ModelRunView]:
        with sessions() as session:
            if session.get(Project, project_id) is None:
                raise HTTPException(404, "Project not found")
            runs = session.scalars(
                select(ModelRun)
                .join(Experiment, ModelRun.experiment_id == Experiment.id)
                .where(Experiment.project_id == project_id)
                .order_by(ModelRun.created_at.desc())
            )
            return [ModelRunView.model_validate(run) for run in runs]

    @app.post(
        "/model-runs/{model_run_id}/evaluate",
        response_model=EvaluationRunView,
        status_code=202,
    )
    def queue_evaluation(
        model_run_id: UUID, body: EvaluationCreate
    ) -> EvaluationRunView:
        from decisionguard.api.evaluations import load_registered_model
        from decisionguard.evaluation.artifacts import UPSTREAM_COMMIT
        from decisionguard.models.artifacts import validate_prior_sources

        with sessions.begin() as session:
            run = session.get(ModelRun, model_run_id, with_for_update=True)
            if run is None:
                raise HTTPException(404, "Model run not found")
            alternative = session.get(ModelRun, body.sensitivity_run_id)
            if alternative is None:
                raise HTTPException(404, "Sensitivity model run not found")
            parent = session.get(Experiment, run.experiment_id)
            other = session.get(Experiment, alternative.experiment_id)
            if (
                parent is None
                or other is None
                or parent.project_id != other.project_id
                or run.dataset_id != alternative.dataset_id
            ):
                raise HTTPException(409, "Sensitivity must use this project's dataset")
            try:
                model = load_registered_model(run, root)
                sensitivity = load_registered_model(alternative, root)
                validate_prior_sources(model, sensitivity)
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise HTTPException(
                    409, "Completed compatible model evidence required"
                ) from error
            active = session.scalar(
                select(EvaluationRun.id).where(
                    EvaluationRun.model_run_id == model_run_id,
                    EvaluationRun.status.in_(["QUEUED", "RUNNING"]),
                )
            )
            if active is not None:
                raise HTTPException(409, "Model already has an active evaluation")
            identifier = uuid4()
            evaluation = EvaluationRun(
                id=identifier,
                model_run_id=model_run_id,
                sensitivity_run_id=alternative.id,
                evaluator="mmm-eval",
                configuration={
                    "commit": UPSTREAM_COMMIT,
                    "draws": body.draws or model["config"]["draws"],
                    "tune": body.tune or model["config"]["tune"],
                    "model_record_hash": run.record_hash,
                    "sensitivity_record_hash": alternative.record_hash,
                },
                artifact_uri=f"projects/{parent.project_id}/evaluations/{identifier}",
                status="QUEUED",
            )
            session.add(evaluation)
            session.flush()
            return EvaluationRunView.model_validate(evaluation)

    @app.get("/model-runs/{model_run_id}/health", response_model=ModelHealthView)
    def model_health(model_run_id: UUID) -> dict[str, Any]:
        from decisionguard.api.evaluations import (
            load_registered_model,
            verified_evaluation,
        )

        with sessions() as session:
            run = session.get(ModelRun, model_run_id)
            if run is None:
                raise HTTPException(404, "Model run not found")
            evaluation = session.scalar(
                select(EvaluationRun)
                .where(EvaluationRun.model_run_id == model_run_id)
                .order_by(EvaluationRun.created_at.desc(), EvaluationRun.id.desc())
                .limit(1)
            )
            diagnostics: dict[str, Any] | None = None
            policy: dict[str, Any] | None = None
            try:
                if evaluation is not None and evaluation.status == "SUCCEEDED":
                    alternative = session.get(ModelRun, evaluation.sensitivity_run_id)
                    if alternative is None:
                        raise ValueError("sensitivity model missing")
                    policy, model = verified_evaluation(
                        evaluation, run, alternative, root
                    )
                    diagnostics = model["diagnostics"]
                elif run.status == "SUCCEEDED":
                    diagnostics = load_registered_model(run, root)["diagnostics"]
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise HTTPException(
                    409, "Model health artifact integrity check failed"
                ) from error
            return {
                "model_run_id": str(run.id),
                "model_status": run.model_status,
                "decision_status": policy["state"] if policy else "BLOCK",
                "diagnostics": diagnostics,
                "policy": policy,
                "evaluation": (
                    EvaluationRunView.model_validate(evaluation).model_dump(mode="json")
                    if evaluation is not None
                    else None
                ),
            }

    return app
