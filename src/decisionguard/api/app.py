"""Synchronous metadata/data endpoints; Bayesian jobs stay outside HTTP workers."""

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime
from hmac import compare_digest
from time import perf_counter
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4
from zipfile import BadZipFile

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from starlette.responses import JSONResponse, Response

from decisionguard.agent.contracts import AgentAnswer, AgentQuery, Provider
from decisionguard.api.database import (
    Dataset,
    EvaluationRun,
    Experiment,
    ModelRun,
    OptimizationRun,
    Project,
    Scenario,
)
from decisionguard.api.history import (
    JobStatus,
    RunHistoryPage,
    RunKind,
    project_run_history,
)
from decisionguard.api.model_evidence import (
    ChannelEvidenceView,
    PredictionSplit,
    PredictionsView,
    channel_evidence,
    prediction_evidence,
)
from decisionguard.api.scenarios import ScenarioCreate
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


class ScenarioView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    model_run_id: UUID
    evaluation_run_id: UUID
    evaluation_record_hash: str
    budget: float
    constraints: dict[str, Any]
    effective_bounds: dict[str, Any]
    release: dict[str, Any]
    horizon_weeks: int
    risk_policy: Literal["expected", "conservative"]
    code_version: str
    created_at: datetime


class OptimizationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    draws: int = Field(default=200, ge=2, le=2000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class OptimizationRunView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    scenario_id: UUID
    configuration: dict[str, Any]
    status: Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]
    summary: dict[str, Any] | None
    error_type: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    evidence_current: bool = False
    recommendation_allowed: bool = False


def create_app(
    settings: Settings | None = None, *, agent_provider: Provider | None = None
) -> FastAPI:
    settings = settings or Settings.from_environment()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    sessions = sessionmaker(engine, expire_on_commit=False)
    root = settings.artifact_root

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        yield
        engine.dispose()

    app = FastAPI(title="mmm-decision-engine", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        # Do not reflect untrusted inputs, including non-JSON finite-number failures.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: item[key] for key in ("type", "loc", "msg")}
                    for item in error.errors()
                ]
            },
        )

    @app.middleware("http")
    async def correlation(request: Request, call_next: Any) -> Response:
        # Generate our own IDs: untrusted headers cannot inject logs or traces.
        request.state.correlation_id = str(uuid4())
        started = perf_counter()
        response: Response | None = None
        if request.url.path not in {"/health", "/ready"}:
            if settings.api_token and not compare_digest(
                request.headers.get("authorization", "").encode(),
                ("Bearer " + settings.api_token).encode(),
            ):
                response = JSONResponse(
                    {"detail": "Authentication required"}, status_code=401
                )
            elif settings.read_only and request.method not in {
                "GET",
                "HEAD",
                "OPTIONS",
            }:
                response = JSONResponse(
                    {"detail": "This workspace is read-only"}, status_code=403
                )
        if response is None:
            response = await call_next(request)
        assert response is not None
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

    @app.get("/ready")
    def ready() -> dict[str, str]:
        try:
            with sessions() as session:
                session.execute(text("SELECT 1"))
            if not root.is_dir():
                raise HTTPException(503, "Artifact storage unavailable")
        except SQLAlchemyError as error:
            raise HTTPException(503, "Database unavailable") from error
        return {"status": "ready"}

    @app.get("/capabilities")
    def capabilities() -> dict[str, bool]:
        return {
            "agent_enabled": agent_provider is not None and not settings.read_only,
            "read_only": settings.read_only,
        }

    @app.get("/projects", response_model=list[ProjectView])
    def list_projects(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
    ) -> list[ProjectView]:
        with sessions() as session:
            projects = session.scalars(
                select(Project)
                .order_by(Project.created_at.desc(), Project.id.desc())
                .offset(offset)
                .limit(limit)
            )
            return [ProjectView.model_validate(item) for item in projects]

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

    @app.get("/projects/{project_id}/datasets", response_model=list[DatasetView])
    def list_datasets(
        project_id: UUID,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
    ) -> list[DatasetView]:
        get_project(project_id)
        with sessions() as session:
            rows = session.scalars(
                select(Dataset)
                .where(Dataset.project_id == project_id)
                .order_by(Dataset.version.desc())
                .offset(offset)
                .limit(limit)
            )
            return [DatasetView.model_validate(row) for row in rows]

    @app.get("/projects/{project_id}/model-comparison")
    def compare_models(
        project_id: UUID,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=20)] = 20,
    ) -> dict[str, Any]:
        from decisionguard.api.comparison import model_comparison

        get_project(project_id)
        try:
            return model_comparison(
                sessions, root, project_id, offset=offset, limit=limit
            )
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise HTTPException(
                409, "Intact model comparison evidence required"
            ) from error

    @app.get("/evaluation-runs/{evaluation_id}/evidence")
    def get_evaluation_evidence(evaluation_id: UUID) -> dict[str, Any]:
        import pandas as pd

        from decisionguard.api.evaluations import verified_evaluation
        from decisionguard.evaluation.artifacts import source_evidence

        with sessions() as session:
            evaluation = session.get(EvaluationRun, evaluation_id)
            if evaluation is None:
                raise HTTPException(404, "Evaluation not found")
            if evaluation.status != "SUCCEEDED":
                raise HTTPException(409, "Evaluation is not complete")
            model = session.get(ModelRun, evaluation.model_run_id)
            alternative = session.get(ModelRun, evaluation.sensitivity_run_id)
            try:
                if model is None or alternative is None:
                    raise ValueError("Evaluation inputs missing")
                policy, record = verified_evaluation(
                    evaluation, model, alternative, root
                )
                path = artifact_path(root, evaluation.artifact_uri)
                raw = pd.read_parquet(path / "mmm_eval_raw.parquet")
                return {
                    "evaluation_run_id": str(evaluation.id),
                    "model_run_id": str(model.id),
                    "record_hash": evaluation.record_hash,
                    "policy": policy,
                    "metrics": [
                        asdict(row)
                        for row in source_evidence(raw, record["config"]["channels"])
                    ],
                    "decision_use": "historical_evidence; check current model health",
                }
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise HTTPException(
                    409, "Intact evaluation evidence required"
                ) from error

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

    @app.get("/projects/{project_id}/runs", response_model=RunHistoryPage)
    def list_runs(
        project_id: UUID,
        kind: RunKind | None = None,
        status: JobStatus | None = None,
        offset: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> RunHistoryPage:
        with sessions() as session:
            if session.get(Project, project_id) is None:
                raise HTTPException(404, "Project not found")
            return project_run_history(
                session,
                project_id,
                kind=kind,
                status=status,
                offset=offset,
                limit=limit,
            )

    def evidence_inputs(model_run_id: UUID) -> tuple[ModelRun, Dataset]:
        # Detach metadata before disk reads; no open transaction while hashing arrays.
        with sessions() as session:
            run = session.get(ModelRun, model_run_id)
            if run is None:
                raise HTTPException(404, "Model run not found")
            dataset = session.get(Dataset, run.dataset_id)
            if dataset is None:
                raise HTTPException(409, "Model dataset missing")
            return run, dataset

    @app.get("/model-runs/{model_run_id}/predictions", response_model=PredictionsView)
    def get_predictions(
        model_run_id: UUID,
        split: PredictionSplit | None = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=520)] = 100,
    ) -> PredictionsView:
        run, dataset = evidence_inputs(model_run_id)
        try:
            return prediction_evidence(
                run, dataset, root, split=split, offset=offset, limit=limit
            )
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise HTTPException(
                409, "Completed intact prediction evidence required"
            ) from error

    @app.get(
        "/model-runs/{model_run_id}/channels/{channel}",
        response_model=ChannelEvidenceView,
    )
    def get_channel_evidence(model_run_id: UUID, channel: str) -> ChannelEvidenceView:
        run, dataset = evidence_inputs(model_run_id)
        if channel not in run.configuration["channels"]:
            raise HTTPException(404, "Channel not found in this model")
        try:
            return channel_evidence(run, dataset, root, channel)
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            EOFError,
            BadZipFile,
        ) as error:
            raise HTTPException(
                409, "Completed intact channel evidence required"
            ) from error

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
            latest_evaluation,
            load_registered_model,
            verified_evaluation,
        )

        with sessions() as session:
            run = session.get(ModelRun, model_run_id)
            if run is None:
                raise HTTPException(404, "Model run not found")
            evaluation = latest_evaluation(session, model_run_id)
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
                "model_run_id": run.id,
                "model_status": run.model_status,
                "decision_status": policy["state"] if policy else "BLOCK",
                "diagnostics": diagnostics,
                "policy": policy,
                "evaluation": (
                    EvaluationRunView.model_validate(evaluation)
                    if evaluation is not None
                    else None
                ),
            }

    @app.post(
        "/model-runs/{model_run_id}/scenarios",
        response_model=ScenarioView,
        status_code=201,
    )
    def create_scenario(model_run_id: UUID, body: ScenarioCreate) -> ScenarioView:
        return ScenarioView.model_validate(
            prepare_scenario(model_run_id, body, persist=True)
        )

    def prepare_scenario(
        model_run_id: UUID, body: ScenarioCreate, *, persist: bool
    ) -> Scenario:
        from decisionguard.api.evaluations import latest_evaluation, verified_evaluation
        from decisionguard.data.artifacts import source_code_hash
        from decisionguard.evaluation.policy import ReleaseDecision, ReleaseState
        from decisionguard.optimization.allocation import BudgetConstraints

        with sessions.begin() as session:
            # Serialize with evaluation requests so creation binds the latest request.
            run = session.get(ModelRun, model_run_id, with_for_update=True)
            if run is None:
                raise HTTPException(404, "Model run not found")
            evaluation = latest_evaluation(session, model_run_id)
            if (
                evaluation is None
                or evaluation.status != "SUCCEEDED"
                or evaluation.record_hash is None
            ):
                raise HTTPException(409, "Latest evaluation must be completed")
            alternative = session.get(ModelRun, evaluation.sensitivity_run_id)
            if alternative is None:
                raise HTTPException(409, "Sensitivity model missing")
            try:
                summary, model = verified_evaluation(evaluation, run, alternative, root)
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise HTTPException(
                    409, "Scenario evidence integrity check failed"
                ) from error
            release = TypeAdapter(ReleaseDecision).validate_python(summary)
            if release.state == ReleaseState.BLOCK:
                raise HTTPException(409, "BLOCK model cannot create a budget scenario")
            quantities = body.model_dump(exclude={"horizon_weeks", "risk_policy"})
            constraints = BudgetConstraints(**quantities)
            channels = tuple(model["config"]["channels"])
            try:
                lower, upper = constraints.bounds(channels, release)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            scenario = Scenario(
                model_run_id=model_run_id,
                evaluation_run_id=evaluation.id,
                evaluation_record_hash=evaluation.record_hash,
                budget=body.total,
                constraints=quantities,
                effective_bounds={
                    channel: [float(lower[index]), float(upper[index])]
                    for index, channel in enumerate(channels)
                },
                release=summary,
                horizon_weeks=body.horizon_weeks,
                risk_policy=body.risk_policy,
                code_version=source_code_hash(),
            )
            if persist:
                session.add(scenario)
                session.flush()
            return scenario

    @app.post("/model-runs/{model_run_id}/constraints/validate")
    def validate_constraints(
        model_run_id: UUID, body: ScenarioCreate
    ) -> dict[str, Any]:
        scenario = prepare_scenario(model_run_id, body, persist=False)
        return {
            "valid": True,
            "model_run_id": str(model_run_id),
            "evaluation_run_id": str(scenario.evaluation_run_id),
            "evaluation_record_hash": scenario.evaluation_record_hash,
            "constraints": scenario.constraints,
            "effective_bounds": scenario.effective_bounds,
            "release": scenario.release,
            "horizon_weeks": scenario.horizon_weeks,
            "risk_policy": scenario.risk_policy,
            "units": "AUD/week",
            "recommendation_allowed": False,
        }

    @app.get("/scenarios/{scenario_id}", response_model=ScenarioView)
    def get_scenario(scenario_id: UUID) -> ScenarioView:
        """Historical evidence snapshot; no current optimization authority."""
        with sessions() as session:
            scenario = session.get(Scenario, scenario_id)
            if scenario is None:
                raise HTTPException(404, "Scenario not found")
            return ScenarioView.model_validate(scenario)

    @app.post(
        "/scenarios/{scenario_id}/optimize",
        response_model=OptimizationRunView,
        status_code=202,
    )
    def queue_optimization(
        scenario_id: UUID, body: OptimizationCreate
    ) -> OptimizationRunView:
        from decisionguard.api.optimizations import (
            current_scenario_inputs,
            scenario_snapshot,
            verify_scenario,
        )

        with sessions.begin() as session:
            scenario = session.get(Scenario, scenario_id)
            if scenario is None:
                raise HTTPException(404, "Scenario not found")
            try:
                model, evaluation, alternative = current_scenario_inputs(
                    session, scenario, lock=True
                )
                verify_scenario(scenario, model, evaluation, alternative, root)
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise HTTPException(
                    409, "Current verified scenario evidence required"
                ) from error
            active = session.scalar(
                select(OptimizationRun.id).where(
                    OptimizationRun.scenario_id == scenario_id,
                    OptimizationRun.status.in_(["QUEUED", "RUNNING"]),
                )
            )
            if active is not None:
                raise HTTPException(409, "Scenario already has an active optimization")
            parent = session.get(Experiment, model.experiment_id)
            if parent is None:
                raise HTTPException(409, "Model experiment missing")
            identifier = uuid4()
            run = OptimizationRun(
                id=identifier,
                scenario_id=scenario_id,
                status="QUEUED",
                configuration={
                    **body.model_dump(),
                    "scenario": scenario_snapshot(scenario),
                    "model_record_hash": model.record_hash,
                },
                artifact_uri=f"projects/{parent.project_id}/optimizations/{identifier}",
            )
            session.add(run)
            session.flush()
            return OptimizationRunView.model_validate(run)

    @app.get("/optimization-runs/{run_id}", response_model=OptimizationRunView)
    def get_optimization(run_id: UUID) -> OptimizationRunView:
        from decisionguard.api.optimizations import (
            current_scenario_inputs,
            verified_optimization,
            verify_scenario,
        )

        with sessions() as session:
            run = session.get(OptimizationRun, run_id)
            if run is None:
                raise HTTPException(404, "Optimization run not found")
            view = OptimizationRunView.model_validate(run)
            if run.status != "SUCCEEDED":
                return view
            scenario = session.get(Scenario, run.scenario_id)
            if scenario is None:
                raise HTTPException(409, "Optimization scenario missing")
            try:
                view.summary = verified_optimization(run, scenario, root)
                # Historical output stays inspectable after a newer release request.
                try:
                    inputs = current_scenario_inputs(session, scenario)
                except ValueError:
                    return view
                verify_scenario(scenario, *inputs, root)
                view.evidence_current = True
                view.recommendation_allowed = view.summary["selected"][
                    "recommendation_allowed"
                ]
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise HTTPException(
                    409, "Optimization evidence integrity check failed"
                ) from error
            return view

    @app.post("/agent/query", response_model=AgentAnswer)
    def agent_query(body: AgentQuery, request: Request) -> AgentAnswer:
        from decisionguard.agent.runtime import run_agent
        from decisionguard.api.agent_tools import AgentTools
        from decisionguard.data.artifacts import source_code_hash

        get_project(body.project_id)
        if agent_provider is None:
            raise HTTPException(503, "Agent provider is not configured")
        agent_tools = AgentTools(
            query=body,
            sessions=sessions,
            root=root,
            quality=get_quality,
            health=model_health,
            channel=get_channel_evidence,
            validate=validate_constraints,
            queue=lambda identifier: queue_optimization(
                identifier, OptimizationCreate()
            ),
            optimization=get_optimization,
        )
        return run_agent(
            body,
            agent_provider,
            schemas=agent_tools.schemas(),
            execute=agent_tools.execute,
            trace_root=root / "agent-traces",
            code_hash=source_code_hash(),
            request_id=UUID(request.state.correlation_id),
        )

    return app
