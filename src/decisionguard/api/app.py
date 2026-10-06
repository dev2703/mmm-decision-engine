"""Synchronous metadata/data endpoints; Bayesian jobs stay outside HTTP workers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from starlette.responses import Response

from decisionguard.api.database import Dataset, Project
from decisionguard.config import Settings


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


def artifact_path(root: Path, uri: str) -> Path:
    path = (root / uri).resolve()
    if Path(uri).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes configured root")
    return path


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    sessions = sessionmaker(engine, expire_on_commit=False)
    root = settings.artifact_root

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        engine.dispose()

    app = FastAPI(title="mmm-decision-engine", lifespan=lifespan)

    @app.middleware("http")
    async def correlation(request: Request, call_next: Any) -> Response:
        # Generate our own IDs: untrusted headers cannot inject logs or traces.
        request.state.correlation_id = str(uuid4())
        response: Response = await call_next(request)
        response.headers["X-Correlation-ID"] = request.state.correlation_id
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
                configuration=asdict(generating),
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

    return app
