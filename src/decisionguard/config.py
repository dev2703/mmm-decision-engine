"""Explicit backend configuration; scientific run parameters live with each run."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.engine import make_url


def artifact_path(root: Path, uri: str) -> Path:
    path = (root / uri).resolve()
    if Path(uri).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes configured root")
    return path


@dataclass(frozen=True)
class Settings:
    database_url: str
    artifact_root: Path = Path("artifacts")
    api_token: str | None = field(default=None, repr=False)
    read_only: bool = False
    environment: str = "local"

    def __post_init__(self) -> None:
        if make_url(self.database_url).drivername != "postgresql+psycopg":
            raise ValueError("DATABASE_URL must use postgresql+psycopg")
        object.__setattr__(self, "artifact_root", self.artifact_root.resolve())
        if self.environment not in {"local", "production"}:
            raise ValueError("APP_ENV must be local or production")
        if self.environment == "production" and (
            self.api_token is None or len(self.api_token) < 32
        ):
            raise ValueError(
                "Production requires an API_TOKEN of at least 32 characters"
            )

    @classmethod
    def from_environment(cls) -> Settings:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        read_only = os.environ.get("READ_ONLY", "false").lower()
        if read_only not in {"true", "false"}:
            raise ValueError("READ_ONLY must be true or false")
        return cls(
            database_url,
            Path(os.environ.get("ARTIFACT_ROOT", "artifacts")),
            api_token=os.environ.get("API_TOKEN") or None,
            read_only=read_only == "true",
            environment=os.environ.get("APP_ENV", "local"),
        )
