"""Explicit backend configuration; scientific run parameters live with each run."""

from __future__ import annotations

import os
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if make_url(self.database_url).drivername != "postgresql+psycopg":
            raise ValueError("DATABASE_URL must use postgresql+psycopg")
        object.__setattr__(self, "artifact_root", self.artifact_root.resolve())

    @classmethod
    def from_environment(cls) -> Settings:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        return cls(database_url, Path(os.environ.get("ARTIFACT_ROOT", "artifacts")))
