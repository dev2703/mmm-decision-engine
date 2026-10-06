"""Local immutable raw snapshots and auditable quality artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

from decisionguard.data.integrity import (
    IntegrityConfig,
    IntegrityResult,
    clean_dataset,
    dataset_hash,
)


def json_date(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=json_date, allow_nan=False) + "\n"
    )


def source_code_hash() -> str:
    """Identify the exact package source used for a data or experiment run."""
    package = Path(__file__).resolve().parent.parent
    code = sha256()
    for path in sorted(package.rglob("*.py")):
        code.update(str(path.relative_to(package)).encode())
        code.update(path.read_bytes())
    return code.hexdigest()


def write_dataset(
    raw: pd.DataFrame,
    output: Path,
    config: IntegrityConfig | None = None,
    *,
    simulation: bool = False,
) -> IntegrityResult:
    """A new run directory only; clean.parquet is a blocked candidate if flagged."""
    result = clean_dataset(raw, config)
    output.mkdir(parents=True, exist_ok=False)
    raw.to_parquet(output / "raw.parquet", index=False)
    result.clean.to_parquet(output / "clean.parquet", index=False)
    write_json(output / "quality.json", asdict(result.report))
    provenance = {
        "simulation": simulation,
        "code_hash": source_code_hash(),
        "artifacts": {
            name: sha256((output / name).read_bytes()).hexdigest()
            for name in ("raw.parquet", "clean.parquet", "quality.json")
        },
    }
    lock = Path(__file__).resolve().parents[3] / "uv.lock"
    if lock.is_file():
        provenance["dependency_lock_hash"] = sha256(lock.read_bytes()).hexdigest()
    write_json(output / "provenance.json", provenance)
    return result


def load_model_ready(output: Path) -> pd.DataFrame:
    """Enforce the quality gate and artifact integrity at the consuming boundary."""
    provenance = json.loads((output / "provenance.json").read_text())
    if provenance.get("simulation") and not all(
        name in provenance["artifacts"] for name in ("truth.parquet", "generation.json")
    ):
        raise ValueError("incomplete simulation artifacts")
    names = ["raw.parquet", "clean.parquet", "quality.json"]
    names.extend(
        name
        for name in ("truth.parquet", "generation.json")
        if name in provenance["artifacts"]
    )
    for name in names:
        path = output / name
        if sha256(path.read_bytes()).hexdigest() != provenance["artifacts"][name]:
            raise ValueError(f"{name} artifact hash mismatch")
    report = json.loads((output / "quality.json").read_text())
    if report.get("status") not in ("RESOLVED", "WARNING"):
        raise ValueError("modeling blocked by data-quality status")
    path = output / "clean.parquet"
    return pd.read_parquet(path)


def load_model_inputs(directory: Path) -> tuple[pd.DataFrame, pd.Series[pd.Timestamp]]:
    """Verified clean data and conservatively aligned raw arrival timestamps."""
    data = load_model_ready(directory)
    raw = pd.read_parquet(directory / "raw.parquet")
    if "available_at" not in raw:
        return data, data["week"] + pd.Timedelta(days=7)
    raw["week"] = pd.to_datetime(raw["week"], format="ISO8601").dt.normalize()
    raw["available_at"] = pd.to_datetime(raw["available_at"], format="ISO8601")
    by_week = raw.groupby("week")["available_at"].max()
    return data, by_week.reindex(data["week"]).reset_index(drop=True)


def dataset_evidence(directory: Path, data: pd.DataFrame) -> dict[str, object]:
    """Shared dataset provenance for predictive and probabilistic runs."""
    quality = json.loads((directory / "quality.json").read_text())
    return {
        "path": str(directory.resolve()),
        "hash": dataset_hash(data),
        **{
            f"{kind}_artifact_hash": sha256((directory / name).read_bytes()).hexdigest()
            for kind, name in (
                ("clean", "clean.parquet"),
                ("raw", "raw.parquet"),
                ("quality", "quality.json"),
            )
        },
        "quality_status": quality["status"],
        "quality_issues": quality["issues"],
    }
