"""Local immutable raw snapshots and auditable quality artifacts."""

import json
from dataclasses import asdict
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

from decisionguard.data.integrity import IntegrityConfig, IntegrityResult, clean_dataset


def json_date(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=json_date, allow_nan=False) + "\n"
    )


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
    package = Path(__file__).resolve().parent.parent
    code = sha256()
    for path in sorted(package.rglob("*.py")):
        code.update(str(path.relative_to(package)).encode())
        code.update(path.read_bytes())
    provenance = {
        "simulation": simulation,
        "code_hash": code.hexdigest(),
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
