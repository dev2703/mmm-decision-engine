"""Verified per-test progress for expensive, explicitly resumed evaluations."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from decisionguard.data.artifacts import file_hash, write_json
from decisionguard.evaluation.policy import REQUIRED_TESTS


def initialize_progress(
    output: Path, identity: dict[str, object], *, resume: bool
) -> None:
    if output.exists():
        if not resume or (output / "evaluation.json").exists():
            raise FileExistsError(output)
        stored = json.loads((output / "progress.json").read_text())
        if stored != identity:
            raise ValueError("evaluation resume identity changed")
    else:
        if resume:
            raise ValueError("no evaluation progress to resume")
        output.mkdir(parents=True)
        write_json(output / "progress.json", identity)


def load_checkpoint(
    output: Path, test: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]] | None:
    if test not in REQUIRED_TESTS:
        raise ValueError("unknown evaluation test")
    manifest = output / f"{test}.json"
    if not manifest.exists():
        return None
    record = json.loads(manifest.read_text())
    table = output / f"{test}.parquet"
    if file_hash(table) != record["table_hash"]:
        raise ValueError(f"evaluation checkpoint changed: {test}")
    refits = record["refits"]
    digest = sha256(
        json.dumps(refits, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if digest != record["refits_hash"]:
        raise ValueError(f"evaluation checkpoint refits changed: {test}")
    return pd.read_parquet(table), refits


def save_checkpoint(
    output: Path,
    test: str,
    table: pd.DataFrame,
    refits: list[dict[str, object]],
) -> None:
    if test not in REQUIRED_TESTS:
        raise ValueError("unknown evaluation test")
    table_path = output / f"{test}.parquet"
    table.to_parquet(table_path, index=False)
    # Publish the manifest last and atomically: a partial test is never reused.
    temporary = output / f"{test}.json.tmp"
    write_json(
        temporary,
        {
            "table_hash": file_hash(table_path),
            "refits": refits,
            "refits_hash": sha256(
                json.dumps(refits, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        },
    )
    temporary.replace(output / f"{test}.json")
