"""End-to-end persistence, raw preservation, and consuming quality gates."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from decisionguard.data.artifacts import load_model_ready, write_dataset
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset


def test_artifact_roundtrip_raw_preservation_and_no_overwrite(tmp_path: Path) -> None:
    raw = generate_dataset(SyntheticConfig(weeks=20)).observations
    snapshot = raw.copy(deep=True)
    output = tmp_path / "run"
    result = write_dataset(raw, output)
    pd.testing.assert_frame_equal(raw, snapshot)
    pd.testing.assert_frame_equal(pd.read_parquet(output / "raw.parquet"), snapshot)
    pd.testing.assert_frame_equal(load_model_ready(output), result.clean)
    original_bytes = (output / "raw.parquet").read_bytes()
    with pytest.raises(FileExistsError):
        write_dataset(raw, output)
    assert (output / "raw.parquet").read_bytes() == original_bytes
    report = json.loads((output / "quality.json").read_text())
    assert report["schema_version"] == "weekly-wide-v1"
    assert report["raw_hash"] != "" and report["clean_hash"] != ""
    assert "code_hash" in json.loads((output / "provenance.json").read_text())
    with (output / "clean.parquet").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_model_ready(output)


def test_blocked_artifact_cannot_be_loaded_for_modeling(tmp_path: Path) -> None:
    raw = generate_dataset(SyntheticConfig(weeks=20)).observations
    raw.loc[0, "revenue"] = float("nan")
    output = tmp_path / "blocked"
    result = write_dataset(raw, output)
    assert result.report.status == "BLOCKER"
    assert (output / "raw.parquet").is_file() and (output / "clean.parquet").is_file()
    with pytest.raises(ValueError, match="blocked"):
        load_model_ready(output)


def test_cli_prepare_and_clean_existing_raw(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    command = [
        sys.executable,
        "-m",
        "decisionguard.cli",
        "prepare-data",
        "--weeks",
        "26",
        "--output",
        str(output),
        "--meta-alias",
        "fb_spend",
        "--unit-error-weeks",
        "3",
        "--duplicate-rows",
        "2",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert (output / "generation.json").is_file()
    assert (output / "truth.parquet").is_file()
    first_bytes = (output / "raw.parquet").read_bytes()
    cleaned = tmp_path / "cleaned"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "decisionguard.cli",
            "clean-data",
            str(output / "raw.parquet"),
            "--output",
            str(cleaned),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pd.testing.assert_frame_equal(load_model_ready(output), load_model_ready(cleaned))
    assert (output / "raw.parquet").read_bytes() == first_bytes


def test_cli_blocked_exit_preserves_diagnostic_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "blocked"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "decisionguard.cli",
            "prepare-data",
            "--weeks",
            "26",
            "--output",
            str(output),
            "--tracking-outage-weeks",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "blocked" in result.stderr.lower()
    assert json.loads((output / "quality.json").read_text())["status"] == "BLOCKER"


def test_quality_report_tampering_is_rejected(tmp_path: Path) -> None:
    raw = generate_dataset(SyntheticConfig(weeks=20)).observations
    raw.loc[0, "revenue"] = float("nan")
    output = tmp_path / "blocked"
    write_dataset(raw, output)
    path = output / "quality.json"
    content = json.loads(path.read_text())
    content["status"] = "RESOLVED"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="hash"):
        load_model_ready(output)


def test_incomplete_simulation_is_not_loadable(tmp_path: Path) -> None:
    raw = generate_dataset(SyntheticConfig(weeks=20)).observations
    write_dataset(raw, tmp_path / "partial", simulation=True)
    with pytest.raises(ValueError, match="incomplete"):
        load_model_ready(tmp_path / "partial")
