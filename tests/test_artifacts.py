"""End-to-end persistence, raw preservation, and consuming quality gates."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from decisionguard.data.artifacts import load_model_ready, load_quality, write_dataset
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


def test_model_identity_includes_raw_arrivals_even_when_clean_data_matches(
    tmp_path: Path,
) -> None:
    from decisionguard.data.artifacts import dataset_evidence, load_model_inputs

    raw = generate_dataset(SyntheticConfig(weeks=26)).observations
    raw["available_at"] = raw["week"] + pd.Timedelta(days=14)
    first, second = tmp_path / "first", tmp_path / "second"
    write_dataset(raw, first)
    clean = load_model_ready(first)
    expected = dataset_evidence(first, clean)
    raw["available_at"] = raw["week"] + pd.Timedelta(days=7)
    write_dataset(raw, second)
    pd.testing.assert_frame_equal(load_model_ready(second), clean)
    with pytest.raises(ValueError, match="raw_artifact_hash"):
        load_model_inputs(second, expected)


def test_relocated_identical_dataset_keeps_identity_and_temporal_evidence(
    tmp_path: Path,
) -> None:
    import shutil

    from decisionguard.data.artifacts import dataset_evidence, load_model_inputs

    first, moved = tmp_path / "first", tmp_path / "relocated"
    write_dataset(generate_dataset(SyntheticConfig(weeks=26)).observations, first)
    expected = dataset_evidence(first, load_model_ready(first))
    shutil.copytree(first, moved)
    data, arrivals = load_model_inputs(moved, expected)
    assert len(data) == 26 and len(arrivals) == 26


def test_decoded_data_must_match_quality_identity_before_modeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "data"
    write_dataset(generate_dataset(SyntheticConfig(weeks=26)).observations, directory)
    changed = pd.read_parquet(directory / "clean.parquet")
    changed.loc[0, "revenue"] = 1.0

    def swapped_read(*args: object, **kwargs: object) -> pd.DataFrame:
        return changed

    monkeypatch.setattr(pd, "read_parquet", swapped_read)
    with pytest.raises(ValueError, match="recorded quality evidence"):
        load_model_ready(directory)


def test_existing_dataset_rejects_before_profiling_invalid_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileExistsError):
        write_dataset(pd.DataFrame(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_registered_quality_snapshot_is_checked_before_modeling(tmp_path: Path) -> None:
    output = tmp_path / "data"
    result = write_dataset(
        generate_dataset(SyntheticConfig(weeks=26)).observations, output
    )
    expected = load_quality(output)
    pd.testing.assert_frame_equal(
        load_model_ready(output, expected_quality=expected), result.clean
    )
    changed = dict(expected, clean_hash="different registered dataset")
    with pytest.raises(ValueError, match="registered dataset identity"):
        load_model_ready(output, expected_quality=changed)
    blocked = tmp_path / "blocked"
    raw = result.clean.copy()
    raw.loc[0, "revenue"] = float("nan")
    write_dataset(raw, blocked)
    with pytest.raises(ValueError, match="blocked"):
        load_model_ready(blocked, expected_quality=load_quality(blocked))


def test_prepare_cli_preserves_all_generation_and_corruption_options(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    from decisionguard.data.corruption import CorruptionConfig

    output = tmp_path / "all-options"
    corruption = CorruptionConfig(
        seed=9,
        missing_weeks=2,
        duplicate_rows=3,
        meta_alias="fb_spend",
        unit_error_weeks=1,
        tracking_outage_weeks=1,
        erroneous_outlier_weeks=1,
        late_arrival_weeks=1,
        arrival_delay_days=21,
        mixed_currency_weeks=1,
        negative_spend_weeks=1,
    )
    flags = [
        argument
        for name, value in asdict(corruption).items()
        for argument in ("--" + name.replace("_", "-"), str(value))
    ]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "decisionguard.cli",
            "prepare-data",
            "--weeks",
            "26",
            "--shift-week",
            "5",
            "--shift-amount",
            "1000",
            "--event-week",
            "2",
            "--event-amount",
            "400",
            "--output",
            str(output),
            *flags,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2, completed.stderr
    manifest = json.loads((output / "generation.json").read_text())
    assert manifest["corruption"] == asdict(corruption)
    expected = {
        "weeks": 26,
        "seed": 9,
        "shift_week": 5,
        "shift_amount": 1000,
        "event_week": 2,
        "event_amount": 400,
    }
    assert {key: manifest["generation"][key] for key in expected} == expected
    assert load_quality(output)["status"] == "BLOCKER"
