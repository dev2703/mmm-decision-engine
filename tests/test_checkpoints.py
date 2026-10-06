"""Interruption recovery preserves evidence and fails closed on changed work."""

from pathlib import Path

import pandas as pd
import pytest

from decisionguard.evaluation.checkpoints import (
    initialize_progress,
    load_checkpoint,
    save_checkpoint,
)


def test_resume_keeps_completed_evidence_and_ignores_unfinished_test(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    identity: dict[str, object] = {"source": "abc", "draws": 2000}
    initialize_progress(output, identity, resume=False)
    raw = pd.DataFrame({"metric_value": [float("nan"), 1.23456789012345]})
    refits: list[dict[str, object]] = [
        {"train_rows": 52, "diagnostics": {"diagnostic_status": "INVESTIGATE"}}
    ]
    save_checkpoint(output, "holdout_accuracy", raw, refits)
    (output / "placebo.parquet").write_bytes(b"interrupted write")
    initialize_progress(output, identity, resume=True)
    restored = load_checkpoint(output, "holdout_accuracy")
    assert restored is not None
    pd.testing.assert_frame_equal(restored[0], raw)
    assert restored[1] == refits
    assert load_checkpoint(output, "placebo") is None


def test_resume_rejects_changed_identity_or_completed_run(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    initialize_progress(output, {"draws": 2000}, resume=False)
    with pytest.raises(ValueError, match="identity changed"):
        initialize_progress(output, {"draws": 100}, resume=True)
    with pytest.raises(FileExistsError):
        initialize_progress(output, {"draws": 2000}, resume=False)
    (output / "evaluation.json").write_text("{}")
    with pytest.raises(FileExistsError):
        initialize_progress(output, {"draws": 2000}, resume=True)
    with pytest.raises(ValueError, match="no evaluation progress"):
        initialize_progress(tmp_path / "absent", {}, resume=True)


def test_changed_checkpoint_fails_instead_of_recomputing_silently(
    tmp_path: Path,
) -> None:
    save_checkpoint(tmp_path, "placebo", pd.DataFrame({"value": [1]}), [])
    (tmp_path / "placebo.parquet").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checkpoint changed"):
        load_checkpoint(tmp_path, "placebo")
