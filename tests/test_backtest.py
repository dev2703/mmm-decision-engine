"""Future assessment cannot tune models or contaminate fitted preprocessing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decisionguard.cli import main
from decisionguard.data.artifacts import file_hash, write_dataset
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset
from decisionguard.experiments.backtest import run_backtest
from decisionguard.experiments.baseline import (
    BaselineConfig,
    TemporalFold,
    evaluate_folds,
)
from decisionguard.models.config import MMMConfig

CONFIG = BaselineConfig(period=13, initial_train=52)


def test_cli_four_blocks_and_frozen_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, output = tmp_path / "data", tmp_path / "backtest"
    write_dataset(
        generate_dataset(SyntheticConfig(weeks=107, seed=1701)).observations, data
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "decisionguard",
            "backtest",
            "--dataset",
            str(data),
            "--output",
            str(output),
            "--period",
            "13",
            "--initial-train",
            "52",
        ],
    )
    main()
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((output / "backtest.json").read_text())
    membership = pd.read_parquet(output / "partitions.parquet")
    assert membership["week"].is_unique
    assert membership["split"].tolist() == [
        *(["train"] * 52),
        *(["validation"] * 29),
        *(["test"] * 13),
        *(["holdout"] * 13),
    ]
    validation = json.loads((output / "validation.json").read_text())
    assert len(validation) == 5
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert len(predictions[predictions["split"] == "validation"]) == 5 * 29
    for context, model in result["selected_by_validation_mae"].items():
        assert validation[model]["metrics"]["mae"] == min(
            row["metrics"]["mae"]
            for row in validation.values()
            if row["prediction_context"] == context
        )
        fit = result["assessment"][model]["fit"][0]
        assert fit["train_rows"] == 81
        assert fit["train_end"] < result["splits"]["test"]["start"]
        for split in ("test", "holdout"):
            frame = predictions.loc[
                (predictions["model"] == str(model)) & (predictions["split"] == split)
            ]
            assert len(frame) == 13
            assert result["metrics"][model][split]["mae"] == pytest.approx(
                float((frame["actual"] - frame["predicted"]).abs().mean())
            )
    for name, digest in result["artifacts"].items():
        assert file_hash(output / name) == digest
    with pytest.raises(FileExistsError):
        run_backtest(data, output, CONFIG)


@pytest.mark.parametrize("change", ["targets", "covariates"])
def test_future_changes_cannot_affect_selection_or_fit(
    tmp_path: Path, change: str
) -> None:
    data = generate_dataset(SyntheticConfig(weeks=104, seed=1702)).observations
    changed = data.copy(deep=True)
    if change == "targets":
        changed.loc[78:, "revenue"] *= 10
    else:
        changed.loc[78:, "search_spend"] *= 10
        changed.loc[78:, "price"] *= 2
    results: list[dict[str, object]] = []
    for name, frame in (("original", data), ("changed", changed)):
        path = tmp_path / name
        write_dataset(frame, path)
        results.append(run_backtest(path, tmp_path / f"{name}-run", CONFIG))
    assert (
        results[0]["selected_by_validation_mae"]
        == results[1]["selected_by_validation_mae"]
    )
    before = json.loads((tmp_path / "original-run" / "validation.json").read_text())
    after = json.loads((tmp_path / "changed-run" / "validation.json").read_text())
    assert before == after
    original = json.loads((tmp_path / "original-run" / "backtest.json").read_text())
    modified = json.loads((tmp_path / "changed-run" / "backtest.json").read_text())
    for model, evidence in original["assessment"].items():
        fit_before = evidence["fit"][0]["diagnostics"].copy()
        fit_after = modified["assessment"][model]["fit"][0]["diagnostics"].copy()
        # Coverage measures future outcomes, rather than fitted model state.
        fit_before.pop("test_coverage_90", None)
        fit_after.pop("test_coverage_90", None)
        assert fit_before == fit_after
    if change == "targets":
        pred_before = pd.read_parquet(tmp_path / "original-run" / "predictions.parquet")
        pred_after = pd.read_parquet(tmp_path / "changed-run" / "predictions.parquet")
        np.testing.assert_array_equal(pred_before["predicted"], pred_after["predicted"])
        assert original["metrics"] != modified["metrics"]


def test_blocked_data_and_insufficient_history_cannot_complete(tmp_path: Path) -> None:
    data = generate_dataset(SyntheticConfig(weeks=104)).observations
    write_dataset(data, tmp_path / "good")
    with pytest.raises(ValueError, match="insufficient data"):
        run_backtest(tmp_path / "good", tmp_path / "too-short")
    assert not (tmp_path / "too-short").exists()
    data.loc[0, "revenue"] = np.nan
    write_dataset(data, tmp_path / "bad")
    with pytest.raises(ValueError, match="data-quality"):
        run_backtest(tmp_path / "bad", tmp_path / "blocked", CONFIG)
    assert not (tmp_path / "blocked").exists()


def test_late_training_data_fails_without_completion(tmp_path: Path) -> None:
    data = generate_dataset(SyntheticConfig(weeks=104)).observations
    data["available_at"] = data["week"] + pd.Timedelta(days=7)
    data.loc[51, "available_at"] = data["week"].iloc[55]
    write_dataset(data, tmp_path / "late")
    with pytest.raises(ValueError, match="unavailable"):
        run_backtest(tmp_path / "late", tmp_path / "run", CONFIG)
    assert not (tmp_path / "run" / "backtest.json").exists()
    assert not (tmp_path / "run" / "selection.json").exists()


def test_selection_is_persisted_before_assessment_and_failure_is_not_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = generate_dataset(SyntheticConfig(weeks=104)).observations
    write_dataset(data, tmp_path / "data")
    output = tmp_path / "run"

    def fail_at_assessment(
        data: pd.DataFrame,
        config: BaselineConfig,
        folds: tuple[TemporalFold, ...],
        available_at: pd.Series[pd.Timestamp] | None = None,
    ) -> tuple[pd.DataFrame, tuple[dict[str, object], ...]]:
        if folds[0].kind == "assessment":
            assert (output / "selection.json").exists()
            raise ValueError("assessment failed")
        assert len(data) == 78
        return evaluate_folds(data, config, folds, available_at)

    monkeypatch.setattr(
        "decisionguard.experiments.backtest.evaluate_folds", fail_at_assessment
    )
    with pytest.raises(ValueError, match="assessment failed"):
        run_backtest(tmp_path / "data", output, CONFIG)
    assert not (output / "backtest.json").exists()


@pytest.mark.scientific
def test_real_mmm_shares_cutoff_and_preserves_uncertainty(tmp_path: Path) -> None:
    data = generate_dataset(SyntheticConfig(weeks=104, seed=1703)).observations
    write_dataset(data, tmp_path / "data")
    output = tmp_path / "run"
    run_backtest(
        tmp_path / "data",
        output,
        CONFIG,
        mmm_config=MMMConfig(draws=8, tune=16, chains=1),
    )
    result = json.loads((output / "backtest.json").read_text())
    model = json.loads((output / "mmm" / "model.json").read_text())
    assert model["config"]["holdout"] == 26
    assert model["training_window"]["rows"] == 78
    assert model["preprocessing"]["mean"]["price"] == pytest.approx(
        float(data["price"].iloc[:78].mean())
    )
    assert model["training_window"]["end"] < result["splits"]["test"]["start"]
    assert model["diagnostics"]["diagnostic_status"] == "INCOMPLETE"
    assert model["model_status"] == "CANDIDATE_UNEVALUATED"
    predictions = pd.read_parquet(output / "predictions.parquet")
    future = predictions[predictions["model"] == "pymc_marketing_mmm"]
    assert len(future) == 26
    assert (future["lower_90"] < future["upper_90"]).all()
    for split in ("test", "holdout"):
        assert 0 <= result["metrics"]["pymc_marketing_mmm"][split]["coverage_90"] <= 1
