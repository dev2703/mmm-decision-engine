"""Mathematical forecasts, independent holdout, and temporal leakage boundaries."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decisionguard.data.artifacts import write_dataset
from decisionguard.data.corruption import CorruptionConfig, corrupt_dataset
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset
from decisionguard.experiments.baseline import (
    BaselineConfig,
    evaluate_baseline,
    forecast_metrics,
    run_baseline,
    seasonal_naive,
    temporal_folds,
)


def seasonal_data(rows: int = 36) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": pd.date_range("2022-01-03", periods=rows, freq="W-MON"),
            "revenue": np.tile([1.0, 2.0, 3.0, 4.0], (rows + 3) // 4)[:rows],
        }
    )


def test_seasonal_naive_repeats_training_season_beyond_one_cycle() -> None:
    training = np.array([10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        seasonal_naive(training, 10, 4), [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    )


def test_metrics_known_values_and_zero_denominator() -> None:
    actual, prediction = np.array([1.0, 3.0]), np.array([2.0, 1.0])
    result = forecast_metrics(actual, prediction)
    assert result["mae"] == 1.5
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["wape"] == 0.75
    assert forecast_metrics(np.zeros(2), np.ones(2))["wape"] is None
    with pytest.raises(ValueError):
        forecast_metrics(np.array([np.nan]), np.zeros(1))
    with pytest.raises(ValueError):
        forecast_metrics(np.zeros(2), np.zeros(1))


@pytest.mark.parametrize("gap", [0, 1, 5])
def test_expanding_folds_are_ordered_and_holdout_is_separate(gap: int) -> None:
    config = BaselineConfig(period=4, initial_train=8, horizon=5, gap=gap)
    folds = temporal_folds(36, config)
    assert folds[-1].kind == "holdout"
    assert folds[-1].test_start == 31 and folds[-1].test_end == 36
    seen: set[int] = set()
    previous_end = 0
    for fold in folds:
        assert fold.train_end > previous_end
        assert fold.train_end + gap == fold.test_start
        assert fold.test_end - fold.test_start == 5
        validation = set(range(fold.test_start, fold.test_end))
        assert seen.isdisjoint(validation)
        seen.update(validation)
        previous_end = fold.train_end
    assert all(fold.test_end <= 31 for fold in folds[:-1])


@pytest.mark.parametrize("gap", [0, 1, 6])
def test_exact_periodic_series_is_recovered_with_gap_and_long_horizon(gap: int) -> None:
    result = evaluate_baseline(
        seasonal_data(), BaselineConfig(period=4, initial_train=8, horizon=5, gap=gap)
    )
    assert result.cv_metrics["mae"] == result.holdout_metrics["mae"] == 0
    np.testing.assert_array_equal(
        result.predictions["actual"], result.predictions["predicted"]
    )


def test_holdout_target_cannot_change_cv_or_any_holdout_forecast() -> None:
    config = BaselineConfig(period=4, initial_train=8, horizon=5)
    data = seasonal_data()
    baseline = evaluate_baseline(data, config)
    changed = data.copy()
    changed.loc[31:, "revenue"] *= 100
    repeated = evaluate_baseline(changed, config)
    assert baseline.cv_metrics == repeated.cv_metrics
    np.testing.assert_array_equal(
        baseline.predictions["predicted"], repeated.predictions["predicted"]
    )
    assert repeated.holdout_metrics["mae"] != baseline.holdout_metrics["mae"]
    pd.testing.assert_frame_equal(
        baseline.predictions.loc[baseline.predictions["split"] == "cv"],
        repeated.predictions.loc[repeated.predictions["split"] == "cv"],
    )


def test_future_fold_targets_never_enter_earlier_predictions() -> None:
    data = seasonal_data()
    config = BaselineConfig(period=4, initial_train=8, horizon=5)
    first = evaluate_baseline(data, config)
    changed = data.copy()
    changed.loc[8:, "revenue"] += 1_000
    second = evaluate_baseline(changed, config)
    np.testing.assert_array_equal(
        first.predictions.iloc[:5]["predicted"],
        second.predictions.iloc[:5]["predicted"],
    )
    assert str(first.fold_records[0]["train_end"]) < str(
        first.fold_records[0]["test_start"]
    )


def test_late_training_arrivals_fail_without_sufficient_gap() -> None:
    data = seasonal_data()
    availability = data["week"] + pd.Timedelta(days=21)
    config = BaselineConfig(period=4, initial_train=8, horizon=5)
    with pytest.raises(ValueError, match="unavailable"):
        evaluate_baseline(data, config, availability)
    result = evaluate_baseline(data, replace(config, gap=2), availability)
    assert result.cv_metrics["mae"] == 0
    for fold in result.fold_records:
        assert str(fold["max_training_available_at"])[:10] <= str(
            fold["forecast_origin"]
        )


def test_malformed_data_and_insufficient_history_fail() -> None:
    config = BaselineConfig(period=4, initial_train=8, horizon=5)
    data = seasonal_data()
    with pytest.raises(ValueError, match="contiguous"):
        evaluate_baseline(data.drop(index=5), config)
    with pytest.raises(ValueError, match="unique"):
        evaluate_baseline(pd.concat([data, data.iloc[:1]]), config)
    with pytest.raises(ValueError, match="insufficient"):
        evaluate_baseline(data.iloc[:12], config)
    with pytest.raises(ValueError, match="finite"):
        seasonal_naive(np.array([1.0, np.inf]), 5, 2)
    with pytest.raises(ValueError):
        BaselineConfig(period=52, initial_train=10)
    with pytest.raises(ValueError):
        BaselineConfig(gap=-1)


def test_artifact_experiment_is_reproducible_and_auditable(tmp_path: Path) -> None:
    data = generate_dataset(SyntheticConfig(weeks=104)).observations
    source = tmp_path / "data"
    write_dataset(data, source)
    first = run_baseline(source, tmp_path / "first")
    second = run_baseline(source, tmp_path / "second")
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    record = json.loads((tmp_path / "first" / "experiment.json").read_text())
    assert record == json.loads((tmp_path / "second" / "experiment.json").read_text())
    assert record["model_status"] == "PREDICTIVE_ONLY"
    assert record["random_seed"] is None
    assert (
        record["dataset"]["hash"] and record["code_hash"] and record["predictions_hash"]
    )
    assert record["feature_config"]["learned_preprocessing"] is None
    assert all(
        key in record
        for key in (
            "hypothesis",
            "change",
            "validation",
            "metrics",
            "result",
            "interpretation",
            "decision",
            "limitations",
        )
    )
    with pytest.raises(FileExistsError):
        run_baseline(source, tmp_path / "first")


def test_blocked_dataset_cannot_create_baseline_run(tmp_path: Path) -> None:
    data = generate_dataset(SyntheticConfig(weeks=104)).observations
    data.loc[0, "revenue"] = np.nan
    source = tmp_path / "data"
    write_dataset(data, source)
    with pytest.raises(ValueError, match="blocked"):
        run_baseline(source, tmp_path / "experiment")
    assert not (tmp_path / "experiment").exists()


def test_raw_arrival_metadata_is_checked_by_artifact_runner(tmp_path: Path) -> None:
    source = generate_dataset(SyntheticConfig(weeks=104))
    dirty = corrupt_dataset(source, CorruptionConfig(late_arrival_weeks=104))
    directory = tmp_path / "data"
    write_dataset(dirty.observations, directory)
    with pytest.raises(ValueError, match="unavailable"):
        run_baseline(directory, tmp_path / "no-gap")
    result = run_baseline(directory, tmp_path / "gap", BaselineConfig(gap=2))
    assert len(result.predictions) > 0


def test_cli_baseline(tmp_path: Path) -> None:
    source = tmp_path / "data"
    write_dataset(generate_dataset(SyntheticConfig(weeks=104)).observations, source)
    output = tmp_path / "experiment"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "decisionguard.cli",
            "baseline",
            "--dataset",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads(completed.stdout)
    assert "cv" in metrics and "holdout" in metrics
    assert (output / "predictions.parquet").is_file()


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_metrics_preserve_scale_without_square_overflow_or_underflow(
    scale: float,
) -> None:
    actual = np.array([scale, scale])
    predicted = np.zeros(2)
    result = forecast_metrics(actual, predicted)
    assert result["mae"] == pytest.approx(scale, rel=1e-12, abs=0)
    assert result["rmse"] == pytest.approx(scale, rel=1e-12, abs=0)
    assert result["wape"] == pytest.approx(1.0)


def test_metrics_reject_unrepresentable_difference() -> None:
    with pytest.raises(ValueError, match="numerical range"):
        forecast_metrics(np.array([1e308]), np.array([-1e308]))


def test_unsigned_inputs_do_not_wrap_subtraction() -> None:
    result = forecast_metrics(
        np.array([3], dtype=np.uint64), np.array([2], dtype=np.uint64)
    )
    assert result["mae"] == 1.0 and result["rmse"] == 1.0
    assert result["wape"] == pytest.approx(1 / 3)
