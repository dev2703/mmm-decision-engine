"""Train-only representation, nested temporal tuning and conditional prediction."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytest

from decisionguard.data.artifacts import write_dataset
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset
from decisionguard.experiments.baseline import (
    BaselineConfig,
    evaluate_baseline,
    run_baseline,
)
from decisionguard.experiments.regression import regression_features


@pytest.mark.parametrize(
    "model", ["ridge_raw", "ridge_domain", "hist_gradient_boosting"]
)
def test_holdout_target_never_changes_fitted_benchmark(
    model: Literal["ridge_raw", "ridge_domain", "hist_gradient_boosting"],
) -> None:
    data = generate_dataset(SyntheticConfig(weeks=104)).observations
    config = BaselineConfig(model=model, initial_train=65)
    original = evaluate_baseline(data, config)
    changed = data.copy()
    changed.loc[91:, "revenue"] *= 2
    repeated = evaluate_baseline(changed, config)
    np.testing.assert_array_equal(
        original.predictions["predicted"], repeated.predictions["predicted"]
    )
    assert original.cv_metrics == repeated.cv_metrics
    assert original.holdout_metrics != repeated.holdout_metrics
    for before, after in zip(original.fold_records, repeated.fold_records, strict=True):
        assert before["diagnostics"] == after["diagnostics"]


def test_domain_transform_has_no_future_state_or_reference_leakage() -> None:
    data = generate_dataset(SyntheticConfig(weeks=104)).observations.drop(
        columns="revenue"
    )
    features, references = regression_features(data, 52, 52, True)
    changed = data.copy()
    changed.loc[52:, "meta_spend"] *= 100
    second, repeated_refs = regression_features(changed, 52, 52, True)
    assert references == repeated_refs
    pd.testing.assert_frame_equal(features.iloc[:52], second.iloc[:52])
    assert not features.iloc[52:].equals(second.iloc[52:])


def test_scaler_and_nested_tuning_use_only_training_prefix() -> None:
    data = generate_dataset(SyntheticConfig(weeks=104)).observations
    config = BaselineConfig(model="ridge_domain", initial_train=65)
    original = evaluate_baseline(data, config)
    changed = data.copy()
    changed.loc[65:, "revenue"] += 100000
    changed.loc[65:, "meta_spend"] *= 100
    second = evaluate_baseline(changed, config)
    assert (
        original.fold_records[0]["diagnostics"] == second.fold_records[0]["diagnostics"]
    )
    diagnostic = cast(dict[str, Any], original.fold_records[0]["diagnostics"])
    assert isinstance(diagnostic, dict)
    features, _ = regression_features(data.drop(columns="revenue"), 65, 52, True)
    np.testing.assert_allclose(
        diagnostic["scaler_mean"], features.iloc[:65].mean().to_numpy()
    )
    inner = diagnostic["inner_validation"]
    assert isinstance(inner, dict)
    assert inner["train_rows"] == 52
    assert inner["validation_rows"] == 13


def test_nested_temporal_tuning_respects_arrival_gap() -> None:
    data = generate_dataset(SyntheticConfig(weeks=104)).observations
    available = data["week"] + pd.Timedelta(days=21)
    config = BaselineConfig(model="ridge_domain", initial_train=67, gap=2)
    result = evaluate_baseline(data, config, available)
    diagnostic = cast(dict[str, Any], result.fold_records[0]["diagnostics"])
    assert isinstance(diagnostic, dict)
    inner = diagnostic["inner_validation"]
    assert isinstance(inner, dict)
    assert inner["gap"] == 2
    assert inner["train_rows"] == 52
    with pytest.raises(ValueError, match="unavailable"):
        evaluate_baseline(data, replace(config, gap=0), available)


def test_conditional_artifact_is_auditable_and_not_causal(tmp_path: Path) -> None:
    directory = tmp_path / "data"
    write_dataset(generate_dataset(SyntheticConfig(weeks=104)).observations, directory)
    result = run_baseline(
        directory,
        tmp_path / "ridge",
        BaselineConfig(model="ridge_domain", initial_train=65),
    )
    assert result.fold_records[0]["diagnostics"]

    record = json.loads((tmp_path / "ridge" / "experiment.json").read_text())
    assert record["model_status"] == "PREDICTIVE_ONLY"
    assert record["prediction_context"] == "conditional_on_realized_covariates"
    assert "Cannot support attribution or budget allocation" in record["limitations"]
    assert record["feature_config"]["learned_preprocessing"]
