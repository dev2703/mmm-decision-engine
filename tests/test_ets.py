"""Classical forecasts preserve temporal boundaries and expose uncertainty."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from decisionguard.experiments.baseline import BaselineConfig, evaluate_baseline


def trend_data() -> pd.DataFrame:
    t = np.arange(64)
    return pd.DataFrame(
        {
            "week": pd.date_range("2022-01-03", periods=len(t), freq="W-MON"),
            "revenue": 100
            + 2 * t
            + 8 * np.sin(t * 2 * np.pi / 4)
            + np.random.default_rng(42).normal(0, 1, len(t)),
        }
    )


def test_ets_learns_trend_and_records_intervals_and_diagnostics() -> None:
    data = trend_data()
    config = BaselineConfig(period=4, initial_train=24, horizon=8, gap=2, model="ets")
    result = evaluate_baseline(data, config)
    naive = evaluate_baseline(data, replace(config, model="seasonal_naive"))
    assert float(result.holdout_metrics["mae"] or 0) < float(
        naive.holdout_metrics["mae"] or 0
    )
    p = result.predictions
    assert (p["lower_90"] <= p["predicted"]).all()
    assert (p["upper_90"] >= p["predicted"]).all()
    assert np.isfinite(p[["predicted", "lower_90", "upper_90"]].to_numpy()).all()
    for fold in result.fold_records:
        diagnostic = fold["diagnostics"]
        assert isinstance(diagnostic, dict)
        assert diagnostic["converged"]
        assert 0 <= diagnostic["test_coverage_90"] <= 1
        assert 0 <= diagnostic["ljung_box_pvalue"] <= 1


def test_ets_future_target_changes_do_not_change_first_forecast_or_intervals() -> None:
    data = trend_data()
    config = BaselineConfig(period=4, initial_train=24, horizon=8, model="ets")
    first = evaluate_baseline(data, config)
    data.loc[24:, "revenue"] += 50
    second = evaluate_baseline(data, config)
    pd.testing.assert_frame_equal(
        first.predictions.iloc[:8][["predicted", "lower_90", "upper_90"]],
        second.predictions.iloc[:8][["predicted", "lower_90", "upper_90"]],
    )


def test_ets_requires_two_seasons_and_checks_late_arrivals() -> None:
    with pytest.raises(ValueError, match="two complete"):
        BaselineConfig(period=52, initial_train=52, model="ets")
    data = trend_data()
    with pytest.raises(ValueError, match="unavailable"):
        evaluate_baseline(
            data,
            BaselineConfig(period=4, initial_train=24, model="ets"),
            data["week"] + pd.Timedelta(days=21),
        )
