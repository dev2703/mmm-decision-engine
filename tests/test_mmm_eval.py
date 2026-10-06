"""Optional upstream adapter contract tests; run in the evaluation environment."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

bridge = pytest.importorskip("decisionguard.evaluation.mmm_eval", exc_type=ImportError)


def adapter(data: pd.DataFrame, ledger: list[dict[str, object]]) -> Any:
    from decisionguard.models.bayesian import MMMConfig, build_mmm

    model = build_mmm(MMMConfig())
    config = bridge.PyMCConfig.from_model_object(
        model, revenue_column="revenue", response_column="response"
    )
    arrivals = pd.Series(
        (data["week"] + pd.Timedelta(days=7)).to_numpy(), index=data["week"]
    )
    return bridge.FoldSafePyMCAdapter(config, arrivals, ledger)


def observations() -> pd.DataFrame:
    from decisionguard.data.synthetic import SyntheticConfig, generate_dataset

    data = generate_dataset(SyntheticConfig(weeks=78)).observations
    data["response"] = data["revenue"]
    data["trend"] = np.arange(len(data), dtype=np.float64)
    return data


def test_upstream_rows_and_units_remain_unmodified() -> None:
    raw = pd.DataFrame(
        [
            {
                "test_name": "holdout_accuracy",
                "general_metric_name": "mape",
                "specific_metric_name": "mape",
                "metric_value": 15.5,
                "metric_pass": False,
            },
            {
                "test_name": "perturbation",
                "general_metric_name": "percentage_change",
                "specific_metric_name": "percentage_change_meta_spend",
                "metric_value": 9.0,
                "metric_pass": False,
            },
            {
                "test_name": "placebo",
                "general_metric_name": "shuffled_channel_roi",
                "specific_metric_name": "shuffled_channel_roi_search_spend_shuffled",
                "metric_value": -60.0,
                "metric_pass": True,
            },
        ]
    )
    before = raw.copy(deep=True)
    evidence = bridge.source_evidence(raw, ["meta_spend", "search_spend"])
    pd.testing.assert_frame_equal(raw, before)
    assert evidence[0].value == 15.5 and evidence[0].passed is False
    assert evidence[1].channel == "meta_spend" and evidence[1].value == 9.0
    assert evidence[2].value == -60.0 and evidence[2].passed is True


def test_adapter_fits_controls_only_on_its_training_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = observations()
    captured: list[pd.DataFrame] = []

    def fake_fit(self: Any, frame: pd.DataFrame) -> None:
        captured.append(frame.copy())
        self.trace = None

    monkeypatch.setattr(bridge.PyMCAdapter, "fit", fake_fit)

    def diagnostics(_: Any) -> dict[str, str]:
        return {"diagnostic_status": "ACCEPTABLE"}

    monkeypatch.setattr(bridge, "posterior_diagnostics", diagnostics)
    ledger: list[dict[str, object]] = []
    fit = adapter(data, ledger)
    fit.fit(data.iloc[:52])
    controls = ["price", "promotion", "macro_index", "competitor_index", "trend"]
    np.testing.assert_allclose(captured[0][controls].mean(), np.zeros(5), atol=1e-12)
    np.testing.assert_allclose(captured[0][controls].std(ddof=0), np.ones(5))
    expected = (data.iloc[52:][controls] - data.iloc[:52][controls].mean()) / data.iloc[
        :52
    ][controls].std(ddof=0)
    pd.testing.assert_frame_equal(fit._transform(data.iloc[52:])[controls], expected)
    assert ledger[0]["train_rows"] == 52
    clone = fit._create_adapter_with_placebo_channel("search_spend_shuffled")
    assert isinstance(clone, bridge.FoldSafePyMCAdapter)
    assert clone.ledger is ledger and clone.mean is None
    assert "search_spend_shuffled" in clone.config.channel_columns


def test_late_arrival_blocks_refit_before_upstream_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = observations()
    fit = adapter(data, [])
    fit.arrivals += pd.Timedelta(days=14)

    def must_not_fit(self: Any, frame: pd.DataFrame) -> None:
        pytest.fail("unavailable training reached the upstream model")

    monkeypatch.setattr(bridge.PyMCAdapter, "fit", must_not_fit)
    with pytest.raises(ValueError, match="unavailable"):
        fit.fit(data.iloc[:52])
