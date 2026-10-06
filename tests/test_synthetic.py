"""Scientific invariants for the clean data-generating process."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from decisionguard.data.synthetic import (
    DEFAULT_CHANNELS,
    ChannelConfig,
    SyntheticConfig,
    generate_dataset,
    geometric_adstock,
)


def test_reproducible_seed_and_preserved_temporal_prefix() -> None:
    config = SyntheticConfig(weeks=104)
    first = generate_dataset(config)
    repeat = generate_dataset(config)
    longer = generate_dataset(replace(config, weeks=156))
    pd.testing.assert_frame_equal(first.observations, repeat.observations)
    pd.testing.assert_frame_equal(first.truth, repeat.truth)
    pd.testing.assert_frame_equal(first.observations, longer.observations.iloc[:104])
    pd.testing.assert_frame_equal(first.truth, longer.truth.iloc[:104])
    other = generate_dataset(replace(config, seed=43))
    assert not first.observations.equals(other.observations)


def test_weekly_schema_and_contribution_reconciliation() -> None:
    dataset = generate_dataset()
    data, truth = dataset.observations, dataset.truth
    expected = ["week", *(f"{c.name}_spend" for c in DEFAULT_CHANNELS)]
    expected += ["price", "promotion", "macro_index", "competitor_index", "revenue"]
    assert list(data.columns) == expected
    assert len(data) == dataset.config.weeks
    assert data["week"].is_unique
    pd.testing.assert_series_equal(
        data["week"],
        pd.Series(
            pd.date_range(
                dataset.config.start, periods=dataset.config.weeks, freq="W-MON"
            ),
            name="week",
        ),
    )
    assert not data.isna().any().any()
    assert np.isfinite(data.drop(columns="week").to_numpy()).all()
    assert (data.filter(like="_spend") >= 0).all().all()
    assert set(data["promotion"]) <= {0, 1}
    pd.testing.assert_series_equal(data["week"], truth["week"])
    np.testing.assert_allclose(
        data["revenue"],
        truth.filter(like="_contribution").sum(axis=1) + truth["noise"],
        rtol=0,
        atol=1e-9,
    )
    assert not any(name.endswith("_contribution") for name in data.columns)


def test_zero_effect_channel_and_truth_isolation() -> None:
    channels = (
        replace(DEFAULT_CHANNELS[0], maximum_contribution=0),
        *DEFAULT_CHANNELS[1:],
    )
    baseline = generate_dataset(SyntheticConfig(noise_std=0))
    zero_effect = generate_dataset(SyntheticConfig(channels=channels, noise_std=0))
    assert (zero_effect.truth["search_contribution"] == 0).all()
    np.testing.assert_allclose(
        baseline.observations["revenue"] - zero_effect.observations["revenue"],
        baseline.truth["search_contribution"],
        rtol=0,
        atol=1e-9,
    )
    snapshot = baseline.truth.copy(deep=True)
    baseline.observations.loc[0, "revenue"] = -1
    pd.testing.assert_frame_equal(baseline.truth, snapshot)


def test_carryover_is_causal_and_decays_after_an_impulse() -> None:
    spend = np.array([10.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(geometric_adstock(spend, 0.5), [10, 5, 2.5, 1.25])
    changed_future = spend.copy()
    changed_future[2:] = 100
    np.testing.assert_array_equal(
        geometric_adstock(spend, 0.5)[:2], geometric_adstock(changed_future, 0.5)[:2]
    )
    np.testing.assert_array_equal(geometric_adstock(spend, 0), spend)


def test_correlated_media_and_bounded_saturation() -> None:
    dataset = generate_dataset(SyntheticConfig(weeks=520))
    correlation = dataset.observations["search_spend"].corr(
        dataset.observations["meta_spend"]
    )
    assert 0.3 < correlation < 0.8
    for channel in dataset.config.channels:
        contribution = dataset.truth[f"{channel.name}_contribution"]
        assert (contribution >= 0).all()
        assert (contribution < channel.maximum_contribution).all()


def test_media_response_increases_with_diminishing_returns() -> None:
    outputs: list[pd.Series[float]] = []
    for multiplier in (1, 2, 4):
        channels = tuple(
            replace(channel, typical_spend=channel.typical_spend * multiplier)
            for channel in DEFAULT_CHANNELS
        )
        dataset = generate_dataset(SyntheticConfig(channels=channels))
        outputs.append(dataset.truth["search_contribution"])
    low, middle, high = outputs
    assert (middle > low).all()
    assert (high > middle).all()
    # Compare equal spend increments: C(2x)-C(x) > (C(4x)-C(2x))/2.
    assert ((middle - low) > (high - middle) / 2).all()


@pytest.mark.parametrize("weeks", [0, -1])
def test_invalid_horizon(weeks: int) -> None:
    with pytest.raises(ValueError, match="weeks"):
        SyntheticConfig(weeks=weeks)


def test_invalid_config() -> None:
    with pytest.raises(ValueError, match="seed"):
        SyntheticConfig(seed=-1)
    with pytest.raises(ValueError, match="Monday"):
        SyntheticConfig(start=date(2022, 1, 4))
    with pytest.raises(ValueError, match="noise_std"):
        SyntheticConfig(noise_std=float("nan"))
    with pytest.raises(ValueError, match="channels"):
        SyntheticConfig(channels=())


@pytest.mark.parametrize("decay", [-0.1, 1.0, float("nan")])
def test_invalid_decay(decay: float) -> None:
    with pytest.raises(ValueError):
        geometric_adstock(np.array([1.0]), decay)
    with pytest.raises(ValueError):
        ChannelConfig("search", 100, decay, 100, 100)


@pytest.mark.parametrize("spend", [[-1.0], [float("inf")], [float("nan")]])
def test_invalid_spend(spend: list[float]) -> None:
    with pytest.raises(ValueError, match="spend"):
        geometric_adstock(np.array(spend), 0.5)


def test_adstock_rejects_numeric_overflow() -> None:
    with pytest.raises(ValueError, match="overflowed"):
        geometric_adstock(np.array([1e308, 1e308, 1e308]), 0.9)


@pytest.mark.parametrize("shift_week", [0, 26, 51, 70])
@pytest.mark.parametrize("shift_amount", [-15_000.0, 0.0, 15_000.0])
def test_structural_shift_boundary_and_reconciliation(
    shift_week: int, shift_amount: float
) -> None:
    config = SyntheticConfig(weeks=52)
    baseline = generate_dataset(config)
    shifted = generate_dataset(
        replace(config, shift_week=shift_week, shift_amount=shift_amount)
    )
    expected = np.where(np.arange(52) >= shift_week, shift_amount, 0.0)
    np.testing.assert_array_equal(
        shifted.truth["structural_shift_contribution"], expected
    )
    np.testing.assert_allclose(
        shifted.observations["revenue"] - baseline.observations["revenue"],
        expected,
        rtol=0,
        atol=1e-9,
    )
    pd.testing.assert_frame_equal(
        shifted.observations.drop(columns="revenue"),
        baseline.observations.drop(columns="revenue"),
    )
    pd.testing.assert_frame_equal(
        shifted.truth.drop(columns="structural_shift_contribution"),
        baseline.truth.drop(columns="structural_shift_contribution"),
    )
    np.testing.assert_allclose(
        shifted.observations["revenue"],
        shifted.truth.filter(like="_contribution").sum(axis=1) + shifted.truth["noise"],
        rtol=0,
        atol=1e-9,
    )


@pytest.mark.parametrize("shift_week", [26, 80])
def test_structural_shift_reproducibility_and_temporal_prefix(shift_week: int) -> None:
    config = SyntheticConfig(weeks=52, shift_week=shift_week, shift_amount=15_000)
    first = generate_dataset(config)
    repeat = generate_dataset(config)
    extended = generate_dataset(replace(config, weeks=104))
    pd.testing.assert_frame_equal(first.observations, repeat.observations)
    pd.testing.assert_frame_equal(first.truth, repeat.truth)
    pd.testing.assert_frame_equal(first.observations, extended.observations.iloc[:52])
    pd.testing.assert_frame_equal(first.truth, extended.truth.iloc[:52])
    assert first.config == config


@pytest.mark.parametrize("shift_week", [-1, True])
def test_invalid_shift_week(shift_week: int) -> None:
    with pytest.raises(ValueError, match="shift_week"):
        SyntheticConfig(shift_week=shift_week)


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), -float("inf")])
def test_invalid_shift_amount(amount: float) -> None:
    with pytest.raises(ValueError, match="shift_amount"):
        SyntheticConfig(shift_week=26, shift_amount=amount)


def test_shift_amount_requires_start_week() -> None:
    with pytest.raises(ValueError, match="requires shift_week"):
        SyntheticConfig(shift_amount=15_000)


@pytest.mark.parametrize("amount", [-50_000.0, 50_000.0])
def test_commercial_event_is_genuine_ground_truth(amount: float) -> None:
    config = SyntheticConfig(weeks=40)
    baseline = generate_dataset(config)
    event = generate_dataset(replace(config, event_week=20, event_amount=amount))
    expected = np.zeros(40)
    expected[20] = amount
    np.testing.assert_allclose(
        event.observations["revenue"] - baseline.observations["revenue"],
        expected,
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        event.truth["commercial_event_contribution"], expected
    )
    pd.testing.assert_frame_equal(
        event.observations.drop(columns="revenue"),
        baseline.observations.drop(columns="revenue"),
    )
    extended = generate_dataset(
        replace(config, weeks=60, event_week=20, event_amount=amount)
    )
    pd.testing.assert_frame_equal(event.truth, extended.truth.iloc[:40])
    np.testing.assert_allclose(
        event.observations["revenue"],
        event.truth.filter(like="_contribution").sum(axis=1) + event.truth["noise"],
        rtol=0,
        atol=1e-9,
    )


def test_invalid_commercial_event() -> None:
    with pytest.raises(ValueError):
        SyntheticConfig(event_week=-1)
    with pytest.raises(ValueError):
        SyntheticConfig(event_week=1, event_amount=float("nan"))
    with pytest.raises(ValueError):
        SyntheticConfig(event_amount=1)
