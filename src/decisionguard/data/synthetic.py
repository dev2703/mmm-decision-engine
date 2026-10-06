"""A controlled weekly marketing world with explicit revenue components."""

from dataclasses import dataclass
from datetime import date
from math import isfinite

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChannelConfig:
    """Spend in AUD/week; carryover is unnormalized; response is AUD/week."""

    name: str
    typical_spend: float
    decay: float
    half_saturation: float
    maximum_contribution: float

    def __post_init__(self) -> None:
        values = (
            self.typical_spend,
            self.decay,
            self.half_saturation,
            self.maximum_contribution,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Channel parameters must be finite")
        if self.typical_spend <= 0 or self.half_saturation <= 0:
            raise ValueError("Spend and half_saturation must be positive")
        if not 0 <= self.decay < 1 or self.maximum_contribution < 0:
            raise ValueError(
                "Decay must be in [0, 1); contribution must be nonnegative"
            )


DEFAULT_CHANNELS = (
    ChannelConfig("search", 8_000, 0.2, 15_000, 25_000),
    ChannelConfig("meta", 6_500, 0.4, 15_000, 20_000),
    ChannelConfig("tv", 12_000, 0.8, 50_000, 30_000),
    ChannelConfig("ooh", 3_500, 0.7, 15_000, 10_000),
    ChannelConfig("youtube", 5_000, 0.3, 10_000, 15_000),
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Immutable simulation settings; weeks are labeled by Monday start date."""

    weeks: int = 156
    seed: int = 42
    start: date = date(2022, 1, 3)
    noise_std: float = 2_500
    channels: tuple[ChannelConfig, ...] = DEFAULT_CHANNELS
    shift_week: int | None = None
    shift_amount: float = 0.0
    event_week: int | None = None
    event_amount: float = 0.0

    def __post_init__(self) -> None:
        if type(self.weeks) is not int or self.weeks <= 0:
            raise ValueError("weeks must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.start) is not date or self.start.weekday() != 0:
            raise ValueError("start must be a Monday date")
        if not isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        if self.shift_week is not None and (
            type(self.shift_week) is not int or self.shift_week < 0
        ):
            raise ValueError("shift_week must be a nonnegative integer or None")
        if not isfinite(self.shift_amount):
            raise ValueError("shift_amount must be finite")
        if self.shift_week is None and self.shift_amount != 0:
            raise ValueError("a nonzero shift_amount requires shift_week")
        if self.event_week is not None and (
            type(self.event_week) is not int or self.event_week < 0
        ):
            raise ValueError("event_week must be a nonnegative integer or None")
        if not isfinite(self.event_amount):
            raise ValueError("event_amount must be finite")
        if self.event_week is None and self.event_amount != 0:
            raise ValueError("a nonzero event_amount requires event_week")
        if tuple(channel.name for channel in self.channels) != tuple(
            channel.name for channel in DEFAULT_CHANNELS
        ):
            raise ValueError("channels must be search, meta, tv, ooh, youtube in order")


@dataclass(frozen=True)
class SyntheticDataset:
    """Separate tables: model inputs/target and privileged simulation truth."""

    observations: pd.DataFrame
    truth: pd.DataFrame
    config: SyntheticConfig


def geometric_adstock(spend: NDArray[np.float64], decay: float) -> NDArray[np.float64]:
    """Compute A[t] = spend[t] + decay*A[t-1], starting from zero history."""
    if not isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("decay must be finite and in [0, 1)")
    if spend.ndim != 1 or not np.all(np.isfinite(spend)) or np.any(spend < 0):
        raise ValueError("spend must be a finite nonnegative one-dimensional array")
    result = np.empty(spend.shape, dtype=np.float64)
    previous = 0.0
    for index, value in enumerate(spend):
        previous = float(value) + decay * previous
        if not isfinite(previous):
            raise ValueError("adstock overflowed; reduce spend or decay")
        result[index] = previous
    return result


def generate_dataset(config: SyntheticConfig | None = None) -> SyntheticDataset:
    """Generate clean observations without fitted transforms or future inputs."""
    if config is None:
        config = SyntheticConfig()
    # One stream per process prevents longer horizons shifting other variables.
    streams = [
        np.random.default_rng(seed)
        for seed in np.random.SeedSequence(config.seed).spawn(11)
    ]
    week = np.arange(config.weeks, dtype=np.float64)
    season = np.sin(2 * np.pi * week / 52)
    dates = pd.date_range(config.start, periods=config.weeks, freq="W-MON")
    observations = pd.DataFrame({"week": dates})
    truth = pd.DataFrame({"week": dates})
    common = streams[0].normal(size=config.weeks)
    for index, channel in enumerate(config.channels):
        private = streams[index + 1].normal(size=config.weeks)
        shock = (common + private) / np.sqrt(2) if index < 2 else private
        spend = channel.typical_spend * np.exp(0.35 * shock - 0.35**2 / 2)
        adstock = geometric_adstock(spend, channel.decay)
        observations[f"{channel.name}_spend"] = spend
        truth[f"{channel.name}_adstock"] = adstock
        truth[f"{channel.name}_contribution"] = channel.maximum_contribution * (
            adstock / (channel.half_saturation + adstock)
        )

    observations["price"] = 100 + streams[6].normal(0, 5, config.weeks)
    observations["promotion"] = streams[7].binomial(1, 0.15, config.weeks)
    observations["macro_index"] = 100 + streams[8].normal(0, 2, config.weeks)
    observations["competitor_index"] = 100 + streams[9].normal(0, 5, config.weeks)
    truth["baseline_contribution"] = 100_000 + 200 * week + 15_000 * season
    shift = np.zeros(config.weeks, dtype=np.float64)
    if config.shift_week is not None:
        shift[config.shift_week :] = config.shift_amount
    truth["structural_shift_contribution"] = shift
    event = np.zeros(config.weeks, dtype=np.float64)
    if config.event_week is not None and config.event_week < config.weeks:
        event[config.event_week] = config.event_amount
    truth["commercial_event_contribution"] = event
    truth["price_contribution"] = -800 * (observations["price"] - 100)
    truth["promotion_contribution"] = 12_000 * observations["promotion"]
    truth["macro_contribution"] = 1_000 * (observations["macro_index"] - 100)
    truth["competitor_contribution"] = -300 * (observations["competitor_index"] - 100)
    truth["noise"] = streams[10].normal(0, config.noise_std, config.weeks)
    components = [name for name in truth.columns if name.endswith("_contribution")]
    observations["revenue"] = truth[components].sum(axis=1) + truth["noise"]
    return SyntheticDataset(observations, truth, config)
