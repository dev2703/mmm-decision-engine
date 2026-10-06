"""Sampler and feature contract, independent of the Bayesian runtime."""

from dataclasses import dataclass
from math import isfinite

from decisionguard.data.integrity import SPEND_COLUMNS


@dataclass(frozen=True)
class MMMConfig:
    draws: int = 2000
    tune: int = 1500
    chains: int = 4
    seed: int = 42
    holdout: int = 13
    adstock_lags: int = 16
    max_tree_depth: int = 12
    target_accept: float = 0.99
    media_prior_mean: float = 0.15
    media_prior_sigma: float = 0.1
    channels: tuple[str, ...] = SPEND_COLUMNS

    def __post_init__(self) -> None:
        for name in (
            "draws",
            "tune",
            "chains",
            "holdout",
            "adstock_lags",
            "max_tree_depth",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0.5 < self.target_accept < 1:
            raise ValueError("target_accept must lie between 0.5 and 1")
        if (
            not self.channels
            or len(set(self.channels)) != len(self.channels)
            or not set(self.channels) <= set(SPEND_COLUMNS)
        ):
            raise ValueError("channels must be unique known spend columns")
        object.__setattr__(self, "channels", tuple(self.channels))
        if not all(
            isfinite(v) and v > 0
            for v in (self.media_prior_mean, self.media_prior_sigma)
        ):
            raise ValueError("media prior parameters must be finite and positive")
