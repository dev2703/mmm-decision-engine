"""Reproducible observation defects, separate from the clean economic world."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from decisionguard.data.synthetic import SyntheticDataset


@dataclass(frozen=True)
class CorruptionConfig:
    """Exact defect counts; zero counts return an independent clean copy."""

    missing_weeks: int = 0
    duplicate_rows: int = 0
    seed: int = 42
    meta_alias: Literal["fb_spend", "facebook_spend"] | None = None
    unit_error_weeks: int = 0
    tracking_outage_weeks: int = 0
    erroneous_outlier_weeks: int = 0
    late_arrival_weeks: int = 0
    arrival_delay_days: int = 14
    mixed_currency_weeks: int = 0
    negative_spend_weeks: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("missing_weeks", self.missing_weeks),
            ("duplicate_rows", self.duplicate_rows),
            ("seed", self.seed),
            ("unit_error_weeks", self.unit_error_weeks),
            ("tracking_outage_weeks", self.tracking_outage_weeks),
            ("erroneous_outlier_weeks", self.erroneous_outlier_weeks),
            ("late_arrival_weeks", self.late_arrival_weeks),
            ("arrival_delay_days", self.arrival_delay_days),
            ("mixed_currency_weeks", self.mixed_currency_weeks),
            ("negative_spend_weeks", self.negative_spend_weeks),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.meta_alias not in (None, "fb_spend", "facebook_spend"):
            raise ValueError("unsupported meta_alias")
        if self.late_arrival_weeks and self.arrival_delay_days == 0:
            raise ValueError("late arrivals require a positive delay")


@dataclass(frozen=True)
class CorruptedDataset:
    """Dirty observations and an injection manifest; clean truth stays upstream."""

    observations: pd.DataFrame
    config: CorruptionConfig
    removed_weeks: tuple[pd.Timestamp, ...]
    duplicated_weeks: tuple[pd.Timestamp, ...]
    defects: tuple[tuple[str, tuple[pd.Timestamp, ...]], ...] = ()


def corrupt_dataset(
    clean: SyntheticDataset, config: CorruptionConfig | None = None
) -> CorruptedDataset:
    """Drop weeks, then copy retained rows; preserve the input tables exactly."""
    if config is None:
        config = CorruptionConfig()
    source = clean.observations
    if "week" not in source.columns:
        raise ValueError("clean observations require a week column")
    weeks = source["week"]
    if (
        not pd.api.types.is_datetime64_any_dtype(weeks.dtype)
        or weeks.isna().any()
        or not weeks.is_unique
        or not weeks.is_monotonic_increasing
    ):
        raise ValueError(
            "clean week keys must be datetime, non-null, unique and ordered"
        )
    if config.missing_weeks > len(source):
        raise ValueError("missing_weeks exceeds the available weeks")
    if config.duplicate_rows and config.missing_weeks == len(source):
        raise ValueError("cannot duplicate rows when no weeks remain")

    # A separate seed domain avoids reusing the economic generator's streams.
    seeds = np.random.SeedSequence([config.seed, 1]).spawn(8)
    missing_seed, duplicate_seed = seeds[:2]
    removed = np.sort(
        np.random.default_rng(missing_seed).choice(
            len(source), size=config.missing_weeks, replace=False
        )
    )
    retained = np.setdiff1d(np.arange(len(source)), removed)
    duplicated = np.random.default_rng(duplicate_seed).choice(
        retained, size=config.duplicate_rows, replace=True
    )
    selected = np.sort(np.concatenate((retained, duplicated)), kind="stable")
    affected = source.reset_index(drop=True).copy(deep=True)
    spend_columns = [name for name in source.columns if name.endswith("_spend")]
    defects: list[tuple[str, tuple[pd.Timestamp, ...]]] = []
    counts = (
        ("unit_error", config.unit_error_weeks),
        ("tracking_outage", config.tracking_outage_weeks),
        ("erroneous_outlier", config.erroneous_outlier_weeks),
        ("late_arrival", config.late_arrival_weeks),
        ("mixed_currency", config.mixed_currency_weeks),
        ("negative_spend", config.negative_spend_weeks),
    )
    for (kind, count), seed in zip(counts, seeds[2:], strict=True):
        if count > len(retained):
            raise ValueError(f"{kind} count exceeds retained weeks")
        if not count:
            continue
        rng = np.random.default_rng(seed)
        if kind == "tracking_outage":
            # A consecutive observed interval; missing weeks remain absent.
            start = int(rng.integers(0, len(retained) - count + 1))
            rows = retained[start : start + count]
        else:
            rows = rng.choice(retained, size=count, replace=False)
        keys = tuple(pd.Timestamp(value) for value in weeks.iloc[rows])
        defects.append((kind, keys))
        if kind == "unit_error":
            if "spend_unit" not in affected:
                affected["spend_unit"] = "base_units"
            affected.loc[affected.index[rows], spend_columns] /= 1_000
            affected.loc[affected.index[rows], "spend_unit"] = "thousands"
        elif kind == "tracking_outage":
            affected.loc[affected.index[rows], "revenue"] = np.nan
        elif kind == "erroneous_outlier":
            affected.loc[affected.index[rows], "revenue"] *= 10
        elif kind == "late_arrival":
            if "available_at" not in affected:
                affected["available_at"] = affected["week"] + pd.Timedelta(days=7)
            affected.loc[affected.index[rows], "available_at"] += pd.Timedelta(
                days=config.arrival_delay_days
            )
        elif kind == "mixed_currency":
            if "currency" not in affected:
                affected["currency"] = "AUD"
            # Values become USD at a known simulated rate: USD -> AUD = 1.5.
            affected.loc[
                affected.index[rows], [*spend_columns, "price", "revenue"]
            ] /= 1.5
            affected.loc[affected.index[rows], "currency"] = "USD"
        elif kind == "negative_spend":
            affected.loc[affected.index[rows], "search_spend"] *= -1
    if config.meta_alias is not None:
        affected = affected.rename(columns={"meta_spend": config.meta_alias})
    dirty = affected.iloc[selected].reset_index(drop=True)
    return CorruptedDataset(
        observations=dirty.copy(deep=True),
        config=config,
        removed_weeks=tuple(pd.Timestamp(value) for value in weeks.iloc[removed]),
        duplicated_weeks=tuple(pd.Timestamp(value) for value in weeks.iloc[duplicated]),
        defects=tuple(defects),
    )
