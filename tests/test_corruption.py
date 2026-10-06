"""Defect counts, reproducibility, and preservation of the clean source."""

from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest

from decisionguard.data.corruption import CorruptionConfig, corrupt_dataset
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset


def test_exact_defects_and_auditable_rows() -> None:
    clean = generate_dataset(SyntheticConfig(weeks=20))
    config = CorruptionConfig(missing_weeks=4, duplicate_rows=7, seed=123)
    result = corrupt_dataset(clean, config)
    dirty = result.observations
    assert result.config == config
    assert len(dirty) == 23
    assert len(result.removed_weeks) == len(set(result.removed_weeks)) == 4
    assert len(result.duplicated_weeks) == 7
    assert set(result.removed_weeks).isdisjoint(result.duplicated_weeks)
    assert dirty["week"].nunique() == 16
    assert dirty.duplicated("week").sum() == 7
    assert dirty["week"].is_monotonic_increasing
    assert set(clean.observations["week"]) - set(dirty["week"]) == set(
        result.removed_weeks
    )
    counts = Counter(result.duplicated_weeks)
    for week, rows in dirty.groupby("week"):
        assert isinstance(week, pd.Timestamp)
        assert len(rows) == 1 + counts[week]
    pd.testing.assert_frame_equal(
        dirty.drop_duplicates("week").reset_index(drop=True),
        clean.observations.loc[
            ~clean.observations["week"].isin(result.removed_weeks)
        ].reset_index(drop=True),
    )
    # Every appended duplicate has exactly the original values, not just its key.
    expected = clean.observations.set_index("week").loc[dirty["week"]].reset_index()
    pd.testing.assert_frame_equal(dirty, expected)


def test_reproducibility_and_independent_missing_selection() -> None:
    clean = generate_dataset()
    config = CorruptionConfig(missing_weeks=10, duplicate_rows=8)
    first = corrupt_dataset(clean, config)
    repeat = corrupt_dataset(clean, config)
    pd.testing.assert_frame_equal(first.observations, repeat.observations)
    assert first.removed_weeks == repeat.removed_weeks
    assert first.duplicated_weeks == repeat.duplicated_weeks
    more_duplicates = corrupt_dataset(clean, replace(config, duplicate_rows=20))
    assert first.removed_weeks == more_duplicates.removed_weeks
    other_seed = corrupt_dataset(clean, replace(config, seed=43))
    assert first.removed_weeks != other_seed.removed_weeks


@pytest.mark.parametrize("missing,duplicates", [(0, 0), (3, 0), (0, 4), (3, 4)])
def test_corruption_and_later_mutation_preserve_clean_tables(
    missing: int, duplicates: int
) -> None:
    clean = generate_dataset(
        SyntheticConfig(weeks=10, shift_week=5, shift_amount=10_000)
    )
    observations = clean.observations.copy(deep=True)
    truth = clean.truth.copy(deep=True)
    result = corrupt_dataset(clean, CorruptionConfig(missing, duplicates))
    pd.testing.assert_frame_equal(clean.observations, observations)
    pd.testing.assert_frame_equal(clean.truth, truth)
    assert not any(
        name.endswith("_contribution") for name in result.observations.columns
    )
    result.observations.loc[0, "revenue"] = -1
    result.observations.loc[0, "week"] = pd.Timestamp("2000-01-03")
    pd.testing.assert_frame_equal(clean.observations, observations)
    pd.testing.assert_frame_equal(clean.truth, truth)


def test_no_op_and_all_missing_preserve_schema() -> None:
    clean = generate_dataset(SyntheticConfig(weeks=10))
    unchanged = corrupt_dataset(clean)
    pd.testing.assert_frame_equal(unchanged.observations, clean.observations)
    assert unchanged.removed_weeks == unchanged.duplicated_weeks == ()
    empty = corrupt_dataset(clean, CorruptionConfig(missing_weeks=10))
    pd.testing.assert_frame_equal(empty.observations, clean.observations.iloc[:0])
    assert len(empty.removed_weeks) == 10


def test_repeated_duplication_of_one_survivor() -> None:
    clean = generate_dataset(SyntheticConfig(weeks=2))
    result = corrupt_dataset(clean, CorruptionConfig(missing_weeks=1, duplicate_rows=5))
    assert len(result.observations) == 6
    assert result.observations["week"].nunique() == 1
    assert len(set(result.duplicated_weeks)) == 1


@pytest.mark.parametrize("value", [-1, True])
def test_invalid_config(value: int) -> None:
    with pytest.raises(ValueError, match="missing_weeks"):
        CorruptionConfig(missing_weeks=value)
    with pytest.raises(ValueError, match="duplicate_rows"):
        CorruptionConfig(duplicate_rows=value)
    with pytest.raises(ValueError, match="seed"):
        CorruptionConfig(seed=value)


def test_infeasible_counts_fail_without_mutating_source() -> None:
    clean = generate_dataset(SyntheticConfig(weeks=2))
    snapshot = clean.observations.copy(deep=True)
    with pytest.raises(ValueError, match="exceeds"):
        corrupt_dataset(clean, CorruptionConfig(missing_weeks=3))
    with pytest.raises(ValueError, match="no weeks remain"):
        corrupt_dataset(clean, CorruptionConfig(missing_weeks=2, duplicate_rows=1))
    pd.testing.assert_frame_equal(clean.observations, snapshot)


def test_dirty_input_is_rejected() -> None:
    clean = generate_dataset(SyntheticConfig(weeks=2))
    clean.observations.loc[1, "week"] = clean.observations.loc[0, "week"]
    with pytest.raises(ValueError, match="unique"):
        corrupt_dataset(clean)


@pytest.mark.parametrize(
    "kind",
    [
        "unit_error_weeks",
        "tracking_outage_weeks",
        "erroneous_outlier_weeks",
        "late_arrival_weeks",
        "mixed_currency_weeks",
        "negative_spend_weeks",
    ],
)
def test_each_extended_defect_is_reproducible_and_preserves_truth(kind: str) -> None:
    clean = generate_dataset(SyntheticConfig(weeks=30))
    observations = clean.observations.copy(deep=True)
    truth = clean.truth.copy(deep=True)
    config = replace(CorruptionConfig(), **{kind: 3})
    first = corrupt_dataset(clean, config)
    repeat = corrupt_dataset(clean, config)
    pd.testing.assert_frame_equal(first.observations, repeat.observations)
    assert first.defects == repeat.defects
    assert len(first.defects[0][1]) == 3
    pd.testing.assert_frame_equal(clean.observations, observations)
    pd.testing.assert_frame_equal(clean.truth, truth)
    dirty = first.observations
    if kind == "unit_error_weeks":
        assert (dirty["spend_unit"] == "thousands").sum() == 3
    elif kind == "tracking_outage_weeks":
        assert dirty["revenue"].isna().sum() == 3
    elif kind == "erroneous_outlier_weeks":
        assert (dirty["revenue"] != observations["revenue"]).sum() == 3
    elif kind == "late_arrival_weeks":
        assert (dirty["available_at"] > dirty["week"] + pd.Timedelta(days=7)).sum() == 3
    elif kind == "mixed_currency_weeks":
        assert (dirty["currency"] == "USD").sum() == 3
    else:
        assert (dirty["search_spend"] < 0).sum() == 3


def test_alias_and_combined_defects_preserve_counts() -> None:
    clean = generate_dataset(SyntheticConfig(weeks=30))
    config = CorruptionConfig(
        missing_weeks=2,
        duplicate_rows=3,
        meta_alias="fb_spend",
        unit_error_weeks=5,
        tracking_outage_weeks=3,
        erroneous_outlier_weeks=2,
        late_arrival_weeks=4,
        mixed_currency_weeks=2,
        negative_spend_weeks=1,
    )
    first = corrupt_dataset(clean, config)
    second = corrupt_dataset(clean, config)
    assert len(first.observations) == 31
    assert "fb_spend" in first.observations and "meta_spend" not in first.observations
    assert set(first.removed_weeks).isdisjoint(
        key for _, keys in first.defects for key in keys
    )
    pd.testing.assert_frame_equal(first.observations, second.observations)
    assert first.defects == second.defects


def test_extended_defect_cannot_exceed_retained_weeks() -> None:
    clean = generate_dataset(SyntheticConfig(weeks=3))
    with pytest.raises(ValueError, match="exceeds retained"):
        corrupt_dataset(clean, CorruptionConfig(missing_weeks=2, unit_error_weeks=2))
