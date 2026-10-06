"""Contracts, safe repairs, intervention gates, and temporal boundaries."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from decisionguard.data.corruption import CorruptionConfig, corrupt_dataset
from decisionguard.data.integrity import IntegrityConfig, clean_dataset, dataset_hash
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset


def test_safe_repairs_recover_source_and_are_deterministic() -> None:
    source = generate_dataset()
    dirty = corrupt_dataset(
        source,
        CorruptionConfig(
            meta_alias="facebook_spend",
            unit_error_weeks=8,
            mixed_currency_weeks=8,
            duplicate_rows=7,
        ),
    )
    snapshot = dirty.observations.copy(deep=True)
    config = IntegrityConfig(currency_rates=(("USD", 1.5),))
    first = clean_dataset(dirty.observations, config)
    second = clean_dataset(dirty.observations, config)
    assert first.report.status != "BLOCKER"
    assert first.report == second.report
    pd.testing.assert_frame_equal(first.clean, second.clean)
    pd.testing.assert_frame_equal(
        first.clean, source.observations, check_dtype=False, rtol=1e-12
    )
    pd.testing.assert_frame_equal(dirty.observations, snapshot)
    assert first.report.raw_hash == dataset_hash(snapshot)
    assert first.report.clean_hash == dataset_hash(first.clean)
    assert len(first.clean) == len(source.observations)
    assert (
        len(
            {issue.code for issue in first.report.issues}
            & {"taxonomy", "units", "currency", "exact_duplicates"}
        )
        == 4
    )


@pytest.mark.parametrize(
    "kind", ["tracking_outage_weeks", "negative_spend_weeks", "mixed_currency_weeks"]
)
def test_unresolved_defects_block_modeling(kind: str) -> None:
    source = generate_dataset()
    config = replace(CorruptionConfig(), **{kind: 3})
    dirty = corrupt_dataset(source, config)
    result = clean_dataset(dirty.observations)
    assert result.report.status == "BLOCKER"
    with pytest.raises(ValueError, match="blocked"):
        result.require_model_ready()


def test_missing_edges_detected_from_explicit_window() -> None:
    source = generate_dataset(SyntheticConfig(weeks=10)).observations
    raw = source.iloc[1:-1].copy()
    result = clean_dataset(
        raw, IntegrityConfig(expected_start="2022-01-03", expected_end="2022-03-07")
    )
    assert result.report.status == "BLOCKER"
    assert result.report.profile["missing_weeks"] == ["2022-01-03", "2022-03-07"]
    assert len(result.clean) == 8  # no invented zero rows or target imputations


def test_conflicting_duplicates_never_aggregate() -> None:
    source = generate_dataset(SyntheticConfig(weeks=10)).observations
    duplicate = source.iloc[:1].copy()
    duplicate["revenue"] *= 2
    raw = pd.concat([source, duplicate], ignore_index=True)
    result = clean_dataset(raw)
    assert result.report.status == "BLOCKER"
    assert len(result.clean) == 11
    assert result.clean["week"].duplicated().sum() == 1
    blocked = clean_dataset(
        pd.concat([source, source.iloc[:1]], ignore_index=True),
        IntegrityConfig(duplicate_policy="block"),
    )
    assert blocked.report.status == "BLOCKER"


def test_as_of_boundary_excludes_future_availability() -> None:
    source = generate_dataset(SyntheticConfig(weeks=10))
    dirty = corrupt_dataset(source, CorruptionConfig(late_arrival_weeks=10))
    config = IntegrityConfig(as_of="2022-02-07")
    assert config.as_of is not None
    result = clean_dataset(dirty.observations, config)
    assert result.report.status == "BLOCKER"
    assert all(
        result.clean["week"] + pd.Timedelta(days=21) <= pd.Timestamp(config.as_of)
    )
    assert len(result.clean) == 3
    earlier = clean_dataset(dirty.observations, replace(config, as_of="2022-01-10"))
    assert len(earlier.clean) == 0
    no_metadata = clean_dataset(
        source.observations, IntegrityConfig(as_of="2022-01-10")
    )
    assert len(no_metadata.clean) == 1


def test_future_changes_do_not_modify_cleaned_training_values() -> None:
    source = generate_dataset(SyntheticConfig(weeks=40)).observations
    training = clean_dataset(source.iloc[:20])
    changed_future = source.copy(deep=True)
    changed_future.loc[20:, "revenue"] *= 100
    whole = clean_dataset(changed_future)
    pd.testing.assert_frame_equal(training.clean, whole.clean.iloc[:20])
    assert len(whole.clean) == 40
    assert any(
        issue.code == "outliers" or issue.code == "structural_change"
        for issue in whole.report.issues
    )


def test_genuine_event_and_structural_shift_are_flagged_and_retained() -> None:
    source = generate_dataset(
        SyntheticConfig(
            weeks=52,
            event_week=20,
            event_amount=1_000_000,
            shift_week=26,
            shift_amount=200_000,
        )
    )
    result = clean_dataset(source.observations)
    assert result.report.status == "WARNING"
    assert any(issue.code == "structural_change" for issue in result.report.issues)
    assert any(issue.code == "outliers" for issue in result.report.issues)
    pd.testing.assert_frame_equal(result.clean, source.observations, check_dtype=False)
    profile = result.report.profile
    assert profile["row_count"] == 52
    assert (
        "distributions" in profile
        and "cardinality" in profile
        and "missingness" in profile
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("revenue", np.nan),
        ("price", 0.0),
        ("promotion", 2.0),
        ("search_spend", -1.0),
        ("macro_index", np.inf),
    ],
)
def test_invalid_values_are_blocked_without_imputation(
    column: str, value: float
) -> None:
    raw = generate_dataset(SyntheticConfig(weeks=10)).observations
    raw.loc[0, column] = value
    result = clean_dataset(raw)
    assert result.report.status == "BLOCKER"


def test_schema_and_alias_collisions_block() -> None:
    raw = generate_dataset(SyntheticConfig(weeks=10)).observations
    missing = clean_dataset(raw.drop(columns="revenue"))
    assert missing.report.status == "BLOCKER"
    extra = raw.assign(search_contribution=1)
    assert clean_dataset(extra).report.status == "BLOCKER"
    collision = raw.assign(fb_spend=raw["meta_spend"])
    assert clean_dataset(collision).report.status == "BLOCKER"
    double_columns = raw.copy()
    double_columns.columns = ["week", *(["value"] * (len(raw.columns) - 1))]
    assert clean_dataset(double_columns).report.status == "BLOCKER"


def test_iso_dates_sort_and_invalid_dates_block() -> None:
    raw = generate_dataset(SyntheticConfig(weeks=10)).observations
    strings = raw.iloc[::-1].copy()
    strings["week"] = strings["week"].dt.strftime("%Y-%m-%d")
    repaired = clean_dataset(strings)
    pd.testing.assert_frame_equal(repaired.clean, raw, check_dtype=False)
    strings.loc[0, "week"] = "bad date"
    assert clean_dataset(strings).report.status == "BLOCKER"
    raw["week"] += pd.Timedelta(days=1)
    assert clean_dataset(raw).report.status == "BLOCKER"


def test_unknown_units_and_null_currency_block() -> None:
    raw = generate_dataset(SyntheticConfig(weeks=10)).observations
    assert clean_dataset(raw.assign(spend_unit="unknown")).report.status == "BLOCKER"
    assert clean_dataset(raw.assign(currency=None)).report.status == "BLOCKER"


def test_invalid_integrity_configuration() -> None:
    with pytest.raises(ValueError):
        IntegrityConfig(as_of="NaT")
    with pytest.raises(ValueError):
        IntegrityConfig(expected_start="2022-01-04")
    with pytest.raises(ValueError):
        IntegrityConfig(expected_start="2022-02-07", expected_end="2022-01-03")
    with pytest.raises(ValueError):
        IntegrityConfig(currency_rates=(("USD", 0),))
    with pytest.raises(ValueError):
        IntegrityConfig(currency_rates=(("USD", 1.5), ("USD", 1.6)))


def test_mixed_timezones_and_numeric_dates_are_blocked() -> None:
    raw = generate_dataset(SyntheticConfig(weeks=10)).observations
    invalid = raw.copy()
    invalid["week"] = raw["week"].dt.strftime("%Y-%m-%d")
    invalid.loc[0, "week"] = "2022-01-03T00:00:00+10:00"
    assert clean_dataset(invalid).report.status == "BLOCKER"
    numeric = raw.assign(week=list(range(10)))
    assert clean_dataset(numeric).report.status == "BLOCKER"
    arrival = raw.assign(available_at="2022-03-14")
    arrival.loc[0, "available_at"] = "2022-01-10T00:00:00Z"
    assert clean_dataset(arrival).report.status == "BLOCKER"
