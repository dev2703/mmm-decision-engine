"""Release gates remain explicit, deterministic and immune to averaging."""

from dataclasses import replace

import pytest

from decisionguard.evaluation.policy import (
    MetricEvidence,
    ReleaseState,
    assess_release,
)

CHANNELS = ["search_spend", "meta_spend"]


def evidence() -> list[MetricEvidence]:
    rows: list[MetricEvidence] = []
    for test in ("in_sample_accuracy", "holdout_accuracy"):
        rows.extend(
            MetricEvidence(test, metric, 1.0, True)
            for metric in ("mape", "smape", "r_squared")
        )
    rows.extend(
        MetricEvidence("cross_validation", metric, 1.0, True)
        for metric in (
            "mean_mape",
            "std_mape",
            "mean_smape",
            "std_smape",
            "mean_r_squared",
        )
    )
    for channel in CHANNELS:
        rows.extend(
            MetricEvidence("refresh_stability", metric, 1.0, True, channel)
            for metric in ("mean_percentage_change", "std_percentage_change")
        )
        rows.append(
            MetricEvidence("perturbation", "percentage_change", 1.0, True, channel)
        )
    rows.append(MetricEvidence("placebo", "shuffled_channel_roi", -80.0, True))
    return rows


def test_complete_evidence_passes_and_warnings_are_retained() -> None:
    passed = assess_release(
        evidence(),
        {"diagnostic_status": "ACCEPTABLE"},
        "RESOLVED",
        CHANNELS,
        prior_sensitivity=dict.fromkeys(CHANNELS, 0.1),
    )
    assert passed.state == ReleaseState.PASS
    assert passed.channel_movement_limits == {}
    warned = assess_release(
        evidence(),
        {"diagnostic_status": "ACCEPTABLE"},
        "WARNING",
        CHANNELS,
        prior_sensitivity=dict.fromkeys(CHANNELS, 0.1),
    )
    assert warned.state == ReleaseState.WARN
    assert warned.reasons[0].code == "data_warning"


def test_placebo_failure_blocks_despite_all_other_passes() -> None:
    rows = [
        replace(row, passed=False) if row.test == "placebo" else row
        for row in evidence()
    ]
    result = assess_release(
        rows,
        {"diagnostic_status": "ACCEPTABLE"},
        "RESOLVED",
        CHANNELS,
        prior_sensitivity=dict.fromkeys(CHANNELS, 0.01),
    )
    assert result.state == ReleaseState.BLOCK
    assert all(value == 0 for value in result.channel_movement_limits.values())
    assert any(reason.code == "placebo" for reason in result.reasons)


def test_channel_failure_and_prior_sensitivity_restrict_only_affected_channels() -> (
    None
):
    rows = [
        replace(row, passed=False)
        if row.test == "perturbation" and row.channel == "meta_spend"
        else row
        for row in evidence()
    ]
    result = assess_release(
        rows,
        {"diagnostic_status": "ACCEPTABLE"},
        "RESOLVED",
        CHANNELS,
        prior_sensitivity={"search_spend": 0.01, "meta_spend": 0.25},
    )
    assert result.state == ReleaseState.RESTRICT
    assert result.channel_movement_limits == {"meta_spend": 0.05}
    assert {reason.code for reason in result.reasons} == {
        "perturbation",
        "prior_sensitivity",
    }
    assert result == assess_release(
        rows,
        {"diagnostic_status": "ACCEPTABLE"},
        "RESOLVED",
        CHANNELS,
        prior_sensitivity={"search_spend": 0.01, "meta_spend": 0.25},
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing_test",
        "undefined",
        "nonfinite",
        "unknown_channel",
        "leakage",
        "diagnostics",
        "sensitivity",
    ],
)
def test_incomplete_or_invalid_evidence_fails_closed(problem: str) -> None:
    rows = evidence()
    if problem == "missing_test":
        rows.pop()
    if problem in ("undefined", "nonfinite", "unknown_channel"):
        rows[0] = replace(
            rows[0],
            value=None
            if problem == "undefined"
            else float("nan")
            if problem == "nonfinite"
            else 1.0,
            channel="bad" if problem == "unknown_channel" else None,
        )
    result = assess_release(
        rows,
        {
            "diagnostic_status": "INVESTIGATE"
            if problem == "diagnostics"
            else "ACCEPTABLE"
        },
        "RESOLVED",
        CHANNELS,
        known_leakage=problem == "leakage",
        prior_sensitivity=None
        if problem == "sensitivity"
        else dict.fromkeys(CHANNELS, 0.01),
    )
    assert result.state == ReleaseState.BLOCK


def test_catastrophic_future_error_blocks_but_moderate_failure_warns() -> None:
    rows = [
        replace(row, passed=False, value=20.0)
        if row.test == "holdout_accuracy"
        else row
        for row in evidence()
    ]
    warning = assess_release(
        rows,
        {"diagnostic_status": "ACCEPTABLE"},
        "RESOLVED",
        CHANNELS,
        prior_sensitivity=dict.fromkeys(CHANNELS, 0.01),
    )
    assert warning.state == ReleaseState.WARN
    rows = [
        replace(row, value=35.0) if row.test == "holdout_accuracy" else row
        for row in rows
    ]
    blocked = assess_release(
        rows,
        {"diagnostic_status": "ACCEPTABLE"},
        "RESOLVED",
        CHANNELS,
        prior_sensitivity=dict.fromkeys(CHANNELS, 0.01),
    )
    assert blocked.state == ReleaseState.BLOCK
