"""Explicit gates: critical failures cannot be averaged away by good predictions."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from decisionguard.models.diagnostics import acceptable_diagnostics


class ReleaseState(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    RESTRICT = "RESTRICT"
    BLOCK = "BLOCK"


REQUIRED_TESTS = frozenset(
    (
        "in_sample_accuracy",
        "holdout_accuracy",
        "cross_validation",
        "refresh_stability",
        "perturbation",
        "placebo",
    )
)


EXPECTED_METRICS = {
    "in_sample_accuracy": {"mape", "smape", "r_squared"},
    "holdout_accuracy": {"mape", "smape", "r_squared"},
    "cross_validation": {
        "mean_mape",
        "std_mape",
        "mean_smape",
        "std_smape",
        "mean_r_squared",
    },
    "refresh_stability": {"mean_percentage_change", "std_percentage_change"},
    "perturbation": {"percentage_change"},
    "placebo": {"shuffled_channel_roi"},
}


@dataclass(frozen=True)
class MetricEvidence:
    test: str
    metric: str
    value: float | None
    passed: bool | None
    channel: str | None = None


@dataclass(frozen=True)
class ReleaseReason:
    code: str
    state: ReleaseState
    message: str
    channel: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class ReleaseDecision:
    state: ReleaseState
    reasons: tuple[ReleaseReason, ...]
    channel_movement_limits: dict[str, float]
    policy_version: str = "release-v1"


def assess_release(
    evidence: Sequence[MetricEvidence],
    diagnostics: Mapping[str, object],
    quality_status: str,
    channels: Sequence[str],
    *,
    known_leakage: bool = False,
    prior_sensitivity: Mapping[str, float | None] | None = None,
) -> ReleaseDecision:
    """Use source pass/fail flags; retain every failure as separate evidence."""
    reasons: list[ReleaseReason] = []
    restricted: set[str] = set()
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("unique decision channels required")
    if quality_status not in ("RESOLVED", "WARNING"):
        reasons.append(
            ReleaseReason(
                "data_contract",
                ReleaseState.BLOCK,
                "Data-quality contract blocks modeling",
            )
        )
    elif quality_status == "WARNING":
        reasons.append(
            ReleaseReason(
                "data_warning",
                ReleaseState.WARN,
                "Data warnings require review; retained in model evidence",
            )
        )
    if known_leakage:
        reasons.append(
            ReleaseReason(
                "leakage", ReleaseState.BLOCK, "Known temporal or preprocessing leakage"
            )
        )
    if diagnostics.get(
        "diagnostic_status"
    ) != "ACCEPTABLE" or not acceptable_diagnostics(diagnostics):
        reasons.append(
            ReleaseReason(
                "sampler_diagnostics",
                ReleaseState.BLOCK,
                "Sampler diagnostics are incomplete or require investigation",
            )
        )
    missing = REQUIRED_TESTS - {item.test for item in evidence}
    if missing:
        reasons.append(
            ReleaseReason(
                "missing_checks",
                ReleaseState.BLOCK,
                "Required evaluation missing: " + ", ".join(sorted(missing)),
            )
        )
    for test, metrics in EXPECTED_METRICS.items():
        expected_channels = (
            channels if test in ("refresh_stability", "perturbation") else (None,)
        )
        for channel in expected_channels:
            observed = {
                row.metric
                for row in evidence
                if row.test == test and (channel is None or row.channel == channel)
            }
            if not metrics <= observed:
                reasons.append(
                    ReleaseReason(
                        "missing_metrics",
                        ReleaseState.BLOCK,
                        f"Incomplete {test} metrics",
                        channel,
                    )
                )
    for item in evidence:
        if (
            item.test not in REQUIRED_TESTS
            or item.metric not in EXPECTED_METRICS.get(item.test, set())
            or type(item.passed) is not bool
            or item.value is None
            or not math.isfinite(item.value)
        ):
            reasons.append(
                ReleaseReason(
                    "invalid_evidence",
                    ReleaseState.BLOCK,
                    "Unknown, undefined or nonfinite evaluation evidence",
                    item.channel,
                )
            )
            continue
        # The shuffled placebo is an intentionally artificial channel.
        if (
            item.channel is not None
            and item.channel not in channels
            and item.test != "placebo"
        ):
            reasons.append(
                ReleaseReason(
                    "unknown_channel",
                    ReleaseState.BLOCK,
                    "Unknown decision channel",
                    item.channel,
                )
            )
            continue
        catastrophic = (
            item.test in ("holdout_accuracy", "cross_validation")
            and item.metric in ("mape", "mean_mape")
            and item.value > 30
        )
        if item.passed and not catastrophic:
            continue
        if item.test == "placebo":
            state = ReleaseState.BLOCK
        elif item.test in ("refresh_stability", "perturbation"):
            state = ReleaseState.RESTRICT
            restricted.update([item.channel] if item.channel is not None else channels)
        elif catastrophic:
            state = ReleaseState.BLOCK
        else:
            state = ReleaseState.WARN
        reasons.append(
            ReleaseReason(
                item.test,
                state,
                f"{item.metric} exceeded the hard release limit"
                if catastrophic
                else f"Upstream {item.metric} failed its source threshold",
                item.channel,
                item.value,
            )
        )
    if prior_sensitivity is None or set(prior_sensitivity) != set(channels):
        reasons.append(
            ReleaseReason(
                "missing_sensitivity",
                ReleaseState.BLOCK,
                "Prior sensitivity must cover every decision channel",
            )
        )
    else:
        for channel, change in prior_sensitivity.items():
            if change is None or not math.isfinite(change) or change < 0:
                reasons.append(
                    ReleaseReason(
                        "invalid_sensitivity",
                        ReleaseState.BLOCK,
                        "Prior sensitivity is undefined",
                        channel,
                    )
                )
            elif change > 0.2:
                restricted.add(channel)
                reasons.append(
                    ReleaseReason(
                        "prior_sensitivity",
                        ReleaseState.RESTRICT,
                        "ROI mean changes over 20% under a plausible prior",
                        channel,
                        change,
                    )
                )
    priority = list(ReleaseState)
    state = max(
        (reason.state for reason in reasons),
        key=priority.index,
        default=ReleaseState.PASS,
    )
    limits = dict.fromkeys(sorted(restricted), 0.05)
    if state == ReleaseState.BLOCK:
        limits = dict.fromkeys(channels, 0.0)
    return ReleaseDecision(state, tuple(reasons), limits)
