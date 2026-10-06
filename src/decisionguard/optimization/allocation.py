"""Finite carryover response and constrained posterior allocation analysis."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import (  # pyright: ignore[reportMissingTypeStubs]
    minimize,  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
)

from decisionguard.evaluation.policy import ReleaseDecision, ReleaseState

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class PosteriorResponse:
    """Joint draws; amplitudes in AUD/week, half-saturation in raw adstocked AUD."""

    channels: tuple[str, ...]
    decay: Vector
    amplitude: Vector
    half_saturation: Vector
    history: Vector
    horizon: int = 13
    lags: int = 16
    _multiplier: Vector = field(init=False, repr=False)
    _carry_in: Vector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        shape = self.decay.shape
        if (
            not self.channels
            or len(set(self.channels)) != len(self.channels)
            or len(shape) != 2
            or not shape[0]
            or shape[1] != len(self.channels)
            or self.amplitude.shape != shape
            or self.half_saturation.shape != shape
            or self.history.ndim != 2
            or not len(self.history)
            or self.history.shape[1] != shape[1]
        ):
            raise ValueError(
                "aligned nonempty joint posterior draws and history required"
            )
        if any(
            not np.isfinite(values).all()
            for values in (
                self.decay,
                self.amplitude,
                self.half_saturation,
                self.history,
            )
        ):
            raise ValueError("response inputs must be finite")
        if (
            (self.decay < 0).any()
            or (self.decay >= 1).any()
            or (self.amplitude <= 0).any()
            or (self.half_saturation <= 0).any()
            or (self.history < 0).any()
        ):
            raise ValueError("invalid decay, amplitude, saturation or historical spend")
        if any(type(v) is not int or v <= 0 for v in (self.horizon, self.lags)):
            raise ValueError("horizon and lags must be positive integers")

        # Own immutable arrays so later caller mutation cannot change cached carry-in.
        for name in ("decay", "amplitude", "half_saturation", "history"):
            values = getattr(self, name).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        multiplier = np.zeros((shape[0], self.horizon, shape[1]))
        carry_in = np.zeros_like(multiplier)
        for t in range(self.horizon):
            for lag in range(self.lags):
                weight = self.decay**lag
                if lag <= t:
                    multiplier[:, t] += weight
                elif len(self.history) + t - lag >= 0:
                    carry_in[:, t] += weight * self.history[len(self.history) + t - lag]
        object.__setattr__(self, "_multiplier", multiplier)
        object.__setattr__(self, "_carry_in", carry_in)

    def response_and_gradient(
        self, allocation: Vector, draw: int | None = None
    ) -> tuple[Vector, Vector]:
        if (
            allocation.shape != (len(self.channels),)
            or not np.isfinite(allocation).all()
            or (allocation < 0).any()
        ):
            raise ValueError("allocation must be finite nonnegative channel spend")
        if draw is not None and not 0 <= draw < len(self.decay):
            raise ValueError("unknown posterior draw")
        selected = slice(None) if draw is None else slice(draw, draw + 1)
        multiplier = self._multiplier[selected]
        adstock = multiplier * allocation + self._carry_in[selected]
        amplitude = self.amplitude[selected, None, :]
        half = self.half_saturation[selected, None, :]
        denominator = half + adstock
        return (
            (amplitude * adstock / denominator).sum(axis=(1, 2)),
            (amplitude * half * multiplier / denominator**2).sum(axis=1),
        )


@dataclass(frozen=True)
class BudgetConstraints:
    total: float
    current: Mapping[str, float]
    floors: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())
    caps: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())
    max_movement: Mapping[str, float] = field(
        default_factory=lambda: dict[str, float]()
    )
    protected_spend: Mapping[str, float] = field(
        default_factory=lambda: dict[str, float]()
    )

    def bounds(
        self, channels: tuple[str, ...], release: ReleaseDecision
    ) -> tuple[Vector, Vector]:
        if release.state == ReleaseState.BLOCK:
            raise ValueError("BLOCK model cannot reach optimization")
        if not np.isfinite(self.total) or self.total <= 0:
            raise ValueError("total budget must be finite and positive")
        if set(self.current) != set(channels):
            raise ValueError("current allocation must cover exactly the model channels")
        for mapping in (
            self.current,
            self.floors,
            self.caps,
            self.max_movement,
            self.protected_spend,
            release.channel_movement_limits,
        ):
            if not set(mapping) <= set(channels) or any(
                not np.isfinite(v) or v < 0 for v in mapping.values()
            ):
                raise ValueError(
                    "constraint quantities must be finite, nonnegative and known"
                )
        lower: list[float] = []
        upper: list[float] = []
        for channel in channels:
            lo = max(self.floors.get(channel, 0), self.protected_spend.get(channel, 0))
            hi = self.caps.get(channel, self.total)
            movement = min(
                self.max_movement.get(channel, float("inf")),
                release.channel_movement_limits.get(channel, float("inf")),
            )
            if np.isfinite(movement):
                lo = max(lo, self.current[channel] * (1 - movement))
                hi = min(hi, self.current[channel] * (1 + movement))
            lower.append(lo)
            upper.append(hi)
        low, high = (
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )
        if (low > high).any() or low.sum() > self.total or high.sum() < self.total:
            raise ValueError("infeasible budget constraints")
        return low, high


def solve_allocation(
    response: PosteriorResponse,
    total: float,
    lower: Vector,
    upper: Vector,
    *,
    draw: int | None = None,
    conservative: bool = False,
    current: Vector | None = None,
) -> Vector:
    """SciPy solves the bounded simplex; independently check its result."""
    capacity = upper - lower
    remaining = total - lower.sum()
    if remaining <= 1e-10:
        return lower.copy()
    if abs(upper.sum() - total) <= 1e-10:
        return upper.copy()
    start = (lower + capacity * remaining / capacity.sum()) / total
    baseline = response.response_and_gradient(current)[0] if current is not None else 0
    scale = max(
        float(np.abs(response.response_and_gradient(start * total)[0]).mean()),
        1e-12,
    )

    def objective(fractions: Vector) -> tuple[float, Vector]:
        values, gradients = response.response_and_gradient(fractions * total, draw)
        if draw is not None:
            value, gradient = float(values[0]), gradients[0]
        elif conservative:
            # Empirical lower-tail mean of improvement, retaining joint states.
            worst = np.argsort(values - baseline)[
                : max(1, int(np.ceil(len(values) * 0.1)))
            ]
            value, gradient = (
                float((values - baseline)[worst].mean()),
                gradients[worst].mean(axis=0),
            )
        else:
            value, gradient = float(values.mean()), gradients.mean(axis=0)
        return -value / scale, -gradient * total / scale

    def budget_sum(x: Vector) -> float:
        return float(np.sum(x)) - 1

    def budget_gradient(x: Vector) -> Vector:
        return np.ones(len(x), dtype=np.float64)

    solver = cast(Any, minimize)
    result = solver(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=list(zip(lower / total, upper / total, strict=True)),
        constraints={
            "type": "eq",
            "fun": budget_sum,
            "jac": budget_gradient,
        },
        options={"ftol": 1e-10, "maxiter": 500},
    )
    allocation = np.asarray(result.x, dtype=np.float64) * total
    tolerance = max(total * 1e-8, 1e-6)
    if (
        not result.success
        or not np.isfinite(allocation).all()
        or abs(allocation.sum() - total) > tolerance
        or (allocation < lower - tolerance).any()
        or (allocation > upper + tolerance).any()
    ):
        raise ValueError(
            f"optimizer failed independent feasibility checks: {result.message}"
        )
    return allocation


def analyze_allocation(
    response: PosteriorResponse,
    constraints: BudgetConstraints,
    release: ReleaseDecision,
    *,
    stability_limit: float = 0.2,
) -> dict[str, object]:
    """Unstable or extrapolated candidates cannot authorize recommendations."""
    lower, upper = constraints.bounds(response.channels, release)
    if not np.isfinite(stability_limit) or not 0 <= stability_limit <= 2:
        raise ValueError("stability limit must lie between zero and two")
    current = np.asarray(
        [constraints.current[c] for c in response.channels], dtype=np.float64
    )
    baseline = response.response_and_gradient(current)[0]
    allocations: Vector = np.stack(
        [
            solve_allocation(response, constraints.total, lower, upper, draw=k)
            for k in range(len(response.decay))
        ]
    )
    alternatives: dict[str, object] = {}
    for preference in ("expected", "conservative"):
        allocation = solve_allocation(
            response,
            constraints.total,
            lower,
            upper,
            conservative=preference == "conservative",
            current=current,
        )
        outcomes = response.response_and_gradient(allocation)[0]
        gain = outcomes - baseline
        distance: Vector = (
            np.abs(allocations - allocation).sum(axis=1) / constraints.total
        )
        support = (allocation < response.history.min(axis=0) - 1e-6) | (
            allocation > response.history.max(axis=0) + 1e-6
        )
        movement = allocation - current
        flips: list[float | None] = [
            float(((allocations[:, c] - current[c]) * movement[c] < -1e-6).mean())
            if abs(movement[c]) > 1e-6
            else None
            for c in range(len(response.channels))
        ]
        p95 = float(np.quantile(distance, 0.95))
        alternatives[preference] = {
            "allocation": dict(
                zip(response.channels, allocation.tolist(), strict=True)
            ),
            "expected_modeled_media_revenue": float(outcomes.mean()),
            "media_revenue_interval_90": np.quantile(outcomes, [0.05, 0.95]).tolist(),
            "expected_improvement": float(gain.mean()),
            "improvement_p05": float(np.quantile(gain, 0.05)),
            "probability_beat_current": float(np.mean(gain > 1e-6)),
            "downside_probability": float(np.mean(gain < -1e-6)),
            "stability_median": float(np.median(distance)),
            "stability_p95": p95,
            "direction_flip_probability": dict(
                zip(response.channels, flips, strict=True)
            ),
            "extrapolated_channels": [
                c for c, flag in zip(response.channels, support, strict=True) if flag
            ],
            "recommendation_allowed": p95 <= stability_limit and not support.any(),
            "decision_state": "RESTRICT"
            if p95 > stability_limit or support.any()
            else release.state.value,
        }
    return {
        "alternatives": alternatives,
        "allocation_draws": allocations.tolist(),
        "channel_direction_probabilities": {
            c: {
                "increase": float((allocations[:, k] > current[k] + 1e-6).mean()),
                "decrease": float((allocations[:, k] < current[k] - 1e-6).mean()),
                "movement_over_20pct": float(
                    (
                        np.abs(allocations[:, k] - current[k]) > current[k] * 0.2 + 1e-6
                    ).mean()
                ),
            }
            for k, c in enumerate(response.channels)
        },
        "channel_allocation_quantiles": {
            c: np.quantile(allocations[:, k], [0.05, 0.5, 0.95]).tolist()
            for k, c in enumerate(response.channels)
        },
        "effective_bounds": {
            c: [float(lower[k]), float(upper[k])]
            for k, c in enumerate(response.channels)
        },
        "stability_limit": stability_limit,
        "outcome_units": "AUD modeled media revenue over planning horizon",
        "limitations": [
            "Model response; not experimentally identified incremental profit",
            "Constant weekly spend; nonmedia revenue and observation noise excluded",
            "Joint posterior uncertainty; conservative tail is empirical worst 10%",
            "L1 counts both sides of transfers; cash moved is half for equal totals",
            "Prototype stability threshold requires commercial calibration",
        ],
    }
