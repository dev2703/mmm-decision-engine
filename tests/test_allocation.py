"""Constraint properties and falsifiable uncertainty-to-decision examples."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from decisionguard.evaluation.policy import ReleaseDecision, ReleaseState
from decisionguard.optimization.allocation import (
    BudgetConstraints,
    PosteriorResponse,
    analyze_allocation,
    solve_allocation,
)

PASS = ReleaseDecision(ReleaseState.PASS, (), {})


def world(amplitudes: Sequence[Sequence[float]]) -> PosteriorResponse:
    values = np.asarray(amplitudes, dtype=np.float64)
    return PosteriorResponse(
        ("a", "b"),
        np.zeros_like(values),
        values,
        np.full_like(values, 1000),
        np.array([[0.0, 0.0], [100.0, 100.0]]),
        horizon=1,
        lags=1,
    )


def alternatives(result: dict[str, object]) -> dict[str, Any]:
    return cast(dict[str, Any], result["alternatives"])


def test_many_feasible_budgets_respect_sum_bounds_and_protected_spend() -> None:
    rng = np.random.default_rng(42)
    response = world([[1000, 3000]])
    for _ in range(40):
        current = rng.uniform(20, 80, 2)
        total = float(current.sum())
        constraints = BudgetConstraints(
            total,
            dict(zip(response.channels, current.tolist(), strict=True)),
            floors={"a": 10},
            caps={"a": 100, "b": 100},
            max_movement={"a": 0.1, "b": 0.1},
            protected_spend={"b": 15},
        )
        low, high = constraints.bounds(response.channels, PASS)
        allocation = solve_allocation(response, total, low, high)
        assert allocation.sum() == pytest.approx(total, abs=1e-6)
        assert (allocation >= low - 1e-6).all()
        assert (allocation <= high + 1e-6).all()


def test_restriction_overrides_requested_movement() -> None:
    response = world([[1000, 3000]])
    constraints = BudgetConstraints(100, {"a": 50, "b": 50}, max_movement={"b": 0.5})
    release = ReleaseDecision(ReleaseState.RESTRICT, (), {"b": 0.05})
    low, high = constraints.bounds(response.channels, release)
    allocation = solve_allocation(response, 100, low, high)
    assert allocation[1] == pytest.approx(52.5, abs=1e-5)


@pytest.mark.parametrize(
    "constraints",
    [
        BudgetConstraints(100, {"a": 50, "b": 50}, floors={"a": 101}),
        BudgetConstraints(100, {"a": 50, "b": 50}, caps={"a": 30, "b": 30}),
        BudgetConstraints(100, {"a": 50, "b": 50}, floors={"a": 60}, caps={"a": 50}),
        BudgetConstraints(200, {"a": 50, "b": 50}, max_movement={"a": 0, "b": 0}),
    ],
)
def test_infeasible_constraints_rejected(constraints: BudgetConstraints) -> None:
    with pytest.raises(ValueError, match="infeasible"):
        constraints.bounds(("a", "b"), PASS)


def test_block_rejected_before_response_or_solver() -> None:
    response = world([[1000, 2000]])
    block = ReleaseDecision(ReleaseState.BLOCK, (), {})
    with pytest.raises(ValueError, match="BLOCK"):
        analyze_allocation(response, BudgetConstraints(100, {"a": 50, "b": 50}), block)


def test_stable_world_has_zero_instability() -> None:
    result = analyze_allocation(
        world([[2000, 1000]] * 10), BudgetConstraints(100, {"a": 50, "b": 50}), PASS
    )
    expected = alternatives(result)["expected"]
    assert expected["stability_p95"] < 1e-7
    assert expected["recommendation_allowed"]
    assert expected["probability_beat_current"] == 1


def test_uncertainty_changes_allowed_recommendations() -> None:
    response = world([[3000, 1000]] * 10 + [[1000, 3000]] * 10)
    constraints = BudgetConstraints(100, {"a": 50, "b": 50})
    open_result = alternatives(analyze_allocation(response, constraints, PASS))[
        "expected"
    ]
    assert open_result["stability_p95"] > 0.9
    assert not open_result["recommendation_allowed"]
    bounded = replace(constraints, max_movement={"a": 0.1, "b": 0.1})
    guarded = alternatives(analyze_allocation(response, bounded, PASS))["expected"]
    assert guarded["stability_p95"] < 0.11
    assert guarded["recommendation_allowed"]


def test_tiny_response_change_can_create_allocation_cliff() -> None:
    response = replace(
        world([[1.001, 1], [1, 1.001]]), half_saturation=np.full((2, 2), 1e8)
    )
    constraints = BudgetConstraints(100, {"a": 50, "b": 50})
    low, high = constraints.bounds(response.channels, PASS)
    first = solve_allocation(response, 100, low, high, draw=0)
    second = solve_allocation(response, 100, low, high, draw=1)
    assert np.abs(first - second).sum() / 100 > 1.9


def test_conservative_tail_avoids_low_probability_downside() -> None:
    response = world([[4000, 2000]] * 18 + [[200, 2000]] * 2)
    result = alternatives(
        analyze_allocation(response, BudgetConstraints(100, {"a": 50, "b": 50}), PASS)
    )
    assert (
        result["conservative"]["allocation"]["a"]
        < result["expected"]["allocation"]["a"]
    )
    assert (
        result["conservative"]["improvement_p05"]
        >= result["expected"]["improvement_p05"]
    )


def test_analytic_gradient_matches_finite_difference_with_carry_in() -> None:
    response = replace(
        world([[3000, 1000], [2000, 2000]]),
        decay=np.array([[0.3, 0.6], [0.4, 0.5]]),
        horizon=4,
        lags=3,
    )
    allocation = np.array([40.0, 60.0])
    _, gradient = response.response_and_gradient(allocation)
    for k in range(2):
        delta: NDArray[np.float64] = np.zeros(2)
        delta[k] = 1e-3
        numerical = (
            response.response_and_gradient(allocation + delta)[0]
            - response.response_and_gradient(allocation - delta)[0]
        ) / 2e-3
        np.testing.assert_allclose(numerical, gradient[:, k], rtol=1e-8)


def test_extrapolation_prevents_authorized_recommendation() -> None:
    response = replace(
        world([[3000, 1000]]), history=np.array([[40.0, 40.0], [60.0, 60.0]])
    )
    result = alternatives(
        analyze_allocation(response, BudgetConstraints(100, {"a": 50, "b": 50}), PASS)
    )["expected"]
    assert result["extrapolated_channels"]
    assert not result["recommendation_allowed"]


def test_response_owns_inputs_and_cannot_be_mutated_after_carry_in_cache() -> None:
    amplitude = np.array([[1000.0, 2000.0]])
    response = PosteriorResponse(
        ("a", "b"),
        np.array([[0.3, 0.6]]),
        amplitude,
        np.array([[100.0, 100.0]]),
        np.array([[40.0, 60.0]]),
    )
    allocation = np.array([50.0, 50.0])
    before = response.response_and_gradient(allocation)[0]
    amplitude[:] = 0
    np.testing.assert_array_equal(response.response_and_gradient(allocation)[0], before)
    with pytest.raises(ValueError, match="read-only"):
        response.amplitude[:] = 0
