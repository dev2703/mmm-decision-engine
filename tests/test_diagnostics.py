"""Numerical release evidence cannot be replaced by a trustworthy-looking label."""

import json
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from decisionguard.models.diagnostics import acceptable_diagnostics


def valid() -> dict[str, object]:
    return {
        "diagnostic_status": "ACCEPTABLE",
        "sufficient_chains_draws": True,
        "max_rhat": 1.01,
        "min_ess_bulk": 400.0,
        "min_ess_tail": 400.0,
        "divergences": 0,
        "maxdepth_reached": 0,
        "bfmi_by_chain": [0.3, 0.3],
    }


def test_diagnostic_threshold_boundaries_are_inclusive() -> None:
    assert acceptable_diagnostics(valid())


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_rhat", 1.01001),
        ("max_rhat", 0),
        ("max_rhat", True),
        ("min_ess_bulk", 399.99),
        ("min_ess_tail", float("inf")),
        ("divergences", 1),
        ("divergences", False),
        ("maxdepth_reached", 1),
        ("sufficient_chains_draws", 1),
        ("bfmi_by_chain", [0.299, 0.4]),
        ("bfmi_by_chain", [0.4]),
        ("bfmi_by_chain", [float("nan"), 0.4]),
        ("bfmi_by_chain", [None, 0.4]),
        ("bfmi_by_chain", "0.4,0.4"),
    ],
)
def test_invalid_diagnostics_fail_despite_acceptable_label(
    key: str, value: object
) -> None:
    assert not acceptable_diagnostics({**valid(), key: value})


@pytest.mark.parametrize("key", list(valid()))
def test_missing_numerical_evidence_fails_closed(key: str) -> None:
    record = valid()
    del record[key]
    # Labels have no numerical authority; the release policy separately checks status.
    assert acceptable_diagnostics(record) is (key == "diagnostic_status")


def test_nonfinite_energy_diagnostic_remains_json_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from decisionguard.models import bayesian

    class Group:
        sizes: ClassVar[dict[str, int]] = {"chain": 2, "draw": 100}
        diverging = np.zeros((2, 100), dtype=np.int64)
        maxdepth_reached = np.zeros((2, 100), dtype=np.int64)
        depth = np.ones((2, 100), dtype=np.int64)

    class Trace:
        posterior = Group()
        sample_stats = Group()

    def summary(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"r_hat": [1.0], "ess_bulk": [500.0], "ess_tail": [500.0]})

    def bfmi(_: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.array([np.nan, 0.5])

    monkeypatch.setattr(bayesian.az, "summary", summary)
    monkeypatch.setattr(bayesian.az, "bfmi", bfmi)
    record = bayesian.posterior_diagnostics(Trace())
    assert record["diagnostic_status"] == "INVESTIGATE"
    assert record["bfmi_by_chain"] == [None, 0.5]
    json.dumps(record, allow_nan=False)
