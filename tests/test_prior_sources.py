"""Sensitivity changes a scientific assumption rather than only sampling noise."""

from copy import deepcopy
from typing import Any

import pytest

from decisionguard.models.artifacts import validate_prior_sources


def sources() -> tuple[dict[str, Any], dict[str, Any]]:
    first: dict[str, Any] = {
        "dataset": {"hash": "data"},
        "training_window": {"rows": 143},
        "config": {
            "channels": ["search_spend"],
            "media_prior_mean": 0.15,
            "media_prior_sigma": 0.1,
            "adstock_lags": 16,
            "draws": 2000,
        },
    }
    second = deepcopy(first)
    second["config"]["media_prior_mean"] = 0.25
    return first, second


def test_changed_prior_and_sampler_precision_are_acceptable() -> None:
    first, second = sources()
    second["config"]["draws"] = 3000
    validate_prior_sources(first, second)


def test_identical_prior_does_not_count_as_prior_sensitivity() -> None:
    first, second = sources()
    second["config"]["media_prior_mean"] = 0.15
    second["config"]["draws"] = 3000
    with pytest.raises(ValueError, match="changed media prior"):
        validate_prior_sources(first, second)


def test_changing_adstock_alongside_prior_confounds_sensitivity() -> None:
    first, second = sources()
    second["config"]["adstock_lags"] = 8
    with pytest.raises(ValueError, match="model specification"):
        validate_prior_sources(first, second)
