"""Planning math matches the pinned library's finite adstock and saturation."""

from typing import Any, cast

import numpy as np
import pytest
from pymc_marketing.mmm import GeometricAdstock, MichaelisMentenSaturation

from decisionguard.optimization.allocation import PosteriorResponse


@pytest.mark.scientific
def test_response_matches_library_with_real_historical_carry_in() -> None:
    history = np.array([[10.0, 80.0], [20.0, 60.0], [30.0, 40.0], [40.0, 20.0]])
    allocation = np.array([45.0, 55.0])
    response = PosteriorResponse(
        ("a", "b"),
        np.array([[0.3, 0.7], [0.5, 0.6]]),
        np.array([[3000.0, 1000.0], [2000.0, 2000.0]]),
        np.array([[80.0, 120.0], [100.0, 100.0]]),
        history,
        horizon=5,
        lags=4,
    )
    whole = np.concatenate([history, np.tile(allocation, (response.horizon, 1))])
    predicted, _ = response.response_and_gradient(allocation)
    adstock = cast(Any, GeometricAdstock(l_max=4, normalize=False))
    saturation = cast(Any, MichaelisMentenSaturation())
    for k in range(2):
        carried = adstock.function(whole, alpha=response.decay[k])
        modeled = saturation.function(
            carried, alpha=response.amplitude[k], lam=response.half_saturation[k]
        ).eval()
        assert predicted[k] == pytest.approx(modeled[len(history) :].sum(), rel=1e-12)
    no_history = PosteriorResponse(
        response.channels,
        response.decay,
        response.amplitude,
        response.half_saturation,
        np.zeros_like(history),
        horizon=5,
        lags=4,
    )
    assert (predicted > no_history.response_and_gradient(allocation)[0]).all()
