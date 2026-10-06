"""Statsmodels additive ETS; all estimation uses the supplied training prefix."""

from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.stats.diagnostic import (  # pyright: ignore[reportMissingTypeStubs]
    acorr_ljungbox,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)
from statsmodels.tsa.exponential_smoothing.ets import (  # pyright: ignore[reportMissingTypeStubs]
    ETSModel,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)


def ets_forecast(
    training: NDArray[np.float64], horizon: int, period: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    if len(training) < 2 * period:
        raise ValueError("seasonal ETS requires two complete training seasons")
    # Statsmodels is untyped; keep its dynamic interface at this library boundary.
    model = cast(Any, ETSModel)(
        pd.Series(training),
        error="add",
        trend="add",
        seasonal="add",
        seasonal_periods=period,
        initialization_method="estimated",
    )
    fitted = model.fit(disp=False, maxiter=2000)
    if not fitted.mle_retvals["converged"]:
        raise ValueError("ETS did not converge; no successful experiment recorded")
    prediction = fitted.get_prediction(
        start=len(training), end=len(training) + horizon - 1, method="exact"
    ).summary_frame(alpha=0.1)
    frame = pd.DataFrame(
        {
            "predicted": np.asarray(prediction["mean"], dtype=np.float64),
            "lower_90": np.asarray(prediction["pi_lower"], dtype=np.float64),
            "upper_90": np.asarray(prediction["pi_upper"], dtype=np.float64),
        }
    )
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("ETS produced nonfinite predictions or intervals")
    residual = np.asarray(fitted.resid, dtype=np.float64)
    lag = min(period, len(training) // 5)
    diagnostic = cast(Any, acorr_ljungbox)(residual, lags=[lag], return_df=True)
    return frame, {
        "converged": True,
        "iterations": int(fitted.mle_retvals["iterations"]),
        "training_residual_mean": float(residual.mean()),
        "ljung_box_lag": lag,
        "ljung_box_statistic": float(diagnostic["lb_stat"].iloc[0]),
        "ljung_box_pvalue": float(diagnostic["lb_pvalue"].iloc[0]),
        "interval_assumption": (
            "Additive ETS Gaussian innovations; parameter uncertainty excluded"
        ),
    }
