"""Shared deterministic forecast metrics in original outcome units."""

import numpy as np
from numpy.typing import NDArray


def forecast_metrics(
    actual: NDArray[np.float64], predicted: NDArray[np.float64]
) -> dict[str, float | None]:
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("actual and predicted must be nonempty matching vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metric inputs must be finite")
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape": float(np.abs(error).sum()) / denominator if denominator else None,
    }
