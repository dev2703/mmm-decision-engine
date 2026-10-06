"""Shared deterministic forecast metrics in original outcome units."""

import numpy as np
from numpy.typing import ArrayLike


def forecast_metrics(
    actual: ArrayLike, predicted: ArrayLike
) -> dict[str, float | None]:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("actual and predicted must be nonempty matching vectors")
    if np.iscomplexobj(actual) or np.iscomplexobj(predicted):
        raise ValueError("metric inputs must be real-valued")
    actual = actual.astype(np.float64, copy=False)
    predicted = predicted.astype(np.float64, copy=False)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metric inputs must be finite")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = np.abs(predicted - actual)
            scale = float(error.max())
            normalized = error / scale if scale else error
            mae = float(normalized.mean()) * scale
            rmse = float(np.sqrt(np.square(normalized).mean())) * scale
            target_scale = float(np.abs(actual).max())
            denominator = (
                float((np.abs(actual) / target_scale).mean()) if target_scale else 0
            )
            wape = (mae / target_scale) / denominator if denominator else None
    except FloatingPointError as failure:
        raise ValueError(
            "metric computation exceeds finite numerical range"
        ) from failure
    if any(value is not None and not np.isfinite(value) for value in (mae, rmse, wape)):
        raise ValueError("metric computation exceeds finite numerical range")
    return {"mae": mae, "rmse": rmse, "wape": wape}
