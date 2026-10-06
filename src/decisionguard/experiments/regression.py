"""Conditional predictive benchmarks; no incremental attribution or ROI claims."""

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import (  # pyright: ignore[reportMissingTypeStubs]
    HistGradientBoostingRegressor,  # pyright: ignore[reportMissingTypeStubs]
)
from sklearn.linear_model import Ridge  # pyright: ignore[reportMissingTypeStubs]
from sklearn.pipeline import (  # pyright: ignore[reportMissingTypeStubs]
    make_pipeline,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)
from sklearn.preprocessing import (  # pyright: ignore[reportMissingTypeStubs]
    StandardScaler,  # pyright: ignore[reportMissingTypeStubs]
)
from threadpoolctl import threadpool_limits  # pyright: ignore[reportMissingTypeStubs]

from decisionguard.data.integrity import CONTROL_COLUMNS, SPEND_COLUMNS
from decisionguard.data.synthetic import geometric_adstock

SPEND = list(SPEND_COLUMNS)
CONTROLS = list(CONTROL_COLUMNS)


def regression_features(
    covariates: pd.DataFrame, train_rows: int, period: int, domain: bool
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Causal-in-time transforms; saturation references use only the fit prefix."""
    if not 0 < train_rows <= len(covariates) or period <= 0:
        raise ValueError("valid training prefix and period required")
    required = ["week", *SPEND, *CONTROLS]
    if not set(required) <= set(covariates.columns):
        raise ValueError("regression requires all weekly spend and control columns")
    numeric = covariates[[*SPEND, *CONTROLS]].astype(float)
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("covariates must be finite")
    if (numeric[SPEND] < 0).any().any():
        raise ValueError("spend must be nonnegative")
    x = numeric.copy()
    t = np.arange(len(x), dtype=np.float64)
    x["trend"] = t
    x["season_sin"] = np.sin(2 * np.pi * t / period)
    x["season_cos"] = np.cos(2 * np.pi * t / period)
    references: dict[str, float] = {}
    if domain:
        for name in SPEND:
            carryover = geometric_adstock(numeric[name].to_numpy(dtype=np.float64), 0.5)
            half = max(float(np.median(carryover[:train_rows])), 1.0)
            references[name] = half
            x[name] = carryover / (half + carryover)
    return x, references


def regression_forecast(
    training: pd.DataFrame,
    covariates: pd.DataFrame,
    period: int,
    model: str,
    inner_horizon: int,
    gap: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Covariates include historical state and test inputs, never test revenue."""
    domain = model != "ridge_raw"
    train_rows = len(training)
    y = training["revenue"].to_numpy(dtype=np.float64)
    x, references = regression_features(covariates, train_rows, period, domain)
    audit: dict[str, object] = {
        "prediction_context": "conditional_on_realized_covariates",
        "saturation_reference": references,
        "adstock_decay": 0.5 if domain else None,
        "feature_columns": list(x.columns),
    }
    if model == "hist_gradient_boosting":
        estimator = cast(Any, HistGradientBoostingRegressor)(
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=10,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        )
        audit["model_config"] = {
            "max_iter": 100,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 10,
            "l2_regularization": 1,
            "early_stopping": False,
            "random_state": 42,
        }
    else:
        # Nested temporal tuning: the last inner horizon belongs only to validation.
        inner_start = train_rows - inner_horizon
        inner_end = inner_start - gap
        if inner_end < max(period, 12):
            raise ValueError("insufficient history for inner temporal Ridge selection")
        inner_x, inner_refs = regression_features(
            covariates.iloc[:train_rows], inner_end, period, domain
        )
        scores: dict[str, float] = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            candidate = cast(Any, make_pipeline)(
                cast(Any, StandardScaler)(), cast(Any, Ridge)(alpha=alpha)
            )
            candidate.fit(inner_x.iloc[:inner_end], y[:inner_end])
            pred = np.asarray(
                candidate.predict(inner_x.iloc[inner_start:]), dtype=np.float64
            )
            scores[str(alpha)] = float(np.abs(pred - y[inner_start:]).mean())
        alpha = float(min(scores, key=lambda key: scores[key]))
        estimator = cast(Any, make_pipeline)(
            cast(Any, StandardScaler)(), cast(Any, Ridge)(alpha=alpha)
        )
        audit["model_config"] = {"alpha": alpha, "scaler": "StandardScaler"}
        audit["inner_validation"] = {
            "train_rows": inner_end,
            "validation_rows": inner_horizon,
            "gap": gap,
            "mae_by_alpha": scores,
            "saturation_reference": inner_refs,
        }
    # Limit native thread oversubscription; these weekly datasets are tiny.
    with cast(Any, threadpool_limits)(limits=1):
        estimator.fit(x.iloc[:train_rows], y)
        predicted = np.asarray(estimator.predict(x.iloc[train_rows:]), dtype=np.float64)
    if model != "hist_gradient_boosting":
        audit["scaler_mean"] = list(map(float, estimator[0].mean_))
        audit["scaler_scale"] = list(map(float, estimator[0].scale_))
    if not np.isfinite(predicted).all():
        raise ValueError("regression predictions must be finite")
    return pd.DataFrame({"predicted": predicted}), audit
