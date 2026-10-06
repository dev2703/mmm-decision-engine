"""Leakage-safe temporal evaluation and auditable predictive experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from decisionguard.data.artifacts import (
    dataset_evidence,
    file_hash,
    load_model_inputs,
    source_code_hash,
    write_json,
)
from decisionguard.experiments.ets import ets_forecast
from decisionguard.experiments.metrics import forecast_metrics
from decisionguard.experiments.regression import regression_forecast


@dataclass(frozen=True)
class BaselineConfig:
    period: int = 52
    initial_train: int = 52
    horizon: int = 13
    gap: int = 0
    model: Literal[
        "seasonal_naive", "ets", "ridge_raw", "ridge_domain", "hist_gradient_boosting"
    ] = "seasonal_naive"

    def __post_init__(self) -> None:
        for name, value in (
            ("period", self.period),
            ("initial_train", self.initial_train),
            ("horizon", self.horizon),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.gap) is not int or self.gap < 0:
            raise ValueError("gap must be a nonnegative integer")
        if self.model not in (
            "seasonal_naive",
            "ets",
            "ridge_raw",
            "ridge_domain",
            "hist_gradient_boosting",
        ):
            raise ValueError("unsupported baseline model")
        if self.model == "ets" and self.initial_train < 2 * self.period:
            raise ValueError("seasonal ETS requires two complete training seasons")
        if self.initial_train < self.period:
            raise ValueError("initial_train must contain at least one complete season")


@dataclass(frozen=True)
class TemporalFold:
    kind: str
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class BaselineResult:
    predictions: pd.DataFrame
    fold_records: tuple[dict[str, object], ...]
    cv_metrics: dict[str, float | None]
    holdout_metrics: dict[str, float | None]
    config: BaselineConfig


def temporal_folds(rows: int, config: BaselineConfig) -> tuple[TemporalFold, ...]:
    """Nonoverlapping validation windows, with the last horizon sealed as holdout."""
    if (
        type(rows) is not int
        or rows < config.initial_train + config.gap + 2 * config.horizon
    ):
        raise ValueError(
            "insufficient data for a season, one CV fold and a final holdout"
        )
    holdout = rows - config.horizon
    folds = [
        TemporalFold("cv", start - config.gap, start, start + config.horizon)
        for start in range(
            config.initial_train + config.gap,
            holdout - config.horizon + 1,
            config.horizon,
        )
    ]
    folds.append(TemporalFold("holdout", holdout - config.gap, holdout, rows))
    return tuple(folds)


def seasonal_naive(
    training: NDArray[np.float64], horizon: int, period: int
) -> NDArray[np.float64]:
    """Repeat the last observed season, using no realized future target values."""
    if (
        type(period) is not int
        or period <= 0
        or type(horizon) is not int
        or horizon <= 0
    ):
        raise ValueError("period and horizon must be positive integers")
    if training.ndim != 1 or len(training) < period or not np.isfinite(training).all():
        raise ValueError("training must be finite and contain a complete season")
    return training[-period:][np.arange(horizon) % period].copy()


def evaluate_baseline(
    data: pd.DataFrame,
    config: BaselineConfig | None = None,
    available_at: pd.Series[pd.Timestamp] | None = None,
) -> BaselineResult:
    """Evaluate forecasts at each origin; availability is checked before training."""
    if config is None:
        config = BaselineConfig()
    if not data.columns.is_unique or not {"week", "revenue"} <= set(data.columns):
        raise ValueError("unique week and revenue columns are required")
    dates = pd.DatetimeIndex(data["week"])
    if (
        dates.hasnans
        or dates.tz is not None
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("weeks must be unique, ordered, timezone-free dates")
    if (
        len(dates) == 0
        or dates[0].dayofweek != 0
        or not dates.equals(pd.date_range(dates[0], periods=len(data), freq="W-MON"))
    ):
        raise ValueError("data must contain contiguous Monday weeks")
    revenue = data["revenue"].to_numpy(dtype=np.float64)
    if not np.isfinite(revenue).all():
        raise ValueError("revenue must be finite")
    availability = (
        pd.DatetimeIndex(available_at)
        if available_at is not None
        else dates + pd.Timedelta(days=7)
    )
    if (
        len(availability) != len(data)
        or availability.hasnans
        or availability.tz is not None
    ):
        raise ValueError("availability must align with all weekly observations")
    if (availability < dates + pd.Timedelta(days=7)).any():
        raise ValueError("availability cannot precede week completion")
    frames: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for fold_id, fold in enumerate(temporal_folds(len(data), config)):
        origin = dates[fold.test_start]
        if (availability[: fold.train_end] > origin).any():
            raise ValueError(
                "training records unavailable at forecast origin; increase gap"
            )
        diagnostics: dict[str, object] = {}
        if config.model == "ets":
            forecast, diagnostics = ets_forecast(
                revenue[: fold.train_end], config.gap + config.horizon, config.period
            )
            forecast = forecast.iloc[config.gap :].reset_index(drop=True)
        elif config.model in ("ridge_raw", "ridge_domain", "hist_gradient_boosting"):
            if config.model.startswith("ridge"):
                inner_start = fold.train_end - config.horizon
                inner_end = inner_start - config.gap
                if (
                    inner_end <= 0
                    or (availability[:inner_end] > dates[inner_start]).any()
                ):
                    raise ValueError(
                        "inner training unavailable at validation origin; increase gap"
                    )
            forecast, diagnostics = regression_forecast(
                data.iloc[: fold.train_end],
                data.iloc[: fold.test_end].drop(columns="revenue"),
                config.period,
                config.model,
                config.horizon,
                config.gap,
            )
            forecast = forecast.iloc[config.gap :].reset_index(drop=True)
        else:
            forecast = pd.DataFrame(
                {
                    "predicted": seasonal_naive(
                        revenue[: fold.train_end],
                        config.gap + config.horizon,
                        config.period,
                    )[config.gap :]
                }
            )
        predicted = forecast["predicted"].to_numpy(dtype=np.float64)
        actual = revenue[fold.test_start : fold.test_end]
        forecast["week"] = dates[fold.test_start : fold.test_end]
        forecast["fold"] = fold_id
        forecast["split"] = fold.kind
        forecast["actual"] = actual
        forecast["residual"] = actual - predicted
        if config.model == "ets":
            inside = (actual >= forecast["lower_90"].to_numpy(dtype=np.float64)) & (
                actual <= forecast["upper_90"].to_numpy(dtype=np.float64)
            )
            diagnostics["test_coverage_90"] = int(np.count_nonzero(inside)) / len(
                actual
            )
        frames.append(forecast)
        records.append(
            {
                "fold": fold_id,
                "split": fold.kind,
                "train_start": str(dates[0].date()),
                "train_end": str(dates[fold.train_end - 1].date()),
                "train_rows": fold.train_end,
                "forecast_origin": str(origin.date()),
                "test_start": str(origin.date()),
                "test_end": str(dates[fold.test_end - 1].date()),
                "max_training_available_at": availability[: fold.train_end]
                .max()
                .isoformat(),
                "metrics": forecast_metrics(actual, predicted),
                "diagnostics": diagnostics,
            }
        )
    predictions = pd.concat(frames, ignore_index=True)
    cv = predictions.loc[predictions["split"] == "cv"]
    holdout = predictions.loc[predictions["split"] == "holdout"]
    return BaselineResult(
        predictions,
        tuple(records),
        forecast_metrics(
            cv["actual"].to_numpy(dtype=np.float64),
            cv["predicted"].to_numpy(dtype=np.float64),
        ),
        forecast_metrics(
            holdout["actual"].to_numpy(dtype=np.float64),
            holdout["predicted"].to_numpy(dtype=np.float64),
        ),
        config,
    )


def run_baseline(
    dataset: Path,
    output: Path,
    config: BaselineConfig | None = None,
    hypothesis: str | None = None,
) -> BaselineResult:
    """Load through the quality gate, evaluate and persist a predictive-only run."""
    code_hash_at_start = source_code_hash()
    data, availability = load_model_inputs(dataset)
    result = evaluate_baseline(data, config, availability)
    output.mkdir(parents=True, exist_ok=False)
    result.predictions.to_parquet(output / "predictions.parquet", index=False)
    record = {
        "model_family": result.config.model,
        "model_status": "PREDICTIVE_ONLY",
        "prediction_context": (
            "forecast_from_origin"
            if result.config.model in ("ets", "seasonal_naive")
            else "conditional_on_realized_covariates"
        ),
        "hypothesis": hypothesis
        or (
            f"A {result.config.period}-week {result.config.model} benchmark provides "
            "a reference for more complex predictive models."
        ),
        "change": (
            "Fit raw-feature Ridge with inner temporal alpha selection"
            if result.config.model == "ridge_raw"
            else "Fit domain adstock/saturation predictive benchmark"
            if result.config.model in ("ridge_domain", "hist_gradient_boosting")
            else "Fit additive error/trend/seasonal ETS on each training prefix"
            if result.config.model == "ets"
            else "Repeat the last training season without observed future targets"
        ),
        "validation": asdict(result.config),
        "folds": result.fold_records,
        "metrics": {"cv": result.cv_metrics, "holdout": result.holdout_metrics},
        "result": "Baseline evaluated; retained as the comparison reference",
        "interpretation": "Forecast errors do not establish incrementality or ROI",
        "metric_units": {"mae": "AUD/week", "rmse": "AUD/week", "wape": "fraction"},
        "decision": "keep_as_baseline",
        "limitations": [
            (
                "90% Gaussian innovation intervals exclude parameter uncertainty"
                if result.config.model == "ets"
                else "Point forecasts only; no uncertainty intervals"
            ),
            "Fixed seasonal period and untested stability under regime changes",
            "Regressors condition on realized test covariates",
            "Domain decay is fixed at 0.5, not estimated or read from privileged truth",
            "Cannot support attribution or budget allocation",
            "Ridge alpha selected inside training only; no holdout selection",
        ],
        "feature_config": {
            "inputs": (
                ["training_revenue"]
                if result.config.model in ("ets", "seasonal_naive")
                else ["weekly_spend", "controls", "calendar"]
            ),
            "learned_preprocessing": (
                "fold-fitted saturation references and StandardScaler for Ridge"
                if result.config.model.startswith("ridge")
                else "fold-fitted saturation references"
                if result.config.model == "hist_gradient_boosting"
                else None
            ),
        },
        "random_seed": 42 if result.config.model == "hist_gradient_boosting" else None,
        "dataset": dataset_evidence(dataset, data),
        "code_hash": code_hash_at_start,
        "predictions_hash": file_hash(output / "predictions.parquet"),
    }
    lock = Path(__file__).resolve().parents[3] / "uv.lock"
    if lock.is_file():
        record["dependency_lock_hash"] = file_hash(lock)
    write_json(output / "experiment.json", record)
    return result
