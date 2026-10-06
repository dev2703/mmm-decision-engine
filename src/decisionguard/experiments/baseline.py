"""Seasonal-naive baseline and a minimal auditable temporal experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from decisionguard.data.artifacts import load_model_ready, source_code_hash, write_json
from decisionguard.data.integrity import dataset_hash


@dataclass(frozen=True)
class BaselineConfig:
    period: int = 52
    initial_train: int = 52
    horizon: int = 13
    gap: int = 0

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
        predicted = seasonal_naive(
            revenue[: fold.train_end], config.gap + config.horizon, config.period
        )[config.gap :]
        actual = revenue[fold.test_start : fold.test_end]
        frames.append(
            pd.DataFrame(
                {
                    "week": dates[fold.test_start : fold.test_end],
                    "fold": fold_id,
                    "split": fold.kind,
                    "actual": actual,
                    "predicted": predicted,
                    "residual": actual - predicted,
                }
            )
        )
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
    data = load_model_ready(dataset)
    raw = pd.read_parquet(dataset / "raw.parquet")
    availability: pd.Series[pd.Timestamp] | None = None
    if "available_at" in raw:
        raw["week"] = pd.to_datetime(raw["week"], format="ISO8601").dt.normalize()
        raw["available_at"] = pd.to_datetime(raw["available_at"], format="ISO8601")
        # One-to-one alignment; max arrival is conservative for repeated records.
        by_week = raw.groupby("week")["available_at"].max()
        availability = by_week.reindex(data["week"]).reset_index(drop=True)
    result = evaluate_baseline(data, config, availability)
    output.mkdir(parents=True, exist_ok=False)
    result.predictions.to_parquet(output / "predictions.parquet", index=False)
    quality = json.loads((dataset / "quality.json").read_text())
    record = {
        "model_family": "seasonal_naive",
        "model_status": "PREDICTIVE_ONLY",
        "hypothesis": hypothesis
        or (
            f"A {result.config.period}-week seasonal-naive forecast provides "
            "a reference for more complex predictive models."
        ),
        "change": "Repeat the last training season without observed future targets",
        "validation": asdict(result.config),
        "folds": result.fold_records,
        "metrics": {"cv": result.cv_metrics, "holdout": result.holdout_metrics},
        "result": "Baseline evaluated; retained as the comparison reference",
        "interpretation": "Forecast errors do not establish incrementality or ROI",
        "metric_units": {"mae": "AUD/week", "rmse": "AUD/week", "wape": "fraction"},
        "decision": "keep_as_baseline",
        "limitations": [
            "Point forecasts only; no uncertainty intervals",
            "Fixed seasonal period; does not learn trend, promotions or media effects",
            "Cannot support attribution or budget allocation",
            "No tuning or model-selection claims from this single baseline",
        ],
        "feature_config": {
            "inputs": ["training_revenue"],
            "learned_preprocessing": None,
        },
        "random_seed": None,
        "dataset": {
            "path": str(dataset.resolve()),
            "hash": dataset_hash(data),
            "clean_artifact_hash": sha256(
                (dataset / "clean.parquet").read_bytes()
            ).hexdigest(),
            "raw_artifact_hash": sha256(
                (dataset / "raw.parquet").read_bytes()
            ).hexdigest(),
            "quality_artifact_hash": sha256(
                (dataset / "quality.json").read_bytes()
            ).hexdigest(),
            "quality_status": quality["status"],
            "quality_issues": quality["issues"],
        },
        "code_hash": source_code_hash(),
        "predictions_hash": sha256(
            (output / "predictions.parquet").read_bytes()
        ).hexdigest(),
    }
    lock = Path(__file__).resolve().parents[3] / "uv.lock"
    if lock.is_file():
        record["dependency_lock_hash"] = sha256(lock.read_bytes()).hexdigest()
    write_json(output / "experiment.json", record)
    return result
