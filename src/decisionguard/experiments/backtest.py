"""Four chronological blocks with selection frozen before future assessment."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from decisionguard.data.artifacts import (
    dataset_evidence,
    file_hash,
    load_model_inputs,
    source_code_hash,
    write_json,
)
from decisionguard.experiments.baseline import (
    BaselineConfig,
    TemporalFold,
    evaluate_folds,
)
from decisionguard.experiments.metrics import forecast_metrics
from decisionguard.models.config import MMMConfig


def _metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    result = forecast_metrics(
        frame["actual"].to_numpy(dtype=np.float64),
        frame["predicted"].to_numpy(dtype=np.float64),
    )
    if "lower_90" in frame and frame["lower_90"].notna().all():
        result["coverage_90"] = float(
            (
                (frame["actual"] >= frame["lower_90"])
                & (frame["actual"] <= frame["upper_90"])
            ).mean()
        )
    return result


def run_backtest(
    dataset: Path,
    output: Path,
    config: BaselineConfig | None = None,
    *,
    test_weeks: int = 13,
    holdout_weeks: int = 13,
    mmm_config: MMMConfig | None = None,
) -> dict[str, object]:
    """Select on expanding validation; fit once before test, never refit on test.

    Remaining weeks between initial training and the two assessment blocks are
    validation. Optional MMM uses a prespecified configuration, not a tuned winner.
    Its holdout length is derived from the combined test and final holdout blocks.
    """
    if output.exists():
        raise FileExistsError(output)
    config = config or BaselineConfig(initial_train=104)
    if config.gap != 0 or config.initial_train < 2 * config.period:
        raise ValueError("backtest requires zero gap and two initial training seasons")
    if any(type(n) is not int or n <= 0 for n in (test_weeks, holdout_weeks)):
        raise ValueError("test_weeks and holdout_weeks must be positive integers")
    code_hash = source_code_hash()
    data, available_at = load_model_inputs(dataset)
    input_evidence = dataset_evidence(dataset, data)
    development_end = len(data) - test_weeks - holdout_weeks
    if development_end < config.initial_train + config.horizon:
        raise ValueError("insufficient data for training, validation, test and holdout")
    if mmm_config is not None:
        if development_end < 52:
            raise ValueError("MMM requires at least 52 development weeks")
        mmm_config = replace(mmm_config, holdout=test_weeks + holdout_weeks)
    boundaries = (
        ("train", 0, config.initial_train),
        ("validation", config.initial_train, development_end),
        ("test", development_end, development_end + test_weeks),
        ("holdout", development_end + test_weeks, len(data)),
    )
    membership = data[["week"]].copy()
    membership["split"] = ""
    splits: dict[str, object] = {}
    for name, start, end in boundaries:
        membership.loc[membership.index[start:end], "split"] = name
        splits[name] = {
            "start": str(data["week"].iloc[start].date()),
            "end": str(data["week"].iloc[end - 1].date()),
            "rows": end - start,
        }
    output.mkdir(parents=True, exist_ok=False)
    membership.to_parquet(output / "partitions.parquet", index=False)
    lock = Path(__file__).resolve().parents[3] / "uv.lock"
    write_json(
        output / "split.json",
        {
            "splits": splits,
            "dataset": input_evidence,
            "baseline_config": asdict(config),
            "mmm_config": asdict(mmm_config) if mmm_config else None,
            "code_hash": code_hash,
            "dependency_lock_hash": file_hash(lock) if lock.exists() else None,
            "partitions_hash": file_hash(output / "partitions.parquet"),
            "protocol": "expanding validation; fit train+validation; no test refit",
            "selection_metric": "validation MAE within each information context",
        },
    )
    folds = tuple(
        TemporalFold(
            "validation", start, start, min(start + config.horizon, development_end)
        )
        for start in range(config.initial_train, development_end, config.horizon)
    )
    scores: dict[str, list[tuple[float, str]]] = {}
    validation: dict[str, object] = {}
    frames: list[pd.DataFrame] = []
    for model in (
        "seasonal_naive",
        "ets",
        "ridge_raw",
        "ridge_domain",
        "hist_gradient_boosting",
    ):
        context = (
            "forecast_from_origin"
            if model in ("seasonal_naive", "ets")
            else "conditional_on_realized_covariates"
        )
        predictions, records = evaluate_folds(
            data.iloc[:development_end],
            replace(config, model=model),
            folds,
            available_at.iloc[:development_end],
        )
        metrics = _metrics(predictions)
        mae = metrics["mae"]
        if mae is None:
            raise ValueError("validation MAE unavailable")
        scores.setdefault(context, []).append((mae, model))
        validation[model] = {
            "prediction_context": context,
            "metrics": metrics,
            "folds": records,
            "config": asdict(replace(config, model=model)),
        }
        frames.append(predictions.assign(model=model, prediction_context=context))
    write_json(output / "validation.json", validation)
    selected = {context: min(values)[1] for context, values in scores.items()}
    # Persist the choice before evaluating either future block, including the MMM.
    write_json(
        output / "selection.json",
        {
            "selected_by_validation_mae": selected,
            "validation_hash": file_hash(output / "validation.json"),
            "split_hash": file_hash(output / "split.json"),
        },
    )
    assessment: dict[str, object] = {}
    for context, model in selected.items():
        predictions, records = evaluate_folds(
            data,
            replace(config, model=model),
            (TemporalFold("assessment", development_end, development_end, len(data)),),
            available_at,
        )
        assessment[model] = {"prediction_context": context, "fit": records}
        frames.append(predictions.assign(model=model, prediction_context=context))
    if mmm_config is not None:
        from decisionguard.models.bayesian import train_mmm

        record = train_mmm(dataset, output / "mmm", mmm_config)
        if record["dataset"] != input_evidence:
            raise ValueError("dataset changed during backtest")
        predictions = pd.read_parquet(output / "mmm" / "predictions.parquet")
        predictions = predictions.loc[predictions["split"] == "holdout"].copy()
        predictions["split"] = "assessment"
        frames.append(
            predictions.assign(
                model="pymc_marketing_mmm",
                prediction_context="conditional_on_realized_covariates",
            )
        )
        assessment["pymc_marketing_mmm"] = {
            "prediction_context": "conditional_on_realized_covariates",
            "model_status": record["model_status"],
            "diagnostics": record["diagnostics"],
            "training_window": record["training_window"],
            "model_record_hash": file_hash(output / "mmm" / "model.json"),
        }
    predictions = pd.concat(frames, ignore_index=True)
    future = predictions["split"] == "assessment"
    test_end = data["week"].iloc[development_end + test_weeks]
    predictions.loc[future & (predictions["week"] < test_end), "split"] = "test"
    predictions.loc[future & (predictions["week"] >= test_end), "split"] = "holdout"
    predictions.to_parquet(output / "predictions.parquet", index=False)
    metrics_by_model = {
        model: {
            split: _metrics(
                predictions.loc[
                    (predictions["model"] == model) & (predictions["split"] == split)
                ]
            )
            for split in ("test", "holdout")
        }
        for model in assessment
    }
    summary: dict[str, object] = {
        "splits": splits,
        "selected_by_validation_mae": selected,
        "metrics": metrics_by_model,
        "assessment": assessment,
        "decision_use": "None; prediction alone never authorizes allocation",
        "limitations": [
            "Regressors and MMM condition on realized future spend and controls",
            "Test and holdout use the same pre-test fit; no target updates or retuning",
            "MMM is prespecified and requires separate robustness/release evaluation",
            "Opened holdout cannot be reused to select changes without new future data",
        ],
        "artifacts": {
            name: file_hash(output / name)
            for name in (
                "split.json",
                "partitions.parquet",
                "validation.json",
                "selection.json",
                "predictions.parquet",
            )
        },
    }
    write_json(output / "backtest.json", summary)
    return summary
