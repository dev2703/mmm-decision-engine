"""Compare real predictive runs while separating their information assumptions."""

from dataclasses import replace
from pathlib import Path

from decisionguard.data.artifacts import file_hash, write_json
from decisionguard.experiments.baseline import BaselineConfig, run_baseline


def run_comparison(
    dataset: Path, output: Path, config: BaselineConfig | None = None
) -> dict[str, object]:
    """Run the model ladder on identical windows; select by CV within each context."""
    config = config or BaselineConfig(initial_train=104)
    if config.initial_train < 2 * config.period:
        raise ValueError("comparison needs two initial training seasons for ETS")
    if output.exists():
        raise FileExistsError(output)
    rows: list[dict[str, object]] = []
    scores: dict[str, list[tuple[float, str]]] = {
        "forecast_from_origin": [],
        "conditional_on_realized_covariates": [],
    }
    for model in (
        "seasonal_naive",
        "ets",
        "ridge_raw",
        "ridge_domain",
        "hist_gradient_boosting",
    ):
        result = run_baseline(dataset, output / model, replace(config, model=model))
        context = (
            "forecast_from_origin"
            if model in ("seasonal_naive", "ets")
            else "conditional_on_realized_covariates"
        )
        mae = result.cv_metrics["mae"]
        if mae is None:
            raise ValueError("CV MAE unavailable")
        scores[context].append((mae, model))
        record_path = output / model / "experiment.json"
        rows.append(
            {
                "model": model,
                "prediction_context": context,
                "cv": result.cv_metrics,
                "holdout": result.holdout_metrics,
                "experiment_path": str(record_path.resolve()),
                "experiment_hash": file_hash(record_path),
            }
        )
    winners = {context: min(values)[1] for context, values in scores.items()}
    summary: dict[str, object] = {
        "models": rows,
        "selected_by_cv_mae": winners,
        "decision_use": "None; all models remain PREDICTIVE_ONLY",
        "interpretation": [
            "Compare raw vs domain Ridge within the conditional group",
            "Keep complexity only if temporal CV improves in the same context",
            "Information assumptions differ; no combined forecast/conditional ranking",
            "Holdout scores performance after CV selection; it never selects",
        ],
    }
    write_json(output / "comparison.json", summary)
    return summary
