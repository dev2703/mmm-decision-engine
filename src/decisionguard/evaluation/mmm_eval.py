"""Pinned upstream tests with fold-fitted controls and unmodified raw results."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from mmm_eval.adapters.pymc import (  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
    PyMCAdapter,  # pyright: ignore[reportUnknownVariableType]
)
from mmm_eval.configs import (  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
    PyMCConfig,  # pyright: ignore[reportUnknownVariableType]
)
from mmm_eval.core.validation_test_orchestrator import (  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
    ValidationTestOrchestrator,  # pyright: ignore[reportUnknownVariableType]
)
from mmm_eval.core.validation_tests_models import (  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
    ValidationTestNames,  # pyright: ignore[reportUnknownVariableType]
)
from numpy.typing import NDArray

from decisionguard.data.artifacts import (
    file_hash,
    load_model_inputs,
    source_code_hash,
    write_json,
)
from decisionguard.data.integrity import dataset_hash
from decisionguard.evaluation.artifacts import UPSTREAM_COMMIT, source_evidence
from decisionguard.evaluation.checkpoints import (
    initialize_progress,
    load_checkpoint,
    save_checkpoint,
)
from decisionguard.evaluation.policy import assess_release
from decisionguard.models.artifacts import (
    load_mmm_record,
    roi_sensitivity,
    validate_prior_sources,
)
from decisionguard.models.bayesian import MMMConfig, build_mmm, posterior_diagnostics


class FoldSafePyMCAdapter(cast(Any, PyMCAdapter)):  # pyright: ignore[reportUntypedBaseClass]
    """Reuse upstream fitting/ROI; contain learned control scaling at each fit."""

    def __init__(
        self,
        config: Any,
        arrivals: pd.Series[pd.Timestamp],
        ledger: list[dict[str, object]],
    ) -> None:
        base: Any = super()
        base.__init__(config)
        self.arrivals = arrivals
        self.ledger = ledger
        self.mean: pd.Series[float] | None = None
        self.scale: pd.Series[float] | None = None

    def fit(self, data: pd.DataFrame) -> None:
        external = cast(Any, self)
        date_column: str = external.date_column
        dates = pd.DatetimeIndex(data[date_column])
        available = self.arrivals.reindex(dates)
        origin = dates.max() + pd.Timedelta(days=7)
        if available.isna().any() or (available > origin).any():
            raise ValueError(
                "upstream refit uses observations unavailable at its origin"
            )
        controls: list[str] = external.control_columns
        mean = data[controls].mean()
        scale = data[controls].std(ddof=0).replace(0, 1)
        self.mean = mean
        self.scale = scale
        transformed = self._transform(data)
        base: Any = super()
        base.fit(transformed)
        self.ledger.append(
            {
                "train_start": dates.min().isoformat(),
                "train_end": dates.max().isoformat(),
                "train_rows": len(data),
                "control_mean": mean.to_dict(),
                "control_scale": scale.to_dict(),
                "diagnostics": posterior_diagnostics(external.trace),
            }
        )

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.mean is None or self.scale is None:
            raise ValueError("adapter control transform is not fitted")
        transformed = data.copy()
        columns = list(self.mean.index)
        transformed[columns] = (transformed[columns] - self.mean) / self.scale
        return transformed

    def predict(self, data: pd.DataFrame | None = None) -> NDArray[np.float64]:
        if data is None:
            raise ValueError("PyMC prediction requires covariates")
        covariates = self._transform(data).drop(
            columns=["response", "revenue"], errors="ignore"
        )
        base: Any = super()
        result = base.predict(covariates)
        return np.asarray(result, dtype=np.float64)

    def fit_and_predict_in_sample(self, data: pd.DataFrame) -> NDArray[np.float64]:
        self.fit(data)
        external = cast(Any, self)
        covariates = self._transform(data).drop(columns=["response", "revenue"])
        return np.asarray(
            external.model.predict(
                covariates,
                extend_idata=False,
                include_last_observations=False,
                **external.predict_kwargs,
            ),
            dtype=np.float64,
        )

    def _create_adapter_with_placebo_channel(
        self, shuffled_channel: str
    ) -> FoldSafePyMCAdapter:
        base: Any = super()
        created = base._create_adapter_with_placebo_channel(shuffled_channel)
        return FoldSafePyMCAdapter(created.config, self.arrivals, self.ledger)


def evaluate_mmm(
    model_run: Path,
    sensitivity_run: Path,
    output: Path,
    *,
    draws: int | None = None,
    tune: int | None = None,
    resume: bool = False,
    dataset: Path | None = None,
) -> dict[str, object]:
    """Evaluate training-only refits, keeping the main model's holdout sealed."""
    if output.exists() and (not resume or (output / "evaluation.json").exists()):
        raise FileExistsError(output)
    code_hash_at_start = source_code_hash()
    model_record = load_mmm_record(model_run)
    alternative = load_mmm_record(sensitivity_run)
    config = MMMConfig(
        **{
            key: value
            for key, value in model_record["config"].items()
            if key not in ("sampler", "cores")
        }
    )
    config = replace(
        config,
        draws=config.draws if draws is None else draws,
        tune=config.tune if tune is None else tune,
    )
    validate_prior_sources(model_record, alternative)
    data, arrivals = load_model_inputs(
        dataset or Path(model_record["dataset"]["path"]), model_record["dataset"]
    )
    if model_record["dataset"]["hash"] != dataset_hash(data):
        raise ValueError("model input identity changed")
    arrival_by_week = pd.Series(arrivals.to_numpy(), index=data["week"])
    rows = model_record["training_window"]["rows"]
    data = data.iloc[:rows].copy()
    data["trend"] = np.arange(len(data), dtype=np.float64)
    data["response"] = data["revenue"]
    model = build_mmm(config)
    # Upstream from_model_object filters extra sampler kwargs: construct then
    # extend the validated schema explicitly so Nutpie/depth settings survive.
    upstream_config = cast(Any, PyMCConfig).from_model_object(
        model,
        revenue_column="revenue",
        response_column="response",
        fit_kwargs={
            "draws": config.draws,
            "tune": config.tune,
            "chains": config.chains,
            "target_accept": config.target_accept,
            "random_seed": config.seed,
            "progressbar": False,
        },
    )
    upstream_config.fit_config = upstream_config.fit_config.model_copy(
        update={
            "nuts_sampler": "nutpie",
            "nuts_sampler_kwargs": {
                "maxdepth": config.max_tree_depth,
                "save_warmup": config.save_warmup,
            },
            "cores": 1,
        }
    )
    runtime_versions = {
        name: version(name)
        for name in (
            "mmm-eval",
            "pymc-marketing",
            "pymc",
            "pytensor",
            "nutpie",
            "numpy",
            "pandas",
            "tensorflow",
            "tf-keras",
        )
    }
    identity: dict[str, object] = {
        "model_record_hash": file_hash(model_run / "model.json"),
        "sensitivity_record_hash": file_hash(sensitivity_run / "model.json"),
        "dataset_hash": dataset_hash(data),
        "code_hash": code_hash_at_start,
        "lock_hash": file_hash(Path("uv.lock")),
        "configuration": upstream_config.fit_config_dict,
        "runtime_versions": runtime_versions,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    initialize_progress(output, identity, resume=resume)
    ledger: list[dict[str, object]] = []
    adapter = FoldSafePyMCAdapter(upstream_config, arrival_by_week, ledger)
    orchestrator = cast(Any, ValidationTestOrchestrator)()
    tables: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for test in cast(Any, ValidationTestNames):
        name = str(test.value)
        cached = load_checkpoint(output, name)
        if cached is not None:
            table, refits = cached
            tables.append(table)
            ledger.extend(refits)
            continue
        first_refit = len(ledger)
        try:
            result = orchestrator.validate(adapter, data, [test])
            table = cast(pd.DataFrame, result.to_df())
            save_checkpoint(output, name, table, ledger[first_refit:])
            tables.append(table)
        except Exception as error:
            # Preserve failures and fail closed; never substitute a passing score.
            errors.append(
                {
                    "test": str(test.value),
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
    raw = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(
            columns=[
                "test_name",
                "general_metric_name",
                "specific_metric_name",
                "metric_value",
                "metric_pass",
                "timestamp",
            ]
        )
    )
    channels = list(config.channels)
    sensitivity = roi_sensitivity(model_record, alternative)
    diagnostic = dict(model_record["diagnostics"])
    if alternative["diagnostics"]["diagnostic_status"] != "ACCEPTABLE" or any(
        cast(dict[str, Any], event["diagnostics"])["diagnostic_status"] != "ACCEPTABLE"
        for event in ledger
    ):
        diagnostic["diagnostic_status"] = "INVESTIGATE"
    decision = assess_release(
        source_evidence(raw, channels),
        diagnostic,
        model_record["dataset"]["quality_status"],
        channels,
        prior_sensitivity=sensitivity,
    )
    raw.to_parquet(output / "mmm_eval_raw.parquet", index=False)
    # JSON null preserves undefined evidence as undefined; Parquet keeps raw NaN.
    write_json(
        output / "mmm_eval_raw.json",
        json.loads(raw.to_json(orient="records", double_precision=15)),
    )
    record: dict[str, object] = {
        "evaluator": "mmm-eval",
        "runtime_versions": runtime_versions,
        "progress_identity": identity,
        "version": version("mmm-eval"),
        "commit": UPSTREAM_COMMIT,
        "model_run": str(model_run.resolve()),
        "model_record_hash": file_hash(model_run / "model.json"),
        "sensitivity_run": str(sensitivity_run.resolve()),
        "sensitivity": sensitivity,
        "policy": asdict(decision),
        "test_execution_errors": errors,
        "refits": ledger,
        "evaluation_window": model_record["training_window"],
        "configuration": {
            "fit": upstream_config.fit_config_dict,
            "upstream_holdout_weeks": 8,
            "upstream_cv_folds": 5,
            "upstream_cv_horizon": 4,
            "seed": 42,
        },
        "limitations": [
            "Upstream perturbation uses common 5% Gaussian spend noise",
            "One randomized shuffled-channel placebo is not exhaustive falsification",
            "Source ROI is net ROI percent; model.json ROI is revenue/spend ratio",
            "Evaluation refits exclude the original sealed holdout",
        ],
        "code_hash": code_hash_at_start,
        "raw_artifact_hash": file_hash(output / "mmm_eval_raw.parquet"),
    }
    write_json(output / "evaluation.json", record)
    return record
