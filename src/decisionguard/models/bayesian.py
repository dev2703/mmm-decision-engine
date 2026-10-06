"""PyMC-Marketing workflow; a fitted candidate never authorizes allocation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import arviz as az  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pymc_marketing.mmm import MMM, GeometricAdstock, MichaelisMentenSaturation
from pymc_marketing.prior import Prior

from decisionguard.data.artifacts import (
    dataset_evidence,
    load_model_inputs,
    source_code_hash,
    write_json,
)
from decisionguard.data.integrity import CONTROL_COLUMNS, SPEND_COLUMNS
from decisionguard.experiments.metrics import forecast_metrics


@dataclass(frozen=True)
class MMMConfig:
    draws: int = 2000
    tune: int = 1500
    chains: int = 4
    seed: int = 42
    holdout: int = 13
    adstock_lags: int = 16
    max_tree_depth: int = 12
    target_accept: float = 0.99
    media_prior_mean: float = 0.15
    media_prior_sigma: float = 0.1
    channels: tuple[str, ...] = SPEND_COLUMNS

    def __post_init__(self) -> None:
        for name in (
            "draws",
            "tune",
            "chains",
            "holdout",
            "adstock_lags",
            "max_tree_depth",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0.5 < self.target_accept < 1:
            raise ValueError("target_accept must lie between 0.5 and 1")
        if (
            not self.channels
            or len(set(self.channels)) != len(self.channels)
            or not set(self.channels) <= set(SPEND_COLUMNS)
        ):
            raise ValueError("channels must be unique known spend columns")
        if not all(
            np.isfinite(v) and v > 0
            for v in (self.media_prior_mean, self.media_prior_sigma)
        ):
            raise ValueError("media prior parameters must be finite and positive")


def prepare_mmm_inputs(
    data: pd.DataFrame, train_rows: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Center/scale controls and trend on training only, preserving AUD media."""
    if not 52 <= train_rows <= len(data):
        raise ValueError("MMM requires at least 52 contiguous training weeks")
    x = data[["week", *SPEND_COLUMNS, *CONTROL_COLUMNS]].copy()
    x["trend"] = np.arange(len(x), dtype=np.float64)
    columns = [*CONTROL_COLUMNS, "trend"]
    mean = x.iloc[:train_rows][columns].mean()
    scale = x.iloc[:train_rows][columns].std(ddof=0).replace(0, 1)
    x[columns] = (x[columns] - mean) / scale
    return x, {
        "columns": columns,
        "mean": mean.to_dict(),
        "scale": scale.to_dict(),
        "train_rows": train_rows,
    }


def build_mmm(config: MMMConfig) -> Any:
    """Priors refer to revenue scaled by its training maximum, not AUD coefficients."""
    prior = cast(Any, Prior)
    return cast(Any, MMM)(
        date_column="week",
        channel_columns=list(config.channels),
        control_columns=[*CONTROL_COLUMNS, "trend"],
        yearly_seasonality=2,
        adstock=cast(Any, GeometricAdstock)(
            l_max=config.adstock_lags,
            normalize=False,
            priors={"alpha": prior("Beta", alpha=2, beta=2, dims="channel")},
        ),
        saturation=cast(Any, MichaelisMentenSaturation)(
            priors={
                "alpha": prior(
                    "Gamma",
                    mu=config.media_prior_mean,
                    sigma=config.media_prior_sigma,
                    dims="channel",
                ),
                "lam": prior("HalfNormal", sigma=1, dims="channel"),
            }
        ),
        model_config={
            "intercept": prior("Normal", mu=0.5, sigma=0.25),
            "gamma_control": prior("Normal", mu=0, sigma=0.1, dims="control"),
            "gamma_fourier": prior("Normal", mu=0, sigma=0.1, dims="fourier_mode"),
            "likelihood": prior("Normal", sigma=prior("HalfNormal", sigma=0.05)),
        },
    )


def posterior_diagnostics(idata: Any) -> dict[str, object]:
    parameters = [
        "adstock_alpha",
        "saturation_alpha",
        "saturation_lam",
        "gamma_control",
        "gamma_fourier",
        "intercept",
        "y_sigma",
    ]
    if idata.posterior.sizes["chain"] < 2 or idata.posterior.sizes["draw"] < 4:
        return {
            "sufficient_chains_draws": False,
            "diagnostic_status": "INCOMPLETE",
            "divergences": int(idata.sample_stats.diverging.sum()),
        }
    summary = cast(Any, az).summary(idata, var_names=parameters, round_to="none")
    values = summary[["r_hat", "ess_bulk", "ess_tail"]].to_numpy(dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    divergences = int(idata.sample_stats.diverging.sum())
    bfmi = np.asarray(cast(Any, az).bfmi(idata), dtype=np.float64)
    maxdepth = int(idata.sample_stats.maxdepth_reached.sum())
    max_rhat = float(values[:, 0].max()) if finite else None
    min_bulk = float(values[:, 1].min()) if finite else None
    min_tail = float(values[:, 2].min()) if finite else None
    adequate = (
        finite
        and max_rhat is not None
        and max_rhat <= 1.01
        and min_bulk is not None
        and min_bulk >= 400
        and min_tail is not None
        and min_tail >= 400
        and divergences == 0
        and bool(np.isfinite(bfmi).all())
        and float(bfmi.min()) >= 0.3
        and maxdepth == 0
    )
    return {
        "sufficient_chains_draws": True,
        "diagnostic_status": "ACCEPTABLE" if adequate else "INVESTIGATE",
        "max_rhat": max_rhat,
        "min_ess_bulk": min_bulk,
        "min_ess_tail": min_tail,
        "divergences": divergences,
        "bfmi_by_chain": bfmi.tolist(),
        "maxdepth_reached": maxdepth,
        "maximum_tree_depth_used": int(idata.sample_stats.depth.max()),
        "parameter_summary": json.loads(
            summary.replace([np.inf, -np.inf], np.nan).to_json(orient="index")
        ),
    }


def train_mmm(
    dataset: Path, output: Path, config: MMMConfig | None = None
) -> dict[str, object]:
    """Run prior screening, sampling, checks and persistence on a gated dataset."""
    config = config or MMMConfig()
    if output.exists():
        raise FileExistsError(output)
    code_hash_at_start = source_code_hash()
    data, availability = load_model_inputs(dataset)
    train_rows = len(data) - config.holdout
    x, preprocessing = prepare_mmm_inputs(data, train_rows)
    if (
        availability.iloc[:train_rows].isna().any()
        or (availability.iloc[:train_rows] > data["week"].iloc[train_rows]).any()
    ):
        raise ValueError("training records unavailable at holdout origin")
    y = data["revenue"].to_numpy(dtype=np.float64)
    model = build_mmm(config)
    model.build_model(x.iloc[:train_rows], y[:train_rows])
    model.sample_prior_predictive(
        x.iloc[:train_rows],
        y[:train_rows],
        samples=200,
        random_seed=config.seed,
        combined=False,
    )
    prior = np.asarray(model.idata.prior_predictive["y"], dtype=np.float64)
    prior_check = {
        "negative_fraction": float(np.mean(prior < 0)),
        "scaled_revenue_quantiles": np.quantile(prior, [0.01, 0.5, 0.99]).tolist(),
    }
    if (
        not np.isfinite(prior).all()
        or prior_check["negative_fraction"] > 0.05
        or np.quantile(prior, 0.99) > 5
    ):
        raise ValueError(
            "prior predictive screening failed; revise priors before sampling"
        )
    model.fit(
        x.iloc[:train_rows],
        y[:train_rows],
        draws=config.draws,
        tune=config.tune,
        chains=config.chains,
        cores=1,
        nuts_sampler="nutpie",
        nuts_sampler_kwargs={"maxdepth": config.max_tree_depth},
        target_accept=config.target_accept,
        random_seed=config.seed,
        progressbar=False,
        compute_convergence_checks=config.chains >= 2,
    )
    diagnostic = posterior_diagnostics(model.idata)
    observed_training = model.idata.observed_data.copy(deep=True)
    # Capture original-scale contributions before setting prediction data.
    contribution = model.compute_channel_contribution_original_scale()
    channel_draws = np.asarray(
        contribution.transpose("chain", "draw", "date", "channel"), dtype=np.float64
    )
    roi: NDArray[np.float64] = channel_draws.sum(axis=2) / data.iloc[:train_rows][
        list(config.channels)
    ].sum().to_numpy(dtype=np.float64)
    if not np.isfinite(roi).all():
        raise ValueError("channel ROI undefined; positive historical spend required")
    in_sample = model.sample_posterior_predictive(
        x.iloc[:train_rows],
        extend_idata=True,
        combined=False,
        random_seed=config.seed,
        progressbar=False,
    )
    insample_draws = np.asarray(
        in_sample["y"].transpose("chain", "draw", "date"), dtype=np.float64
    )
    future = model.sample_posterior_predictive(
        x.iloc[train_rows:],
        extend_idata=False,
        combined=False,
        include_last_observations=True,
        random_seed=config.seed,
        progressbar=False,
    )
    holdout_draws = np.asarray(
        future["y"].transpose("chain", "draw", "date"), dtype=np.float64
    )
    curve = model.saturation.sample_curve(model.idata.posterior, max_value=4)
    curve_draws = np.asarray(
        curve.transpose("chain", "draw", "x", "channel"), dtype=np.float64
    )
    target_scale = float(
        model.get_target_transformer().inverse_transform(np.ones((1, 1)))[0, 0]
    )
    spend_scale = np.asarray(
        model.channel_transformer.inverse_transform(np.ones((1, len(config.channels))))[
            0
        ],
        dtype=np.float64,
    )
    curve_draws *= target_scale
    adstocked_spend_grid = (
        np.asarray(curve.coords["x"], dtype=np.float64)[:, None] * spend_scale
    )
    # Prediction internally sets dummy targets and can replace observed_data.
    # Retain the real fitted observations for valid posterior predictive auditing.
    model.idata.observed_data = observed_training
    output.mkdir(parents=True, exist_ok=False)
    model.save(str(output / "posterior.nc"))
    np.savez_compressed(
        output / "channel_draws.npz",
        contributions=channel_draws,
        roi=roi,
        holdout=holdout_draws,
        channels=np.asarray(config.channels),
        saturation_response=curve_draws,
        adstocked_spend_grid=adstocked_spend_grid,
    )
    frames: list[pd.DataFrame] = []
    for split, actual, draws, weeks in (
        ("training", y[:train_rows], insample_draws, data["week"].iloc[:train_rows]),
        ("holdout", y[train_rows:], holdout_draws, data["week"].iloc[train_rows:]),
    ):
        flat = draws.reshape(-1, draws.shape[-1])
        frames.append(
            pd.DataFrame(
                {
                    "week": weeks.to_numpy(),
                    "split": split,
                    "actual": actual,
                    "predicted": flat.mean(axis=0),
                    "lower_90": np.quantile(flat, 0.05, axis=0),
                    "upper_90": np.quantile(flat, 0.95, axis=0),
                }
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    ppc: dict[str, object] = {}
    for split in ("training", "holdout"):
        frame = predictions[predictions["split"] == split]
        ppc[split] = {
            "metrics": forecast_metrics(
                frame["actual"].to_numpy(dtype=np.float64),
                frame["predicted"].to_numpy(dtype=np.float64),
            ),
            "coverage_90": float(
                (
                    (frame["actual"] >= frame["lower_90"])
                    & (frame["actual"] <= frame["upper_90"])
                ).mean()
            ),
        }
    total_draws: NDArray[np.float64] = channel_draws.sum(axis=2).reshape(
        -1, len(config.channels)
    )
    corr: NDArray[np.float64] = (
        np.asarray(np.corrcoef(total_draws, rowvar=False), dtype=np.float64)
        if len(config.channels) > 1
        else np.ones((1, 1))
    )
    evidence = {
        name: {
            "roi_mean": float(roi[..., index].mean()),
            "roi_interval_90": np.quantile(roi[..., index], [0.05, 0.95]).tolist(),
            "contribution_mean": float(total_draws[:, index].mean()),
            "contribution_interval_90": np.quantile(
                total_draws[:, index], [0.05, 0.95]
            ).tolist(),
        }
        for index, name in enumerate(config.channels)
    }
    record: dict[str, object] = {
        "model_family": "pymc_marketing_mmm",
        "model_status": "CANDIDATE_UNEVALUATED",
        "prediction_context": "conditional_on_realized_covariates",
        "config": {**asdict(config), "sampler": "nutpie", "cores": 1},
        "hypothesis": "Test predictive structure and signed control recovery",
        "decision": "candidate_requires_external_evaluation",
        "response_curve_axis": "adstocked_spend_AUD; conditional saturation response",
        "preprocessing": preprocessing,
        "training_window": {
            "start": str(data["week"].iloc[0].date()),
            "end": str(data["week"].iloc[train_rows - 1].date()),
            "rows": train_rows,
        },
        "prior_predictive": prior_check,
        "diagnostics": diagnostic,
        "posterior_predictive": ppc,
        "channels": evidence,
        "channel_contribution_correlation": corr.tolist(),
        "dataset": dataset_evidence(dataset, data),
        "code_hash": code_hash_at_start,
        "limitations": [
            "Observational model assumptions do not prove causality",
            "Positive media priors cannot independently establish positive effects",
            "Realized held-out controls/spend condition prediction",
            "Finite geometric kernel truncates carryover",
            "No optimization before mmm-eval and deterministic release gating",
        ],
        "artifacts": {
            name: sha256((output / name).read_bytes()).hexdigest()
            for name in ("posterior.nc", "channel_draws.npz", "predictions.parquet")
        },
    }
    lock = Path(__file__).resolve().parents[3] / "uv.lock"
    record["dependency_lock_hash"] = (
        sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    )
    write_json(output / "model.json", record)
    return record
