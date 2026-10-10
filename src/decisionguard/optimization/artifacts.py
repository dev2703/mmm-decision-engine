"""Planning from joint posterior draws after deterministic release checks."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from pymc_marketing.mmm import MMM
from sklearn.preprocessing import (  # pyright: ignore[reportMissingTypeStubs]
    MaxAbsScaler,  # pyright: ignore[reportMissingTypeStubs]
)

from decisionguard.data.artifacts import (
    file_hash,
    load_model_inputs,
    source_code_hash,
    write_json,
)
from decisionguard.data.integrity import dataset_hash
from decisionguard.evaluation.artifacts import load_release
from decisionguard.evaluation.policy import ReleaseState
from decisionguard.models.artifacts import load_mmm_record
from decisionguard.optimization.allocation import (
    BudgetConstraints,
    PosteriorResponse,
    analyze_allocation,
)


def load_response(
    model_run: Path,
    *,
    horizon: int = 13,
    draws: int = 200,
    seed: int = 42,
    dataset: Path | None = None,
) -> tuple[PosteriorResponse, list[int]]:
    """Sample joint states without replacement; never use synthetic ground truth."""
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0:
        raise ValueError("at least two posterior draws and a nonnegative seed required")
    record = load_mmm_record(model_run)
    data, _ = load_model_inputs(
        dataset or Path(record["dataset"]["path"]), record["dataset"]
    )
    if dataset_hash(data) != record["dataset"]["hash"]:
        raise ValueError("model input identity changed")
    model = cast(Any, MMM.load(str(model_run / "posterior.nc")))
    if (
        model.adstock.lookup_name != "geometric"
        or model.adstock.normalize
        or model.saturation.lookup_name != "michaelis_menten"
        or not model.adstock_first
    ):
        raise ValueError("unsupported planning response specification")
    channels = tuple(record["config"]["channels"])
    posterior = model.idata.posterior
    if list(posterior.coords["channel"].values) != list(channels):
        raise ValueError("posterior channel order differs from model record")
    count = posterior.sizes["chain"] * posterior.sizes["draw"]
    selected = np.random.default_rng(seed).choice(
        count, size=min(draws, count), replace=False
    )

    def parameter(name: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.asarray(
            posterior[name].transpose("chain", "draw", "channel"), dtype=np.float64
        ).reshape(count, len(channels))[selected]

    for transform in (model.get_target_transformer(), model.channel_transformer):
        if len(transform.steps) != 1 or not isinstance(
            transform.named_steps.get("scaler"), MaxAbsScaler
        ):
            raise ValueError(
                "planning requires recorded MaxAbs transforms without centering"
            )
    target_scale = float(model.get_target_transformer().named_steps["scaler"].scale_[0])
    spend_scale = np.asarray(
        model.channel_transformer.named_steps["scaler"].scale_, dtype=np.float64
    )
    response = PosteriorResponse(
        channels=channels,
        decay=parameter("adstock_alpha"),
        amplitude=parameter("saturation_alpha") * target_scale,
        half_saturation=parameter("saturation_lam") * spend_scale,
        history=data.iloc[: record["training_window"]["rows"]][list(channels)].to_numpy(
            dtype=np.float64
        ),
        horizon=horizon,
        lags=record["config"]["adstock_lags"],
    )
    return response, selected.tolist()


def optimize_run(
    model_run: Path,
    evaluation: Path,
    output: Path,
    constraints: BudgetConstraints,
    *,
    horizon: int = 13,
    draws: int = 200,
    seed: int = 42,
    dataset: Path | None = None,
    sensitivity_run: Path | None = None,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    release, _ = load_release(evaluation, model_run, sensitivity_run=sensitivity_run)
    if release.state == ReleaseState.BLOCK:
        raise ValueError("BLOCK model cannot reach optimization")
    response, selected = load_response(
        model_run, horizon=horizon, draws=draws, seed=seed, dataset=dataset
    )
    result = analyze_allocation(response, constraints, release)
    result.update(
        {
            "model_record_hash": file_hash(model_run / "model.json"),
            "evaluation_record_hash": file_hash(evaluation / "evaluation.json"),
            "release": asdict(release),
            "constraints": asdict(constraints),
            "horizon_weeks": horizon,
            "seed": seed,
            "posterior_indices": selected,
            "code_hash": source_code_hash(),
            "dependency_lock_hash": file_hash(Path("uv.lock")),
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "optimization.json", result)
    return result
