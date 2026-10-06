"""Bayesian boundaries, actual small-chain serialization and signed recovery."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from decisionguard.data.artifacts import write_dataset
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset
from decisionguard.models.bayesian import (
    MMMConfig,
    build_mmm,
    prepare_mmm_inputs,
    train_mmm,
)


def test_mmm_preprocessing_cannot_learn_from_holdout() -> None:
    data = generate_dataset(SyntheticConfig(weeks=78)).observations
    original, state = prepare_mmm_inputs(data, 65)
    changed = data.copy()
    changed.loc[65:, "price"] += 10000
    repeated, second = prepare_mmm_inputs(changed, 65)
    assert state == second
    pd.testing.assert_frame_equal(original.iloc[:65], repeated.iloc[:65])
    assert not original.iloc[65:].equals(repeated.iloc[65:])
    assert "revenue" not in original


def test_invalid_sampler_and_channel_configuration_fails() -> None:
    with pytest.raises(ValueError):
        MMMConfig(draws=0)
    with pytest.raises(ValueError):
        MMMConfig(channels=("unknown",))
    with pytest.raises(ValueError):
        MMMConfig(media_prior_mean=np.nan)


def test_bayesian_blocked_data_cannot_build_or_create_run(tmp_path: Path) -> None:
    data = generate_dataset(SyntheticConfig(weeks=78)).observations
    data.loc[0, "revenue"] = np.nan
    directory = tmp_path / "data"
    write_dataset(data, directory)
    with pytest.raises(ValueError, match="blocked"):
        train_mmm(directory, tmp_path / "mmm")
    assert not (tmp_path / "mmm").exists()


@pytest.mark.scientific
def test_fast_mmm_build_sample_and_reload_preserve_real_posterior(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "data"
    write_dataset(generate_dataset(SyntheticConfig(weeks=78)).observations, directory)
    config = MMMConfig(draws=8, tune=16, chains=1, channels=("search_spend",))
    output = tmp_path / "mmm"
    record = train_mmm(directory, output, config)
    assert record["model_status"] == "CANDIDATE_UNEVALUATED"
    diagnostic = cast(dict[str, Any], record["diagnostics"])
    assert diagnostic["diagnostic_status"] == "INCOMPLETE"
    from pymc_marketing.mmm import MMM

    loaded = cast(Any, MMM).load(str(output / "posterior.nc"))
    assert loaded.idata.posterior.sizes["draw"] == 8
    np.testing.assert_allclose(
        loaded.idata.observed_data["y"].to_numpy(),
        generate_dataset(SyntheticConfig(weeks=78))
        .observations["revenue"]
        .iloc[:65]
        .to_numpy()
        / generate_dataset(SyntheticConfig(weeks=78))
        .observations["revenue"]
        .iloc[:65]
        .max(),
    )
    archive = np.load(output / "channel_draws.npz", allow_pickle=False)
    assert archive["contributions"].shape == (1, 8, 65, 1)
    assert archive["holdout"].shape == (1, 8, 13)
    assert np.isfinite(archive["roi"]).all()
    assert archive["saturation_response"].shape == (1, 8, 100, 1)
    assert (np.diff(archive["saturation_response"], axis=2) >= 0).all()
    model = build_mmm(config)
    assert model.control_columns == [
        "price",
        "promotion",
        "macro_index",
        "competitor_index",
        "trend",
    ]


@pytest.mark.scientific
def test_clean_synthetic_signed_controls_recover_without_sign_constrained_priors(
    tmp_path: Path,
) -> None:
    world = generate_dataset(SyntheticConfig(weeks=104))
    directory = tmp_path / "data"
    write_dataset(world.observations, directory)
    config = MMMConfig(draws=100, tune=200, chains=2)
    train_mmm(directory, tmp_path / "mmm", config)
    from pymc_marketing.mmm import MMM

    fitted = cast(Any, MMM).load(str(tmp_path / "mmm" / "posterior.nc"))
    controls = fitted.idata.posterior["gamma_control"]
    assert float((controls.sel(control="price") < 0).mean()) > 0.9
    assert float((controls.sel(control="promotion") > 0).mean()) > 0.9
