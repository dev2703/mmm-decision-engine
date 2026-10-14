"""Media recovery against privileged synthetic truth from a well-specified world."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from decisionguard.data.artifacts import write_dataset
from decisionguard.data.synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    generate_dataset,
)
from decisionguard.models.bayesian import train_mmm
from decisionguard.models.config import MMMConfig

CONFIG = MMMConfig(draws=500, tune=500, chains=2)


@dataclass(frozen=True)
class RecoveryFit:
    world: SyntheticDataset
    rows: int
    roi: NDArray[np.float64]
    contributions: NDArray[np.float64]
    prior_total_scaled: NDArray[np.float64]
    posterior_total_scaled: NDArray[np.float64]


@pytest.fixture(scope="module")
def fit(tmp_path_factory: pytest.TempPathFactory) -> RecoveryFit:
    root: Path = tmp_path_factory.mktemp("recovery")
    world = generate_dataset(SyntheticConfig(weeks=156, seed=42))
    write_dataset(world.observations, root / "data")
    record = train_mmm(root / "data", root / "mmm", CONFIG)
    from pymc_marketing.mmm import MMM

    idata = cast(Any, MMM).load(str(root / "mmm" / "posterior.nc")).idata
    with np.load(root / "mmm" / "channel_draws.npz", allow_pickle=False) as archive:
        roi = np.asarray(archive["roi"], dtype=np.float64)
        contributions = np.asarray(archive["contributions"], dtype=np.float64)

    def total_scaled(group: str) -> NDArray[np.float64]:
        draws = idata[group]["channel_contribution"].sum(("date", "channel"))
        return np.asarray(draws, dtype=np.float64).ravel()

    return RecoveryFit(
        world,
        cast(dict[str, int], record["training_window"])["rows"],
        roi.reshape(-1, len(CONFIG.channels)),
        contributions.sum(axis=(2, 3)).ravel(),
        total_scaled("prior"),
        total_scaled("posterior"),
    )


def true_contribution(fit: RecoveryFit, channel: str) -> float:
    name = channel.removesuffix("_spend")
    return float(fit.world.truth[f"{name}_contribution"].iloc[: fit.rows].sum())


@pytest.mark.scientific
def test_total_media_posterior_is_informed_by_data(fit: RecoveryFit) -> None:
    """Saturation alpha and lam trade off; their identified product is contribution."""
    shrinkage = fit.posterior_total_scaled.std() / fit.prior_total_scaled.std()
    assert shrinkage < 0.75


@pytest.mark.scientific
def test_media_roi_and_total_contribution_recover_truth(fit: RecoveryFit) -> None:
    """Total media is the baseline/media split that budget decisions depend on."""
    misses: list[str] = []
    for index, channel in enumerate(CONFIG.channels):
        spend = float(fit.world.observations[channel].iloc[: fit.rows].sum())
        truth = true_contribution(fit, channel) / spend
        lower, upper = np.quantile(fit.roi[:, index], [0.05, 0.95])
        if not lower <= truth <= upper:
            misses.append(
                f"{channel} ROI {truth:.3f} outside [{lower:.3f}, {upper:.3f}]"
            )
    total = sum(true_contribution(fit, channel) for channel in CONFIG.channels)
    lower, upper = np.quantile(fit.contributions, [0.05, 0.95])
    if not lower <= total <= upper:
        misses.append(f"total media {total:.0f} outside [{lower:.0f}, {upper:.0f}]")
    assert not misses, "; ".join(misses)
