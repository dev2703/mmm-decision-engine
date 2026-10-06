"""Comparison grouping and deterministic repairs preserve scientific meaning."""

import json
from pathlib import Path

import numpy as np
import pytest

from decisionguard.data.artifacts import write_dataset
from decisionguard.data.corruption import CorruptionConfig, corrupt_dataset
from decisionguard.data.integrity import clean_dataset
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset
from decisionguard.experiments.baseline import BaselineConfig, evaluate_baseline
from decisionguard.experiments.comparison import run_comparison


def test_comparison_separates_information_context_and_records_real_runs(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    write_dataset(generate_dataset(SyntheticConfig(weeks=104)).observations, data)
    output = tmp_path / "comparison"
    summary = run_comparison(data, output, BaselineConfig(period=13, initial_train=52))
    record = json.loads((output / "comparison.json").read_text())
    assert summary == record
    models = {row["model"]: row for row in record["models"]}
    assert len(models) == 5
    for row in models.values():
        assert Path(row["experiment_path"]).exists()
        assert row["experiment_hash"]
    selected = record["selected_by_cv_mae"]
    for context, winner in selected.items():
        assert models[winner]["prediction_context"] == context
        assert models[winner]["cv"]["mae"] == min(
            row["cv"]["mae"]
            for row in models.values()
            if row["prediction_context"] == context
        )
    with pytest.raises(FileExistsError):
        run_comparison(data, output, BaselineConfig(period=13, initial_train=52))


def test_unit_alias_and_duplicate_repairs_recover_clean_predictive_evidence() -> None:
    world = generate_dataset(SyntheticConfig(weeks=104))
    corrupted = corrupt_dataset(
        world,
        CorruptionConfig(
            unit_error_weeks=52,
            duplicate_rows=8,
            meta_alias="fb_spend",
        ),
    )
    repaired = clean_dataset(corrupted.observations).require_model_ready()
    original = clean_dataset(world.observations).require_model_ready()
    config = BaselineConfig(model="ridge_raw", initial_train=65)
    before = evaluate_baseline(original, config)
    after = evaluate_baseline(repaired, config)
    np.testing.assert_allclose(
        before.predictions["predicted"], after.predictions["predicted"], rtol=1e-12
    )
    assert after.holdout_metrics["mae"] == pytest.approx(before.holdout_metrics["mae"])
