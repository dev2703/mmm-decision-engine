"""Downstream policy is bound to actual raw evidence and model identities."""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from decisionguard.data.artifacts import write_json
from decisionguard.evaluation.artifacts import (
    UPSTREAM_COMMIT,
    load_release,
    source_evidence,
)
from decisionguard.evaluation.policy import (
    EXPECTED_METRICS,
    ReleaseState,
    assess_release,
)
from decisionguard.optimization import artifacts as optimization
from decisionguard.optimization.allocation import BudgetConstraints

DIAGNOSTICS = {
    "diagnostic_status": "ACCEPTABLE",
    "sufficient_chains_draws": True,
    "max_rhat": 1.0,
    "min_ess_bulk": 500.0,
    "min_ess_tail": 500.0,
    "divergences": 0,
    "maxdepth_reached": 0,
    "bfmi_by_chain": [0.4, 0.4],
}


def release_fixture(
    root: Path, *, placebo_passes: bool = True, channels: list[str] | None = None
) -> tuple[Path, Path]:
    model, alternative, evaluation = (
        root / n for n in ("model", "alternative", "evaluation")
    )
    channels = channels or ["a", "b"]
    for path in (model, alternative):
        path.mkdir()
        for name in ("posterior.nc", "channel_draws.npz", "predictions.parquet"):
            (path / name).write_bytes(b"unit fixture; no scientific run")
        write_json(
            path / "model.json",
            {
                "model_family": "pymc_marketing_mmm",
                "config": {
                    "channels": channels,
                    "media_prior_mean": 0.15 if path == model else 0.25,
                    "media_prior_sigma": 0.1,
                },
                "artifacts": {
                    n: sha256((path / n).read_bytes()).hexdigest()
                    for n in (
                        "posterior.nc",
                        "channel_draws.npz",
                        "predictions.parquet",
                    )
                },
                "dataset": {"hash": "same", "quality_status": "RESOLVED"},
                "training_window": {"rows": 143},
                "diagnostics": DIAGNOSTICS,
                "channels": {c: {"roi_mean": 1.0} for c in channels},
            },
        )
    rows: list[dict[str, object]] = []
    for test, metrics in EXPECTED_METRICS.items():
        for metric in sorted(metrics):
            rows.extend(
                {
                    "test_name": test,
                    "general_metric_name": metric,
                    "specific_metric_name": metric + ("_" + channel if channel else ""),
                    "metric_value": 1.0,
                    "metric_pass": placebo_passes if test == "placebo" else True,
                }
                for channel in (
                    channels
                    if test in ("refresh_stability", "perturbation")
                    else [None]
                )
            )
    raw = pd.DataFrame(rows)
    decision = assess_release(
        source_evidence(raw, channels),
        DIAGNOSTICS,
        "RESOLVED",
        channels,
        prior_sensitivity={"a": 0.0, "b": 0.0},
    )
    evaluation.mkdir()
    raw.to_parquet(evaluation / "mmm_eval_raw.parquet", index=False)
    write_json(
        evaluation / "evaluation.json",
        {
            "evaluator": "mmm-eval",
            "commit": UPSTREAM_COMMIT,
            "model_record_hash": sha256(
                (model / "model.json").read_bytes()
            ).hexdigest(),
            "raw_artifact_hash": sha256(
                (evaluation / "mmm_eval_raw.parquet").read_bytes()
            ).hexdigest(),
            "evaluation_window": {"rows": 143},
            "sensitivity_run": str(alternative),
            "progress_identity": {
                "sensitivity_record_hash": sha256(
                    (alternative / "model.json").read_bytes()
                ).hexdigest()
            },
            "policy": asdict(decision),
            "test_execution_errors": [],
            "refits": [{"diagnostics": DIAGNOSTICS}],
        },
    )
    return evaluation, model


def test_verified_release_round_trip(tmp_path: Path) -> None:
    evaluation, model = release_fixture(tmp_path)
    decision, _ = load_release(evaluation, model)
    assert decision.state == ReleaseState.PASS


def test_claimed_pass_cannot_override_failed_placebo(tmp_path: Path) -> None:
    evaluation, model = release_fixture(tmp_path, placebo_passes=False)
    record = json.loads((evaluation / "evaluation.json").read_text())
    record["policy"]["state"] = "PASS"
    write_json(evaluation / "evaluation.json", record)
    with pytest.raises(ValueError, match="differs from verified evidence"):
        load_release(evaluation, model)


def test_block_prevents_loading_posterior_or_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation, model = release_fixture(tmp_path, placebo_passes=False)

    def must_not_load(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("blocked model reached posterior loading")

    monkeypatch.setattr(optimization, "load_response", must_not_load)
    output = tmp_path / "optimization"
    with pytest.raises(ValueError, match="BLOCK"):
        optimization.optimize_run(
            model, evaluation, output, BudgetConstraints(100, {"a": 50, "b": 50})
        )
    assert not output.exists()


@pytest.mark.parametrize("target", ["raw", "model", "alternative"])
def test_changed_artifacts_cannot_reuse_release(tmp_path: Path, target: str) -> None:
    evaluation, model = release_fixture(tmp_path)
    if target == "raw":
        (evaluation / "mmm_eval_raw.parquet").write_bytes(b"changed")
    else:
        path = model if target == "model" else tmp_path / "alternative"
        record = json.loads((path / "model.json").read_text())
        record["channels"]["a"]["roi_mean"] = 5
        write_json(path / "model.json", record)
    with pytest.raises(ValueError):
        load_release(evaluation, model)


def test_upstream_rows_and_units_remain_unmodified() -> None:
    raw = pd.DataFrame(
        [
            {
                "test_name": "holdout_accuracy",
                "general_metric_name": "mape",
                "specific_metric_name": "mape",
                "metric_value": 15.5,
                "metric_pass": False,
            },
            {
                "test_name": "perturbation",
                "general_metric_name": "percentage_change",
                "specific_metric_name": "percentage_change_meta_spend",
                "metric_value": 9.0,
                "metric_pass": False,
            },
            {
                "test_name": "placebo",
                "general_metric_name": "shuffled_channel_roi",
                "specific_metric_name": "shuffled_channel_roi_search_spend_shuffled",
                "metric_value": -60.0,
                "metric_pass": True,
            },
        ]
    )
    before = raw.copy(deep=True)
    evidence = source_evidence(raw, ["meta_spend", "search_spend"])
    pd.testing.assert_frame_equal(raw, before)
    assert evidence[0].value == 15.5 and evidence[0].passed is False
    assert evidence[1].channel == "meta_spend" and evidence[1].value == 9.0
    assert evidence[2].value == -60.0 and evidence[2].passed is True


@pytest.mark.parametrize("value", [None, "not-a-number", float("inf"), True])
def test_undefined_source_values_remain_invalid_instead_of_coerced_to_pass(
    value: object,
) -> None:
    raw = pd.DataFrame(
        [
            {
                "test_name": "holdout_accuracy",
                "general_metric_name": "mape",
                "specific_metric_name": "mape",
                "metric_value": value,
                "metric_pass": True,
            }
        ]
    )
    original = raw.copy(deep=True)
    evidence = source_evidence(raw, ["a"])
    assert evidence[0].value is None
    assert evidence[0].passed is True
    pd.testing.assert_frame_equal(raw, original)


def test_duplicate_source_columns_cannot_hide_failed_flag() -> None:
    raw = pd.DataFrame(
        [["placebo", "shuffled_channel_roi", "roi", 100.0, False, True]],
        columns=[
            "test_name",
            "general_metric_name",
            "specific_metric_name",
            "metric_value",
            "metric_pass",
            "metric_pass",
        ],
    )
    with pytest.raises(ValueError, match="unique upstream metric columns"):
        source_evidence(raw, ["a"])
