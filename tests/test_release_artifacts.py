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


def release_fixture(root: Path, *, placebo_passes: bool = True) -> tuple[Path, Path]:
    model, alternative, evaluation = (
        root / n for n in ("model", "alternative", "evaluation")
    )
    channels = ["a", "b"]
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
                "diagnostics": {"diagnostic_status": "ACCEPTABLE"},
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
        {"diagnostic_status": "ACCEPTABLE"},
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
            "refits": [{"diagnostics": {"diagnostic_status": "ACCEPTABLE"}}],
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
