"""Consume verified release evidence without importing the evaluation runtime."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decisionguard.data.artifacts import file_hash
from decisionguard.evaluation.policy import (
    MetricEvidence,
    ReleaseDecision,
    assess_release,
)
from decisionguard.models.artifacts import (
    load_mmm_record,
    roi_sensitivity,
    validate_prior_sources,
)

UPSTREAM_COMMIT = "71d20009feaa30dd9606ffface62f16fb1134265"


def source_evidence(raw: pd.DataFrame, channels: list[str]) -> list[MetricEvidence]:
    """Interpret source flags/units directly, preserving the separate raw table."""
    required = {
        "test_name",
        "general_metric_name",
        "specific_metric_name",
        "metric_value",
        "metric_pass",
    }
    if not raw.columns.is_unique or not required <= set(raw.columns):
        raise ValueError("unique upstream metric columns required")
    evidence: list[MetricEvidence] = []
    for row in raw.to_dict(orient="records"):
        specific = str(row["specific_metric_name"])
        channel = next(
            (name for name in channels if specific.endswith("_" + name)), None
        )
        try:
            value = (
                float(row["metric_value"])
                if type(row["metric_value"]) is not bool
                else float("nan")
            )
        except (TypeError, ValueError):
            value = float("nan")
        evidence.append(
            MetricEvidence(
                str(row["test_name"]),
                str(row["general_metric_name"]),
                value if np.isfinite(value) else None,
                bool(row["metric_pass"]) if type(row["metric_pass"]) is bool else None,
                channel,
            )
        )
    return evidence


def load_release(
    evaluation: Path, model_run: Path, *, sensitivity_run: Path | None = None
) -> tuple[ReleaseDecision, dict[str, Any]]:
    """Recompute policy from verified evidence before downstream decisions."""
    record = json.loads((evaluation / "evaluation.json").read_text())
    if record["evaluator"] != "mmm-eval" or record["commit"] != UPSTREAM_COMMIT:
        raise ValueError("unsupported evaluator source")
    model = load_mmm_record(model_run)
    model_hash = file_hash(model_run / "model.json")
    if model_hash != record["model_record_hash"]:
        raise ValueError("evaluation belongs to a different model")
    raw_file = evaluation / "mmm_eval_raw.parquet"
    if file_hash(raw_file) != record["raw_artifact_hash"]:
        raise ValueError("evaluation raw artifact hash mismatch")
    if record["evaluation_window"] != model["training_window"]:
        raise ValueError("evaluation training window mismatch")
    diagnostic = dict(model["diagnostics"])
    refits = record.get("refits", [])
    if (
        not refits
        or record.get("test_execution_errors")
        or any(
            event["diagnostics"]["diagnostic_status"] != "ACCEPTABLE"
            for event in refits
        )
    ):
        diagnostic["diagnostic_status"] = "INVESTIGATE"
    # Sensitivity is separately bound to a checksum-verified source model.
    alternative_path = sensitivity_run or Path(record["sensitivity_run"])
    alternative = load_mmm_record(alternative_path)
    identity = record.get("progress_identity", {})
    if identity.get("sensitivity_record_hash") != file_hash(
        alternative_path / "model.json"
    ):
        raise ValueError("evaluation lacks verified sensitivity identity")
    validate_prior_sources(model, alternative)
    if alternative["diagnostics"]["diagnostic_status"] != "ACCEPTABLE":
        diagnostic["diagnostic_status"] = "INVESTIGATE"
    channels = model["config"]["channels"]
    sensitivity = roi_sensitivity(model, alternative)
    decision = assess_release(
        source_evidence(pd.read_parquet(raw_file), channels),
        diagnostic,
        model["dataset"]["quality_status"],
        channels,
        prior_sensitivity=sensitivity,
    )
    # JSON round-trip normalizes enum/tuple representations in the persisted policy.
    if json.loads(json.dumps(asdict(decision))) != record["policy"]:
        raise ValueError("stored release policy differs from verified evidence")
    return decision, model
