"""Load complete immutable MMM evidence and record real sensitivity comparisons."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from decisionguard.data.artifacts import source_code_hash, write_json


def load_mmm_record(directory: Path) -> dict[str, Any]:
    record = json.loads((directory / "model.json").read_text())
    for name in ("posterior.nc", "channel_draws.npz", "predictions.parquet"):
        if (
            sha256((directory / name).read_bytes()).hexdigest()
            != record["artifacts"][name]
        ):
            raise ValueError(f"{name} model artifact hash mismatch")
    if record["model_family"] != "pymc_marketing_mmm":
        raise ValueError("not a PyMC-Marketing model run")
    return record


def validate_prior_sources(first: dict[str, Any], second: dict[str, Any]) -> None:
    if (
        first["dataset"]["hash"] != second["dataset"]["hash"]
        or first["training_window"] != second["training_window"]
        or first["config"]["channels"] != second["config"]["channels"]
    ):
        raise ValueError(
            "sensitivity requires matching data, training window and channels"
        )
    prior_keys = {"media_prior_mean", "media_prior_sigma"}
    if not any(first["config"][key] != second["config"][key] for key in prior_keys):
        raise ValueError("prior sensitivity requires a changed media prior")
    sampler_keys = {
        "draws",
        "tune",
        "chains",
        "seed",
        "target_accept",
        "max_tree_depth",
        "sampler",
        "cores",
    }
    for key in first["config"].keys() | second["config"].keys():
        if key not in prior_keys | sampler_keys and first["config"].get(key) != second[
            "config"
        ].get(key):
            raise ValueError("prior sensitivity cannot also change model specification")


def compare_prior_sensitivity(
    baseline: Path, alternative: Path, output: Path
) -> dict[str, object]:
    first, second = load_mmm_record(baseline), load_mmm_record(alternative)
    validate_prior_sources(first, second)
    if output.exists():
        raise FileExistsError(output)
    differences = {
        key: [value, second["config"].get(key)]
        for key, value in first["config"].items()
        if value != second["config"].get(key)
    }
    if not differences:
        raise ValueError("sensitivity requires a changed configuration")
    channels = {}
    for name, evidence in first["channels"].items():
        base = evidence["roi_mean"]
        alt = second["channels"][name]["roi_mean"]
        channels[name] = {
            "baseline_roi_mean": base,
            "alternative_roi_mean": alt,
            "relative_change": abs(alt - base) / abs(base) if base else None,
            "baseline_interval_90": evidence["roi_interval_90"],
            "alternative_interval_90": second["channels"][name]["roi_interval_90"],
        }
    record: dict[str, object] = {
        "hypothesis": "Test whether plausible prior changes alter channel evidence",
        "changed_configuration": differences,
        "channels": channels,
        "diagnostics": {
            "baseline": first["diagnostics"],
            "alternative": second["diagnostics"],
        },
        "runs": {
            str(path.resolve()): sha256((path / "model.json").read_bytes()).hexdigest()
            for path in (baseline, alternative)
        },
        "decision": "retain sensitivity evidence; no automatic promotion",
        "code_hash": source_code_hash(),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "sensitivity.json", record)
    return record
