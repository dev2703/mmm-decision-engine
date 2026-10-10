"""Current release authority and persisted scenario/result identity checks."""

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from sqlalchemy.orm import Session

from decisionguard.api.database import (
    EvaluationRun,
    ModelRun,
    OptimizationRun,
    Scenario,
)
from decisionguard.api.evaluations import latest_evaluation, verified_evaluation
from decisionguard.config import artifact_path
from decisionguard.data.artifacts import file_hash
from decisionguard.evaluation.policy import ReleaseDecision
from decisionguard.optimization.allocation import BudgetConstraints


def scenario_snapshot(scenario: Scenario) -> dict[str, Any]:
    return {
        "scenario_id": str(scenario.id),
        "model_run_id": str(scenario.model_run_id),
        "evaluation_run_id": str(scenario.evaluation_run_id),
        "evaluation_record_hash": scenario.evaluation_record_hash,
        "budget": scenario.budget,
        "constraints": scenario.constraints,
        "effective_bounds": scenario.effective_bounds,
        "release": scenario.release,
        "horizon_weeks": scenario.horizon_weeks,
        "risk_policy": scenario.risk_policy,
    }


def current_scenario_inputs(
    session: Session,
    scenario: Scenario,
    *,
    lock: bool = False,
) -> tuple[ModelRun, EvaluationRun, ModelRun]:
    """Serialize consequential boundaries with new evaluation requests."""
    model = session.get(ModelRun, scenario.model_run_id, with_for_update=lock)
    evaluation = latest_evaluation(session, scenario.model_run_id)
    if (
        model is None
        or evaluation is None
        or evaluation.status != "SUCCEEDED"
        or evaluation.id != scenario.evaluation_run_id
        or evaluation.record_hash != scenario.evaluation_record_hash
    ):
        raise ValueError("scenario evidence is stale; create a new scenario")
    alternative = session.get(ModelRun, evaluation.sensitivity_run_id)
    if alternative is None:
        raise ValueError("sensitivity model missing")
    return model, evaluation, alternative


def verify_scenario(
    scenario: Scenario,
    model: ModelRun,
    evaluation: EvaluationRun,
    alternative: ModelRun,
    root: Path,
) -> dict[str, Any]:
    summary, record = verified_evaluation(evaluation, model, alternative, root)
    release = TypeAdapter(ReleaseDecision).validate_python(summary)
    constraints = BudgetConstraints(**scenario.constraints)
    channels = tuple(record["config"]["channels"])
    lower, upper = constraints.bounds(channels, release)
    bounds = {c: [float(lower[i]), float(upper[i])] for i, c in enumerate(channels)}
    if (
        summary != scenario.release
        or constraints.total != scenario.budget
        or bounds != scenario.effective_bounds
    ):
        raise ValueError("scenario differs from verified constraints or release")
    return record


def optimization_summary(result: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    """Keep small decision evidence in PostgreSQL; allocation draws stay on disk."""
    return {
        "baseline_allocation": scenario.constraints["current"],
        "risk_policy": scenario.risk_policy,
        "selected": result["alternatives"][scenario.risk_policy],
        "alternatives": result["alternatives"],
        "release": result["release"],
        "horizon_weeks": result["horizon_weeks"],
        "outcome_units": result["outcome_units"],
        "code_hash": result["code_hash"],
    }


def verified_optimization(
    run: OptimizationRun,
    scenario: Scenario,
    root: Path,
) -> dict[str, Any]:
    path = artifact_path(root, run.artifact_uri) / "optimization.json"
    if run.record_hash is None or file_hash(path) != run.record_hash:
        raise ValueError("optimization artifact differs from registered identity")
    result = json.loads(path.read_text())
    if (
        run.configuration["scenario"] != scenario_snapshot(scenario)
        or result["model_record_hash"] != run.configuration["model_record_hash"]
        or result["constraints"] != scenario.constraints
        or result["evaluation_record_hash"] != scenario.evaluation_record_hash
        or result["effective_bounds"] != scenario.effective_bounds
        or result["release"] != scenario.release
        or result["horizon_weeks"] != scenario.horizon_weeks
        or result["seed"] != run.configuration["seed"]
        or len(result["posterior_indices"]) > run.configuration["draws"]
    ):
        raise ValueError("optimization differs from queued scenario")
    summary = optimization_summary(result, scenario)
    if run.summary is not None and run.summary != summary:
        raise ValueError("optimization summary differs from artifact")
    return summary
