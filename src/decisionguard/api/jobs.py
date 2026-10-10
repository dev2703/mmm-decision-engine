"""CLI-owned heavy work with short claim and publication transactions."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from decisionguard.api.database import (
    Dataset,
    EvaluationRun,
    ModelRun,
    OptimizationRun,
    Scenario,
)
from decisionguard.api.evaluations import load_registered_model, verified_evaluation
from decisionguard.config import Settings, artifact_path
from decisionguard.data.artifacts import file_hash, load_model_ready
from decisionguard.models.artifacts import load_mmm_record
from decisionguard.models.config import MMMConfig


def _fail_running_job(
    engine: Engine,
    table: type[ModelRun] | type[EvaluationRun] | type[OptimizationRun],
    identifier: UUID | None,
    error: Exception,
) -> None:
    if identifier is None:
        return
    with Session(engine) as session, session.begin():
        job = session.get(table, identifier, with_for_update=True)
        if job is not None and job.status == "RUNNING":
            job.status = "FAILED"
            job.error_type = type(error).__name__
            job.finished_at = datetime.now(UTC)


def train_experiment(experiment_id: UUID, settings: Settings) -> dict[str, Any]:
    """A second worker cannot claim a RUNNING run; failures never publish results."""
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    run_id: UUID | None = None
    try:
        with Session(engine) as session, session.begin():
            run = session.scalar(
                select(ModelRun)
                .where(
                    ModelRun.experiment_id == experiment_id, ModelRun.status == "QUEUED"
                )
                .order_by(ModelRun.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if run is None:
                raise ValueError("no queued model run available for this experiment")
            dataset = session.get(Dataset, run.dataset_id)
            if dataset is None:
                raise ValueError("model run dataset missing")
            run_id = run.id
            configuration = dict(run.configuration)
            dataset_path = artifact_path(settings.artifact_root, dataset.artifact_uri)
            expected_hash = dataset.hash
            expected_quality = dict(dataset.quality)
            output = artifact_path(settings.artifact_root, run.artifact_uri)
            run.status = "RUNNING"
            run.started_at = datetime.now(UTC)
        # No open transaction while sampling, and no scientific imports at API startup.
        if expected_quality["clean_hash"] != expected_hash:
            raise ValueError("registered dataset identity changed")
        load_model_ready(dataset_path, expected_quality=expected_quality)
        from decisionguard.models.bayesian import train_mmm

        train_mmm(dataset_path, output, MMMConfig(**configuration))
        result = load_mmm_record(output)
        if result["dataset"]["hash"] != expected_hash:
            raise ValueError("completed model uses a different dataset")
        record_hash = file_hash(output / "model.json")
        with Session(engine) as session, session.begin():
            run = session.get(ModelRun, run_id, with_for_update=True)
            if run is None or run.status != "RUNNING":
                raise ValueError("model run no longer belongs to this worker")
            run.status = "SUCCEEDED"
            run.model_status = result["model_status"]
            run.code_version = result["code_hash"]
            run.record_hash = record_hash
            run.training_window = result["training_window"]
            run.metrics = result["posterior_predictive"]
            run.finished_at = datetime.now(UTC)
        return result
    except Exception as error:
        _fail_running_job(engine, ModelRun, run_id, error)
        raise
    finally:
        engine.dispose()


def optimize_scenario(scenario_id: UUID, settings: Settings) -> dict[str, Any]:
    """Claim one optimization, refusing stale evidence before solve and publication."""
    from decisionguard.api.optimizations import (
        current_scenario_inputs,
        scenario_snapshot,
        verified_optimization,
        verify_scenario,
    )

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    run_id: UUID | None = None
    try:
        with Session(engine, expire_on_commit=False) as session, session.begin():
            run = session.scalar(
                select(OptimizationRun)
                .where(
                    OptimizationRun.scenario_id == scenario_id,
                    OptimizationRun.status == "QUEUED",
                )
                .order_by(OptimizationRun.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if run is None:
                raise ValueError("no queued optimization available for this scenario")
            run_id = run.id
            run.status = "RUNNING"
            run.started_at = datetime.now(UTC)
        with Session(engine, expire_on_commit=False) as session, session.begin():
            scenario = session.get(Scenario, scenario_id)
            if scenario is None:
                raise ValueError("optimization scenario missing")
            model, evaluation, alternative = current_scenario_inputs(session, scenario)
            dataset = session.get(Dataset, model.dataset_id)
            if dataset is None:
                raise ValueError("optimization dataset missing")
        # No transaction is open during posterior loading or numerical solves.
        if run.configuration["scenario"] != scenario_snapshot(scenario):
            raise ValueError("queued scenario changed")
        primary = verify_scenario(
            scenario, model, evaluation, alternative, settings.artifact_root
        )
        if (
            model.record_hash != run.configuration["model_record_hash"]
            or dataset.hash != primary["dataset"]["hash"]
            or dataset.quality["clean_hash"] != dataset.hash
        ):
            raise ValueError("registered optimization inputs changed")
        dataset_path = artifact_path(settings.artifact_root, dataset.artifact_uri)
        load_model_ready(dataset_path, expected_quality=dataset.quality)
        from decisionguard.optimization.allocation import BudgetConstraints
        from decisionguard.optimization.artifacts import optimize_run

        output = artifact_path(settings.artifact_root, run.artifact_uri)
        result = optimize_run(
            artifact_path(settings.artifact_root, model.artifact_uri),
            artifact_path(settings.artifact_root, evaluation.artifact_uri),
            output,
            BudgetConstraints(**scenario.constraints),
            horizon=scenario.horizon_weeks,
            draws=run.configuration["draws"],
            seed=run.configuration["seed"],
            dataset=dataset_path,
            sensitivity_run=artifact_path(
                settings.artifact_root, alternative.artifact_uri
            ),
        )
        run.record_hash = file_hash(output / "optimization.json")
        summary = verified_optimization(run, scenario, settings.artifact_root)
        with Session(engine) as session, session.begin():
            current = session.get(Scenario, scenario_id)
            if (
                current is None
                or scenario_snapshot(current) != run.configuration["scenario"]
            ):
                raise ValueError("queued scenario changed")
            inputs = current_scenario_inputs(session, current, lock=True)
            verify_scenario(current, *inputs, settings.artifact_root)
            stored = session.get(OptimizationRun, run_id, with_for_update=True)
            if (
                stored is None
                or stored.status != "RUNNING"
                or stored.configuration != run.configuration
            ):
                raise ValueError("optimization no longer belongs to this worker")
            stored.status = "SUCCEEDED"
            stored.summary = summary
            stored.record_hash = run.record_hash
            stored.finished_at = datetime.now(UTC)
        return result
    except Exception as error:
        _fail_running_job(engine, OptimizationRun, run_id, error)
        raise
    finally:
        engine.dispose()


def evaluate_model(model_run_id: UUID, settings: Settings) -> dict[str, Any]:
    """Claim one registered evaluation; BLOCK is a successful scientific result."""
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    evaluation_id: UUID | None = None
    try:
        with Session(engine, expire_on_commit=False) as session, session.begin():
            evaluation = session.scalar(
                select(EvaluationRun)
                .where(
                    EvaluationRun.model_run_id == model_run_id,
                    EvaluationRun.status == "QUEUED",
                )
                .order_by(EvaluationRun.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if evaluation is None:
                raise ValueError("no queued evaluation available for this model")
            model = session.get(ModelRun, evaluation.model_run_id)
            alternative = session.get(ModelRun, evaluation.sensitivity_run_id)
            dataset = session.get(Dataset, model.dataset_id) if model else None
            if model is None or alternative is None or dataset is None:
                raise ValueError("registered evaluation inputs missing")
            evaluation_id = evaluation.id
            dataset_path = artifact_path(settings.artifact_root, dataset.artifact_uri)
            expected_hash, expected_quality = dataset.hash, dict(dataset.quality)
            evaluation.status = "RUNNING"
            evaluation.started_at = datetime.now(UTC)
        # Detached snapshots: no database transaction is held during external fits.
        primary = load_registered_model(model, settings.artifact_root)
        load_registered_model(alternative, settings.artifact_root)
        config = evaluation.configuration
        from decisionguard.evaluation.artifacts import UPSTREAM_COMMIT

        if (
            config["commit"] != UPSTREAM_COMMIT
            or model.record_hash != config["model_record_hash"]
            or alternative.record_hash != config["sensitivity_record_hash"]
            or primary["dataset"]["hash"] != expected_hash
            or expected_quality["clean_hash"] != expected_hash
        ):
            raise ValueError("registered evaluation inputs changed")
        load_model_ready(dataset_path, expected_quality=expected_quality)
        from decisionguard.evaluation.mmm_eval import evaluate_mmm

        output = artifact_path(settings.artifact_root, evaluation.artifact_uri)
        result = evaluate_mmm(
            artifact_path(settings.artifact_root, model.artifact_uri),
            artifact_path(settings.artifact_root, alternative.artifact_uri),
            output,
            draws=config["draws"],
            tune=config["tune"],
            dataset=dataset_path,
        )
        summary, _ = verified_evaluation(
            evaluation, model, alternative, settings.artifact_root
        )
        record_hash = file_hash(output / "evaluation.json")
        with Session(engine) as session, session.begin():
            stored = session.get(EvaluationRun, evaluation_id, with_for_update=True)
            if stored is None or stored.status != "RUNNING":
                raise ValueError("evaluation no longer belongs to this worker")
            stored.status = "SUCCEEDED"
            stored.summary = summary
            stored.record_hash = record_hash
            stored.finished_at = datetime.now(UTC)
        return result
    except Exception as error:
        _fail_running_job(engine, EvaluationRun, evaluation_id, error)
        raise
    finally:
        engine.dispose()
