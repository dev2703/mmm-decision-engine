"""Register reproducible demo evidence without changing scientific artifacts."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from decisionguard.api.database import (
    Dataset,
    EvaluationRun,
    Experiment,
    ModelRun,
    Project,
)
from decisionguard.api.evaluations import verified_evaluation
from decisionguard.config import Settings
from decisionguard.data.artifacts import (
    file_hash,
    load_model_inputs,
    load_quality,
    write_simulation,
)
from decisionguard.data.corruption import CorruptionConfig, corrupt_dataset
from decisionguard.data.integrity import IntegrityConfig
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset
from decisionguard.evaluation.artifacts import load_release
from decisionguard.models.artifacts import load_mmm_record


def seed_demo(
    settings: Settings,
    *,
    dataset: Path | None = None,
    model: Path | None = None,
    sensitivity: Path | None = None,
    evaluation: Path | None = None,
) -> dict[str, Any]:
    """One transaction; import paths must already be inside ARTIFACT_ROOT.

    Existing immutable paths are registered in place. Re-imports fail their unique
    constraints instead of overwriting provenance. Registration time is not fit time.
    """
    if settings.read_only:
        raise ValueError("Seeding requires a writable workspace")
    if any((model, sensitivity, evaluation)) and not all(
        (dataset, model, sensitivity, evaluation)
    ):
        raise ValueError("Import requires dataset, model, sensitivity and evaluation")
    root = settings.artifact_root
    for path in (dataset, model, sensitivity, evaluation):
        if path is not None and not path.resolve().is_relative_to(root):
            raise ValueError("Demo artifacts must be inside ARTIFACT_ROOT")
    if dataset is None:
        dataset = root / "demo" / str(uuid4())
        clean = generate_dataset(SyntheticConfig(seed=42, weeks=156))
        write_simulation(
            clean,
            corrupt_dataset(clean, CorruptionConfig(seed=42)),
            dataset,
            IntegrityConfig(
                expected_start=str(clean.observations["week"].min())[:10],
                expected_end=str(clean.observations["week"].max())[:10],
            ),
        )
    quality = load_quality(dataset)
    data, _ = load_model_inputs(dataset)
    provenance = json.loads((dataset / "provenance.json").read_text())
    if not provenance.get("simulation"):
        raise ValueError("Seeded demo requires a recorded synthetic dataset")
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in (model, sensitivity):
        if path is not None:
            record = load_mmm_record(path)
            load_model_inputs(dataset, record["dataset"])
            records.append((path, record))
    engine = create_engine(settings.database_url)
    try:
        with Session(engine) as session, session.begin():
            project = Project(name="Evidence in practice · seeded demo")
            session.add(project)
            session.flush()
            registered = Dataset(
                project_id=project.id,
                version=1,
                source="SYNTHETIC",
                artifact_uri=str(dataset.resolve().relative_to(root)),
                schema_version=quality["schema_version"],
                date_start=data["week"].min().date(),
                date_end=data["week"].max().date(),
                hash=quality["clean_hash"],
                quality_status=quality["status"],
                configuration=json.loads((dataset / "generation.json").read_text())[
                    "generation"
                ],
                quality=quality,
            )
            session.add(registered)
            session.flush()
            project.current_dataset_id = registered.id
            runs: list[ModelRun] = []
            for path, record in records:
                config = {
                    key: value
                    for key, value in record["config"].items()
                    if key not in {"sampler", "cores"}
                }
                experiment = Experiment(
                    project_id=project.id,
                    dataset_id=registered.id,
                    hypothesis=record["hypothesis"],
                    model_family=record["model_family"],
                    configuration=config,
                )
                session.add(experiment)
                session.flush()
                run = ModelRun(
                    experiment_id=experiment.id,
                    dataset_id=registered.id,
                    model_family=record["model_family"],
                    configuration=config,
                    artifact_uri=str(path.resolve().relative_to(root)),
                    status="SUCCEEDED",
                    model_status=record["model_status"],
                    record_hash=file_hash(path / "model.json"),
                    code_version=record["code_hash"],
                    training_window=record["training_window"],
                    metrics=record["posterior_predictive"],
                )
                session.add(run)
                session.flush()
                runs.append(run)
            result: dict[str, Any] = {
                "project_id": str(project.id),
                "dataset_id": str(registered.id),
                "model_run_ids": [str(run.id) for run in runs],
            }
            if evaluation is not None and model is not None and sensitivity is not None:
                decision, _ = load_release(
                    evaluation, model, sensitivity_run=sensitivity
                )
                source = json.loads((evaluation / "evaluation.json").read_text())
                evaluated = EvaluationRun(
                    model_run_id=runs[0].id,
                    sensitivity_run_id=runs[1].id,
                    evaluator=source["evaluator"],
                    configuration={
                        "commit": source["commit"],
                        "draws": source["configuration"]["fit"]["draws"],
                        "tune": source["configuration"]["fit"]["tune"],
                        "model_record_hash": runs[0].record_hash,
                        "sensitivity_record_hash": runs[1].record_hash,
                    },
                    artifact_uri=str(evaluation.resolve().relative_to(root)),
                    status="SUCCEEDED",
                    summary=json.loads(json.dumps(asdict(decision))),
                    record_hash=file_hash(evaluation / "evaluation.json"),
                )
                # Reuse the same verification boundary as the HTTP health view.
                verified_evaluation(evaluated, runs[0], runs[1], root)
                session.add(evaluated)
                session.flush()
                result["evaluation_run_id"] = str(evaluated.id)
                result["release"] = decision.state.value
            return result
    finally:
        engine.dispose()
