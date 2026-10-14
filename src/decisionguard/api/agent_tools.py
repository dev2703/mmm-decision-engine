"""Seven project-scoped tools backed by the same checks as the public API."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from decisionguard.agent.contracts import (
    AgentQuery,
    Evidence,
    StrictModel,
    ToolName,
    ToolResult,
)
from decisionguard.api.comparison import model_comparison
from decisionguard.api.database import (
    Dataset,
    Experiment,
    ModelRun,
    OptimizationRun,
    Project,
    Scenario,
)
from decisionguard.api.scenarios import ScenarioCreate


class NoArguments(StrictModel):
    pass


class CompareArguments(StrictModel):
    offset: int = Field(default=0, ge=0, le=10000)
    limit: int = Field(default=5, ge=1, le=5)


class ModelArguments(StrictModel):
    model_run_id: UUID


class ChannelArguments(ModelArguments):
    channel: str = Field(min_length=1, max_length=100)


class ConstraintArguments(ModelArguments):
    scenario: ScenarioCreate


class ScenarioArguments(StrictModel):
    scenario_id: UUID


class StabilityArguments(StrictModel):
    optimization_run_id: UUID


TOOL_INPUTS: dict[ToolName, tuple[type[BaseModel], str]] = {
    "get_data_quality": (
        NoArguments,
        "Inspect verified quality for the current dataset.",
    ),
    "compare_models": (
        CompareArguments,
        "Inspect registered candidates, verified metrics and training windows. "
        "Different data/windows are not a fair ranking. No automatic winner.",
    ),
    "get_model_health": (
        ModelArguments,
        "Retrieve current release state, diagnostics and restriction reasons.",
    ),
    "get_channel_evidence": (
        ChannelArguments,
        "Retrieve channel ROI, contribution intervals and response curves.",
    ),
    "validate_constraints": (
        ConstraintArguments,
        "Check hypothetical weekly constraints against current release. "
        "Read-only; missing budget assumptions must be supplied by the user.",
    ),
    "run_scenario": (
        ScenarioArguments,
        "Queue the caller-authorized existing scenario. Does not run the "
        "solver or change spending. The CLI must execute the queued optimization.",
    ),
    "get_decision_stability": (
        StabilityArguments,
        "Retrieve verified optimization uncertainty, stability and "
        "current recommendation authority. Pending jobs have no result.",
    ),
}


@dataclass
class AgentTools:
    query: AgentQuery
    sessions: sessionmaker[Session]
    root: Path
    quality: Callable[[UUID], dict[str, Any]]
    health: Callable[[UUID], dict[str, Any]]
    channel: Callable[[UUID, str], BaseModel]
    validate: Callable[[UUID, ScenarioCreate], dict[str, Any]]
    queue: Callable[[UUID], BaseModel]
    optimization: Callable[[UUID], BaseModel]

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": description,
                "input_schema": schema.model_json_schema(),
                "output_schema": ToolResult.model_json_schema(),
            }
            for name, (schema, description) in TOOL_INPUTS.items()
        ]

    def _model(self, identifier: UUID) -> ModelRun:
        with self.sessions() as session:
            model = session.scalar(
                select(ModelRun)
                .join(Experiment)
                .where(
                    ModelRun.id == identifier,
                    Experiment.project_id == self.query.project_id,
                )
            )
            if model is None:
                raise HTTPException(404, "Model not found in project")
            return model

    def execute(
        self, name: str, arguments: dict[str, Any], identifier: str
    ) -> ToolResult:
        if name not in TOOL_INPUTS:
            return ToolResult(id=identifier, tool=None, status="policy_denied")
        tool = name
        try:
            parsed = TOOL_INPUTS[tool][0].model_validate_json(
                json.dumps(arguments, allow_nan=False)
            )
        except (ValueError, ValidationError):
            return ToolResult(id=identifier, tool=tool, status="invalid_arguments")
        try:
            evidence = self._execute(tool, parsed)
            # Validate nested results too: no nonfinite values reach model context.
            json.dumps(evidence.model_dump(mode="json"), allow_nan=False)
            return ToolResult(id=identifier, tool=tool, status="ok", evidence=evidence)
        except HTTPException as error:
            status = (
                "unavailable"
                if error.status_code == 404
                else "invalid_arguments"
                if error.status_code == 422
                else "policy_denied"
                if error.status_code in (403, 409)
                else "tool_error"
            )
            return ToolResult(id=identifier, tool=tool, status=status)
        except Exception:
            # Do not disclose raw paths, database errors or artifact text.
            return ToolResult(id=identifier, tool=tool, status="tool_error")

    def _execute(self, name: ToolName, args: BaseModel) -> Evidence:
        if isinstance(args, ModelArguments):
            self._model(args.model_run_id)
        if name == "get_data_quality":
            with self.sessions() as session:
                project = session.get(Project, self.query.project_id)
                if project is None or project.current_dataset_id is None:
                    raise HTTPException(404, "No dataset")
                dataset = session.get(Dataset, project.current_dataset_id)
                if dataset is None or dataset.project_id != project.id:
                    raise HTTPException(409, "Dataset project differs")
                dataset_id = dataset.id
            return Evidence(
                source=f"/datasets/{dataset_id}/quality",
                data={
                    "dataset_id": str(dataset_id),
                    "quality": self.quality(dataset_id),
                },
            )
        if name == "compare_models":
            assert isinstance(args, CompareArguments)
            return self._compare(args)
        if name == "get_model_health":
            assert isinstance(args, ModelArguments)
            # API health contains a Pydantic evaluation view; normalize nested values.
            from fastapi.encoders import jsonable_encoder

            return Evidence(
                source=f"/model-runs/{args.model_run_id}/health",
                data=jsonable_encoder(self.health(args.model_run_id)),
            )
        if name == "get_channel_evidence":
            assert isinstance(args, ChannelArguments)
            return Evidence(
                source=f"/model-runs/{args.model_run_id}/channels/{args.channel}",
                data=self.channel(args.model_run_id, args.channel).model_dump(
                    mode="json"
                ),
            )
        if name == "validate_constraints":
            assert isinstance(args, ConstraintArguments)
            return Evidence(
                source=f"/model-runs/{args.model_run_id}/constraints/validate",
                data=self.validate(args.model_run_id, args.scenario),
            )
        if name == "run_scenario":
            assert isinstance(args, ScenarioArguments)
            if args.scenario_id != self.query.authorized_scenario_id:
                raise HTTPException(403, "Caller authorization required")
            with self.sessions() as session:
                scenario = session.get(Scenario, args.scenario_id)
                if scenario is None:
                    raise HTTPException(404, "Scenario not found")
                model_id = scenario.model_run_id
            self._model(model_id)
            # Recheck health here even if the planner skipped the suggested order.
            if self.health(model_id)["decision_status"] == "BLOCK":
                raise HTTPException(409, "BLOCK model")
            queued = self.queue(args.scenario_id).model_dump(mode="json")
            return Evidence(
                source=f"/optimization-runs/{queued['id']}",
                data=queued,
                decision_use="queued_hypothetical_scenario",
            )
        assert isinstance(args, StabilityArguments)
        with self.sessions() as session:
            run = session.get(OptimizationRun, args.optimization_run_id)
            scenario = session.get(Scenario, run.scenario_id) if run else None
            if scenario is None:
                raise HTTPException(404, "Optimization not found")
            self._model(scenario.model_run_id)
        result = self.optimization(args.optimization_run_id).model_dump(mode="json")
        current = result["evidence_current"]
        return Evidence(
            source=f"/optimization-runs/{args.optimization_run_id}",
            data=result,
            decision_use="current" if current else "historical_only",
        )

    def _compare(self, args: CompareArguments) -> Evidence:
        return Evidence(
            source=f"/projects/{self.query.project_id}/model-comparison",
            data=model_comparison(
                self.sessions,
                self.root,
                self.query.project_id,
                offset=args.offset,
                limit=args.limit,
            ),
        )
