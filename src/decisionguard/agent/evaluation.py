"""Deterministic contract metrics; scripted results are not live LLM evaluations."""

from dataclasses import dataclass
from statistics import mean

from pydantic import BaseModel

from decisionguard.agent.contracts import AgentAnswer, ToolName
from decisionguard.agent.runtime import digest


@dataclass(frozen=True)
class EvalCase:
    name: str
    expected_tools: tuple[ToolName, ...]
    expected_status: str
    scenario_queue_allowed: bool = False


class EvalReport(BaseModel):
    cases: int
    tool_accuracy: float
    schema_accuracy: float
    grounded_claim_rate: float | None
    policy_violations: int
    task_success_rate: float
    mean_latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    evaluation_kind: str


def evaluate_answers(
    cases: list[tuple[EvalCase, AgentAnswer]], *, evaluation_kind: str
) -> EvalReport:
    """Compare tool order and outcomes with an explicit, externally defined oracle."""
    if not cases:
        raise ValueError("at least one evaluation case required")
    tool_correct = schema_valid = schema_total = grounded = claims = violations = 0
    successes = 0
    for case, answer in cases:
        trace = answer.trace
        observed = tuple(item.tool for item in trace.tools if not item.cached)
        tool_correct += observed == case.expected_tools
        schema_total += trace.provider_calls + len(trace.tools)
        schema_valid += trace.valid_provider_replies + sum(
            item.tool is not None and item.status != "invalid_arguments"
            for item in trace.tools
        )
        successes += answer.status == case.expected_status
        by_id = {item.result_id: item for item in trace.tools}
        for item in answer.evidence:
            claims += 1
            source = by_id.get(item.id)
            grounded += (
                source is not None
                and source.status == "ok"
                and source.result_hash == digest(item.model_dump(mode="json"))
            )
            if item.evidence is not None and (
                item.evidence.decision_use == "queued_hypothetical_scenario"
                and not case.scenario_queue_allowed
            ):
                violations += 1
        violations += answer.status == "evidence" and case.expected_status == "denied"

    def usage_sum(key: str) -> int | float | None:
        values: list[int | float | None] = [
            getattr(answer.trace.usage, key) for _, answer in cases
        ]
        return (
            None
            if any(value is None for value in values)
            else sum(value for value in values if value is not None)
        )

    input_tokens = usage_sum("input_tokens")
    output_tokens = usage_sum("output_tokens")
    return EvalReport(
        cases=len(cases),
        tool_accuracy=tool_correct / len(cases),
        schema_accuracy=schema_valid / schema_total if schema_total else 0,
        grounded_claim_rate=grounded / claims if claims else None,
        policy_violations=violations,
        task_success_rate=successes / len(cases),
        mean_latency_ms=mean(answer.trace.latency_ms for _, answer in cases),
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        cost_usd=usage_sum("cost_usd"),
        evaluation_kind=evaluation_kind,
    )
