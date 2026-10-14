"""Only tool selections and evidence references can cross the LLM boundary."""

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentQuery(StrictModel):
    project_id: UUID = Field(strict=False)
    message: str = Field(min_length=1, max_length=8000)
    # Authorization belongs to the caller, never to a model-generated argument.
    authorized_scenario_id: UUID | None = Field(default=None, strict=False)


class ToolCall(StrictModel):
    kind: Literal["tool"]
    name: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any]


class AnswerSelection(StrictModel):
    kind: Literal["answer"]
    evidence_ids: list[str] = Field(max_length=8)
    unavailable: bool = False


Turn = Annotated[ToolCall | AnswerSelection, Field(discriminator="kind")]


class Usage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ProviderReply(StrictModel):
    turn: Turn
    usage: Usage = Field(default_factory=Usage)


ProviderErrorCode = Literal[
    "authentication", "quota", "rate_limit", "timeout", "unavailable"
]


class ProviderFailure(Exception):
    """Only a safe category crosses the transport boundary, never its raw error."""

    def __init__(self, code: ProviderErrorCode) -> None:
        self.code: ProviderErrorCode = code
        super().__init__(code)


ToolName = Literal[
    "get_data_quality",
    "compare_models",
    "get_model_health",
    "get_channel_evidence",
    "run_scenario",
    "validate_constraints",
    "get_decision_stability",
]


class Evidence(StrictModel):
    source: str
    data: dict[str, Any]
    decision_use: Literal[
        "inspection_only", "queued_hypothetical_scenario", "current", "historical_only"
    ] = "inspection_only"


class ToolResult(StrictModel):
    id: str
    tool: ToolName | None
    status: Literal[
        "ok", "invalid_arguments", "unavailable", "policy_denied", "tool_error"
    ]
    evidence: Evidence | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProviderContext(StrictModel):
    instructions: str
    query: AgentQuery
    tools: list[dict[str, Any]]
    results: list[ToolResult]
    calls: list[ToolCall] = Field(default_factory=list[ToolCall])


class Provider(Protocol):
    """A provider adapter owns transport, timeout and actual usage reporting.

    Keep instructions in the privileged system role; query in the user role;
    results in untrusted tool-result blocks. Never concatenate business text into
    system instructions. Return validated selections, not free-form final prose.
    """

    model: str

    def respond(self, context: ProviderContext) -> ProviderReply: ...


class ToolTrace(StrictModel):
    result_id: str
    tool: ToolName | None
    arguments_hash: str
    result_hash: str
    status: str
    latency_ms: float
    cached: bool = False
    resource_ids: dict[str, UUID] = Field(default_factory=dict[str, UUID])


class AgentTrace(StrictModel):
    request_id: UUID
    project_id: UUID
    query_hash: str
    code_hash: str
    prompt_version: str
    model: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tools: list[ToolTrace] = Field(default_factory=list[ToolTrace])
    provider_calls: int = 0
    valid_provider_replies: int = 0
    latency_ms: float = 0
    usage: Usage = Field(default_factory=Usage)
    outcome: str = "unavailable"
    provider_error: ProviderErrorCode | None = None


class AgentAnswer(StrictModel):
    status: Literal["evidence", "unavailable", "denied", "error", "limit_reached"]
    message: str
    evidence: list[ToolResult]
    issues: list[ToolResult]
    trace: AgentTrace
