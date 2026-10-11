"""Shared scenario request contract for HTTP and agent constraint validation."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Money = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class ScenarioCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
    current: dict[str, Money]
    floors: dict[str, Money] = Field(default_factory=dict)
    caps: dict[str, Money] = Field(default_factory=dict)
    max_movement: dict[str, Money] = Field(default_factory=dict)
    protected_spend: dict[str, Money] = Field(default_factory=dict)
    horizon_weeks: int = Field(default=13, strict=True, ge=1, le=52)
    risk_policy: Literal["expected", "conservative"] = "conservative"
