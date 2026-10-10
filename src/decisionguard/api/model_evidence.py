"""Typed, immutable chart evidence; reads never fit models or grant release."""

from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, FiniteFloat, TypeAdapter

from decisionguard.api.database import Dataset, ModelRun
from decisionguard.api.evaluations import load_registered_model
from decisionguard.config import artifact_path

PredictionSplit = Literal["training", "holdout"]


class TrainingWindow(BaseModel):
    start: date
    end: date
    rows: int = Field(gt=0)


class EvidenceProvenance(BaseModel):
    model_run_id: UUID
    dataset_id: UUID
    dataset_hash: str
    model_record_hash: str
    model_status: str
    training_window: TrainingWindow
    prediction_context: Literal["conditional_on_realized_covariates"]
    limitations: list[str]
    health_url: str
    decision_use: Literal["requires_current_health_and_scenario_validation"] = (
        "requires_current_health_and_scenario_validation"
    )


class PredictionPoint(BaseModel):
    week: date
    split: PredictionSplit
    actual: FiniteFloat
    predicted: FiniteFloat
    lower_90: FiniteFloat
    upper_90: FiniteFloat
    residual: FiniteFloat


class PredictionsView(BaseModel):
    provenance: EvidenceProvenance
    units: Literal["AUD/week"] = "AUD/week"
    interval_probability: float = 0.9
    total: int
    offset: int
    limit: int
    rows: list[PredictionPoint]


class ChannelSummary(BaseModel):
    roi_mean: FiniteFloat
    roi_interval_90: tuple[FiniteFloat, FiniteFloat]
    contribution_mean: FiniteFloat
    contribution_interval_90: tuple[FiniteFloat, FiniteFloat]


class ResponsePoint(BaseModel):
    adstocked_spend: FiniteFloat
    median: FiniteFloat
    lower_90: FiniteFloat
    upper_90: FiniteFloat


class ChannelEvidenceView(BaseModel):
    provenance: EvidenceProvenance
    channel: str
    summary: ChannelSummary
    roi_definition: Literal["training_contribution / training_spend"] = (
        "training_contribution / training_spend"
    )
    roi_units: Literal["AUD/AUD"] = "AUD/AUD"
    contribution_units: Literal["AUD over training window"] = "AUD over training window"
    response_curve_axis: Literal["adstocked_spend_AUD; conditional saturation response"]
    response_units: Literal["AUD/week"] = "AUD/week"
    interval_probability: float = 0.9
    response_curve: list[ResponsePoint]


def _load_evidence(
    run: ModelRun, dataset: Dataset, root: Path
) -> tuple[EvidenceProvenance, dict[str, Any]]:
    record = load_registered_model(run, root)
    if dataset.id != run.dataset_id or dataset.hash != record["dataset"]["hash"]:
        raise ValueError("model dataset differs from registration")
    provenance = EvidenceProvenance.model_validate(
        {
            "model_run_id": run.id,
            "dataset_id": run.dataset_id,
            "dataset_hash": dataset.hash,
            "model_record_hash": run.record_hash,
            "model_status": record["model_status"],
            "training_window": record["training_window"],
            "prediction_context": record["prediction_context"],
            "limitations": record["limitations"],
            "health_url": f"/model-runs/{run.id}/health",
        }
    )
    return provenance, record


def prediction_evidence(
    run: ModelRun,
    dataset: Dataset,
    root: Path,
    *,
    split: PredictionSplit | None,
    offset: int,
    limit: int,
) -> PredictionsView:
    provenance, record = _load_evidence(run, dataset, root)
    frame = pd.read_parquet(
        artifact_path(root, run.artifact_uri) / "predictions.parquet"
    )
    window = provenance.training_window
    expected_splits = ["training"] * window.rows + ["holdout"] * record["config"][
        "holdout"
    ]
    frame["residual"] = frame["actual"] - frame["predicted"]
    # Validate the full artifact before filtering so a page cannot hide bad evidence.
    points = TypeAdapter(list[PredictionPoint]).validate_python(
        frame.to_dict("records")
    )
    if (
        [point.split for point in points] != expected_splits
        or points[0].week != window.start
        or points[window.rows - 1].week != window.end
        or any(
            second.week - first.week != timedelta(weeks=1)
            for first, second in pairwise(points)
        )
        or any(point.lower_90 > point.upper_90 for point in points)
    ):
        raise ValueError(
            "prediction dates, split or intervals contradict model evidence"
        )
    if split is not None:
        points = [point for point in points if point.split == split]
    return PredictionsView(
        provenance=provenance,
        total=len(points),
        offset=offset,
        limit=limit,
        rows=points[offset : offset + limit],
    )


def channel_evidence(
    run: ModelRun, dataset: Dataset, root: Path, channel: str
) -> ChannelEvidenceView:
    provenance, record = _load_evidence(run, dataset, root)
    channels = record["config"]["channels"]
    index = channels.index(channel)
    summary = ChannelSummary.model_validate(record["channels"][channel])
    if any(
        lower > upper
        for lower, upper in (summary.roi_interval_90, summary.contribution_interval_90)
    ):
        raise ValueError("channel intervals are reversed")
    with np.load(
        artifact_path(root, run.artifact_uri) / "channel_draws.npz", allow_pickle=False
    ) as archive:
        grid = archive["adstocked_spend_grid"]
        draws = archive["saturation_response"]
        if (
            archive["channels"].tolist() != channels
            or grid.ndim != 2
            or grid.shape[1] != len(channels)
            or not 1 <= grid.shape[0] <= 1000
            or draws.shape
            != (
                record["config"]["chains"],
                record["config"]["draws"],
                grid.shape[0],
                len(channels),
            )
            or not np.isfinite(grid).all()
            or not np.isfinite(draws).all()
            or (grid < 0).any()
            or (np.diff(grid, axis=0) <= 0).any()
        ):
            raise ValueError("response curve dimensions or values are invalid")
        lower, median, upper = np.quantile(
            draws[..., index].reshape(-1, grid.shape[0]), [0.05, 0.5, 0.95], axis=0
        )
        curve = [
            ResponsePoint(adstocked_spend=x, lower_90=lo, median=mid, upper_90=hi)
            for x, lo, mid, hi in zip(grid[:, index], lower, median, upper, strict=True)
        ]
    return ChannelEvidenceView(
        provenance=provenance,
        channel=channel,
        summary=summary,
        response_curve_axis=record["response_curve_axis"],
        response_curve=curve,
    )
