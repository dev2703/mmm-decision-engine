"""Optional upstream adapter contract tests; run in the evaluation environment."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

bridge = pytest.importorskip("decisionguard.evaluation.mmm_eval", exc_type=ImportError)


def adapter(data: pd.DataFrame, ledger: list[dict[str, object]]) -> Any:
    from decisionguard.models.bayesian import MMMConfig, build_mmm

    model = build_mmm(MMMConfig())
    config = bridge.PyMCConfig.from_model_object(
        model, revenue_column="revenue", response_column="response"
    )
    arrivals = pd.Series(
        (data["week"] + pd.Timedelta(days=7)).to_numpy(), index=data["week"]
    )
    return bridge.FoldSafePyMCAdapter(config, arrivals, ledger)


def observations() -> pd.DataFrame:
    from decisionguard.data.synthetic import SyntheticConfig, generate_dataset

    data = generate_dataset(SyntheticConfig(weeks=78)).observations
    data["response"] = data["revenue"]
    data["trend"] = np.arange(len(data), dtype=np.float64)
    return data


def test_adapter_fits_controls_only_on_its_training_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = observations()
    captured: list[pd.DataFrame] = []

    def fake_fit(self: Any, frame: pd.DataFrame) -> None:
        captured.append(frame.copy())
        self.trace = None

    monkeypatch.setattr(bridge.PyMCAdapter, "fit", fake_fit)

    def diagnostics(_: Any) -> dict[str, str]:
        return {"diagnostic_status": "ACCEPTABLE"}

    monkeypatch.setattr(bridge, "posterior_diagnostics", diagnostics)
    ledger: list[dict[str, object]] = []
    fit = adapter(data, ledger)
    fit.fit(data.iloc[:52])
    controls = ["price", "promotion", "macro_index", "competitor_index", "trend"]
    np.testing.assert_allclose(captured[0][controls].mean(), np.zeros(5), atol=1e-12)
    np.testing.assert_allclose(captured[0][controls].std(ddof=0), np.ones(5))
    expected = (data.iloc[52:][controls] - data.iloc[:52][controls].mean()) / data.iloc[
        :52
    ][controls].std(ddof=0)
    pd.testing.assert_frame_equal(fit._transform(data.iloc[52:])[controls], expected)
    assert ledger[0]["train_rows"] == 52
    clone = fit._create_adapter_with_placebo_channel("search_spend_shuffled")
    assert isinstance(clone, bridge.FoldSafePyMCAdapter)
    assert clone.ledger is ledger and clone.mean is None
    assert "search_spend_shuffled" in clone.config.channel_columns


def test_late_arrival_blocks_refit_before_upstream_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = observations()
    fit = adapter(data, [])
    fit.arrivals += pd.Timedelta(days=14)

    def must_not_fit(self: Any, frame: pd.DataFrame) -> None:
        pytest.fail("unavailable training reached the upstream model")

    monkeypatch.setattr(bridge.PyMCAdapter, "fit", must_not_fit)
    with pytest.raises(ValueError, match="unavailable"):
        fit.fit(data.iloc[:52])


@pytest.mark.parametrize("field", ["draws", "tune"])
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_sampling_overrides_fail_before_loading_data(
    monkeypatch: pytest.MonkeyPatch, field: str, value: int, tmp_path: Any
) -> None:
    from dataclasses import asdict

    from decisionguard.models.bayesian import MMMConfig

    record = {
        "dataset": {"hash": "same"},
        "training_window": {"rows": 143},
        "config": asdict(MMMConfig()),
    }

    def load_record(_: Any) -> dict[str, Any]:
        return record

    monkeypatch.setattr(bridge, "load_mmm_record", load_record)
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        bridge.evaluate_mmm(tmp_path, tmp_path, tmp_path / "output", **{field: value})
    assert not (tmp_path / "output").exists()


def test_interrupted_evaluation_resumes_without_repeating_completed_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from copy import deepcopy
    from dataclasses import asdict

    from decisionguard.models.bayesian import MMMConfig

    data = observations()
    model = tmp_path / "model"
    alternative = tmp_path / "alternative"
    for path in (model, alternative):
        path.mkdir()
        (path / "model.json").write_text("unit fixture")
    first: dict[str, Any] = {
        "dataset": {
            "hash": bridge.dataset_hash(data),
            "path": str(tmp_path),
            "quality_status": "RESOLVED",
        },
        "training_window": {"rows": len(data)},
        "config": asdict(MMMConfig()),
        "channels": {c: {"roi_mean": 1.0} for c in MMMConfig().channels},
        "diagnostics": {"diagnostic_status": "ACCEPTABLE"},
    }
    second = deepcopy(first)
    second["config"]["media_prior_mean"] = 0.25

    def load_record(path: Any) -> dict[str, Any]:
        return first if path == model else second

    def load_inputs(_: Any, expected: Any = None) -> tuple[pd.DataFrame, Any]:
        return data, data["week"] + pd.Timedelta(days=7)

    monkeypatch.setattr(bridge, "load_mmm_record", load_record)
    monkeypatch.setattr(bridge, "load_model_inputs", load_inputs)
    calls: list[str] = []
    interrupted = False

    class Result:
        def __init__(self, name: str) -> None:
            self.name = name

        def to_df(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "test_name": self.name,
                        "general_metric_name": "mape",
                        "specific_metric_name": "mape",
                        "metric_value": 1.0,
                        "metric_pass": True,
                    }
                ]
            )

    class Orchestrator:
        def validate(self, fit: Any, data: Any, tests: Any) -> Result:
            nonlocal interrupted
            name = str(tests[0].value)
            calls.append(name)
            if len(calls) == 2 and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("simulated worker interruption")
            fit.ledger.append({"diagnostics": {"diagnostic_status": "ACCEPTABLE"}})
            return Result(name)

    monkeypatch.setattr(bridge, "ValidationTestOrchestrator", Orchestrator)
    output = tmp_path / "evaluation"
    with pytest.raises(KeyboardInterrupt):
        bridge.evaluate_mmm(model, alternative, output)
    assert not (output / "evaluation.json").exists()
    record = bridge.evaluate_mmm(model, alternative, output, resume=True)
    assert calls.count(calls[0]) == 1
    assert len(record["refits"]) == 6
    assert record["test_execution_errors"] == []
    assert (
        record["policy"]["state"] == "BLOCK"
    )  # fixture metrics are intentionally incomplete


def test_external_config_text_cannot_reach_upstream_expression_rehydration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from dataclasses import asdict

    from decisionguard.models.config import MMMConfig

    marker = tmp_path / "must-not-be-written"
    payload = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"
    configuration = asdict(MMMConfig())
    configuration["media_prior_mean"] = payload
    record: dict[str, Any] = {
        "dataset": {"hash": "same"},
        "training_window": {"rows": 143},
        "config": configuration,
    }

    def load_record(_: Any) -> dict[str, Any]:
        return record

    def forbidden_rehydration(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("untrusted config reached the upstream expression loader")

    monkeypatch.setattr(bridge, "load_mmm_record", load_record)
    monkeypatch.setattr(
        bridge.PyMCConfig, "load_model_config_from_json", forbidden_rehydration
    )
    with pytest.raises((TypeError, ValueError)):
        bridge.evaluate_mmm(tmp_path, tmp_path, tmp_path / "output")
    assert not marker.exists()
    assert not (tmp_path / "output").exists()
