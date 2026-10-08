"""Registered CLI calls preserve the queued configuration and identity."""

import sys
from typing import Any
from uuid import UUID, uuid4

import pytest

from decisionguard.cli import main
from decisionguard.config import Settings


def test_registered_evaluation_cli_uses_stored_job(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from decisionguard.api import jobs

    identifier = uuid4()
    calls: list[UUID] = []

    def evaluate(model_run_id: UUID, settings: Settings) -> dict[str, Any]:
        assert settings.database_url.startswith("postgresql+psycopg://")
        calls.append(model_run_id)
        return {"policy": {"state": "BLOCK"}}

    monkeypatch.setattr(jobs, "evaluate_model", evaluate)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost/unit_test")
    monkeypatch.setattr(
        sys, "argv", ["decisionguard", "evaluate", "--model-run-id", str(identifier)]
    )
    main()
    assert calls == [identifier]
    assert '"BLOCK"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "override",
    [
        ["--draws", "10"],
        ["--tune", "20"],
        ["--output", "new"],
        ["--sensitivity-run", "other"],
        ["--resume"],
    ],
)
def test_registered_cli_rejects_configuration_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    override: list[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["decisionguard", "evaluate", "--model-run-id", str(uuid4()), *override],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "stored configuration" in capsys.readouterr().err


def test_path_evaluation_requires_sensitivity_and_output_before_import(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["decisionguard", "evaluate", "--model-run", "model"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "requires --sensitivity-run and --output" in capsys.readouterr().err
