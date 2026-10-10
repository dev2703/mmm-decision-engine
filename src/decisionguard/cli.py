"""Reproducible local data and predictive experiment workflows."""

import argparse
import json
from dataclasses import Field, fields, replace
from pathlib import Path
from typing import Any
from uuid import UUID


def _config_arguments(
    args: argparse.Namespace, config_fields: tuple[Field[Any], ...]
) -> dict[str, Any]:
    values = vars(args)
    return {
        field.name: values[field.name]
        for field in config_fields
        if values.get(field.name) is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="decisionguard")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-data", help="Generate, corrupt, validate, and persist a dataset"
    )
    prepare.add_argument("--weeks", type=int, default=156)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--shift-week", type=int)
    prepare.add_argument("--shift-amount", type=float, default=0)
    prepare.add_argument("--event-week", type=int)
    prepare.add_argument("--event-amount", type=float, default=0)
    prepare.add_argument("--meta-alias", choices=["fb_spend", "facebook_spend"])
    for name in (
        "missing-weeks",
        "duplicate-rows",
        "unit-error-weeks",
        "tracking-outage-weeks",
        "erroneous-outlier-weeks",
        "late-arrival-weeks",
        "mixed-currency-weeks",
        "negative-spend-weeks",
    ):
        prepare.add_argument(f"--{name}", type=int, default=0)
    prepare.add_argument("--arrival-delay-days", type=int, default=14)
    imported = commands.add_parser(
        "clean-data", help="Validate an existing raw Parquet dataset"
    )
    imported.add_argument("input", type=Path)
    imported.add_argument("--expected-start")
    imported.add_argument("--expected-end")
    baseline = commands.add_parser("baseline", help="Evaluate one predictive benchmark")
    compare = commands.add_parser(
        "compare", help="Run the predictive model ladder on identical windows"
    )
    backtest = commands.add_parser(
        "backtest", help="Freeze model selection, then score separate test and holdout"
    )
    backtest.add_argument("--test-weeks", type=int, default=13)
    backtest.add_argument("--holdout-weeks", type=int, default=13)
    backtest.add_argument("--include-mmm", action="store_true")
    for command in (baseline, compare, backtest):
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--period", type=int, default=52)
        command.add_argument(
            "--initial-train", type=int, default=52 if command is baseline else 104
        )
        command.add_argument("--horizon", type=int, default=13)
        command.add_argument("--gap", type=int, default=0)
    baseline.add_argument(
        "--model",
        choices=[
            "seasonal_naive",
            "ets",
            "ridge_raw",
            "ridge_domain",
            "hist_gradient_boosting",
        ],
        default="seasonal_naive",
    )
    baseline.add_argument("--hypothesis")
    evaluation = commands.add_parser(
        "evaluate", help="Run upstream MMM checks and release policy"
    )
    evaluation_source = evaluation.add_mutually_exclusive_group(required=True)
    evaluation_source.add_argument("--model-run", type=Path)
    evaluation_source.add_argument("--model-run-id", type=UUID)
    evaluation.add_argument("--sensitivity-run", type=Path)
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--draws", type=int)
    evaluation.add_argument("--tune", type=int)
    evaluation.add_argument("--resume", action="store_true")
    optimization = commands.add_parser(
        "optimize", help="Analyze release-gated budget allocations"
    )
    optimization_source = optimization.add_mutually_exclusive_group(required=True)
    optimization_source.add_argument("--model-run", type=Path)
    optimization_source.add_argument("--scenario-id", type=UUID)
    optimization.add_argument("--evaluation", type=Path)
    optimization.add_argument("--constraints", type=Path)
    optimization.add_argument("--output", type=Path)
    optimization.add_argument("--horizon", type=int)
    optimization.add_argument("--draws", type=int)
    optimization.add_argument("--seed", type=int)
    training = commands.add_parser(
        "train", help="Fit and audit a Bayesian MMM candidate"
    )
    training_source = training.add_mutually_exclusive_group(required=True)
    training_source.add_argument("--dataset", type=Path)
    training_source.add_argument("--experiment-id", type=UUID)
    training.add_argument("--output", type=Path)
    for command in (training, backtest):
        command.add_argument("--draws", type=int)
        command.add_argument("--tune", type=int)
        command.add_argument("--chains", type=int)
        command.add_argument("--seed", type=int)
    training.add_argument("--save-warmup", action="store_true", default=None)
    training.add_argument("--holdout", type=int)
    training.add_argument("--target-accept", type=float)
    training.add_argument("--adstock-lags", type=int)
    training.add_argument("--max-tree-depth", type=int)
    training.add_argument("--media-prior-mean", type=float)
    training.add_argument("--media-prior-sigma", type=float)
    for command in (prepare, imported):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--as-of")
        command.add_argument(
            "--duplicate-policy", choices=["drop_exact", "block"], default="drop_exact"
        )
        command.add_argument(
            "--currency-rate", action="append", default=[], metavar="CODE=AUD_FACTOR"
        )
    args = parser.parse_args()
    try:
        if args.command == "optimize":
            if args.scenario_id is not None:
                from decisionguard.api.jobs import optimize_scenario
                from decisionguard.config import Settings

                if any(
                    value is not None
                    for value in (
                        args.evaluation,
                        args.constraints,
                        args.output,
                        args.horizon,
                        args.draws,
                        args.seed,
                    )
                ):
                    raise ValueError(
                        "registered optimization uses its stored configuration"
                    )
                record = optimize_scenario(
                    args.scenario_id, Settings.from_environment()
                )
            else:
                if any(
                    value is None
                    for value in (args.evaluation, args.constraints, args.output)
                ):
                    raise ValueError(
                        "Path optimization requires --evaluation, --constraints "
                        "and --output"
                    )
                from decisionguard.optimization.allocation import BudgetConstraints
                from decisionguard.optimization.artifacts import optimize_run

                quantities = json.loads(args.constraints.read_text())
                record = optimize_run(
                    args.model_run,
                    args.evaluation,
                    args.output,
                    BudgetConstraints(**quantities),
                    horizon=13 if args.horizon is None else args.horizon,
                    draws=200 if args.draws is None else args.draws,
                    seed=42 if args.seed is None else args.seed,
                )
            print(json.dumps(record["alternatives"]))
            return
        if args.command == "evaluate":
            if args.model_run_id is not None:
                from decisionguard.api.jobs import evaluate_model
                from decisionguard.config import Settings

                if (
                    any(
                        value is not None
                        for value in (
                            args.sensitivity_run,
                            args.output,
                            args.draws,
                            args.tune,
                        )
                    )
                    or args.resume
                ):
                    raise ValueError(
                        "registered evaluation uses its stored configuration"
                    )
                record = evaluate_model(args.model_run_id, Settings.from_environment())
            else:
                if args.sensitivity_run is None or args.output is None:
                    raise ValueError(
                        "Path evaluation requires --sensitivity-run and --output"
                    )
                from decisionguard.evaluation.mmm_eval import evaluate_mmm

                record = evaluate_mmm(
                    args.model_run,
                    args.sensitivity_run,
                    args.output,
                    draws=args.draws,
                    tune=args.tune,
                    resume=args.resume,
                )
            print(json.dumps(record["policy"]))
            return
        if args.command == "train":
            from decisionguard.models.config import MMMConfig

            overrides = _config_arguments(args, fields(MMMConfig))
            if args.experiment_id is not None:
                from decisionguard.api.jobs import train_experiment
                from decisionguard.config import Settings

                if args.output is not None or overrides:
                    raise ValueError(
                        "registered training uses its stored configuration"
                    )
                record = train_experiment(
                    args.experiment_id, Settings.from_environment()
                )
            else:
                from decisionguard.models.bayesian import train_mmm

                if args.output is None:
                    raise ValueError("--output is required for dataset training")
                record = train_mmm(args.dataset, args.output, MMMConfig(**overrides))
            print(
                json.dumps(
                    {
                        "model_status": record["model_status"],
                        "diagnostics": record["diagnostics"],
                    }
                )
            )
            return
        if args.command in ("compare", "baseline", "backtest"):
            from decisionguard.experiments.baseline import BaselineConfig

            config = BaselineConfig(
                period=args.period,
                initial_train=args.initial_train,
                horizon=args.horizon,
                gap=args.gap,
                model=getattr(args, "model", "seasonal_naive"),
            )
            if args.command == "backtest":
                from decisionguard.experiments.backtest import run_backtest
                from decisionguard.models.config import MMMConfig

                overrides = _config_arguments(args, fields(MMMConfig))
                if overrides and not args.include_mmm:
                    raise ValueError("sampler options require --include-mmm")
                summary = run_backtest(
                    args.dataset,
                    args.output,
                    config,
                    test_weeks=args.test_weeks,
                    holdout_weeks=args.holdout_weeks,
                    mmm_config=MMMConfig(**overrides) if args.include_mmm else None,
                )
            elif args.command == "compare":
                from decisionguard.experiments.comparison import run_comparison

                summary = run_comparison(args.dataset, args.output, config)
            else:
                from decisionguard.experiments.baseline import run_baseline

                result = run_baseline(
                    args.dataset, args.output, config, hypothesis=args.hypothesis
                )
                summary = {"cv": result.cv_metrics, "holdout": result.holdout_metrics}
            print(json.dumps(summary))
            return
        import pandas as pd

        from decisionguard.data.artifacts import write_dataset, write_simulation
        from decisionguard.data.corruption import CorruptionConfig, corrupt_dataset
        from decisionguard.data.integrity import IntegrityConfig
        from decisionguard.data.synthetic import SyntheticConfig, generate_dataset

        rates: list[tuple[str, float]] = []
        for text in args.currency_rate:
            currency, factor = text.split("=", 1)
            rates.append((currency, float(factor)))
        config = IntegrityConfig(
            expected_start=getattr(args, "expected_start", None),
            expected_end=getattr(args, "expected_end", None),
            as_of=args.as_of,
            duplicate_policy=args.duplicate_policy,
            currency_rates=tuple(rates),
        )
        if args.command == "prepare-data":
            generating = SyntheticConfig(
                **_config_arguments(args, fields(SyntheticConfig))
            )
            clean = generate_dataset(generating)
            corrupting = CorruptionConfig(
                **_config_arguments(args, fields(CorruptionConfig))
            )
            dirty = corrupt_dataset(clean, corrupting)
            result = write_simulation(
                clean,
                dirty,
                args.output,
                replace(
                    config,
                    expected_start=str(clean.observations["week"].min())[:10],
                    expected_end=str(clean.observations["week"].max())[:10],
                ),
            )
        else:
            result = write_dataset(pd.read_parquet(args.input), args.output, config)
        print(
            f"{result.report.status}: {len(result.clean)} candidate rows; "
            f"artifacts in {args.output}"
        )
        if result.report.status == "BLOCKER":
            parser.exit(
                2, "Modeling blocked; inspect quality.json for required intervention.\n"
            )
    except ImportError as error:
        parser.exit(
            2,
            f"Optional evaluation dependency unavailable: {error}. "
            "Use the documented evaluation environment.\n",
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
