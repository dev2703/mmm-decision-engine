"""Reproducible local data and predictive experiment workflows."""

import argparse
import json
from dataclasses import fields
from pathlib import Path
from uuid import UUID


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
    for command in (baseline, compare):
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--period", type=int, default=52)
        command.add_argument(
            "--initial-train", type=int, default=104 if command is compare else 52
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
    evaluation.add_argument("--model-run", type=Path, required=True)
    evaluation.add_argument("--sensitivity-run", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--draws", type=int)
    evaluation.add_argument("--tune", type=int)
    evaluation.add_argument("--resume", action="store_true")
    optimization = commands.add_parser(
        "optimize", help="Analyze release-gated budget allocations"
    )
    optimization.add_argument("--model-run", type=Path, required=True)
    optimization.add_argument("--evaluation", type=Path, required=True)
    optimization.add_argument("--constraints", type=Path, required=True)
    optimization.add_argument("--output", type=Path, required=True)
    optimization.add_argument("--horizon", type=int, default=13)
    optimization.add_argument("--draws", type=int, default=200)
    optimization.add_argument("--seed", type=int, default=42)
    training = commands.add_parser(
        "train", help="Fit and audit a Bayesian MMM candidate"
    )
    training_source = training.add_mutually_exclusive_group(required=True)
    training_source.add_argument("--dataset", type=Path)
    training_source.add_argument("--experiment-id", type=UUID)
    training.add_argument("--output", type=Path)
    training.add_argument("--draws", type=int)
    training.add_argument("--tune", type=int)
    training.add_argument("--chains", type=int)
    training.add_argument("--seed", type=int)
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
            from decisionguard.optimization.allocation import BudgetConstraints
            from decisionguard.optimization.artifacts import optimize_run

            quantities = json.loads(args.constraints.read_text())
            record = optimize_run(
                args.model_run,
                args.evaluation,
                args.output,
                BudgetConstraints(**quantities),
                horizon=args.horizon,
                draws=args.draws,
                seed=args.seed,
            )
            print(json.dumps(record["alternatives"]))
            return
        if args.command == "evaluate":
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

            overrides = {
                field.name: getattr(args, field.name)
                for field in fields(MMMConfig)
                if hasattr(args, field.name) and getattr(args, field.name) is not None
            }
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
        if args.command == "compare":
            from decisionguard.experiments.baseline import BaselineConfig
            from decisionguard.experiments.comparison import run_comparison

            print(
                json.dumps(
                    run_comparison(
                        args.dataset,
                        args.output,
                        BaselineConfig(
                            period=args.period,
                            initial_train=args.initial_train,
                            horizon=args.horizon,
                            gap=args.gap,
                        ),
                    )
                )
            )
            return
        if args.command == "baseline":
            from decisionguard.experiments.baseline import BaselineConfig, run_baseline

            result = run_baseline(
                args.dataset,
                args.output,
                BaselineConfig(
                    period=args.period,
                    initial_train=args.initial_train,
                    horizon=args.horizon,
                    gap=args.gap,
                    model=args.model,
                ),
                hypothesis=args.hypothesis,
            )
            print(
                json.dumps({"cv": result.cv_metrics, "holdout": result.holdout_metrics})
            )
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
        if args.command == "prepare-data":
            generating = SyntheticConfig(
                weeks=args.weeks,
                seed=args.seed,
                shift_week=args.shift_week,
                shift_amount=args.shift_amount,
                event_week=args.event_week,
                event_amount=args.event_amount,
            )
            clean = generate_dataset(generating)
            corrupting = CorruptionConfig(
                missing_weeks=args.missing_weeks,
                duplicate_rows=args.duplicate_rows,
                seed=args.seed,
                meta_alias=args.meta_alias,
                unit_error_weeks=args.unit_error_weeks,
                tracking_outage_weeks=args.tracking_outage_weeks,
                erroneous_outlier_weeks=args.erroneous_outlier_weeks,
                late_arrival_weeks=args.late_arrival_weeks,
                arrival_delay_days=args.arrival_delay_days,
                mixed_currency_weeks=args.mixed_currency_weeks,
                negative_spend_weeks=args.negative_spend_weeks,
            )
            dirty = corrupt_dataset(clean, corrupting)
            config = IntegrityConfig(
                expected_start=str(clean.observations["week"].min())[:10],
                expected_end=str(clean.observations["week"].max())[:10],
                as_of=args.as_of,
                duplicate_policy=args.duplicate_policy,
                currency_rates=tuple(rates),
            )
            result = write_simulation(clean, dirty, args.output, config)
        else:
            config = IntegrityConfig(
                expected_start=args.expected_start,
                expected_end=args.expected_end,
                as_of=args.as_of,
                duplicate_policy=args.duplicate_policy,
                currency_rates=tuple(rates),
            )
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
