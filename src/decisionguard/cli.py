"""Reproducible local data workflows; no training or model APIs yet."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pandas as pd

from decisionguard.data.artifacts import write_dataset, write_json
from decisionguard.data.corruption import CorruptionConfig, corrupt_dataset
from decisionguard.data.integrity import IntegrityConfig
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset


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
            result = write_dataset(
                dirty.observations, args.output, config, simulation=True
            )
            clean.truth.to_parquet(args.output / "truth.parquet", index=False)
            write_json(
                args.output / "generation.json",
                {
                    "generation": asdict(generating),
                    "corruption": asdict(corrupting),
                    "removed_weeks": dirty.removed_weeks,
                    "duplicated_weeks": dirty.duplicated_weeks,
                    "defects": dirty.defects,
                },
            )
            provenance = json.loads((args.output / "provenance.json").read_text())
            for name in ("truth.parquet", "generation.json"):
                provenance["artifacts"][name] = sha256(
                    (args.output / name).read_bytes()
                ).hexdigest()
            write_json(args.output / "provenance.json", provenance)
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
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
