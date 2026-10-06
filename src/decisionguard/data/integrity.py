"""Deterministic weekly data validation and conservative normalization."""

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd

from decisionguard.data.synthetic import DEFAULT_CHANNELS

Status = Literal["RESOLVED", "WARNING", "BLOCKER"]
SPEND_COLUMNS = tuple(f"{channel.name}_spend" for channel in DEFAULT_CHANNELS)
CONTROL_COLUMNS = ("price", "promotion", "macro_index", "competitor_index")
NUMERIC_COLUMNS = (
    *SPEND_COLUMNS,
    *CONTROL_COLUMNS,
    "revenue",
)
COLUMNS = ("week", *NUMERIC_COLUMNS)
ALIASES = {
    "fb_spend": "meta_spend",
    "facebook_spend": "meta_spend",
    "FB_spend": "meta_spend",
    "Facebook_spend": "meta_spend",
}
METADATA_COLUMNS = ("currency", "spend_unit", "available_at")


@dataclass(frozen=True)
class IntegrityConfig:
    """Explicit dates/rates; no statistics are learned for cleaning."""

    duplicate_policy: Literal["drop_exact", "block"] = "drop_exact"
    expected_start: str | None = None
    expected_end: str | None = None
    as_of: str | None = None
    currency_rates: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.duplicate_policy not in ("drop_exact", "block"):
            raise ValueError("duplicate_policy must be drop_exact or block")
        for name, value in (
            ("expected_start", self.expected_start),
            ("expected_end", self.expected_end),
            ("as_of", self.as_of),
        ):
            if value is not None:
                timestamp = pd.Timestamp(value)
                if (
                    pd.isna(timestamp)
                    or timestamp.tz is not None
                    or value != timestamp.strftime("%Y-%m-%d")
                ):
                    raise ValueError(f"{name} must be a timezone-free calendar date")
                if name != "as_of" and timestamp.dayofweek != 0:
                    raise ValueError(f"{name} must be Monday")
        if (
            self.expected_start
            and self.expected_end
            and self.expected_start > self.expected_end
        ):
            raise ValueError("expected_start must precede expected_end")
        currencies = [currency for currency, _ in self.currency_rates]
        if len(set(currencies)) != len(currencies):
            raise ValueError("currency_rates must have unique currency codes")
        for currency, rate in self.currency_rates:
            if (
                not currency
                or not isfinite(rate)
                or rate <= 0
                or (currency == "AUD" and rate != 1)
            ):
                raise ValueError(
                    "currency rates must be positive, finite AUD conversion factors"
                )


@dataclass(frozen=True)
class QualityIssue:
    code: str
    status: Status
    message: str
    weeks: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityReport:
    schema_version: str
    raw_hash: str
    clean_hash: str
    status: Status
    issues: tuple[QualityIssue, ...]
    profile: dict[str, object]
    config: IntegrityConfig


@dataclass(frozen=True)
class IntegrityResult:
    clean: pd.DataFrame
    report: QualityReport

    def require_model_ready(self) -> pd.DataFrame:
        if self.report.status == "BLOCKER":
            raise ValueError("modeling blocked by unresolved data-quality issues")
        return self.clean.copy(deep=True)


def dataset_hash(frame: pd.DataFrame) -> str:
    header = repr([(str(name), str(dtype)) for name, dtype in frame.dtypes.items()])
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return sha256(header.encode() + values).hexdigest()


def profile_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Descriptive full-window profiling; never used to fit/remove/impute values."""
    if not frame.columns.is_unique:
        return {"row_count": len(frame), "column_names_unique": False}
    numeric = frame.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    distributions: object = {}
    if len(numeric.columns) and len(frame):
        summary = numeric.describe().T
        summary["median"] = numeric.median()
        distributions = json.loads(summary.to_json())
    return {
        "row_count": len(frame),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "missingness": {
            str(name): int(count) for name, count in frame.isna().sum().items()
        },
        "cardinality": {
            str(name): int(count) for name, count in frame.nunique().items()
        },
        "distributions": distributions,
    }


def clean_dataset(
    raw: pd.DataFrame, config: IntegrityConfig | None = None
) -> IntegrityResult:
    """Normalize known defects; preserve ambiguous evidence and block its use."""
    if config is None:
        config = IntegrityConfig()
    clean = raw.copy(deep=True)
    issues: list[QualityIssue] = []
    profile = profile_dataset(raw)
    raw_hash = dataset_hash(raw)

    def add(
        code: str, status: Status, message: str, keys: tuple[str, ...] = ()
    ) -> None:
        issues.append(QualityIssue(code, status, message, keys))

    def finish() -> IntegrityResult:
        status: Status = "RESOLVED"
        if any(issue.status == "BLOCKER" for issue in issues):
            status = "BLOCKER"
        elif any(issue.status == "WARNING" for issue in issues):
            status = "WARNING"
        return IntegrityResult(
            clean,
            QualityReport(
                "weekly-wide-v1",
                raw_hash,
                dataset_hash(clean),
                status,
                tuple(issues),
                profile,
                config,
            ),
        )

    if not clean.columns.is_unique:
        add("duplicate_columns", "BLOCKER", "Column names must be unique")
        return finish()
    for alias, canonical in ALIASES.items():
        if alias not in clean:
            continue
        if canonical in clean:
            add(
                "alias_collision",
                "BLOCKER",
                f"Both {alias} and {canonical} exist; no automatic merge",
            )
        else:
            clean = clean.rename(columns={alias: canonical})
            add("taxonomy", "RESOLVED", f"Renamed {alias} to {canonical}")
    missing = sorted(set(COLUMNS) - set(clean.columns))
    extra = sorted(set(clean.columns) - set(COLUMNS) - set(METADATA_COLUMNS))
    if missing or extra:
        add(
            "schema",
            "BLOCKER",
            f"Missing columns: {missing}; unsupported columns: {extra}",
        )
        return finish()
    timezone_labels = (
        clean["week"]
        .astype(str)
        .str.contains(r"[T ].*(?:Z|[+-]\d{2}(?::?\d{2})?)$", case=False, regex=True)
    )
    if timezone_labels.any() or pd.api.types.is_numeric_dtype(clean["week"]):
        add("dates", "BLOCKER", "Week labels must be timezone-free ISO dates")
        return finish()
    try:
        dates = pd.to_datetime(clean["week"], errors="coerce", format="ISO8601")
        if dates.dt.tz is not None:
            add(
                "timezone",
                "BLOCKER",
                "Week labels must be timezone-free calendar dates",
            )
            return finish()
        clean["week"] = dates.dt.normalize()
    except (ValueError, AttributeError):
        add("dates", "BLOCKER", "Week labels must be unambiguous ISO dates")
        return finish()
    invalid_dates = clean["week"].isna() | (clean["week"].dt.dayofweek != 0)
    if invalid_dates.any():
        add("dates", "BLOCKER", "Invalid dates or non-Monday week labels")
        return finish()
    add(
        "temporal_normalization",
        "RESOLVED",
        "Normalized ISO week labels to calendar dates and sorted chronologically",
    )
    for name in NUMERIC_COLUMNS:
        clean[name] = pd.to_numeric(clean[name], errors="coerce").astype(float)

    if "spend_unit" in clean:
        units = clean["spend_unit"]
        unknown = ~units.isin(["base_units", "thousands", "AUD", "AUD_thousands"])
        if unknown.any():
            add("units", "BLOCKER", "Unknown or missing spend units; no scale inferred")
        thousands = units.isin(["thousands", "AUD_thousands"])
        if thousands.any():
            clean.loc[thousands, list(SPEND_COLUMNS)] *= 1_000
            clean.loc[thousands, "spend_unit"] = "base_units"
            add(
                "units",
                "RESOLVED",
                f"Converted {int(thousands.sum())} declared thousand-unit rows",
            )
    if "currency" in clean:
        rates = {"AUD": 1.0, **dict(config.currency_rates)}
        for currency in clean["currency"].drop_duplicates():
            if currency not in rates:
                add(
                    "currency",
                    "BLOCKER",
                    f"No approved AUD conversion rate for {currency!r}",
                )
            elif currency != "AUD":
                mask = clean["currency"] == currency
                clean.loc[mask, [*SPEND_COLUMNS, "price", "revenue"]] *= rates[currency]
                clean.loc[mask, "currency"] = "AUD"
                add(
                    "currency",
                    "RESOLVED",
                    f"Converted {int(mask.sum())} {currency} rows "
                    f"at factor {rates[currency]}",
                )

    exact = int(clean.duplicated().sum())
    if exact and config.duplicate_policy == "drop_exact":
        clean = clean.drop_duplicates().copy()
        add(
            "exact_duplicates",
            "RESOLVED",
            f"Removed {exact} exact duplicate rows; values never aggregated",
        )
    if clean["week"].duplicated().any():
        add(
            "duplicate_keys",
            "BLOCKER",
            "Unresolved duplicate week keys require intervention",
        )

    if "available_at" in clean:
        timezone_labels = (
            clean["available_at"]
            .astype(str)
            .str.contains(r"[T ].*(?:Z|[+-]\d{2}(?::?\d{2})?)$", case=False, regex=True)
        )
        if timezone_labels.any() or pd.api.types.is_numeric_dtype(
            clean["available_at"]
        ):
            add("availability", "BLOCKER", "Availability must use naive ISO timestamps")
            return finish()
        availability = pd.to_datetime(
            clean["available_at"], errors="coerce", format="ISO8601"
        )
        if availability.dt.tz is not None or availability.isna().any():
            add("availability", "BLOCKER", "Invalid availability timestamps")
        else:
            earliest = clean["week"] + pd.Timedelta(days=7)
            if (availability < earliest).any():
                add(
                    "availability",
                    "BLOCKER",
                    "Weekly observations cannot be available before the week ends",
                )
            if (availability > earliest).any():
                add(
                    "late_arrival",
                    "WARNING",
                    "Some weekly records arrived after the normal week-end release",
                )
            if config.as_of is not None:
                unavailable = availability.gt(pd.Timestamp(config.as_of))
                if unavailable.any():
                    clean = clean.loc[~unavailable, :].copy()
                    add(
                        "unavailable",
                        "BLOCKER",
                        f"Excluded {int(unavailable.sum())} unavailable records "
                        f"as of {config.as_of}",
                    )
    elif config.as_of is not None:
        week_end = pd.to_datetime(clean["week"]) + pd.Timedelta(days=7)
        unavailable = week_end.gt(pd.Timestamp(config.as_of))
        if unavailable.any():
            clean = clean.loc[~unavailable, :].copy()
            add(
                "unavailable",
                "BLOCKER",
                f"Excluded {int(unavailable.sum())} not-yet-complete weekly records",
            )

    clean = clean.sort_values("week", kind="stable").reset_index(drop=True)
    if not len(clean):
        add("empty", "BLOCKER", "No observations remain")
    else:
        start = (
            pd.Timestamp(config.expected_start)
            if config.expected_start
            else clean["week"].min()
        )
        end = (
            pd.Timestamp(config.expected_end)
            if config.expected_end
            else clean["week"].max()
        )
        expected = pd.date_range(start, end, freq="W-MON")
        gaps = expected.difference(pd.DatetimeIndex(clean["week"]))
        profile["date_start"] = str(start)[:10]
        profile["date_end"] = str(end)[:10]
        profile["missing_weeks"] = [str(key)[:10] for key in gaps]
        if len(gaps):
            add(
                "time_gaps",
                "BLOCKER",
                "Missing weeks are unknown observations, not zero activity",
                tuple(str(key)[:10] for key in gaps),
            )
        if ((clean["week"] < start) | (clean["week"] > end)).any():
            add(
                "date_window",
                "BLOCKER",
                "Observations fall outside the declared window",
            )
    for name in NUMERIC_COLUMNS:
        values: pd.Series[float] = clean[name]
        invalid = ~np.isfinite(values.to_numpy(dtype=float))
        if invalid.any():
            add(
                "missing_or_nonfinite",
                "BLOCKER",
                f"{name}: {int(invalid.sum())} invalid/missing values; no imputation",
            )
        if name in (*SPEND_COLUMNS, "revenue") and (values < 0).any():
            add(
                "negative_values",
                "BLOCKER",
                f"Negative {name} is impossible under the data contract",
            )
        if name == "price" and (values <= 0).any():
            add("price", "BLOCKER", "Price must be positive")
        if name == "promotion" and not values.dropna().isin([0, 1]).all():
            add("promotion", "BLOCKER", "Promotion must be binary")
        if values.count() >= 4:
            q25, q75 = values.quantile([0.25, 0.75])
            spread = q75 - q25
            outliers = values.lt(q25 - 3 * spread) | values.gt(q75 + 3 * spread)
            if outliers.any():
                add(
                    "outliers",
                    "WARNING",
                    f"{name}: descriptive 3-IQR extremes retained; "
                    "commercial and erroneous events require investigation",
                    tuple(str(key)[:10] for key in clean.loc[outliers, "week"]),
                )
    if len(clean) >= 16:
        median = clean["revenue"].rolling(8, min_periods=8).median()
        prior = median.shift(8)
        changes = median.sub(prior).abs().gt(0.3 * prior.abs())
        if changes.any():
            add(
                "structural_change",
                "WARNING",
                "Adjacent eight-week median revenue differs by over 30%; "
                "investigate; this is not a causal change-point test",
                tuple(str(key)[:10] for key in clean.loc[changes, "week"]),
            )
    clean = clean.loc[:, list(COLUMNS)].copy()
    profile["clean_row_count"] = len(clean)
    return finish()
