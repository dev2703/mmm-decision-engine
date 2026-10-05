---
name: data-integrity
description: Data Science cleaning, leakage, loading, feature engineering, and loading checklist
---

# Data Integrity

## Goal

Build a dataset whose semantics are trustworthy before asking a model to learn from it.

A sophisticated model cannot repair a corrupted data-generating process.

## Step 1 — Establish the data contract

Before cleaning or joining, identify:

- observation grain;
- primary/unique key;
- event time;
- ingestion time;
- timezone;
- frequency;
- units;
- currency;
- source system;
- allowed categories;
- null semantics;
- mutable vs immutable fields.

For marketing data, explicitly identify:
- channel;
- platform;
- campaign;
- geography;
- product/brand;
- spend;
- impressions/clicks if used;
- revenue/sales target;
- price/promotions;
- external controls.

## Step 2 — Profile before modifying

Measure:

- row counts;
- uniqueness;
- missingness;
- duplicate keys;
- impossible values;
- distribution shape;
- extreme values;
- category cardinality;
- date coverage;
- gaps in time;
- change points;
- source-level inconsistencies.

Do not silently "clean" anomalies before understanding them.

## Step 3 — Classify missingness semantically

For each important missing field ask what `NaN` means:

- true zero;
- unknown;
- not measured;
- not applicable;
- not yet arrived;
- source failure.

Do not replace missing values with zero unless zero is the correct business meaning.

Track imputation flags when the fact of imputation may itself matter.

## Step 4 — Validate temporal completeness

For each entity/time series:

- verify expected frequency;
- detect missing periods;
- detect duplicates;
- distinguish absent row from true zero activity;
- identify late-arriving observations;
- identify revisions/backfills.

Never forward-fill a variable simply because it is convenient.

Confirm that persistence is meaningful for that variable.

## Step 5 — Normalize taxonomies and units

Detect and resolve examples such as:

- `Facebook`, `FB`, `Meta`;
- AUD vs USD;
- gross vs net spend;
- weekly vs monthly observations;
- local vs UTC timestamps;
- campaign renames;
- region code changes.

Keep raw values and normalized values separately when traceability matters.

## Step 6 — Join defensively

Before a join, state expected cardinality:

- one-to-one;
- many-to-one;
- one-to-many.

After joining, assert:
- row count expectations;
- duplicate-key behavior;
- null introduction;
- unmatched key counts.

Do not allow silent many-to-many row multiplication.

## Step 7 — Prevent leakage

Any learned preprocessing step must be fit using training data only.

This includes:
- scalers;
- imputers;
- encoders;
- target encoding;
- PCA;
- feature selection;
- outlier thresholds when learned from data;
- hyperparameter search;
- learned seasonal features.

For time series, future observations must not influence historical training through preprocessing.

Prefer pipeline objects that make this boundary explicit.

## Step 8 — Encode categories according to semantics

Use one-hot encoding when:
- categories are nominal;
- cardinality is manageable;
- interpretability matters.

Do not ordinal-encode nominal categories merely for compactness.

For high-cardinality identifiers, consider:
- hierarchical/group effects;
- carefully leakage-safe target encoding;
- embeddings for suitable models;
- frequency/count encoding;
- domain aggregation.

Do not one-hot thousands of arbitrary campaign IDs without a reason.

## Step 9 — Scale only when needed

Ask whether the selected model benefits from scaling.

Examples:
- regularized linear models often benefit;
- distance-based methods require it;
- many tree methods do not;
- Bayesian parameterization may benefit for sampler geometry.

Fit scaling parameters on the training window only.

Preserve meaningful original units for reporting and optimization.

## Step 10 — Treat outliers as questions, not mistakes

An extreme observation may represent:
- data corruption;
- tracking outage;
- genuine promotion;
- holiday;
- product launch;
- competitor event;
- structural change.

Investigate before dropping or winsorizing.

Record any removal rule and test sensitivity to it.

## Step 11 — Produce data-quality artifacts

Each dataset version should expose:

- schema version;
- date range;
- row count;
- missingness summary;
- duplicates;
- taxonomy mappings;
- transformations;
- excluded rows with reasons;
- data-quality warnings;
- dataset hash/version.

## Blocking conditions

Stop downstream modeling when:
- target semantics are unresolved;
- time ordering is ambiguous;
- currency/units are inconsistent and unresolved;
- joins create unexplained duplication;
- leakage is known;
- a critical source has unexplained structural breaks.

## Completion checklist

- [ ] Grain and key are explicit.
- [ ] Time semantics are explicit.
- [ ] Missingness meanings are documented.
- [ ] Joins have cardinality checks.
- [ ] Units/currencies/taxonomies are consistent.
- [ ] Learned preprocessing is training-only.
- [ ] Temporal splits are respected.
- [ ] Outlier handling is justified.
- [ ] Dataset version is reproducible.