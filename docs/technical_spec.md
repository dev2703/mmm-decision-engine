1. Architecture Objective

Build the smallest production-style architecture capable of supporting the scientific workflow.

Architecture:

Next.js Dashboard
        │
        ▼
FastAPI
        │
        ├──────── PostgreSQL
        │
        ├──────── Artifact Storage
        │
        ├──────── Model/Evaluation Code
        │
        ├──────── Optimizer
        │
        └──────── Agent Tools

mmm-decision-engine starts as a:

modular monolith

not microservices.

2. Runtime

Python

Use:

Python 3.12

Dependency management

Use:

uv
pyproject.toml
uv.lock

Do not maintain parallel:

Conda;

Poetry;

requirements.txt;

without a real compatibility reason.

3. Backend Stack

Use:

FastAPI;

Pydantic;

Uvicorn;

SQLAlchemy 2.x;

Alembic;

psycopg 3.

Why FastAPI:

typed boundaries;

OpenAPI;

lightweight;

relevant to the role;

strong Pydantic integration.

4. Database

Use PostgreSQL.

Store:

projects;

dataset metadata;

experiments;

model runs;

evaluation runs;

scenarios;

optimization runs;

agent trace metadata.

Do not store giant posterior arrays in PostgreSQL.

5. Artifact Storage

Local:

./artifacts/

Production target:

Google Cloud Storage

Artifacts include:

Parquet datasets;

ArviZ/PyMC NetCDF;

model diagnostics;

mmm-eval JSON;

optimization results.

Avoid building a generic storage interface until both local and cloud implementations are actually needed.

6. Data / ML Libraries

Use:

pandas
numpy
scipy
statsmodels
scikit-learn
PyMC
PyMC-Marketing
ArviZ
Mutinex mmm-eval

Potential later dependency:

XGBoost

Only add if sklearn's gradient boosting is insufficient for a meaningful experiment.

7. Data Validation

Use:

Pydantic for application schemas;

optionally Pandera for dataframe contracts.

If Pandera introduces more ceremony than value for our relatively small domain:

use clear explicit validation functions.

8. Optimization

Start with:

scipy.optimize

Do not immediately add CVXPY.

Use CVXPY only if the eventual mathematical formulation benefits clearly from convex-program abstractions.

After every solve independently verify:

total budget;

channel minimums;

channel maximums;

movement limits;

restricted-channel limits;

objective;

feasibility.

Do not trust:

solver.success == True

by itself.

9. Frontend

Use:

Next.js
React
TypeScript

Charts:

ECharts or

Plotly.js.

Choose one.

Avoid pulling in several overlapping visualization frameworks.

10. Agent

Use one provider's official SDK.

Do not begin with:

LangChain;

LangGraph;

CrewAI;

AutoGen.

Initial control flow is explicit Python.

A framework becomes justified only when the orchestration itself becomes a genuine complexity problem.

11. Repository Shape

Target:

mmm-decision-engine/
├── agents.md
├── README.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .pre-commit-config.yaml
│
├── docs/
│   ├── scope.md
│   ├── prd.md
│   ├── technical_spec.md
│   ├── roadmap.md
│   └── adr/
│
├── src/
│   └── decisionguard/
│       ├── api/
│       ├── data/
│       ├── experiments/
│       ├── models/
│       ├── evaluation/
│       ├── optimization/
│       ├── agent/
│       ├── db/
│       ├── config.py
│       └── cli.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── scientific/
│   └── agent/
│
├── web/
└── artifacts/

Important:

This is a target shape, not instructions to create every empty directory on day one.

Create modules only when real code needs them.

12. Synthetic Data-Generating Process

Clean process:

[
Revenue_t =
Baseline_t
+
Media_t
+
Promotion_t
+
Price_t
+
Controls_t
+
\epsilon_t
]

Baseline

Includes:

trend;

annual seasonality;

optional structural shift.

Media

Each channel gets:

spend process;

adstock;

saturation;

known contribution.

Controls

Include:

price;

promotions;

macro;

competitor activity.

Noise

Use an explicitly defined stochastic process.

Document the assumption.

### Clean DGP implementation decision

Context: Phase 1 needs known contributions before evaluating fitted models.
`decisionguard.data.synthetic.generate_dataset` returns observations, separate
privileged truth, and a frozen configuration. Both tables are caller-owned pandas
dataframes; freezing the result object does not make dataframe contents immutable.
Future corruption must copy its input and preserve the clean tables.

Contract: one row per week for one synthetic business, keyed by `week`, a
timezone-free Monday calendar date marking the start of the period. There is no
ingestion timestamp in this clean world. Search, Meta, TV, OOH, and YouTube spend
and revenue are AUD/week; price is AUD/unit, promotion is binary, and macro and
competitor indices are dimensionless and centered on 100. Clean values have no
missingness or duplicates. Truth is never a modeling input.

For channel i and week t:

```
spend[i,t] = typical_spend[i] * exp(0.35*z[i,t] - 0.35**2/2)
A[i,t] = spend[i,t] + decay[i]*A[i,t-1], A[i,-1] = 0
media[i,t] = maximum_contribution[i] * A[i,t] / (half_saturation[i] + A[i,t])
baseline[t] = 100000 + 200*t + 15000*sin(2*pi*t/52)
structural_shift[t] = shift_amount if t >= shift_week else 0 (disabled if None)
commercial_event[t] = event_amount if t == event_week else 0 (disabled if None)
price_effect[t] = -800*(price[t] - 100)
promotion_effect[t] = 12000*promotion[t]
macro_effect[t] = 1000*(macro_index[t] - 100)
competitor_effect[t] = -300*(competitor_index[t] - 100)
revenue[t] = baseline[t] + sum(media[:,t]) + price_effect[t]
             + promotion_effect[t] + macro_effect[t] + competitor_effect[t]
             + structural_shift[t] + commercial_event[t] + noise[t]
```

All spend shocks have standard-normal marginals. Search and Meta share one normal
shock: each uses `(shared + independent)/sqrt(2)`, giving latent correlation 0.5.
Other channels have independent shocks. Spend is lognormal and always positive;
`typical_spend` is its population mean. Price is Normal(100, 5), macro is
Normal(100, 2), competitor is Normal(100, 5), and promotion is Bernoulli(0.15).
Noise is independent Normal(0, noise_std), with default standard deviation 2500.
These controls and innovations are independent through time. Seasonality uses a
simplified 52-week cycle anchored at the configured start, not a holiday calendar.

Decision: use unnormalized geometric carryover and a Hill response with exponent
1, rather than a custom MMM library. Channel-specific parameters are recorded in
`SyntheticConfig.channels`; a zero maximum contribution creates a placebo channel.
Independent seeded NumPy streams preserve historical values when the requested
horizon grows. pandas provides ecosystem-compatible tables; no transforms are
learned from the generated data. The source revision and uv lock are part of the
reproduction contract.

Consequences: zero initial carryover causes an intentional startup transient.
Gaussian controls/noise have unbounded support, so extreme configurations can
produce commercially implausible outcomes; revenue is not silently clipped, as
clipping would break the additive ground truth. This simulation is a controlled
test world, not an empirically calibrated causal model. `event_week` and the signed
`event_amount` create a genuine one-week commercial outlier, separately recorded
as `commercial_event_contribution`. This does not alter observed media or controls.

Structural shift decision: `shift_week` is an optional nonnegative, zero-based
week index; `shift_amount` is a finite, signed AUD/week baseline level change.
The shift starts at that week and persists thereafter, with no extra random draws.
Its default is disabled (`None`, zero amount); a nonzero amount requires an index.
A future index beyond the current horizon is allowed to preserve the same world
when the horizon grows. `structural_shift_contribution` is always present in truth,
including as zeros when disabled. It is added separately to revenue so the existing
trend/seasonality baseline and each economic component remain auditable. It is
privileged truth and is not exposed as an observed feature.

The technical spec places structural shifts in the baseline, while the scope
also lists regime shifts among dirty-data scenarios. We model a genuine economic
shift in the clean DGP; future measurement errors belong in the corruption layer.
An additive level change is chosen over slope or channel-response changes to
provide a simple known temporal boundary. Revisit that choice when robustness
experiments require changing marketing effectiveness or trend regimes.

Alternatives: standard-library random/row dictionaries avoid dependencies but
require more tabular plumbing; using PyMC-Marketing now couples simulation to the
model being tested and adds unnecessary Phase 4 dependencies. Revisit the fixed
noise, control, and response assumptions when robustness experiments need serial
dependence, realistic promotion timing, or model misspecification.

13. Corruption Layer

Separate from clean DGP.

Example API:

dirty = corrupt_dataset(clean, CorruptionConfig(
    missing_weeks=4, duplicate_rows=3, meta_alias="fb_spend",
    unit_error_weeks=2, tracking_outage_weeks=2, seed=42,
))

Ground truth remains intact.

Initial corruption implementation: `corrupt_dataset(clean, CorruptionConfig(...))`
accepts a clean `SyntheticDataset` and returns an independent observations table,
the corruption configuration, `removed_weeks`, and `duplicated_weeks`. Week keys
must be datetime, non-null, unique, and ordered before injection. No ground-truth
columns enter the dirty observations, and neither upstream table is modified.

`missing_weeks` and `duplicate_rows` are exact nonnegative integer counts, default
zero. Missing weeks are selected uniformly without replacement. Duplicates are
exact row copies selected uniformly with replacement from retained weeks, then
sorted stably by week. Repeated copies of a single week are allowed; every copy
appears in the duplicate manifest. Removed weeks represent unknown/absent records,
not zero spend or zero revenue. No values are imputed or recomputed.

Separate seeded NumPy streams isolate missing-week selection from the number of
duplicates requested. Corruption uses seed domain `[seed, 1]` to avoid reusing
the clean DGP's streams when both configurations use the same seed.
Reproduction requires the same clean source, configuration,
code revision, and dependency lock. Sampling covers the supplied source window;
changing that window can change the sampled defects. The manifest is privileged
simulation evidence, not an observed model feature.

Decision: exact counts make small test fixtures and failure scenarios predictable,
whereas probabilistic rates yield variable counts. Rates can be added if needed
for corruption severity experiments. Removing more weeks than exist fails;
removing every week produces an empty table with preserved schema, and requesting
duplicates with no retained weeks fails.

The completed Phase 1 corruption options also include:

- `meta_alias`: rename the wide Meta column to `fb_spend` or `facebook_spend`.
  This models channel taxonomy aliases without inventing campaign IDs or changing
  the observation grain.
- `unit_error_weeks`: divide all channel spend by 1000 on sampled retained weeks
  and declare `spend_unit=thousands`. Other rows declare `base_units`.
- `tracking_outage_weeks`: replace revenue with unknown values over consecutive
  retained observations. Calendar gaps caused by separate removal remain gaps.
- `erroneous_outlier_weeks`: multiply reported revenue by 10. Clean economic truth
  is unchanged; this is separate from a genuine commercial event.
- `late_arrival_weeks`: weekly data normally become available at `week + 7 days`;
  sampled weeks receive `arrival_delay_days` extra delay (default 14).
- `mixed_currency_weeks`: convert monetary observations to simulated USD by
  dividing by 1.5, and record currency per row. 1.5 is a synthetic assumption,
  not a current exchange rate.
- `negative_spend_weeks`: negate Search spend to produce impossible observations.

All counts refer to distinct retained weeks before row duplication. Defects use
separate seeded streams and may overlap. Copies of an affected week inherit its
measurement defects. The manifest records defect kind and affected week keys.
Neither clean table is mutated. Unit metadata describes magnitude independently
of currency, so combined unit/currency corruption is unambiguous.

14. Data Pipeline

Raw Layer

Immutable.

Profile Layer

Calculate:

schema validity;

row counts;

missingness;

date coverage;

duplicates;

category cardinality;

outliers;

taxonomy;

unit issues.

Cleaning Layer

Only deterministic/configured transformations.

Feature Layer

Leakage-safe feature engineering.

Learned transforms must fit on training data only.

### ADR: conservative weekly integrity pipeline (Phases 1–2)

Context: the controlled world needs an auditable raw-to-clean boundary before
temporal modeling. One business is represented by a wide weekly table; merging
multiple entity tables and campaign-level encoders are not current requirements.

Decision: use explicit pandas validation/normalization instead of adding Pandera
or a generic cleaning framework. `clean_dataset` returns an independent candidate
table and a typed quality report. The contract requires a timezone-free Monday
`week` key; five spend columns; price, binary promotion, macro/competitor indices,
and revenue. Spend/revenue are nonnegative AUD; price is positive AUD/unit;
controls are finite. Unsupported columns, including privileged truth, block use.
Numeric strings can be parsed; missing and nonfinite values are never imputed.
Week labels use ISO dates, not ambiguous locale formats or numeric epochs.

Known aliases are renamed only when there is no canonical-column collision.
Explicit `base_units`/`thousands` spend metadata permits deterministic scaling.
Legacy `AUD`/`AUD_thousands` scale labels are also accepted. Unknown units block.
Mixed currencies require positive supplied conversion factors to AUD; no rate
is inferred from magnitudes. The simulation's factor 1.5 must also be explicitly
supplied to the integrity pipeline. Exact duplicates may be dropped under the
versioned `drop_exact` policy; conflicting measurements and the alternative
`block` policy never silently aggregate revenue or choose a convenient row.

Temporal normalization sorts observations and standardizes calendar date labels.
Missing periods are reported and block modeling, with no invented zero rows.
Explicit expected start/end detect missing leading or trailing periods; otherwise
coverage is inferred from observed endpoints. An `as_of` calendar date is an
availability cutoff at that date's midnight. Records arriving later, or weeks
not yet complete, are excluded and produce a blocker. Invalid timestamps block.
Full-window profiling is descriptive and never determines cleaned feature values.

`quality.json` records schema version, raw/clean dataframe hashes, quality status,
configuration, issue codes, affected diagnostic weeks, and resolved transformation
records. Raw profiling contains row counts, missingness, exact duplicates, category
cardinality, and numeric summaries including mean, median and quantiles. Raw
summaries describe recorded values before unit/currency harmonization and are
not commercially comparable aggregates when those metadata are inconsistent. Summary
statistics omit nonfinite values only in their temporary descriptive view; actual
nonfinite observations remain in the candidate and cause blockers. Outliers use
descriptive 3-IQR flags. Adjacent eight-week revenue medians differing by over 30%
flag a suspected structural change. These are exploratory warnings, not calibrated
hypothesis tests, causal evidence, or fitted removal thresholds.

Consequences: clean artifacts with blockers remain inspectable candidates;
`IntegrityResult.require_model_ready()` and `load_model_ready()` reject them.
The artifact loader verifies raw, clean, and report hashes (plus simulation
artifacts when present). Local checksums detect accidental changes; they are not
cryptographic authorization against an operator rewriting the entire manifest.
Outliers remain present because commercial events and data errors can look alike.
No joins occur in this wide-table workflow; duplicate/key tests enforce cardinality
directly. Other domains will require explicit join contracts before adding joins.

Alternatives/revisit: Pandera may help if schema complexity grows. Probabilistic
anomaly detection or change-point inference requires separate temporal experiments
and calibration, not a threshold chosen to classify this generator perfectly.
Missing predictors could later use train-fitted imputation with flags; target
imputation remains blocked. Real exchange-rate histories require time-aligned
approved rates, rather than this MVP's explicitly configured constant per currency.

### Statistical/preprocessing choices

| Question | Phase 2 decision | Revisit evidence |
| --- | --- | --- |
| Mean vs median | Report both; median/IQR describe skewed spend robustly. Neither imputes data. | A model-specific, train-only imputation experiment. |
| Scaling/standardization | Keep original monetary units; no learned scaler in cleaning. | Phase 3 regularized models or Phase 4 sampler geometry. |
| One-hot/ordinal encoding | Channels already have fixed columns; promotion is a real binary indicator. No nominal category is assigned an artificial order. | A genuine low-cardinality nominal feature. |
| Target encoding | Not used; no high-cardinality categories and it could leak future targets. | A justified category with nested temporal validation. |
| PCA | Not used; five channel identities are needed for later attribution. Correlation alone does not justify losing those identities. | Large external-control sets and measured conditioning problems. |
| Outlier removal | Flag and retain; never winsorize a commercial event automatically. | Verified measurement error with an explicit correction policy. |
| Statistical tests | No shuffled permutation tests, IID t-tests, or p-values for serial revenue. Structural-change flags are descriptive. | Temporal/block-aware validation of a specific hypothesis. |

### Reproducible local artifacts

`decisionguard prepare-data --output artifacts/demo` creates the synthetic source,
optional defects, and the integrity outputs in one command. `decisionguard clean-data
RAW.parquet --output artifacts/rechecked` processes an existing file without
modifying it. Output directories must be new. Each run contains `raw.parquet`,
`clean.parquet`, `quality.json`, and `provenance.json`; simulation runs additionally
contain `truth.parquet` and `generation.json`. Provenance records artifact hashes,
a hash of package source files, and a dependency-lock hash when available. Run
configuration and seeds are recorded. Blocked runs exit with code 2 after writing
diagnostic artifacts; resolved/warning runs exit 0.

PyArrow supplies Parquet I/O rather than a custom persistence format. NumPy is
constrained to `<2.4`: tests found pandas 2.3.3 datetime construction warnings with
NumPy 2.5.3, and passed with locked NumPy 2.3.5. Revisit this bound when upgrading
the pandas/NumPy pair with the scientific and artifact tests.

15. Validation

Primary strategy:

expanding window
/
rolling origin
/
leave-future-out

Not shuffled K-fold.

Hyperparameter tuning must remain inside temporal validation.

16. Model Ladder

Seasonal Naive

Baseline.

ARIMA / ETS

Classical forecasting comparison.

Linear / Regularized

Domain-engineered interpretable model.

Gradient Boosting

Predictive benchmark.

PyMC-Marketing

Probabilistic MMM.

17. Feature Engineering

Potential features:

Calendar

week/year;

annual seasonality;

holiday;

promotion periods.

Media

Adstock:

[
A_t =
x_t
+
\lambda x_{t-1}
+
\lambda^2x_{t-2}
+\cdots
]

Saturation using appropriate nonlinear transformation.

Controls

price;

macro;

competitor activity.

All learned transformations must be fold-safe.

18. Bayesian Workflow

Required sequence:

generative story
      ↓
priors
      ↓
prior predictive checks
      ↓
sampling
      ↓
R-hat / ESS / divergences
      ↓
posterior predictive checks
      ↓
identifiability
      ↓
sensitivity
      ↓
mmm-eval
      ↓
decision layer

Never jump:

fit()
→ posterior mean
→ business recommendation

19. mmm-eval

Pin to exact source version/commit.

Minimal adapter.

Do not rewrite its existing tests.

Persist:

test_name
configuration
raw_result
status
policy_threshold
artifact_reference

20. Release Policy

Implement deterministic:

PASS
WARN
RESTRICT
BLOCK

Policy configuration is versioned.

Example:

if data_leakage:
    BLOCK

elif placebo_failed:
    BLOCK

elif refresh_instability > threshold:
    RESTRICT

elif holdout_error > warning_threshold:
    WARN

else:
    PASS

Real policy will be more explicit, but should remain understandable.

21. Decision Stability

Start with:

[
DS =
\frac{
|a_{perturbed}-a_{baseline}|_1
}{
B
}
]

Estimate using:

posterior draws;

plausible data perturbations;

refreshes.

Report:

median;

P95;

direction flip;

channel range.

22. Budget Optimizer

Inputs:

model
posterior draws
budget
current allocation
min bounds
max bounds
movement bounds
channel restrictions
risk preference

Outputs:

allocation
expected outcome
uncertainty
P(beat baseline)
downside
stability
constraint report
extrapolation warnings

23. Extrapolation

Optimizer must check whether proposed spend lies outside historical support.

Example:

Observed Meta spend:
$300k–$850k

Recommended:
$1.4m

Status:
RESTRICT

Smooth mathematical response curves do not prove the extrapolated area is credible.

24. Persistence Objects

Project

id
name
created_at
current_dataset_id
current_model_run_id

Dataset

id
project_id
version
source
raw_uri
clean_uri
schema_version
date_start
date_end
hash
quality_status

Experiment

id
project_id
hypothesis
model_family
feature_config
validation_config
status

ModelRun

id
experiment_id
dataset_id
model_family
code_version
config
artifact_uri
training_window
metrics
status

EvaluationRun

id
model_run_id
evaluator
config
summary
artifact_uri

Scenario

id
model_run_id
budget
constraints
risk_policy

OptimizationRun

id
scenario_id
baseline_allocation
proposed_allocation
expected_outcome
uncertainty
stability
release_status

25. API

Projects

POST /projects
GET /projects/{id}

Data

POST /projects/{id}/datasets/generate
POST /datasets/{id}/validate
GET /datasets/{id}/quality

Models

POST /projects/{id}/experiments
POST /experiments/{id}/run
GET /model-runs/{id}
GET /projects/{id}/model-runs

Evaluation

POST /model-runs/{id}/evaluate
GET /model-runs/{id}/health

Decisions

POST /model-runs/{id}/scenarios
POST /scenarios/{id}/optimize
GET /optimization-runs/{id}

Agent

POST /agent/query

Do not create CRUD endpoints for every database object just because the object exists.

26. Agent Tools

Initial tool surface:

get_data_quality(project_id)

compare_models(project_id)

get_model_health(model_run_id)

get_channel_evidence(
    model_run_id,
    channel,
)

run_scenario(
    model_run_id,
    scenario,
)

validate_constraints(scenario)

get_decision_stability(
    optimization_run_id
)

explain_experiment(
    experiment_id
)

Never expose:

run_python()
run_shell()
execute_sql()
fetch_any_url()

27. Long-Running Jobs

Local:

decisionguard train --experiment-id ...
decisionguard evaluate --model-run-id ...

Production target:

Cloud Run service
+
Cloud Run Jobs

Avoid Celery/Redis unless job orchestration later becomes a real requirement.

28. Artifact Layout

artifacts/
└── projects/
    └── {project_id}/
        ├── datasets/
        │   └── {dataset_id}/
        │       ├── raw.parquet
        │       ├── clean.parquet
        │       └── quality.json
        │
        ├── model-runs/
        │   └── {model_run_id}/
        │       ├── config.json
        │       ├── metrics.json
        │       └── posterior.nc
        │
        ├── evaluations/
        │   └── {evaluation_id}/
        │       └── mmm_eval.json
        │
        └── optimizations/
            └── {optimization_id}/
                └── result.json

29. Testing

Use:

pytest
hypothesis
httpx

Unit

Test:

transformations;

metrics;

release policy;

constraints;

stability calculation.

Scientific

Test:

temporal leakage;

DGP reproducibility;

zero-effect channel sanity;

directional recovery;

adstock transformation.

Integration

Test:

raw
→ clean
→ experiment
→ model
→ evaluation
→ release
→ scenario
→ optimization

Agent

Test:

correct tools;

correct order;

invalid argument handling;

policy bypass;

prompt injection;

unsupported claims;

numerical grounding.

Property Tests

Useful invariants:

[
\sum_i allocation_i = B
]

and:

min_i <= allocation_i <= max_i

for every channel.

30. Quality Tooling

Use:

ruff
pyright
pytest
pre-commit

Suggested Ruff rules:

E
F
I
UP
B
SIM
C4
PERF
RUF

Do not blindly enable everything.

CI:

ruff format --check .
ruff check .
pyright
pytest

31. Docker

One backend image.

Local Compose:

backend
postgres

Frontend may run natively during development.

Do not create:

model-service
optimizer-service
agent-service
data-service

as four containers merely to make a diagram impressive.

32. Deployment

Frontend

Vercel.

Backend

Google Cloud Run.

Heavy model jobs

Cloud Run Jobs if useful.

Database

Managed PostgreSQL.

Artifacts

Google Cloud Storage.

For interview reliability, expensive model runs may be precomputed as long as:

they are real;

reproducible;

reproduction commands exist.

Do not fake training outputs.

33. Security

Requirements:

environment/secret-manager credentials;

no secrets in Git;

parameterized SQL;

Pydantic validation;

safe artifact paths;

no arbitrary execution tools;

no unrestricted network tool;

prompt/data separation;

deterministic release policy;

explicit approval for consequential actions;

secret redaction.

Treat all data strings as potentially adversarial.

34. Observability

Every operation receives a correlation ID.

Log:

project ID;

dataset ID;

experiment ID;

model run;

evaluation run;

optimization run;

latency;

status.

Agent traces:

request
  ↓
model call
  ↓
tool call
  ↓
tool result
  ↓
policy result
  ↓
final response

Track:

latency;

errors;

model version;

tokens;

cost;

tool selection.

35. Major Tradeoffs

Modular Monolith vs Microservices

Choose modular monolith.

Reason:

one developer;

cohesive domain;

easier debugging;

fewer deployment boundaries;

easier refactoring.

PostgreSQL vs NoSQL

Choose PostgreSQL.

Reason:

relational provenance;

transactions;

constraints;

JSONB when necessary.

pandas vs Polars

Start with pandas.

Reason:

ecosystem compatibility;

manageable data size;

PyMC/MMM interoperability.

Revisit only if profiling indicates a real bottleneck.

PyMC-Marketing vs Custom MMM

Use PyMC-Marketing.

The learning objective is not:

reinvent PyMC.

Direct Agent Orchestration vs LangGraph

Start direct.

Reason:

less abstraction;

visible state;

easy debugging;

easier interview explanation.

Cloud Job vs Celery

Avoid Celery initially.

Use:

CLI locally;

cloud job later.

Aggregate Health Score vs Explicit Gates

Use explicit gates.

Do not allow:

Placebo FAIL
+
Holdout PASS
+
PPC PASS
=
83/100 HEALTHY

A critical failure must remain visible.

36. Technical Definition of Done

A feature is finished when:

acceptance criteria pass;

relevant tests pass;

Ruff passes;

Pyright passes;

scientific assumptions are documented;

dependencies are justified;

abstractions are justified;

failure behaviour is clear;

useful logs exist;

simplification review has occurred.
