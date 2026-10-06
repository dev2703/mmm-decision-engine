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

### ADR: first Phase 3 predictive benchmark

Context: future predictive models need a reproducible error reference before
complexity is justified. The target is weekly AUD revenue over the next configured
horizon; this is a predictive question, not channel attribution or budget allocation.

Decision: start with seasonal naive. It repeats the last complete training season
for each future step, including horizons longer than a season. Defaults are period
52, initial training length 52, and horizon 13. These match the synthetic world's
52-week cycle; real calendar and regime assumptions require later sensitivity
experiments. No coefficients, transforms, uncertainty intervals, or random seed
are fitted/generated by this baseline.

Temporal validation uses explicit contiguous prefix slices. CV forecast windows
do not overlap; each training prefix grows. The final horizon is reserved as an
independent holdout and excluded from CV metrics. Extra pre-holdout weeks that do
not fill a CV horizon are still eligible for the final training prefix. At least
one CV fold and a holdout are required. Forecasts are fixed at each origin; actual
targets within a forecast window never update those predictions.

An optional gap withholds the latest training weeks. The baseline forecasts across
the gap before returning test-horizon predictions, preserving seasonal phase.
All training records must have become available by the first test week's start.
Normal weekly availability is week + 7 days; late-arrival timestamps are aligned
from the verified raw artifact. Grouping takes the latest arrival per week
conservatively and reindexes one value per unique clean week; missing alignment
or unavailable records fail. This prevents a finalized historical snapshot from
silently being treated as information available at every historical origin.

Metrics are deterministic MAE and RMSE in AUD/week and WAPE as sum absolute errors
divided by sum absolute actual values. Zero-denominator WAPE is explicitly null.
CV summaries pool the nonoverlapping CV prediction rows; holdout scores are separate.
Fold-level metrics and train/test/availability boundaries remain inspectable.

`decisionguard baseline --dataset DATASET_DIRECTORY --output NEW_RUN_DIRECTORY`
passes through the Phase 2 quality/checksum gate. It saves predictions and an
experiment record containing hypothesis, change, validation, metrics, result,
interpretation, decision, limitations, feature/model configuration, input hashes,
source-code hash, and dependency-lock hash when available. A null random seed
denotes a deterministic baseline. Data warnings are retained in the record. Model
status is `PREDICTIVE_ONLY`; no release status authorizes optimization.

Alternatives: a constant/last-value baseline would ignore the known seasonal
structure. scikit-learn TimeSeriesSplit is useful once the broader model stack
needs it, but a few explicit prefix slices meet this one-model experiment without
a new dependency. Revisit the evaluator when model-specific fitting and nested
temporal tuning require a shared interface; do not scaffold that interface now.
The remaining Phase 3 models must share temporal/availability boundaries and fit
all learned preprocessing on their training folds only.

### ADR: classical ETS comparison and research

Additive error/trend/seasonal ETS is the first classical model. The synthetic
revenue contains changing level and additive annual seasonality, so additive ETS
fits the generative story better than imposing stationarity on raw revenue.
Stationary innovations remain an assumption, not a proven property; structural
shifts, promotions and changing variance can invalidate it. We do not choose
ARIMA differencing orders from the sealed holdout or add tests with IID assumptions.
A seasonal fit requires at least two complete training cycles (104 weeks at period
52); seasonal naive must use the same starting window for a fair comparison.

Statsmodels supplies estimation and exact 90% innovation prediction intervals.
Its installed 0.14.6 interval method needs a pandas Series, as a NumPy input failed
with an index AttributeError in an executed probe. Optimization nonconvergence and
nonfinite predictions fail the run. Record residual mean, Ljung–Box diagnostic lag,
statistic and p-value, and test interval coverage by fold. The Ljung–Box value is
exploratory (no fitted-model degrees-of-freedom correction), not a release gate.
Intervals condition on estimated parameters and omit their uncertainty; they are
not posterior MMM intervals or proof of causal validity.

Alternatives: SARIMA adds differencing/order choices and seasonal computation
without evidence of benefit yet. A separate temporal evaluator would duplicate
leakage guards; the two baselines share the existing evaluator and artifact path.
Revisit ETS when CV, residuals or coverage indicate its assumptions fail.

Research sources: [Statsmodels ETS](https://www.statsmodels.org/stable/examples/notebooks/generated/ets.html)
and [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
Pipelines should own fold-fitted scaling in the next regularized benchmark.

### ADR: Phase 3 conditional regression comparison

Ridge on raw spend and controls, Ridge on domain adstock/saturation features, and
Histogram Gradient Boosting share the baseline's outer temporal windows and input
quality gate. Every representation includes trend and calendar Fourier terms.
Domain carryover uses fixed decay 0.5; each channel's saturation half-point is its
training-prefix median carryover (with a one-AUD lower bound). Neither transform
reads the simulator's privileged parameters or truth. This simple misspecified
representation is an experiment to test, not a promise of improvement.

Ridge uses a StandardScaler/Ridge pipeline, refitted per outer fold. Alpha from
0.1/1/10/100 is selected by MAE on the last horizon of the outer training prefix,
using a separate preceding inner prefix and the configured availability gap.
Inner saturation references and scaler fitting exclude inner validation. Both
inner and outer training observations must be available at their respective
origins. Audit selected alpha, inner scores, references and outer scaler state.
Histogram Gradient Boosting uses 100 iterations, seven leaves, minimum ten rows
per leaf, L2 regularization one, seed 42, and no automatic early stopping; its
otherwise random validation split would violate the temporal design. Scaling
is unnecessary for this tree model. Its fixed configuration is not holdout tuned.

Important information contract: regressors condition on realized test spend and
controls. These covariates are not claimed available at the historical forecast
origin. Their outputs are conditional predictions. Seasonal naive and ETS forecast
from historical revenue alone. `comparison.json` keeps these contexts separate,
selects the lowest CV MAE within each, and then reports the untouched holdout.
There is no mixed leaderboard or automatic causal/decision-model promotion.
Actual planned covariates or their forecast uncertainty would be required to turn
conditional predictions into operational multiweek forecasts.

Executed seed-42, 156-week experiment with initial training 104 and horizon 13:

| Context | Model | CV MAE AUD/week | Holdout MAE AUD/week |
| --- | --- | ---: | ---: |
| Origin-only forecast | Seasonal naive | 13,373.06 | 14,230.85 |
| Origin-only forecast | Additive ETS | 21,631.41 | 13,189.64 |
| Conditional prediction | Raw Ridge | 2,705.12 | 2,803.76 |
| Conditional prediction | Domain Ridge | 3,318.31 | 3,020.69 |
| Conditional prediction | Histogram Gradient Boosting | 7,717.58 | 5,882.56 |

Interpretation: retain seasonal naive and raw Ridge in their respective roles.
ETS' better holdout does not override worse CV. Fixed-decay feature engineering
was not beneficial on this run, and tree complexity was not justified. Shared
trend and known control covariates explain much of raw Ridge's advantage; this
does not establish causal media effects. Additional seeds, altered carryover,
and regimes can change the ranking; this is not a universal model claim.
Quality tests show explicit unit/alias/duplicate repairs recover clean predictive
outputs, while unresolved target/currency/calendar defects block all experiments.

Native thread pools are limited to one during fit/predict for these tiny datasets.
In a restricted macOS sandbox, Joblib physical-core discovery emitted a warning;
`LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1` makes the intended resource limit explicit
for verification and reproducible demos. The warning is not filtered or ignored.
Partial comparisons retain completed run artifacts but have no `comparison.json`;
that final summary is the completion marker. New output directories prevent
silently overwriting previous evidence.

Sources: [scikit-learn pipelines and leakage](https://scikit-learn.org/stable/common_pitfalls.html),
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
and [Histogram Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
Revisit fold-specific hyperparameter grids only through temporal CV; never choose
feature representations or priors by inspecting the final holdout repeatedly.

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

### Architecture research: observed compatibility and resource costs

Phase 4 dependency resolution alone was insufficient: PyMC-Marketing 0.14.1
imported a removed PyTensor compatibility symbol when paired with PyTensor 2.38.
Pinning PyMC 5.23 / PyTensor 2.31 restored imports. That marketing version is
selected to match `mmm-eval` commit `71d20009feaa30dd9606ffface62f16fb1134265`,
whose source requires PyMC-Marketing <0.15. Updating to the current marketing major
would require an evaluated upstream adapter migration, not a silent version bump.

On the actual macOS toolchain, PyTensor's C build failed with `library 'd64' not
found`. The supported `PYTENSOR_FLAGS=cxx=` fallback avoids that linker. Interpreted
NUTS emitted numerical proposal warnings and was slower; the official PyMC Nutpie
sampler provides a compiled Numba backend. A convolution operation can fall back
to object mode, with an unsuppressed library warning. This is a performance risk,
separate from R-hat/ESS/divergences; real scientific tests must still execute.
Compiled Linux deployment should be verified separately rather than assuming
macOS fallback performance represents production.

Sources: [upstream linker issue](https://github.com/pymc-devs/pytensor/issues/2268),
[version-pinned mmm-eval requirements](https://github.com/mutinex/mmm-eval/blob/71d20009feaa30dd9606ffface62f16fb1134265/pyproject.toml),
and [PyMC-Marketing model workflow](https://www.pymc-marketing.io/en/0.14.1/notebooks/mmm/mmm_example.html).

The same `mmm-eval` source imports Meridian/TensorFlow through adapter/config
initializers even for the PyMC path. This increases install size and cold-start
cost without a second framework requirement in our product. Prefer isolating
heavy evaluation in CLI/job execution and retaining raw evaluation artifacts for
the API; do not fork the upstream scientific tests or build microservices solely
to hide package imports. Revisit when upstream supports optional framework imports.

### ADR: auditable Bayesian MMM candidate (Phase 4)

The likelihood is Normal revenue, with an intercept, linear standardized controls
(price, promotion, macro, competitor and trend), two yearly Fourier harmonics,
and positive media contributions. PyMC-Marketing supplies unnormalized finite
geometric adstock and Michaelis–Menten saturation. Unlike the infinite-carryover
DGP, this model truncates history at 16 lags. Calendar seasonality uses the library's
yearly calendar rather than silently reading the privileged simulator parameters.
Controls/trend are centered and scaled using only the training prefix; the library
fits channel and target MaxAbs scaling on that same prefix. Holdout is the final
13 weeks; prediction explicitly carries forward the last training lag history.
Realized holdout covariates condition predictions and are not origin-known inputs.

Priors, in target-scaled units: intercept Normal(0.5, 0.25), signed controls and
Fourier coefficients Normal(0, 0.1), noise HalfNormal(0.05), channel carryover
Beta(2, 2), saturation amplitude Gamma(mean 0.15, SD 0.1), and saturation half-point
HalfNormal(1). These regularize noisy weekly fits while allowing wide media
uncertainty. Positive media priors encode a monotone response assumption; recovered
positive media means alone cannot validate incrementality. Signed-control recovery
is checked under symmetric priors. No ground-truth effects configure the model.

Generate 200 prior predictive draws before sampling. Reject nonfinite draws, more
than 5% negative scaled revenue, or a 99th percentile above five times training
maximum revenue. These are coarse simulation plausibility screens, not empirically
calibrated commercial release thresholds. Retain the complete prior draws in the
posterior NetCDF of successful candidates.

Default production sampling: four chains, 2,000 draws, 1,500 tuning steps, seed 42,
Nutpie, one core, target acceptance 0.99, maximum tree depth 12. Earlier executed
500-draw / acceptance-0.95 fits showed divergences. Acceptance 0.99 removed those,
but the default depth ten still saturated; the increased depth is independently
checked. Diagnostic acceptance requires finite parameter diagnostics, R-hat at
most 1.01, minimum bulk/tail ESS 400, zero divergences, BFMI at least 0.3 in each
chain and zero maximum-depth hits. A one-chain smoke run is explicitly INCOMPLETE.
These checks do not by themselves release a model for decisions.

Save `posterior.nc`, `channel_draws.npz`, `predictions.parquet` and a completion
manifest `model.json`. Preserve original-unit channel contribution and ROI draws,
90% prediction intervals, in-sample/holdout metrics and coverage, parameter
summaries, channel-contribution correlations, and conditional saturation curves.
Their x-axis is adstocked AUD spend, not raw weekly budget; the optimizer must
account for the finite carryover process when evaluating allocations. ROI here is
modeled attributed revenue divided by historical spend, not experimentally proven
incremental profit or marginal return.

An executed regression test found that posterior predictive sampling can replace
`idata.observed_data` with the library's dummy targets. Preserve the real fitted
observations before prediction and restore that group before saving. Reload tests
compare it against actual target-scaled training revenue. Build/prior sampling use
the real target from the start, avoiding the documented placeholder-target hazard.
Artifacts are checksum-verified before consumption; the model record retains input
quality evidence, training window, configuration, preprocessing state, source hash
at job start and dependency-lock hash. Candidates remain CANDIDATE_UNEVALUATED
regardless of predictive accuracy; external validation and release policy follow.

Prior sensitivity compares the same data/window/channels under amplitude mean
0.25 and SD 0.15, retaining both runs and their sampler diagnostics. Similar holdout
error can coexist with materially different channel ROI. The original exploratory
runs are retained, including diagnostics failures; they are not silently upgraded
or substituted as released model evidence.


Final executed default-prior run with 2,000 draws per chain: R-hat maximum
1.00334, minimum bulk ESS 1,350.31, minimum tail ESS 1,641.75, zero divergences,
zero depth-limit hits, and minimum BFMI 0.83717. The wider-prior 2,000-draw run
also passes the diagnostic screen. The 1,000-draw deeper-tree repeat still failed
intercept R-hat, which motivated longer chains rather than a relaxed threshold.
These are sampling checks, not release or causal-validity claims.

The first optional evaluation install exposed a real macOS import deadlock in
TensorFlow 2.20 (`RAW: Lock blocking`), independently reproduced with a bounded
30-second subprocess and a faulthandler stack in TensorFlow's native wrapper.
The [upstream TensorFlow issue](https://github.com/tensorflow/tensorflow/issues/99464)
reports the same failure. Pin TensorFlow/tf-keras to 2.19 in the evaluation group;
verify import and rerun the scientific suite after the resulting NumPy change.
An available wheel or a successful dependency solve is not a runtime check.

### Phase 5 runtime separation and interruption recovery

Use one lockfile with mutually exclusive `dev` and `evaluation` dependency groups.
The supported TensorFlow 2.19 stack needs NumPy below 2.2; current pandas typing
stubs require NumPy 2.3. An isolated evaluation environment satisfies both published
contracts without overriding requirements or weakening development typing. This
costs a second local environment and duplicated scientific wheels. Revisit when
upstream evaluator support permits the same current scientific stack.

The actual six-test smoke evaluation completed 20 refits and produced BLOCK with
no test execution errors. This is a valid policy outcome, not a released model:
placebo failure and short-refit diagnostic failures remain visible. The longer
production evaluation was interrupted after seven completed fits, revealing that
end-only persistence discards expensive partial progress.

Persist each successful upstream test's exact Parquet table and refit evidence,
then atomically publish its checksum manifest. Explicit resume validates input,
source, lock, sampler, and installed-version identity before reusing results.
Unfinished/failed tests rerun; changed checkpoints fail closed. Write the overall
completion manifest last. This is a narrow checkpoint facility, not a job queue or
workflow engine. It supports one worker per directory; process supervision and
transactional claims belong to the backend phase. Code changes intentionally
invalidate resume to avoid mixing results from different implementations.

### Phase 6 planning estimand and optimizer choice

Question: how should a constant weekly media budget be allocated over a declared
horizon, conditional on joint posterior response states and observed carry-in?
Compare against the caller's current allocation under those same states. Outcomes
are modeled media revenue in AUD, excluding future nonmedia revenue and observation
noise. They are neither experimentally identified incremental profit nor total
revenue forecasts. Keep joint draws intact rather than independently sampling
channel marginals.

Use the existing SciPy SLSQP solver with budget fractions, analytic derivatives,
and an independent feasibility check. Scale the objective by its response at the
feasible starting point: the executed nearly-linear decision-cliff test exposed
premature convergence when scaling only by saturation amplitude. The NumPy response
calculation matches the pinned PyMC-Marketing geometric/Michaelis-Menten transforms
in an executed scientific test, including observed carry-in and the finite lag
window. Cache carry-in once; slice the chosen state for per-draw solves. The pinned
0.14 library budget optimizer starts from zero carry-in and scores an added tail,
which answers a different planning question. Revisit its use when a compatible
version supports this historical/horizon contract without recompiling every draw.

Expected allocation maximizes posterior mean media response. Conservative allocation
maximizes the empirical mean of the worst 10% of improvements against current spend
(at least one state, rounded up). Report parameter uncertainty, downside probability,
probability of improvement, allocation quantiles, direction flips and normalized-L1
stability. For equal-budget plans, L1 counts both ends of a transfer; cash moved is
half that distance. A 0.2 P95 stability limit is an explicit prototype threshold,
not an established commercial standard. An unstable or historically unsupported
candidate cannot authorize a recommendation. Tighter movement constraints may
make recommendations admissible; retain the original instability evidence.

The release loader recomputes policy from checksum-verified raw evaluation evidence,
model identity and prior source, rejecting inconsistent stored labels. BLOCK is
checked before loading posterior arrays or solving. Floors, caps, protected spend,
relative movement and source channel restrictions intersect; infeasible requests
fail rather than silently relaxing them. Prior sensitivity must actually change
priors while holding the model specification fixed; sampler precision can differ.

Executed checks include seeded feasible-budget properties, infeasibility, restricted
channels, stable/unstable decisions, a tiny-response allocation cliff, conservative
downside, input immutability and analytical-gradient finite differences. Twenty
joint states loaded from the actual reviewed candidate gave finite original-unit
planning responses. That candidate remains unreleased pending its production
external evaluation; numerical feasibility does not override release evidence.

### Phase 7 initial persistence boundary

Start with projects and immutable generated datasets, using SQLAlchemy 2 sessions
directly from synchronous FastAPI endpoints. `sessionmaker.begin()` commits on
successful exit and rolls back exceptions, as specified in the
[SQLAlchemy session documentation](https://docs.sqlalchemy.org/en/20/orm/session_basics.html).
Lock a project row while assigning its next dataset version; enforce project/version
uniqueness and quality/date invariants in PostgreSQL. Alembic owns schema changes;
application startup does not create tables. Explicit settings require the psycopg
PostgreSQL URL and a local artifact root.

Artifacts use server-generated relative UUID paths with root/symlink containment
checks. PostgreSQL stores metadata and small JSONB evidence, not posterior arrays.
Shared CLI/API simulation persistence retains separate truth artifacts and hashes;
quality inspection can expose BLOCKER evidence without authorizing modeling.
The quality endpoint verifies files against provenance and persisted metadata.
Filesystem writes and a database commit cannot form one transaction: a failed
commit can leave inspectable orphan artifacts, but must not publish a partial
dataset row. Backend worker/reconciliation work must preserve that distinction.

The currently installed Starlette test client prefers httpx2 and uses its types,
while its fallback httpx path emitted deprecation warnings and lost strict typing.
Use its supported client rather than adding typing suppressions or pinning an old
framework solely for tests; see [Starlette TestClient](https://starlette.dev/testclient/).
The first actual PostgreSQL/API tests also exposed date-valued generation config
that JSONB cannot serialize directly; persist the same JSON configuration already
written by the shared artifact workflow.

This is currently a local single-user API. Authentication, shared-store ownership,
worker claims, bounded heavy jobs and deployment remain subsequent work; do not
expose the local service publicly before those boundaries are completed.

### Phase 7 registered model jobs

Experiments snapshot a project-owned dataset, hypothesis and validated MMM config.
Model-run requests queue immutable input/configuration references. The CLI claims
a QUEUED run in a short `FOR UPDATE SKIP LOCKED` transaction, closes the transaction
before sampling, verifies input identity, and publishes only checksum-verified
completion evidence in a second transaction. Duplicate active runs for one
experiment are rejected under an experiment-row lock. This uses PostgreSQL and
existing CLI execution, without Celery/Redis or a speculative workflow framework.

Separate job lifecycle from scientific status: SUCCEEDED means artifacts were
produced, while CANDIDATE_UNEVALUATED still forbids budget recommendations. Failures
retain their type and transition to FAILED without results. Process death can leave
RUNNING; recovery needs operator confirmation that the old worker stopped. Never
blindly reclaim a possibly live scientific job. A database failure during publication
can leave complete but unpublished artifacts; reconciliation must verify manifests.

Configuration validation now lives separately from the Bayesian runtime, allowing
HTTP to validate requests without importing PyMC. Actual fresh-process verification
found neither PyMC nor sklearn imported at API construction. Sampling has explicit
local request limits; each request gets a generated correlation ID and structured
operation/status/duration logs without request bodies or credentials. PostgreSQL
integration checks exercise snapshot isolation, duplicate claims, rollback/failure
and a real one-chain training smoke whose diagnostics remain INCOMPLETE.

### Phases 0–5 audit and strengthening

The full upstream production evaluation completed all six tests and 20 actual
four-chain refits (2,000 draws, 1,500 tuning steps each), without execution errors.
Its release state is BLOCK: shuffled-channel net ROI is 34.1895%, four repeated
refit executions record a divergence, several channels fail refresh/perturbation
checks, and OOH changes by 52.56% under the wider prior. The base and wider-prior
candidate fits individually pass diagnostics; that does not override refit or
placebo evidence. Recomputed release interpretation agrees with the stored record.
These failures are retained, not fixed by relaxing thresholds.

Close two integrity gaps. Clean-data identity alone cannot detect changed raw
arrival timestamps; consumers now verify raw, clean and quality artifact identities
against the model's recorded evidence. Identical relocated artifacts remain valid;
absolute historical paths remain provenance, with optional explicit data-location
overrides. Diagnostic status labels alone are insufficient: a shared pure numeric
gate verifies R-hat/ESS/divergence/BFMI/depth evidence, and undefined BFMI values
persist as null with INVESTIGATE status. Catastrophic future error blocks even when
an inconsistent source flag says pass. Duplicate metric columns and undefined
source values cannot silently become passing evidence. Checkpoints constrain test
names and checksum refit evidence as well as their source tables.

Measured checksum bottleneck: the actual 249,328,477-byte reviewed posterior needed
249,333,327 bytes of peak Python allocation with `read_bytes()`, versus 267,553 bytes
using standard-library `hashlib.file_digest`; digests matched exactly. Use that
streaming helper across artifact consumers instead of caching unverified checksums.
This reduces memory without changing artifact formats or weakening integrity.
Nutpie also defaults to retaining warmup output; the reviewed file contains roughly
107 MiB of warmup posterior arrays. New runs default to supported `save_warmup=False`,
with an explicit troubleshooting option. A real sampling/reload test verifies that
warmup groups are absent while posterior and observed-target evidence remain intact.
Existing immutable runs are preserved.

Numerical metric tests now cover large/small scales and unsigned inputs. Normalize
absolute errors before squaring to avoid representable RMSE overflowing or
underflowing; reject unrepresentable differences explicitly. The data contract
blocks complex values rather than discarding their imaginary parts. Profiling omits
unsupported complex distributions so that it can return a blocker report.

Keep the modular monolith and isolated evaluation environment. No evidence here
justifies a cache of mutable fitted MMM objects, a second model framework, or a
workflow engine. Upstream repeated reference refits are a remaining runtime cost;
changing that execution needs equivalence evidence and correct state isolation.
Local manifests provide integrity against accidental/tampered components, not
cryptographic authorization against an operator rewriting the entire evidence set.
Shared or externally uploaded artifacts require a separately trusted ownership and
record-hash boundary before public deployment. Phase 0 automation remains explicitly
deferred; no GitHub changes were attempted.

Upstream security boundary: the pinned evaluator's JSON configuration rehydrator
falls back from `ast.literal_eval` to an expression `eval` with a restricted class
registry. Our bridge does not expose that loader: construct approved native model
objects from validated primitive `MMMConfig` values, then use
`PyMCConfig.from_model_object`. Do not accept repr-encoded external model objects.
A regression test rejects instruction-like prior text before external computation
and traps any call to the upstream JSON expression loader. See the
[pinned implementation](https://github.com/mutinex/mmm-eval/blob/71d20009feaa30dd9606ffface62f16fb1134265/mmm_eval/configs/rehydrators.py).
