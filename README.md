# mmm-decision-engine

Marketing mix model decision support: evaluate whether model evidence justifies
a budget recommendation, then propagate uncertainty into constrained decisions.

Phases 1 (synthetic marketing world) and 2 (data integrity) are implemented.
Phase 3 predictive benchmarks and auditable temporal comparisons are implemented.
Phase 4 Bayesian candidates and prior sensitivity are implemented.
External evaluation, release gating and the decision/product phases follow.

## Local setup

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

Python 3.12 is the supported runtime. uv selects it using `.python-version`.
The installed Python package is `decisionguard`; the distribution retains the
repository name `mmm-decision-engine`.

## Project context

Read [agents.md](agents.md) before making changes, then the canonical documents:

- [Scope](docs/scope.md): product boundaries.
- [PRD](docs/prd.md): product behavior and user experience.
- [Technical spec](docs/technical_spec.md): architecture and technical decisions.
- [Roadmap](docs/roadmap.md): implementation order and phase exit criteria.

These four documents are intended for Git; other local research under `docs/`
remains ignored. The product name is `mmm-decision-engine`; `decisionguard` remains
the Python package and planned CLI identifier.

Source lives in `src/decisionguard/`; tests live in `tests/`.

## Foundation decisions

- Ruff handles formatting and linting, Pyright checks strict types, and pytest
  runs tests. These are development dependencies, locked by uv. NumPy supplies
  seeded numerical generation; pandas supplies the specified table format.
  pandas-stubs is a development dependency for strict dataframe type checking.
- Hatchling builds the standard `src` package layout. The packaging test imports
  from an isolated interpreter outside the repository, so source-path shortcuts
  cannot hide a broken installation.
- Python is constrained to 3.12 to match the project contract. Supporting newer
  runtimes can be revisited when the scientific dependency stack is verified.
- Configuration has only a module boundary for now; add settings when actual
  runtime requirements exist.

Phase 0 pre-commit hooks and GitHub Actions remain deferred.

## Generate clean synthetic data

After `uv sync --locked`, run this Python example with `uv run python`:

```python
from decisionguard.data.synthetic import SyntheticConfig, generate_dataset

dataset = generate_dataset(SyntheticConfig(weeks=156, seed=42))
print(dataset.observations.head())
print(dataset.truth.head())
```

`observations` contains weekly spend, controls, and revenue. `truth` separately
records adstock, contributions, and noise; never pass this privileged table to a
prediction model. `dataset.config` retains the seed and generating parameters.
Reproduce with the same configuration, source revision, and dependency lock.

To simulate a permanent baseline revenue increase of AUD 15,000/week from the
53rd observation (zero-based index 52):

```python
shifted = generate_dataset(SyntheticConfig(shift_week=52, shift_amount=15_000))
```

Negative amounts simulate a decline. The shift is disabled by default and is
recorded as `structural_shift_contribution` in truth. A shift beyond the current
horizon takes effect when that horizon is extended.

## Inject observation defects

```python
from decisionguard.data.corruption import CorruptionConfig, corrupt_dataset

dirty = corrupt_dataset(
    dataset, CorruptionConfig(missing_weeks=4, duplicate_rows=7, seed=123)
)
print(dirty.observations.head())
print(dirty.removed_weeks, dirty.duplicated_weeks)
```

Counts are exact. Duplicates are sampled with replacement from retained weeks;
missing weeks cannot be reintroduced. The result contains an independent table
and a seed/configuration plus affected week keys. Clean observations and truth
remain unchanged. Use the clean dataset as the source for each corruption run.

Further corruption options include `meta_alias`, `unit_error_weeks`,
`tracking_outage_weeks`, `erroneous_outlier_weeks`, `late_arrival_weeks`,
`mixed_currency_weeks`, and `negative_spend_weeks`. Genuine economic outliers use
`SyntheticConfig(event_week=20, event_amount=100_000)` and retain their contribution
in clean truth. Each measurement defect records its affected week keys.

## Auditable data preparation

One command generates a real synthetic source, injects safely repairable defects,
and writes raw/clean Parquet data, quality, truth, configuration, and provenance:

```sh
uv run decisionguard prepare-data --output artifacts/demo \
  --meta-alias fb_spend --unit-error-weeks 4 --duplicate-rows 3
```

Run directories must be new: artifacts are not overwritten. To validate an
existing Parquet file without changing it:

```sh
uv run decisionguard clean-data artifacts/demo/raw.parquet \
  --output artifacts/rechecked --expected-start 2022-01-03 --expected-end 2024-12-23
```

`quality.json` contains raw profiling, schema version, dataset hashes, transformation
and issue records, and `RESOLVED`, `WARNING`, or `BLOCKER` status. `provenance.json`
contains artifact and source-code hashes, plus the dependency-lock hash when present.
Known aliases, explicitly declared thousand-unit spend, and exact duplicates are
normalized. Conflicting duplicates, missing targets/weeks, negative spend, unknown
units, and currencies without supplied rates block modeling.

Mixed-currency simulation uses USD -> AUD factor 1.5 solely as a controlled test
assumption. Pass `--currency-rate USD=1.5` to approve that conversion in a simulated
run; real imports require their own documented rates. `--as-of YYYY-MM-DD` excludes
records unavailable at that date's midnight. Never estimate missing revenue or
assume a missing week means zero activity.

Blocked runs still write diagnostic artifacts and exit with code 2. Their
`clean.parquet` is a candidate for inspection, not model-ready data. Consumers use
the deterministic gate:

```python
from pathlib import Path
from decisionguard.data.artifacts import load_model_ready

model_inputs = load_model_ready(Path("artifacts/demo"))
```

This checks artifact hashes and rejects blocked quality status. Warnings retain
commercial outliers and suspected structural shifts for investigation. No learned
imputers, scalers, encoders, or outlier-removal thresholds are fitted in Phase 2.
See the [technical spec](docs/technical_spec.md) for the data contract and rationale.

## Evaluate the forecasting baseline

Use a model-ready dataset directory and a new experiment directory:

```sh
uv run decisionguard baseline --dataset artifacts/demo \
  --output artifacts/baseline-demo
```

Defaults use a 52-week season, at least 52 training weeks, and 13-week forecast
windows. Expanding-window CV runs before an independent final 13-week holdout.
`--period`, `--initial-train`, `--horizon`, and `--gap` configure the experiment;
`--hypothesis` supplies a hypothesis for its record. No hyperparameters are tuned.

`predictions.parquet` records actual/predicted revenue, residuals, dates, and fold
IDs. `experiment.json` records metrics, train/test boundaries, availability,
configuration, dataset/source hashes, hypothesis, interpretation, and limitations.
MAE/RMSE are AUD/week; WAPE is a fraction and is null if all actual values are zero.

Training records must be available by each forecast origin. With the simulation's
14-day late-arrival delay, `--gap 2` withholds the two latest training weeks and
advances forecasts across that gap. Insufficient history or missing weeks fail
explicitly. Data-quality blockers and artifact changes are rejected before a run.

The baseline is `PREDICTIVE_ONLY`: it supplies a forecast-error reference, without
uncertainty intervals, attribution, ROI, or budget recommendations. The full Phase 3 ladder adds classical and conditional regression comparisons
with fold-fitted preprocessing, as described below.

Classical ETS comparison (requires two annual training seasons):

```sh
uv run decisionguard baseline --dataset artifacts/demo --output artifacts/ets --model ets --initial-train 104
uv run decisionguard baseline --dataset artifacts/demo --output artifacts/naive-104 --initial-train 104
```

Both runs use identical validation windows. ETS saves 90% prediction intervals,
convergence and residual diagnostics; these remain predictive-only experiments.

Complete predictive comparison:

```sh
LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 uv run decisionguard compare --dataset artifacts/demo --output artifacts/comparison
```

Inspect `comparison.json` and each model's `experiment.json` / `predictions.parquet`.
Ridge and gradient boosting condition on test spend/controls; compare them within
that group, separately from origin-only ETS/seasonal forecasts. CV selects models;
holdout reports performance. No benchmark is authorized for budget optimization.


Fit a real Bayesian candidate and a prior-sensitivity alternative:

```sh
# The cxx flag is the documented macOS linker workaround; omit on a working C toolchain.
PYTENSOR_FLAGS=cxx= LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 uv run decisionguard train --dataset artifacts/demo --output artifacts/mmm
PYTENSOR_FLAGS=cxx= LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 uv run decisionguard train --dataset artifacts/demo --output artifacts/mmm-prior --media-prior-mean 0.25 --media-prior-sigma 0.15
```

Defaults use four chains, 2,000 draws, 1,500 tuning steps, target acceptance 0.99,
and depth 12. `model.json` records diagnostics and input/configuration provenance;
`posterior.nc` preserves prior/posterior draws; `channel_draws.npz` preserves ROI,
contributions and saturation-response draws; predictions contain 90% intervals.
Candidates remain `CANDIDATE_UNEVALUATED` even when diagnostics pass. A short
one-chain smoke run cannot meet the scientific diagnostic criteria.

Record sensitivity without overwriting either fitted run:

```python
from pathlib import Path
from decisionguard.models.artifacts import compare_prior_sensitivity

compare_prior_sensitivity(
    Path("artifacts/mmm"),
    Path("artifacts/mmm-prior"),
    Path("artifacts/prior-sensitivity"),
)
```

## External evaluation and release policy

The pinned upstream evaluator has a separate dependency environment. Its supported
TensorFlow/NumPy versions conflict with development typing dependencies; do not
combine the `dev` and `evaluation` groups.

```sh
UV_PROJECT_ENVIRONMENT=.venv-evaluation uv sync --locked --no-default-groups --group evaluation
PYTENSOR_FLAGS=cxx= LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 .venv-evaluation/bin/decisionguard evaluate --model-run artifacts/mmm --sensitivity-run artifacts/mmm-prior --output artifacts/evaluation
.venv-evaluation/bin/pytest tests/test_mmm_eval.py
```

All six actual upstream tests run on training-only refits, preserving the original
model's sealed holdout. Raw source scores and flags remain separate from the
versioned PASS/WARN/RESTRICT/BLOCK interpretation. A failed placebo blocks release;
predictive accuracy cannot override it. Short sampling overrides are smoke checks,
not evidence that diagnostics pass.

Interrupted runs retain completed tests. Repeat the exact command with `--resume`
to reuse verified checkpoints; changed data, model records, code, lockfile, runtime
versions or sampling settings require a new output directory. A completed
`evaluation.json` prevents overwriting. Failed tests are retried rather than cached
as successes. Stop the original worker before resuming; concurrent writers to one
run directory are unsupported.

## Posterior decision analysis

`decisionguard optimize --model-run artifacts/mmm --evaluation artifacts/evaluation
--constraints constraints.json --output artifacts/optimization` accepts a JSON
object with `total` (AUD/week), `current` (each model channel), and optional
`floors`, `caps`, `protected_spend` (AUD/week) and `max_movement` (relative fractions).
For example, `0.1` allows 10% movement around current spend. Source restrictions
can tighten that limit. `--horizon 13 --draws 200 --seed 42` are defaults.

BLOCK models are rejected before solving. Otherwise the result preserves joint
posterior allocations, expected/conservative alternatives, conditional modeled
media revenue, downside, stability and historical-support warnings. An unstable
or extrapolated candidate cannot authorize a recommendation. This command analyzes
plans; it does not change spend in an advertising account.

## Local PostgreSQL API

Provide `DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost/decisionguard`
and optionally `ARTIFACT_ROOT=/absolute/path/to/artifacts`. Then:

```sh
uv run alembic upgrade head
uv run uvicorn decisionguard.api.app:create_app --factory --host 127.0.0.1
```

Open `/docs` for the typed API. Projects and generated datasets persist in
PostgreSQL; their quality endpoint verifies immutable artifact checksums. The
current slice supports project creation/retrieval, dataset generation and quality
evidence. Further model/job/scenario endpoints are still being implemented.
This local single-user API does not yet provide authentication.

Real PostgreSQL integration tests use an isolated schema per test and require a
dedicated database whose name ends in `_test`:

```sh
DECISIONGUARD_TEST_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost/decisionguard_test uv run pytest tests/test_api.py
```

Tests skip with an explicit reason when that variable is absent. SQLite is not
used as a substitute for PostgreSQL migration, JSONB or transaction verification.

Registered Bayesian training runs are queued through `POST /experiments/{id}/run`
after creating an experiment with `POST /projects/{id}/experiments`. Its dataset
and scientific configuration are snapshots. Run the queued job outside HTTP:

```sh
uv run decisionguard train --experiment-id EXPERIMENT_UUID
```

Do not combine registered training with dataset/output or sampler overrides.
Use `GET /model-runs/{id}` or `GET /projects/{id}/model-runs` to inspect lifecycle
and evidence. A successful job remains an unevaluated candidate; job success is
not scientific release. A crashed process may leave RUNNING metadata; current
recovery is an explicit operator task, never automatic duplicate execution.
