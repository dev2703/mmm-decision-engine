---
name: verification-debugging
description: Verify software and scientific behavior and debug from reproducible evidence. Use for tests, regressions, flaky behavior, model failures, data bugs, and incident-style investigation.
---

# Verification and Debugging

## Goal

Replace speculation with a tight feedback loop.

# Part A — Verification strategy

Different work requires different verification.

## Deterministic software

Prefer:
- unit tests for domain logic;
- property tests for invariants;
- integration tests at external boundaries;
- API contract tests.

## Data pipelines

Verify:
- schema;
- row/cardinality invariants;
- temporal ordering;
- train-only preprocessing;
- deterministic transformations;
- expected handling of missing/late data.

## Scientific/modeling code

Verify:
- reproducibility;
- experiment configuration;
- diagnostic thresholds;
- synthetic-data recovery where feasible;
- holdout behavior;
- robustness tests.

## Agent systems

Verify:
- tool selection;
- tool argument schemas;
- forbidden-action handling;
- evidence grounding;
- policy compliance;
- prompt-injection resistance.

## Test meaningful boundaries

Prefer tests such as:

- allocation always sums to total budget within tolerance;
- channel min/max bounds always hold;
- infeasible constraints return a clear failure;
- future observations never affect fitted preprocessing;
- failed model gate blocks optimization/execution;
- shuffled placebo signal does not retain stable attribution;
- malicious text in campaign names is treated as data;
- same data/config/seed reproduces the same deterministic artifact where expected.

Avoid:
- tests for trivial getters;
- tests that mirror implementation line by line;
- mocks of every internal call;
- coverage-only tests.

# Part B — Debugging workflow

## Step 1 — Reproduce

Create the smallest deterministic reproduction possible.

Record:
- input;
- environment;
- seed;
- version/commit;
- exact observed vs expected behavior.

If you cannot reproduce, gather evidence before changing code.

## Step 2 — Minimize

Reduce:
- dataset;
- function path;
- number of services;
- model configuration;
- agent tools.

Keep the failure.

## Step 3 — State competing hypotheses

Write 2–4 plausible causes.

Do not immediately patch the first plausible line.

Examples:
- join duplicated rows;
- scaler fitted on full data;
- wrong timezone alignment;
- posterior sampler instability;
- optimizer units mismatch;
- stale cached artifact;
- LLM selected a valid but semantically wrong tool.

## Step 4 — Instrument to discriminate

Add the smallest measurement that separates hypotheses.

Examples:
- row counts before/after join;
- fitted scaler means;
- model version hash;
- constraint residuals;
- posterior diagnostic summary;
- tool-call trace.

Remove temporary instrumentation if it has no ongoing operational value.

## Step 5 — Fix root cause

Prefer a fix that restores the violated invariant.

Do not:
- add sleeps for race conditions without understanding them;
- broaden exception handling to hide failures;
- change thresholds solely to make tests pass;
- suppress model diagnostics without explanation.

## Step 6 — Add regression protection

Add a test or check that would have caught the defect earlier.

The regression test should target the violated behavior, not the exact previous implementation.

## Step 7 — Simplify after fixing

Debug patches often accumulate scaffolding.

Review for:
- temporary flags;
- duplicated checks;
- excessive logs;
- compatibility code no longer needed.

## Failure report template

### Symptom
...

### Minimal reproduction
...

### Root cause
...

### Violated invariant
...

### Fix
...

### Regression protection
...

### Remaining risk
...