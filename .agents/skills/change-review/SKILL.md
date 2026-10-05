---
name: change-review
description: Review a completed change across specification, engineering quality, scientific validity, security, and operational risk. Use before merge or release of meaningful changes.
---

# Change Review

## Goal

Review the diff as a skeptical staff engineer and applied scientist.

Do not manufacture stylistic comments to appear productive.

## Step 1 — Understand intent

Read:
- issue/spec;
- relevant ADR;
- changed files in full where feasible;
- tests;
- surrounding call sites.

Summarize what the change is intended to accomplish in one paragraph.

## Step 2 — Review against specification

Ask:
- Does the behavior actually meet the requirement?
- Are important edge cases missing?
- Did scope creep introduce unrelated behavior?
- Are acceptance criteria covered?

## Step 3 — Engineering review

Look for:

### Accidental complexity
- speculative generality;
- one-implementation interfaces;
- pass-through wrappers;
- unnecessary factories;
- indirection without policy;
- generic managers/services;
- configuration sprawl.

### Maintainability
- unclear naming;
- hidden side effects;
- mixed responsibilities;
- duplicated logic;
- shotgun-surgery risk;
- inappropriate coupling;
- dead code.

### Dependency hygiene
- unused dependency;
- library reimplementation;
- unnecessary heavyweight dependency.

Ask the deletion question:

"What can be removed while preserving behavior and clarity?"

## Step 4 — Scientific validity review

For data/model changes inspect:

- temporal leakage;
- preprocessing fit boundary;
- target leakage;
- invalid random CV;
- metric/task mismatch;
- cherry-picked model comparisons;
- unsupported causal language;
- untested statistical assumptions;
- multiple-testing issues;
- discarded uncertainty;
- instability hidden by averages;
- baseline absence;
- experiment reproducibility.

High predictive performance does not override invalid methodology.

## Step 5 — Bayesian/model review

Where relevant:
- priors justified?
- diagnostics clean?
- posterior predictive checks?
- identifiability/confounding?
- sensitivity analysis?
- `mmm-eval` or equivalent robustness checks?
- decision stability?

## Step 6 — Security review

Trace untrusted inputs to sensitive sinks.

Check:
- SQL construction;
- filesystem paths;
- deserialization;
- credentials/secrets;
- external URLs;
- shell execution;
- arbitrary Python execution;
- LLM tool parameters;
- prompt/data separation;
- privilege boundaries.

For consequential changes, estimate blast radius:
- what can this component read?
- what can it mutate?
- what happens if it is wrong?
- is the action idempotent?
- is approval required?

## Step 7 — Operational review

Ask:
- failure behavior explicit?
- timeouts?
- retries safe?
- logs useful without leaking secrets?
- metrics/traces sufficient?
- long-running work handled appropriately?
- artifact/version traceability maintained?

## Severity model

### Blocker

Can cause:
- incorrect business action;
- invalid science;
- security exposure;
- data corruption;
- major reliability failure.

### Major

Likely bug, significant maintainability problem, or important missing test.

### Minor

Concrete improvement with low risk.

Do not block on subjective style preferences already handled by formatters/linters.

## Required final review output

### Summary
...

### Blockers
- ...

### Major issues
- ...

### Minor issues
- ...

### What is good
- ...

### Simplification opportunities
- ...

### Verification still required
- ...