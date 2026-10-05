---
name: decision-stability
description: Evaluate whether an MMM is trustworthy enough to drive budget decisions. Use when connecting model validation, mmm-eval results, posterior uncertainty, optimization, scenario planning, or release gating.
---

# Decision Stability

## Goal

Prevent a model from driving consequential budget decisions merely because it predicts well.

DecisionGuard must answer two separate questions:

1. **Is the model statistically and scientifically trustworthy?**
2. **Are the decisions produced from that model stable enough to act on?**

A model can pass predictive validation and still produce unstable or unsafe allocations.

The downstream decision is part of the model evaluation problem.

---

## Core principle

Do not evaluate only:

`data -> model -> prediction`

Evaluate the complete chain:

`data -> model -> response estimates -> optimizer -> allocation`

A small change upstream should not create an unjustifiably large change downstream.

---

## Step 1 — Validate the model before the decision

Run the applicable model-level checks before optimization.

Prefer integrating Mutinex `mmm-eval` rather than recreating equivalent tests.

Relevant evaluation dimensions may include:

- holdout accuracy;
- in-sample accuracy;
- leave-future-out / temporal cross-validation;
- refresh stability;
- perturbation robustness;
- placebo / falsifiability tests.

Do not treat these as interchangeable.

Each tests a different failure mode.

### Predictive accuracy

Question:

> Can the model predict unseen outcomes reasonably well?

Necessary, but not sufficient.

### Refresh stability

Question:

> Do attribution/ROI estimates remain reasonably stable when new legitimate observations arrive?

Large unexplained changes may indicate weak identification, data problems, or model fragility.

### Perturbation robustness

Question:

> Does small realistic noise in the input produce disproportionately large changes in estimates?

### Placebo testing

Question:

> Does a signal with its temporal relationship deliberately destroyed still receive meaningful attribution?

A model that fails falsification should not drive consequential decisions.

---

## Step 2 — Separate hard gates from soft warnings

Not every failed metric should have the same consequence.

Classify evaluation outcomes as:

### BLOCK

The model must not drive optimization.

Examples:
- known leakage;
- failed placebo test;
- severe sampler pathology;
- broken data contract;
- unresolved unit/currency mismatch;
- catastrophic future holdout performance;
- optimizer input quantities not identifiable.

### RESTRICT

Optimization may run, but decision freedom is reduced.

Examples:
- one channel has elevated refresh instability;
- credible intervals are unusually wide;
- perturbation sensitivity is high;
- training data provide weak support for large spend changes.

### WARN

Decision can proceed but uncertainty must be surfaced.

Examples:
- moderate degradation in forecast error;
- small increase in posterior variance;
- mild sensitivity to a prior choice.

### PASS

No material issue detected for the evaluated criterion.

Do not create one opaque weighted score that hides blocking failures.

A single critical failure must not be averaged away by five good metrics.

---

## Step 3 — Evaluate channel-level trust

Model quality is not always uniform across channels.

For each decision-relevant channel, maintain evidence such as:

- posterior uncertainty;
- spend variation;
- response-curve identifiability;
- refresh stability;
- perturbation sensitivity;
- placebo behavior;
- training support;
- extrapolation distance.

Example:

| Channel | Predictive support | Refresh stability | Perturbation | Uncertainty | Status |
|---|---|---|---|---|---|
| Search | strong | strong | strong | low | PASS |
| TV | strong | moderate | strong | medium | PASS |
| Meta | strong | weak | moderate | high | RESTRICT |
| TikTok | weak | moderate | weak | high | RESTRICT |

Do not allow a globally "healthy" model label to hide a weak individual channel.

---

## Step 4 — Define decision stability

Decision stability asks:

> If we make small plausible changes to the data, model, or posterior draw, how much does the recommended allocation change?

Let:

- `x` = original data/model state;
- `a(x)` = optimized allocation;
- `x'` = plausible perturbed or refreshed state;
- `a(x')` = resulting allocation.

A simple normalized allocation instability metric is:

`DS(x, x') = ||a(x') - a(x)||_1 / B`

where:

- `||.||_1` is total absolute allocation movement;
- `B` is total budget.

Interpretation:

`DS = 0.05`

means 5% of the total budget would be moved differently under the alternative plausible state.

This metric is a starting point, not a universal truth.

Document why the selected distance measure matches the decision problem.

---

## Step 5 — Use multiple perturbation sources

Decision stability must not depend on one arbitrary perturbation.

Evaluate sensitivity across plausible sources such as:

### Data noise

Add realistic measurement noise to:
- spend;
- revenue;
- control variables.

Noise scale must reflect plausible data-quality uncertainty, not arbitrary percentages chosen to generate an interesting graph.

### Dataset refresh

Add newly arriving observations and refit.

Compare:
- ROI distributions;
- response curves;
- optimized allocations.

### Training-window sensitivity

Shift the historical window within defensible bounds.

### Prior sensitivity

For Bayesian models, test reasonable alternative priors.

### Model specification

Where justified, compare plausible model specifications.

### Posterior uncertainty

Optimize under multiple posterior draws rather than only posterior means.

### External control variation

Test sensitivity to uncertain macroeconomic or competitor variables where relevant.

---

## Step 6 — Use Monte Carlo decision stability

For stochastic or Bayesian models, do not evaluate stability from one perturbation.

Generate plausible states:

`x^(1), x^(2), ..., x^(N)`

and optimize each:

`a^(k) = optimize(x^(k))`

Then estimate the allocation distribution.

Report quantities such as:

- median allocation per channel;
- 5th/95th or other appropriate intervals;
- median decision instability;
- P95 decision instability;
- probability a channel changes direction;
- probability allocation movement exceeds a business threshold.

Example:

```text
Meta allocation

Median                  $1.20m
90% interval            $0.82m–$1.56m

P(increase vs current)     71%
P(change > 20%)            18%

Median decision instability  6.2%
P95 instability             21.8%