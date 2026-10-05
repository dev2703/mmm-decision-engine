---
name: scientific-ml
description: Choose and validate statistical/ML methods based on the question, assumptions, temporal structure, uncertainty, and downstream decision. Use for EDA, hypothesis tests, forecasting, classical ML, feature engineering, model comparison, and metrics.
---

# Scientific ML

## Goal

Make defensible modeling decisions rather than demonstrating a catalogue of algorithms.

The right technique is the simplest one that answers the actual question under defensible assumptions.

## Step 1 — Classify the question

Before selecting a method, classify the task.

### Descriptive
What happened?

### Predictive
What will happen?

### Inferential
What population parameter or relationship is plausible?

### Causal
What would happen under an intervention?

### Decisional
What action should be chosen under uncertainty and constraints?

Do not answer a causal or decisional question using predictive evidence alone.

## Step 2 — Write the estimand or target

State precisely what is being estimated.

Examples:
- next 4 weeks of revenue;
- incremental contribution of a channel under model assumptions;
- probability an allocation outperforms current plan;
- expected regret of a budget decision.

If the target is vague, stop and clarify it in the design.

## Step 3 — Establish baselines first

Always compare against a simpler baseline appropriate to the task.

Possible baselines:
- seasonal naive;
- mean/median;
- simple linear model;
- regularized linear model;
- ARIMA/ETS for forecasting;
- current business allocation.

Advanced methods must add something measurable:
- better generalization;
- calibrated uncertainty;
- better interpretability for the decision;
- robustness;
- causal structure;
- decision quality.

## Step 4 — Match validation to the data-generating process

For temporal prediction, prefer:
- expanding-window validation;
- rolling-origin validation;
- leave-future-out validation.

Do not randomly shuffle temporal observations unless exchangeability is justified.

Hyperparameter tuning must occur inside the validation design, not before it.

## Step 5 — Choose metrics from the decision

### Regression

- MAE: interpretable absolute error.
- RMSE: penalizes large errors more.
- MAPE: problematic near zero.
- sMAPE/WAPE: consider scale and use case.
- R²: descriptive fit, not a causal guarantee.

### Classification

Use precision/recall/F1 only when there is a genuine classification problem.

If false negatives are more costly, prioritize recall or expected cost rather than generic accuracy.

### Probabilistic

Consider:
- log predictive density;
- calibration;
- interval/coverage behavior;
- posterior predictive checks;
- expected utility/regret downstream.

Do not select metrics because they are common.

## Step 6 — Treat feature engineering as modeling

For marketing/time series, consider domain transformations such as:

### Seasonality
- calendar effects;
- holidays;
- Fourier terms where appropriate.

### Carryover/adstock

A generic geometric form:

`A_t = x_t + λ x_{t-1} + λ² x_{t-2} + ...`

### Saturation

Use an appropriate nonlinear response such as Hill/logistic-type transformations when supported by the model.

### External controls

Price, promotions, macro conditions, competitor activity, distribution, availability.

Do not engineer features using future information.

## Step 7 — Use statistical tests deliberately

Before a test, state:

- null/alternative;
- assumptions;
- dependence structure;
- multiple-testing implications;
- effect size of practical interest.

A t-test may be inappropriate when observations are serially correlated.

Alternatives may include:
- block bootstrap;
- permutation tests with valid exchangeability;
- regression with appropriate error structure;
- Bayesian posterior comparison;
- change-point analysis.

ANOVA does not establish causality.

A p-value does not measure effect importance.

## Step 8 — Handle dimensionality with interpretability in mind

Do not apply PCA merely because predictors are correlated.

Ask:
- is conditioning actually a problem?
- would regularization suffice?
- does downstream interpretation require original features?

For media channels, preserving channel identity may be more valuable than compression.

PCA may be more defensible for a large correlated set of external macro controls.

## Step 9 — Compare models by role, not one leaderboard

A useful Decision Engine ladder:

1. Seasonal naive baseline.
2. Classical forecast baseline such as ARIMA/ETS.
3. Domain-engineered interpretable regression.
4. Nonlinear predictive benchmark such as gradient boosting.
5. Probabilistic MMM.

Do not automatically promote the lowest-error model to the production decision model.

Example:

XGBoost may be retained as a predictive benchmark while the Bayesian MMM is used for probabilistic channel contribution and decision analysis.

## Step 10 — Separate prediction from attribution

Never infer:

`feature is important for prediction => feature caused outcome`

Specifically:
- SHAP is not causal attribution;
- feature importance is not incrementality;
- correlation is not intervention response;
- predictive accuracy is necessary but not sufficient for trustworthy allocation.

## Step 11 — Quantify uncertainty

When a downstream decision depends on a model estimate, propagate uncertainty rather than using only point estimates.

Prefer reporting:
- posterior/interval distributions;
- probability of exceeding a threshold;
- probability one option beats another;
- expected regret;
- downside risk.

## Step 12 — Falsify

Ask what result would make the model or conclusion unacceptable.

Examples:
- placebo channel retains strong attributed ROI;
- ROI changes drastically under small perturbations;
- refresh with a few new observations reverses recommendations;
- model fails future holdout;
- residual structure remains strongly autocorrelated.

## Experiment record

For every meaningful experiment capture:

### Hypothesis
...

### Change
...

### Validation design
...

### Metric(s)
...

### Result
...

### Interpretation
...

### Decision
keep / reject / investigate

### Limitations
...

## Completion checklist

- [ ] Question type is explicit.
- [ ] Target/estimand is explicit.
- [ ] Baseline exists.
- [ ] Validation respects time/dependence.
- [ ] Metric matches the decision.
- [ ] Assumptions are discussed.
- [ ] Leakage was checked.
- [ ] Predictive claims are not misrepresented as causal.
- [ ] Uncertainty is retained when consequential.
- [ ] A falsification path exists.