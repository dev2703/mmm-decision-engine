---
name: bayesian-workflow
description: Build, diagnose, compare, and report probabilistic models using a disciplined Bayesian workflow. Use for PyMC/ArviZ-style MMMs, priors, posterior diagnostics, uncertainty propagation, and probabilistic decisions.
---

# Bayesian Workflow

## Goal

Treat Bayesian modeling as an iterative scientific workflow, not `fit()` followed by posterior means.

## Step 1 — Define the generative story

Before coding, describe:

- observed variables;
- latent parameters;
- likelihood;
- priors;
- temporal structure;
- hierarchical structure;
- transformations such as adstock/saturation;
- what each parameter means in business terms.

If the generative story is unclear, the implementation is premature.

## Step 2 — Standardize parameterization where helpful

Improve sampling geometry when useful:

- center/scale covariates using training statistics;
- use non-centered hierarchical parameterizations where appropriate;
- avoid pathological unconstrained parameterizations;
- preserve mapping back to business units.

Do not standardize blindly.

## Step 3 — Choose priors deliberately

For each consequential prior answer:

- what values are plausible before observing current data?
- what values are implausible?
- is the prior weakly informative, informative, or regularizing?
- what domain knowledge justifies it?
- could the prior dominate a small dataset?

Avoid both:
- absurdly broad "uninformative" priors;
- overly narrow priors that predetermine the answer.

## Step 4 — Run prior predictive checks

Before fitting:

1. sample from priors;
2. generate outcomes;
3. inspect whether generated worlds are plausible.

Reject priors that produce impossible sales, absurd ROI, or nonsensical response curves.

## Step 5 — Fit with reproducibility

Record:
- model version;
- data version;
- sampler;
- chains;
- draws;
- tuning;
- target acceptance settings;
- random seed(s);
- environment/package versions where material.

## Step 6 — Inspect sampler diagnostics

At minimum inspect:
- R-hat;
- effective sample size;
- divergences;
- tree depth or sampler warnings when available;
- energy diagnostics where useful;
- trace behavior for critical parameters.

Do not report business conclusions from a model with unresolved pathological diagnostics.

Do not treat a single threshold mechanically.

Investigate context and severity.

## Step 7 — Run posterior predictive checks

Ask whether posterior-generated data reproduce the structures relevant to the use case:

- level;
- variance;
- seasonality;
- trend;
- tail behavior;
- promotional spikes;
- held-out future behavior.

A model can have "converged" while still being a poor model of the data.

## Step 8 — Check identifiability and confounding

Marketing channels are often correlated.

Investigate:
- posterior correlations;
- wide/unstable channel effects;
- sensitivity to priors;
- collinearity;
- weak variation in spend;
- channels that move together.

Do not hide non-identifiability behind precise-looking posterior summaries.

## Step 9 — Validate externally

Use predictive and robustness evaluation such as:
- holdout;
- leave-future-out CV;
- refresh stability;
- perturbation tests;
- placebo/falsifiability tests;
- experiment calibration when legitimate external experimental evidence exists.

Integrate Mutinex `mmm-eval` where supported rather than duplicating its tests.

## Step 10 — Sensitivity analysis

For consequential conclusions, test sensitivity to:
- reasonable prior changes;
- training window;
- outlier treatment;
- alternative lag/adstock assumptions;
- control-variable inclusion;
- model specification.

If the decision flips under small plausible changes, surface that instability.

## Step 11 — Propagate posterior uncertainty

Do not reduce posterior distributions to point estimates before optimization.

For posterior draws `θ^(k)` and allocation `a`, evaluate:

`Outcome^(k)(a) = f(a, θ^(k))`

Then summarize:
- expected outcome;
- probability of beating current plan;
- downside probability;
- credible intervals;
- expected regret;
- CVaR or another risk metric if justified.

## Step 12 — Report honestly

Prefer statements like:

"Given the model and assumptions, the posterior probability that Meta ROI exceeds 1 is 0.93."

Avoid:

"Meta definitely causes $3.20 for every dollar."

Always distinguish:
- model-based uncertainty;
- structural/model uncertainty;
- assumptions not captured by the posterior.

## Blocking conditions

Do not release a model for consequential optimization when:
- serious sampler pathologies remain unresolved;
- posterior predictive checks fail materially;
- key conclusions are non-identifiable;
- robustness/placebo tests fail;
- small updates cause extreme decision instability without explanation.

## Completion checklist

- [ ] Generative story is explicit.
- [ ] Priors are justified.
- [ ] Prior predictive checks passed.
- [ ] Sampler diagnostics reviewed.
- [ ] Posterior predictive checks reviewed.
- [ ] Identifiability/confounding assessed.
- [ ] External validation executed where appropriate.
- [ ] Sensitivity analysis performed for consequential claims.
- [ ] Posterior uncertainty flows into decisions.