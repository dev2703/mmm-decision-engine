Product Requirements Document

1. Product Summary

mmm-decision-engine is a marketing-model decision-support application.

Its primary question is:

Can I trust this model enough to change the marketing budget, and what action is justified by the available evidence?

The product combines:

data quality;

statistical analysis;

model comparison;

Bayesian MMM;

Mutinex mmm-eval;

model release gating;

budget optimization;

decision stability;

evidence-grounded AI assistance.

2. Target Users

Applied ML Engineer / Data Scientist

Needs to:

understand the dataset;

reproduce transformations;

detect leakage;

compare models;

inspect diagnostics;

understand why validation failed;

defend model selection.

Marketing Effectiveness / Analytics Lead

Needs to:

understand channel performance;

identify weakly supported recommendations;

compare current vs proposed budget;

inspect uncertainty;

understand why movement is restricted.

CMO / Commercial User

Needs to answer:

Where should we spend more?

Where should we spend less?

How certain are we?

How much could we lose?

Why is Meta restricted?

What happens if my total budget falls 10%?

What happens if I protect brand investment?

3. Product Principles

Evidence before recommendation

Recommendations must always be traceable to evidence.

Uncertainty is first-class

Uncertainty must appear directly in the product.

Not hidden in:

advanced settings.

Explain restrictions

Prefer:

Meta increases are limited to 5% because ROI changed materially after the last refresh and the requested spend moves beyond strong historical support.

over:

Trust score 52%.

Progressive disclosure

Executives should understand the result quickly.

Technical users should be able to inspect:

raw metrics;

assumptions;

diagnostics;

data transformations.

AI is not the source of truth

The agent:

chooses tools;

translates intent;

explains evidence.

Deterministic code:

computes;

validates;

optimizes;

enforces policy.

4. Primary User Journey

Step 1 — Project Overview

Display:

dataset version;

analysis period;

current decision-model status;

latest model run;

current marketing budget;

critical warnings;

current recommendation summary.

Main call to action:

Review Data

Step 2 — Data Quality

Display:

missing periods;

duplicate records;

taxonomy inconsistencies;

currency problems;

suspicious values;

structural breaks;

transformation history.

Each issue must have a status:

RESOLVED
WARNING
BLOCKER

Example:

Raw:
FB

Normalized:
Meta

Reason:
known taxonomy alias

Another example:

Revenue missing:
2025-03-16

Status:
BLOCKER

Reason:
Target value unavailable and cannot be safely imputed.

5. Model Lab

Purpose:

Show what we tried and why.

Example:

Model

Purpose

Future Error

Uncertainty

Decision Use

Seasonal Naive

baseline

18.4%

no

no

ARIMA

forecast baseline

14.9%

forecast interval

no

Ridge

interpretable benchmark

12.8%

limited

diagnostic

Gradient Boosting

nonlinear benchmark

9.7%

limited

predictive only

Bayesian MMM

decision model

10.4%

posterior

yes if validated

Important message:

The model with the lowest prediction error is not automatically the best decision model.

Each experiment should expose:

hypothesis;

features;

validation method;

metrics;

results;

decision;

limitations.

6. Model Health

Top-level states:

PASS
WARN
RESTRICT
BLOCK

Display separate evidence cards for:

predictive accuracy;

temporal cross-validation;

refresh stability;

perturbation stability;

placebo test;

Bayesian diagnostics;

posterior predictive checks;

extrapolation.

Each failure/warning explains:

what was tested;

result;

expected threshold/policy;

why it matters;

what downstream action changes.

7. Decision Lab

Inputs:

total budget;

current allocation;

channel minimums;

channel maximums;

max movement;

brand-investment floor;

risk tolerance.

Display:

Channel

Current

Proposed

Movement

Evidence

Search

$X

$Y

+Z%

PASS

Meta

$X

$Y

+Z%

RESTRICT

TV

$X

$Y

-Z%

PASS

Summary:

expected revenue;

interval;

probability of beating current plan;

downside probability;

total money moved;

extrapolation warnings;

decision instability.

8. Risk Alternatives

Where appropriate show:

Maximum Expected Value

Expected improvement: +4.7%
Probability of beating current: 63%
Downside exposure: higher

Risk-Adjusted

Expected improvement: +3.8%
Probability of beating current: 89%
Downside exposure: lower

Neither should automatically be called:

best.

The product should explain the tradeoff.

9. Ask mmm-decision-engine

Example user query:

Can we increase Meta by 30%?

Expected process:

user request
    ↓
get_model_health()
    ↓
get_channel_evidence("Meta")
    ↓
run_scenario(...)
    ↓
validate_constraints(...)
    ↓
get_decision_stability(...)
    ↓
answer

Possible response:

A 30% increase is not currently supported.

Meta's ROI estimate showed elevated refresh instability and the requested allocation moves beyond the strongest region of historical spend support.

Under the requested scenario, the probability of outperforming the current plan is X%, compared with Y% for a 10% increase.

Current policy therefore restricts Meta increases to Z%.

Every numerical statement must originate from deterministic tool output.

10. Product Information Architecture

Overview

Question:

What should I care about right now?

Data Quality

Question:

Can I trust the inputs?

Model Lab

Question:

What modeling approaches did we test and what did we learn?

Model Health

Question:

Can the selected model support business decisions?

Decision Lab

Question:

Given the model and constraints, what allocation is justified?

Ask mmm-decision-engine

Question:

Can I explore decisions conversationally without bypassing governance?

Runs

Question:

What exactly produced this result?

11. Visual Direction

mmm-decision-engine should look like an analytical B2B platform.

Desired

restrained;

sophisticated;

clear typography;

high-information density;

generous whitespace;

good tables;

useful charts;

clear warning states.

Avoid

neon AI gradients;

giant chatbot homepage;

unnecessary cards;

3D charts;

gauges;

decorative animations;

fake AI confidence;

excessive red/green.

12. Chart Requirements

Actual vs Predicted

Answers:

Where is the model right or wrong?

Residuals Through Time

Answers:

Is there systematic structure left unexplained?

Response Curves

Answers:

How does marginal return change with spend?

Posterior ROI Distribution

Answers:

How uncertain is the ROI estimate?

Refresh Stability

Answers:

How much did channel conclusions change after additional data?

Decision Stability Distribution

Answers:

How much does the recommended budget vary under plausible model uncertainty?

Current vs Proposed Budget

Answers:

Where does money move?

Historical Support

Answers:

Is the optimizer recommending spend levels outside observed evidence?

13. Agent UX

For every decision answer, expose:

tools called;

evidence/run IDs;

warnings;

policy state;

scenario ID;

assumptions.

The agent must be comfortable saying:

insufficient evidence;

model blocked;

scenario unstable;

requested allocation extrapolates too far;

optimization is infeasible.

14. Error States

Data Blocked

Example:

Modeling is blocked because the revenue dataset contains unresolved mixed-currency observations.

Show:

affected observations;

expected remediation.

Model Blocked

Example:

Optimization is unavailable because the placebo validation failed.

Show:

test;

result;

interpretation;

recommended investigation.

Infeasible Optimization

Example:

No allocation satisfies all constraints.

Show:

conflicting constraints;

possible minimal relaxation.

Agent Tool Failure

Example:

mmm-decision-engine could not retrieve model-health evidence, therefore no recommendation was produced.

Never fabricate fallback numbers.