Project Scope

1. Purpose

mmm-decision-engine is a portfolio-grade applied machine-learning system for evaluating whether a Marketing Mix Model (MMM) is trustworthy enough to influence a commercial budget decision, and for producing a risk-aware allocation when the evidence is strong enough.

mmm-decision-engine is intentionally not a clone of Mutinex GrowthOS.

Its narrower thesis is:

A model should not be allowed to move money merely because it predicts well. Model validity, uncertainty, robustness, and downstream decision stability must all influence what the optimizer is allowed to recommend.

The complete path should be auditable:

raw data
    ↓
validated data
    ↓
cleaned/model-ready data
    ↓
candidate models
    ↓
model validation
    ↓
uncertainty
    ↓
budget optimization
    ↓
decision stability
    ↓
recommendation
    ↓
explanation / governance

2. Why This Project Exists

The project should demonstrate practical depth across four areas without becoming a collection of disconnected demos.

Data Science

Demonstrate:

messy-data reasoning;

data profiling;

missingness analysis;

distributions;

statistical assumptions;

outlier reasoning;

leakage-safe preprocessing;

categorical encoding decisions;

normalization and standardization decisions;

feature engineering;

time-series reasoning;

forecasting;

model comparison;

uncertainty;

experimentation.

Machine Learning

Demonstrate:

simple baselines before complex models;

classical forecasting;

linear and regularized models;

nonlinear predictive models;

Bayesian MMM;

temporal validation;

hyperparameter tuning;

robustness;

falsification;

probabilistic reasoning.

Agentic AI / GenAI

Demonstrate:

tool calling;

typed tool schemas;

deterministic calculations;

context engineering;

prompt/data separation;

security guardrails;

agent evaluation;

tool-trace telemetry;

prompt engineering;

grounded explanations;

human approval for consequential actions.

Software Engineering / MLOps

Demonstrate:

maintainable Python;

FastAPI;

PostgreSQL;

SQLAlchemy;

Alembic;

reproducible experiments;

Docker;

CI;

type checking;

linting;

testing;

structured logs;

deployment;

model/version provenance.

The objective is not to demonstrate every technique.

The objective is to demonstrate:

why a technique was selected, rejected, or constrained.

3. Core Product Question

For a given marketing dataset and fitted MMM:

Is this model reliable enough to support a budget decision, and if so, what allocation is justified by the evidence and business constraints?

This decomposes into five questions:

Is the underlying data trustworthy?

Does the model generalize to future observations?

Are the estimated channel effects sufficiently stable and identifiable?

Are the recommendations stable under plausible uncertainty?

Can the recommendation be explained and governed safely?

4. MVP Scope

4.1 Synthetic Marketing Dataset

mmm-decision-engine includes a reproducible synthetic data generator.

The synthetic world contains known ground truth so we can evaluate whether models recover meaningful patterns rather than merely generating attractive charts.

The dataset should contain weekly observations including:

revenue or sales;

Search spend;

Meta spend;

TV spend;

OOH spend;

one additional digital channel;

price;

promotions;

macroeconomic indicator;

competitor activity;

seasonality;

trend.

Each media channel should have its own:

spend process;

carryover/adstock;

saturation behaviour;

ground-truth contribution.

At least two channels should be partially correlated.

4.2 Dirty Data Layer

The clean ground-truth data is then passed through a separate corruption layer.

Possible defects include:

missing weeks;

duplicate rows;

taxonomy aliases such as Facebook, FB, and Meta;

mixed currencies;

impossible negative spend;

tracking outage;

late-arriving observations;

campaign rename;

unit inconsistency;

genuine commercial outlier;

erroneous data outlier;

structural regime shift.

The corruption layer must be reproducible from configuration and seed.

Ground truth must never be destroyed.

4.3 Data Integrity Pipeline

mmm-decision-engine must produce:

raw dataset
+
data-quality report
+
cleaned dataset
+
transformation provenance

The pipeline should reason about:

observation grain;

primary keys;

temporal frequency;

missingness semantics;

categories;

units;

currency;

duplicates;

joins;

outliers;

leakage boundaries.

Not every anomaly should be automatically fixed.

Ambiguous situations should produce:

warning;

blocker;

request for intervention.

4.4 Model Ladder

mmm-decision-engine deliberately compares different kinds of models.

Model 0 — Seasonal Naive

Purpose:

Establish whether sophisticated models beat an extremely simple baseline.

Model 1 — Classical Time-Series Model

ARIMA / SARIMA / ETS depending on the generated process.

Purpose:

Establish a traditional statistical forecasting baseline.

Model 2 — Domain-Engineered Linear Model

Potential features:

trend;

seasonal terms;

promotions;

price;

controls;

adstock;

saturation.

Purpose:

Demonstrate how good feature engineering and data preparation can make a simple model competitive.

Model 3 — Nonlinear Predictive Benchmark

Start with scikit-learn Histogram Gradient Boosting.

Purpose:

Measure how much nonlinear predictive performance we gain.

This model is a predictive benchmark, not a causal attribution model.

Its feature importance must never automatically be interpreted as incremental marketing ROI.

Model 4 — Probabilistic MMM

Use PyMC-Marketing.

Purpose:

probabilistic channel contribution;

response curves;

posterior uncertainty;

priors;

posterior predictive checks;

decision propagation.

This becomes the principal decision model only if validation supports it.

4.5 MMM Evaluation

mmm-decision-engine integrates Mutinex's open-source mmm-eval.

Use existing evaluation functionality instead of duplicating it.

Relevant tests include:

in-sample accuracy;

holdout accuracy;

temporal cross-validation;

refresh stability;

perturbation testing;

placebo/falsifiability testing.

Raw results must be preserved.

Do not convert everything into one mysterious number.

4.6 Model Release Policy

Every decision model receives one of:

PASS
WARN
RESTRICT
BLOCK

Examples:

BLOCK

known leakage;

broken data contract;

serious placebo failure;

catastrophic diagnostics;

unresolved unit/currency corruption.

RESTRICT

high refresh instability;

weakly identified channel;

high perturbation sensitivity;

major extrapolation;

unstable downstream allocation.

WARN

moderate degradation;

elevated uncertainty;

non-critical sensitivity.

PASS

Required checks are within policy.

The policy is deterministic code.

The LLM cannot override it.

4.7 Decision Stability

mmm-decision-engine extends model evaluation into decision evaluation.

Core question:

If small plausible changes to the data or model produce huge changes in recommended spending, should we trust the recommendation?

For baseline state (x):

[
a(x) = \text{optimized allocation}
]

For plausible alternative state (x'):

[
a(x') = \text{alternative allocation}
]

Define a simple normalized instability metric:

[
DS(x,x') =
\frac{|a(x') - a(x)|_1}{B}
]

where:

[
B = \text{total budget}
]

This metric is a project hypothesis, not a claimed industry standard.

Decision stability should be evaluated across:

posterior samples;

input perturbations;

model refreshes;

reasonable prior changes where appropriate;

training-window changes where appropriate.

Report:

median decision instability;

P95 instability;

allocation distribution by channel;

direction-flip probability;

probability of large movement.

4.8 Budget Optimization

The optimizer supports:

total budget;

channel minimum;

channel maximum;

maximum movement from current allocation;

protected brand floor;

restricted-channel movement;

risk preference.

Possible outputs:

maximum expected-value allocation;

risk-adjusted allocation.

The system should estimate:

expected outcome;

uncertainty;

probability of beating current allocation;

downside probability;

extrapolation risk.

4.9 Agent

mmm-decision-engine contains one agent.

Not a multi-agent swarm.

The agent may:

explain data-quality problems;

compare models;

inspect model health;

retrieve channel evidence;

run budget scenarios;

validate constraints;

retrieve decision-stability results;

explain why a decision is blocked or restricted.

The agent may not:

calculate production statistics itself;

execute arbitrary Python;

execute shell commands;

bypass model policy;

make live advertising changes;

treat data as trusted instructions.

The agent is an interface to deterministic software.

It is not the source of truth.

4.10 Dashboard

Primary product areas:

Overview

Data Quality

Model Lab

Model Health

Decision Lab

Ask mmm-decision-engine

Runs / Experiment History

The application should remain useful even if the AI assistant is removed.

5. Explicit Non-Goals

The MVP does not attempt to build:

full GrowthOS clone;

live Meta/Google/The Trade Desk integrations;

automated ad execution;

enterprise multi-tenancy;

enterprise RBAC;

Kafka;

Kubernetes;

streaming architecture;

feature store;

microservice architecture;

custom Bayesian sampling framework;

custom MMM framework;

custom LLM framework;

multi-agent architecture;

reinforcement-learning budget allocation;

general-purpose BI system;

general-purpose data-cleaning platform.

6. Scientific Success Criteria

The project should make it possible to clearly defend:

Why not shuffled K-fold?

Where could leakage occur?

Why scale this feature?

Why not scale that model?

Why one-hot encode this category?

Why not one-hot encode campaign IDs?

Why use or reject PCA?

Why ARIMA?

Why maintain a naive baseline?

What does feature engineering contribute?

Why doesn't XGBoost feature importance establish causality?

Why Bayesian MMM?

Why these priors?

What does posterior predictive checking tell us?

What does mmm-eval provide beyond normal error metrics?

Why can a good model still produce a bad decision?

Why can a stable model produce an unstable optimizer?

How does uncertainty change the recommended allocation?

7. Engineering Success Criteria

The repository should:

target Python 3.12;

use uv;

commit the dependency lock;

use Ruff;

use Pyright;

use pytest;

use meaningful property tests;

expose FastAPI;

use PostgreSQL;

use Alembic migrations;

Dockerize the backend;

record model runs;

record dataset versions;

record evaluation runs;

use structured logs;

expose agent tool traces;

support reproducible local execution.

Production logic must not exist only in notebooks.

8. Product Quality Bar

The product should feel:

analytical;

calm;

precise;

technically credible;

commercially understandable;

uncertainty-aware.

Avoid:

over-designed AI visuals;

fake confidence scores;

unexplained gauges;

green dashboards hiding failed tests;

chatbot-first UX.

9. Scope-Control Rule

A feature enters the MVP only if it strengthens at least one of:

understanding the data;

understanding the model;

understanding uncertainty;

understanding the downstream decision;

safely operationalizing the workflow.

Otherwise:

defer it.