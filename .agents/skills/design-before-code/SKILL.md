---
name: design-before-code
description: Pressure-test non-trivial changes before implementation. Use when architecture, data contracts, public APIs, model methodology, or multi-module behavior will change.
---

# Design Before Code

## Goal

Avoid expensive wrong turns without turning small changes into planning bureaucracy.

Do not invoke for trivial edits.

## Trigger conditions

Use this skill when one or more are true:

- a change crosses multiple modules;
- a public API or data schema changes;
- model methodology changes;
- a new runtime dependency is proposed;
- a new persistence boundary is introduced;
- a consequential agent tool/action is added;
- an irreversible or expensive design choice is being made.

## Step 1 — State the problem

Write one paragraph answering:

- What user/business problem exists?
- What behavior is missing today?
- What evidence shows this is worth solving?

Do not start with a preferred technology.

## Step 2 — Define the constraints

Capture only real constraints:

- latency;
- reliability;
- data volume;
- reproducibility;
- interpretability;
- uncertainty;
- privacy/security;
- deployment target;
- operational cost;
- time available.

Separate:
- must have;
- nice to have;
- unknown.

## Step 3 — Inspect the current system

Before proposing architecture:

- identify the current path through the code;
- locate extension points that already exist;
- note existing libraries and conventions;
- identify invariants that must remain true;
- identify what would become duplicated if we build a new path.

## Step 4 — Compare a small set of options

Usually compare 2–3 approaches.

For each option state:

- simplest description;
- benefits;
- risks;
- complexity added;
- what becomes harder later;
- what assumptions must hold.

Do not generate fake alternatives only to justify the preferred option.

## Step 5 — Choose the smallest sufficient design

Prefer:
- fewer moving parts;
- existing boundaries;
- reversible decisions;
- explicit control flow;
- deterministic behavior around high-risk operations.

Reject:
- event buses for one producer/consumer;
- microservices for module-boundary problems;
- plugin architectures for one implementation;
- generic workflow engines for a fixed flow;
- databases when an immutable artifact/file suffices.

## Step 6 — Define acceptance criteria before implementation

Acceptance criteria must be observable.

Good:
- temporal preprocessing never fits on validation observations;
- optimizer respects channel bounds and total budget;
- model failing a blocking validation cannot reach execution;
- agent answer cites tool evidence for every numerical claim.

Bad:
- code is clean;
- architecture is scalable;
- AI is intelligent.

## Step 7 — Identify failure modes

Ask:

- How can inputs be malformed?
- What happens with missing/late data?
- How can the model be wrong?
- What happens if optimization is infeasible?
- What happens if an LLM selects the wrong tool?
- What happens if an external service times out?
- What happens if we retry?

Design explicit failure behavior.

## Step 8 — Record the decision

For meaningful decisions, create a short ADR with:

- context;
- decision;
- alternatives;
- consequences;
- revisit trigger.

Keep ADRs short.

Do not document obvious implementation trivia.

## Output template

### Problem
...

### Constraints
...

### Existing system
...

### Options

1. ...
2. ...

### Decision
...

### Acceptance criteria
- ...

### Failure modes
- ...

### Revisit when
...