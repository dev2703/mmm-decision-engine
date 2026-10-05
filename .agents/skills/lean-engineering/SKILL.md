name: lean-engineering
description: Build maintainable software with minimal accidental complexity. Use for implementation, refactoring, dependency choices, module boundaries, and code-structure decisions.
---

# Lean Engineering

## Goal

Produce the smallest clear implementation that solves the current requirement and remains easy to change.

Do not optimize for line count alone. Optimize for low accidental complexity.

## Step 1 — Search before building

Before adding code:

1. Search for existing implementations, helpers, patterns, tests, and dependencies.
2. Check whether the current module can absorb the behavior cleanly.
3. Check the Python standard library.
4. Check already-installed dependencies.
5. Consider a mature external library only if it materially reduces maintained complexity.
6. Write custom infrastructure only when the above are insufficient.

Do not reimplement:
- HTTP serving;
- schema validation;
- optimization solvers;
- Bayesian samplers;
- retries;
- tracing protocols;
- serialization formats;
- database connection pools;
- common dataframe operations.

## Step 2 — Challenge the abstraction

Before introducing an interface, base class, factory, adapter, registry, plugin system, or service layer, answer:

- What current variation does this abstraction represent?
- Are there at least two real implementations or a hard external boundary?
- Does it hide substantial complexity behind a smaller interface?
- Would deleting the abstraction make the code easier to understand without increasing coupling?

Prefer deep modules: small interfaces that hide meaningful behavior.

Avoid shallow chains such as:

`Controller -> Service -> Manager -> Repository -> Adapter -> library_call()`

when the intermediate layers add no policy, invariants, or behavior.

## Step 3 — Prefer simple language constructs

Prefer, in order:

- pure functions;
- small stateful objects when state/lifecycle is real;
- composition;
- protocols/interfaces when multiple implementations actually exist;
- inheritance only when subtype behavior is genuinely useful.

Do not create:
- classes containing only static methods;
- one-implementation abstract base classes;
- factories that build one object;
- generic `utils.py` or `helpers.py` dumping grounds;
- wrappers that only forward arguments;
- configuration frameworks for a handful of values.

## Step 4 — Keep modules cohesive

A module should answer one recognizable question.

Good examples:
- `data/validation.py`
- `models/adstock.py`
- `evaluation/decision_stability.py`
- `optimization/constraints.py`

Bad examples:
- `common.py`
- `misc.py`
- `helpers.py`
- `manager.py`

A file is not automatically too large because it has many lines.

Split when responsibilities diverge, not to satisfy arbitrary size targets.

## Step 5 — Treat dependencies as maintained code

Before adding a dependency, record:

- problem solved;
- why stdlib/current dependencies are insufficient;
- maintenance/activity signal;
- API stability;
- transitive complexity;
- whether the dependency will be used in core runtime or only dev/test.

Before writing custom code, ask the inverse:

Are we about to maintain a worse version of a mature library?

## Step 6 — Preserve explicitness

Prefer code a new engineer can trace without framework archaeology.

Avoid:
- hidden global state;
- implicit registries;
- magic decorators that obscure control flow;
- metaprogramming unless clearly justified;
- runtime import tricks;
- broad dependency injection frameworks for small services.

## Step 7 — Error handling

Handle errors at boundaries where you can add value.

Do:
- raise domain-specific errors when callers can act on them;
- validate external input early;
- preserve root-cause information;
- log once at an appropriate boundary.

Do not:
- wrap every function in `try/except`;
- catch `Exception` and continue;
- convert all failures into `None`;
- log and re-raise repeatedly at every layer.

## Step 8 — Refactor by deletion first

After a feature works, ask:

- Can a new file disappear?
- Can a class become a function?
- Can duplicated branches merge?
- Can an abstraction be replaced by direct library use?
- Can configuration be collapsed?
- Did we build for a hypothetical future?
- Can code be deleted without reducing clarity or capability?

A successful feature may add 40 lines and delete 150.

## Completion checklist

Before finishing an implementation:

- [ ] Existing code was searched first.
- [ ] New abstractions represent current, not hypothetical, variation.
- [ ] Every new dependency has a concrete benefit.
- [ ] No pass-through layers were added without policy/behavior.
- [ ] Names reflect domain concepts.
- [ ] Error handling exists at meaningful boundaries.
- [ ] The implementation was reviewed for deletion/simplification.
- [ ] Tests verify important behavior rather than implementation details.