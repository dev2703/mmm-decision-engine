# mmm-decision-engine

Marketing mix model decision support: evaluate whether model evidence justifies
a budget recommendation, then propagate uncertainty into constrained decisions.

## Local setup

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

Python 3.12. See the [scope](docs/scope.md), [technical spec](docs/technical_spec.md)
and [roadmap](docs/roadmap.md) for details. Run `uv run decisionguard --help` for commands.

## Agent exploration (Phase 8)

The provider-independent agent core exposes `POST /agent/query` with a project ID
and message. Claude integration is deferred: the default app returns HTTP 503
until a provider is supplied to `create_app(agent_provider=...)`. No provider SDK
or API key is needed for local tests.

The seven tools inspect quality, models, health, channel evidence, constraints,
scenarios and decision stability. Answers contain cited tool evidence; generated
prose and invented metrics cannot become an answer. Queuing an existing scenario
requires the caller's `authorized_scenario_id`; the CLI executes the job. No tool
changes live spending. See the [adapter contract and design](docs/technical_spec.md#phase-8-design-decision-bounded-provider-independent-exploration).

Run the scripted safety tests and PostgreSQL integration tests:

```sh
uv run pytest tests/test_agent.py
# Use an existing dedicated PostgreSQL database whose name ends in _test.
DECISIONGUARD_TEST_DATABASE_URL='postgresql+psycopg:///decisionguard_test' \
  uv run pytest tests/test_agent_api.py
```

These verify deterministic boundaries, not the quality of a live Claude model.
