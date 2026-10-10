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

