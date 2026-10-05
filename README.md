# DecisionGuard

Marketing mix model decision support: evaluate whether model evidence justifies
a budget recommendation, then propagate uncertainty into constrained decisions.

The repository is in Phase 0 (foundation). Data generation and modeling are not
implemented yet.

## Local setup

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

Python 3.12 is the supported runtime. uv selects it using `.python-version`.
The installed Python package is `decisionguard`; the distribution retains the
repository name `mmm-decision-engine`.

## Project context

Read `agents.md` before making changes. The canonical documents currently live
locally in `docs/scope.md`, `docs/prd.md`, `docs/technical_spec.md`, and
`docs/roadmap.md`. The directory is ignored by Git and is absent from fresh clones.

Source lives in `src/decisionguard/`; tests live in `tests/`.

## Foundation decisions

- Ruff handles formatting and linting, Pyright checks strict types, and pytest
  runs tests. These are development dependencies, locked by uv. No scientific or
  application runtime libraries are needed for this slice.
- Hatchling builds the standard `src` package layout. The packaging test imports
  from an isolated interpreter outside the repository, so source-path shortcuts
  cannot hide a broken installation.
- Python is constrained to 3.12 to match the project contract. Supporting newer
  runtimes can be revisited when the scientific dependency stack is verified.
- Configuration has only a module boundary for now; add settings when actual
  runtime requirements exist.

Remaining Phase 0 work: make canonical docs available to fresh clones and add
pre-commit hooks and GitHub Actions using the same verification commands.
