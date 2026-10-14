FROM ghcr.io/astral-sh/uv:0.11.19 AS uv
FROM python:3.12-slim-bookworm AS base
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends g++ libgomp1 git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --locked --no-dev && useradd --create-home --uid 10001 app && mkdir /artifacts && chown app:app /artifacts
ENV ARTIFACT_ROOT=/artifacts PORT=8080
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn decisionguard.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}"]

# Heavy upstream evaluation dependencies are opt-in, separate from the API image.
FROM base AS evaluation
USER root
RUN uv sync --locked --no-dev --group evaluation
USER app

FROM base AS api
