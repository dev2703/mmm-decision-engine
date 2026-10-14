.DEFAULT_GOAL := demo

DATABASE_URL ?= postgresql+psycopg:///decisionguard_demo
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
WEB_HOST ?= 127.0.0.1
WEB_PORT ?= 3000

.PHONY: dev demo demo-build demo-check demo-stop stop check

# Docker workflow for a clean environment.
dev:
	docker compose up --build --wait
	cd web && npm ci && npm run dev

# Native presentation workflow using the existing local demo database/artifacts.
# Keep this target in the foreground so Ctrl-C stops both child processes.
demo:
	@set -eu; \
	trap 'kill $$api_pid $$web_pid 2>/dev/null || true' EXIT INT TERM; \
	DATABASE_URL='$(DATABASE_URL)' .venv/bin/uvicorn decisionguard.api.app:create_app --factory --host '$(API_HOST)' --port '$(API_PORT)' & \
	api_pid=$$!; \
	(cd web && npm run dev -- --hostname '$(WEB_HOST)' --port '$(WEB_PORT)') & \
	web_pid=$$!; \
	printf '\nDecisionGuard demo: http://%s:%s\nAPI readiness:      http://%s:%s/ready\nPress Ctrl-C to stop both servers.\n\n' '$(WEB_HOST)' '$(WEB_PORT)' '$(API_HOST)' '$(API_PORT)'; \
	wait $$web_pid

# Build and serve the stable production frontend for presentations.
demo-build:
	cd web && npm run build
	@if curl --fail --silent 'http://$(WEB_HOST):$(WEB_PORT)/' >/dev/null && curl --fail --silent 'http://$(API_HOST):$(API_PORT)/ready' >/dev/null; then \
		printf '\nAn active demo is already serving http://%s:%s\nRun make demo-stop first if you want to replace it.\n\n' '$(WEB_HOST)' '$(WEB_PORT)'; \
	exit 0; \
	fi; \
	@set -eu; \
	trap 'kill $$api_pid $$web_pid 2>/dev/null || true' EXIT INT TERM; \
	DATABASE_URL='$(DATABASE_URL)' .venv/bin/uvicorn decisionguard.api.app:create_app --factory --host '$(API_HOST)' --port '$(API_PORT)' & \
	api_pid=$$!; \
	(cd web && npm run start -- --hostname '$(WEB_HOST)' --port '$(WEB_PORT)') & \
	web_pid=$$!; \
	printf '\nDecisionGuard production demo: http://%s:%s\nAPI readiness:               http://%s:%s/ready\nPress Ctrl-C to stop both servers.\n\n' '$(WEB_HOST)' '$(WEB_PORT)' '$(API_HOST)' '$(API_PORT)'; \
	wait $$web_pid

demo-check:
	@curl --fail --silent --show-error 'http://$(API_HOST):$(API_PORT)/ready' >/dev/null
	@curl --fail --silent --show-error 'http://$(WEB_HOST):$(WEB_PORT)/' >/dev/null
	@printf 'Demo is ready at http://%s:%s\n' '$(WEB_HOST)' '$(WEB_PORT)'

demo-stop:
	@pkill -f 'uvicorn decisionguard.api.app:create_app --factory --host $(API_HOST) --port $(API_PORT)' 2>/dev/null || true
	@pkill -f 'next (dev|start).*--port $(WEB_PORT)' 2>/dev/null || true

stop:
	docker compose stop

check:
	uv run ruff check .
	uv run pyright
	cd web && npm run format:check && npm run typecheck && npm run build
