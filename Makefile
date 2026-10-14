.PHONY: dev stop check
dev:
	docker compose up --build --wait
	cd web && npm ci && npm run dev

stop:
	docker compose stop

check:
	uv run ruff check .
	uv run pyright
	cd web && npm run format:check && npm run typecheck && npm run build
