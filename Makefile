.PHONY: test migrate backfill-rag migrations docker-build docker-up docker-up-d ollama-pull docker-down docker-logs docker-restart docker-rebuild

test:
	uv run python -m pytest tests/ -v

migrate:
	uv run alembic upgrade head

backfill-rag:
	uv run python scripts/backfill_rag.py

migrations:
	uv run alembic revision --autogenerate -m "$(msg)"

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	mkdir -p config/agents
	docker compose up

docker-up-d:
	mkdir -p config/agents
	docker compose up -d

ollama-pull:
	docker compose exec ollama ollama pull mistral

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose restart actus

docker-rebuild:
	docker compose down && docker compose up --build -d
