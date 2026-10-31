.PHONY: help setup lint typecheck test check up down obs-up obs-down config sandbox-build clean
help:
	@echo "Targets: setup lint typecheck test check up down obs-up obs-down config sandbox-build clean"

setup:
	uv sync --frozen

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

typecheck:
	uv run --frozen mypy src

test:
	uv run --frozen pytest -q

check: lint typecheck test

# Core services only (no profile)
up:
	docker compose up -d --build api
down:
	docker compose down

# Observability (Langfuse). Requires secrets in .env.
obs-up:
	@test -f .env || (echo ".env missing: copy .env.example and replace CHANGEME" && exit 1)
	@! grep -q CHANGEME .env || (echo "Replace CHANGEME values in .env" && exit 1)
	docker compose --profile observability up -d
obs-down:
	docker compose --profile observability down

config:
	docker compose config -q
	docker compose --profile observability config -q

sandbox-build:
	docker compose build sandbox

clean:
	docker compose --profile observability down -v --remove-orphans
