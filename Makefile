.PHONY: help api setup lint typecheck test check up down obs-up obs-down config sandbox-build clean
help:
	@echo "Targets: api setup lint typecheck test check up down obs-up obs-down config sandbox-build clean"

api:
	uv run --frozen uvicorn patchpilot.api:app --host 127.0.0.1 --port 8000

setup:
	uv sync --frozen
	@uv run --frozen python -c "import pathlib, sysconfig; sp = pathlib.Path(sysconfig.get_path('purelib')); (sp / 'patchpilot.pth').write_text(str(pathlib.Path('src').resolve()) + '\n', encoding='utf-8')"

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

typecheck:
	uv run --frozen mypy src

test:
	uv run --frozen pytest -q

check: lint typecheck test

# Core Compose services only (Qdrant; no profile). FastAPI runs on the host via `api`.
up:
	docker compose up -d --wait qdrant
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
	@uv run --frozen python -c "import pathlib, sysconfig; sp = pathlib.Path(sysconfig.get_path('purelib')); (sp / 'patchpilot.pth').write_text(str(pathlib.Path('src').resolve()) + '\n', encoding='utf-8')"
	docker build -t patchpilot-sandbox:0.0.0 sandbox

clean:
	docker compose --profile observability down -v --remove-orphans
