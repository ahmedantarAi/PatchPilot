FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.11.18-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONPATH=/app/src PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src ./src
RUN useradd --uid 10001 --no-create-home app
USER 10001
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "patchpilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
