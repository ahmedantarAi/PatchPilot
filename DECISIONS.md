# DECISIONS

| ID | Decision | Rationale |
|---|---|---|
| D-01 | Python 3.11 only (`==3.11.*`), uv + `uv.lock`, exact `==` pins | Reproducibility; user requirement |
| D-02 | Ollama native on Windows host, not Docker | User requirement; direct GPU access on RTX 5050 |
| D-03 | Provider interface (`LLMProvider` Protocol) with Ollama, Gemini, Mock | Swappable backends; deterministic CI |
| D-04 | Ollama/Gemini will use `httpx` directly, no vendor SDKs | Fewer pinned dependencies, uniform error handling |
| D-05 | Core services have no compose profile; observability = `profiles: [observability]` | User requirement |
| D-06 | Langfuse **v4** official 6-container topology, images pinned (web/worker 4.72.0, ClickHouse 26.3.62.3, Postgres 17.11-alpine, Redis 7.4.12-alpine). Checked against upstream `docker-compose.yml` on 2026-10-31: no lighter self-hosted topology exists (ClickHouse + Postgres + Redis + MinIO are all required). Memory limits trimmed to fit 6 GB total (5376 MB); ClickHouse at 1280 MB and web at 1024 MB are my judgement, not Langfuse-published minimums | Fit budget; upstream docs advise ≥ 4 GB for the whole stack, so this is tight and unverified under load |
| D-07 | MinIO image `cgr.dev/chainguard/minio:latest` is unpinned (upstream uses it unpinned) | Unresolved; see open questions |
| D-08 | `sandbox` is a compose service used with `docker compose run --rm`, hardened and network-less; it exits immediately on `up` | Single source of truth for limits |
| D-09 | `package = false` in uv; code runs from `src/` via PYTHONPATH | No packaging needed yet |
| D-10 | Gemini/Ollama default model names are placeholders, to be validated in phase 1 | Free-tier models change |
| D-11 | GitHub Actions pinned to major tags (`@v4`, `@v6`), uv pinned to 0.13.6 | Convenience; SHA pinning is an open question |
