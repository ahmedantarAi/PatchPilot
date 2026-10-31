# PatchPilot

Autonomous code review and patch verification agent: inspects pull requests, runs static analysis,
generates targeted fixes, and validates them by running tests in a sandbox.

> **Status: scaffold only.** No feature code yet. See [SPEC.md](SPEC.md) and [PLAN.md](PLAN.md).

## Requirements (Windows 11)
- Docker Desktop (WSL2 backend; allow the VM ≥ 6 GB, e.g. `memory=6GB` in `%UserProfile%\.wslconfig`)
- [uv](https://docs.astral.sh/uv/) (Python 3.11 is fetched automatically)
- [Ollama](https://ollama.com) installed **natively on Windows** (not in Docker)
- Optional: Gemini API key (free tier)

## Quick start
```powershell
Copy-Item .env.example .env
.\tasks.ps1 setup
.\tasks.ps1 check        # lint, types, tests (mock provider)
.\tasks.ps1 up           # Compose core: Qdrant only (no profile)
.\tasks.ps1 api          # FastAPI on the host (uv run), not in Docker
.\tasks.ps1 obs-up       # + Langfuse (needs real secrets in .env)
```
`make <task>` is equivalent where `make` is available (Linux/CI). Run `.\tasks.ps1 help` for tasks.
Optional lint-on-commit: `uvx pre-commit@4.7.0 install` (ruff only; `.python-version` pins Python 3.11).

## Notes
- **Langfuse is optional** (`obs-up`). Its memory limits are estimates to be measured later. Fallback: Langfuse Cloud
  free tier (traces then leave your machine; see SPEC §5).
- **Ollama** runs natively on Windows (`qwen2.5-coder:7b`). Containers could reach it at `host.docker.internal:11434`
  (tested OK on Docker Desktop, see SPEC §4), but nothing needs that today. The `OLLAMA_HOST=0.0.0.0` fallback
  exposes Ollama to your LAN: set a firewall rule first.
- **Gemini** model comes from `PP_GEMINI_MODEL` in `.env` (see `.env.example` for current free-tier names).
- **Sandbox limitation:** hardened Docker shares the WSL2 VM kernel; see SPEC §6 "Known limitation".

## Architecture in one paragraph
FastAPI, the orchestrator and Ollama all run on the host. Compose runs only Qdrant and the optional
Langfuse stack. Untrusted patches/tests run in sandbox containers launched one at a time by the host
orchestrator (SPEC §6).

## LLM providers
`PP_LLM_PROVIDER` = `ollama` | `gemini` | `mock`. CI uses `mock` only.
Never send private code to the Gemini free tier (SPEC §9).

## Layout
`src/patchpilot/` code · `tests/` · `sandbox/` untrusted-code image · `evals/` datasets (planned) ·
`SPEC.md` `PLAN.md` `AGENTS.md` `DECISIONS.md`

## Performance
Not yet measured. Numbers will be added only after the eval harness produces them.
