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
.\tasks.ps1 up           # core services (no profile)
.\tasks.ps1 obs-up       # + Langfuse (needs real secrets in .env)
```
`make <task>` is equivalent where `make` is available (Linux/CI). Run `.\tasks.ps1 help` for tasks.

## LLM providers
`PP_LLM_PROVIDER` = `ollama` | `gemini` | `mock`. CI uses `mock` only.
Containers reach host Ollama at `http://host.docker.internal:11434`.

## Layout
`src/patchpilot/` code · `tests/` · `sandbox/` untrusted-code image · `evals/` datasets (planned) ·
`SPEC.md` `PLAN.md` `AGENTS.md` `DECISIONS.md`

## Performance
Not yet measured. Numbers will be added only after the eval harness produces them.
