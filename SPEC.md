# PatchPilot — Specification

Status: DRAFT, awaiting approval. Scaffold only; no feature code exists yet.

## 1. Purpose
PatchPilot is an autonomous code review and patch verification agent. Given a pull request it:
1. inspects the diff, 2. runs static analysis, 3. generates targeted fixes, 4. validates each fix by
running the project's test suite inside a sandbox, 5. reports only verified patches.

Portfolio project. Constraints: Windows 11, 16 GB RAM, RTX 5050 8 GB VRAM, free tools only.

## 2. Non-goals (v1)
Merging PRs automatically; languages other than Python targets; multi-tenant hosting; any paid service.

## 3. Architecture
```
GitHub PR ─► FastAPI ─► LangGraph pipeline ─► LLMProvider ─► Ollama (host) | Gemini | Mock
                          │  analyze → propose → verify(sandbox) → report
                          ├─► Sandbox (ephemeral container, §6)
                          └─► Langfuse traces (optional profile)
```
- Python 3.11, FastAPI, Pydantic v2, LangGraph; uv with pinned lockfile.
- Pipeline nodes are pure functions over a typed state; LLM calls go only through `LLMProvider`.

## 4. LLM provider interface
Defined in `src/patchpilot/providers/base.py`.
| Provider | Where it runs | Use |
|---|---|---|
| `ollama` | Native on Windows host (`localhost:11434`; `host.docker.internal:11434` from containers). Never in Docker. | local, private, GPU |
| `gemini` | Gemini API free tier (HTTP) | stronger model, rate-limited |
| `mock` | in-process, deterministic, no network | unit tests and CI (mandatory in CI) |

Contract: `async generate(LLMRequest) -> LLMResponse`; failures raise `ProviderError`.
Selection: `PP_LLM_PROVIDER` or explicit argument. Dynamic routing (v1+) is a separate `Router` that
itself implements `LLMProvider`, choosing among the three by task difficulty, quota, and failure.
Gemini free-tier limits and model names change; they are read from config, not hardcoded, and must be
re-verified at implementation time.

## 5. Observability
Langfuse v4 self-hosted, Compose profile `observability` only. Requires 6 containers (web, worker,
ClickHouse, Postgres, Redis, MinIO); this is the minimum official topology, so there is no lighter
self-hosted variant. Core services never depend on it; tracing must degrade to a no-op if Langfuse
is down or unconfigured.

## 6. Sandbox for untrusted patches and tests
Threat model: the PR author, the LLM output, and the repo's test code are all untrusted. They may try
to exfiltrate secrets, reach the network or LAN, escape the container, fork-bomb, fill disk, or hang.

**Execution model.** One fresh container per verification run, destroyed afterwards. Created from
`sandbox/Dockerfile` (toolchain only: no secrets, no PatchPilot code). The repo is copied in
(not bind-mounted from the host working tree); the patch is applied inside the container.

**Mandatory controls** (the Compose `sandbox` service is the reference configuration):
| Control | Setting |
|---|---|
| Network | `network_mode: none` (no egress, no host access). Dependencies must be pre-baked or vendored in a prior, separate, network-enabled *fetch* step that runs no repo code. |
| Privileges | non-root uid 10001, `cap_drop: ALL`, `no-new-privileges`, default seccomp profile |
| Filesystem | read-only root; writable only tmpfs `/work` (256 MB) and `/tmp` (128 MB, noexec) |
| Resources | 768 MB RAM, no swap, 1 CPU, `pids_limit` 256 |
| Time | hard wall-clock timeout (default 120 s) enforced by the orchestrator, which kills and removes the container |
| Secrets | no env vars passed through, no Docker socket, no mounted host paths |
| Output | stdout/stderr captured with a size cap (1 MB); results returned as structured JSON (exit code, junit XML) |
| Logging | each run logs image digest, limits, duration, exit status |

**Rules.** The API process must never run repo code itself. The Docker socket is never mounted into any
container; the orchestrator runs on the host (or a trusted worker) and calls Docker directly.
Known residual risk: Docker containers share the WSL2 VM kernel; a kernel escape is out of scope for v1.
Hardening upgrade path: gVisor/`runsc` or a disposable WSL distro (see open questions).

**Acceptance tests (to be written before the sandbox is trusted):** a corpus of hostile fixtures that
must all be contained: outbound HTTP, reading `/proc/1/environ` of host, fork bomb, `dd` to fill disk,
infinite loop, memory balloon, writing outside `/work`, symlink escape from the copied repo.

## 7. Eval dataset plan
Goal: measure (a) whether proposed patches are *correct* and (b) whether the system routes/traces sanely.
No performance claims are published until the harness has produced them (see README policy).

**Datasets (all free):**
1. **Seeded-bug set (primary, owned):** small Python repos with injected bugs from a taxonomy (off-by-one,
   wrong operator, None handling, missing await, resource leak, bad default arg, SQL-injection style
   string building). Each case = repo snapshot + failing test + known-good fix. Target 100 cases, built
   by script so it is reproducible and licence-clean.
2. **Real-world set:** a curated sample of merged bug-fix commits from permissively licensed Python
   repos (pre-fix commit, failing test from the fix commit, reference diff). Target 50 cases; record
   license per case. Optionally compare with SWE-bench Lite style tasks, subject to licence and disk checks.
3. **Review-quality set:** PRs with human-labelled true/false findings for static-analysis triage
   (target 100 findings).
4. **Safety set:** the hostile fixtures from §6, as pass/fail.

**Splits.** dev (60%), held-out test (40%), frozen before any prompt tuning. For QLoRA distillation
data, training examples are generated only from dev; test cases never enter training. Check for
contamination against public benchmarks and base-model data where possible.

**Metrics.** Patch applies cleanly; target failing test passes; no previously passing test regresses
(pass@1, and pass@k with fixed k); review precision/recall on labelled findings; sandbox containment
rate; cost proxies (tokens, latency) per provider. Report with run count, seed, model version, and
confidence intervals; LLM nondeterminism means ≥3 runs per config.

**Layout.** `evals/datasets/<name>/<case_id>/{repo/,meta.json}`, results in `evals/results/` (gitignored,
summaries committed). Each eval run is a Langfuse dataset run when observability is up, and writes a
local JSON when not. CI runs only a tiny mock-provider eval for plumbing, never model quality.

## 8. Resource budget
Compose memory limits total ≤ 6144 MB (currently 5376 MB: core 1536 MB, observability 3840 MB);
see `docker-compose.yml`. Ollama and its VRAM/RAM use are outside this budget (host process); the
8 GB VRAM bounds usable local models. These are *limits*, not measurements.

## 9. Security & secrets
`.env` is gitignored; `.env.example` holds placeholders. Langfuse telemetry disabled. All ports bind to
127.0.0.1. No secret ever enters the sandbox.

## 10. Definition of done for the scaffold
`make check` / `.\tasks.ps1 check` passes; `docker compose config` validates with and without
`--profile observability`; CI is green using only the mock provider.
