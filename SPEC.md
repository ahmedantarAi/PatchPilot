# PatchPilot — Specification

Status: APPROVED (rev 3, pre-Phase 1 design review complete). Scaffold only; no feature code exists yet.

## 1. Purpose
PatchPilot is an autonomous code review and patch verification agent. Given a pull request it:
1. inspects the diff, 2. runs static analysis, 3. generates targeted fixes, 4. validates each fix by
running the project's test suite inside a sandbox, 5. reports only verified patches.

Portfolio project. Constraints: Windows 11, 16 GB RAM, RTX 5050 8 GB VRAM, free tools only.

## 2. Non-goals (v1)
Merging PRs automatically; languages other than Python targets; multi-tenant hosting; any paid service.

## 3. Architecture and process boundaries
```
HOST (Windows, `uv run`)                                   DOCKER (Compose)
┌──────────────────────────────────────────────┐          ┌───────────────────────────┐
│ FastAPI + LangGraph orchestrator             │─ HTTP ──►│ Qdrant (core, no profile) │
│  analyze → propose ⇄ verify → report         │          ├───────────────────────────┤
│        │            │                        │─ OTLP ──►│ Langfuse v4 stack         │
│        │            └─► LLMProvider ─► Ollama (host)    │ (profile: observability)  │
│        │                              Gemini | Mock      └───────────────────────────┘
│        └─► `docker run` ─► SANDBOX container (one at a time, §6)   (not in Compose)
└──────────────────────────────────────────────┘
```
- **FastAPI and the orchestrator run on the host** (`uv run`). They are never containerised.
- **Compose runs only Qdrant and the observability stack.** It contains no API service and no sandbox service.
- **Sandbox containers are launched by the host orchestrator** with `docker run`, using the flags in §6.
- Python 3.11, FastAPI, Pydantic v2, LangGraph; uv with pinned lockfile.
- Pipeline nodes are pure functions over a typed state; LLM calls go only through `LLMProvider`.
- **Pipeline nodes and state machine (`analyze → propose ⇄ verify → report`):**
  - **`analyze`:** Runs Python stdlib `ast.parse` and `ruff check --output-format json --isolated --no-config`
    (never loading target-repo configuration files on the host), retrieves relevant repository context from
    a per-case Qdrant collection (§5) when available, and performs LLM fault localization and diff inspection.
    Receives only the repository file list (non-test `.py` files), the failing test output, and file contents —
    never the ground-truth answer (`target_file` does not exist in `PipelineState` or in any prompt; for
    multi-file repositories, the model chooses the file itself). Emits typed `Finding` objects
    (`file`, `line`, `category`, `explanation`, `fixable`). Used directly by `findings_eval` (PR review
    mode) and feeds fixable findings / failing test context into `propose`.
  - **`propose`:** Prompts the LLM for `SEARCH/REPLACE` blocks (while also accepting raw unified diffs, e.g.
    from `MockProvider` or reference patches) and normalizes the result on the host via `difflib.unified_diff`
    against the repository snapshot without any ground-truth `default_file` hint. Enforces the
    **reward-hacking guard** at normalization time: any patch that touches a test file (`test_*.py`,
    `*_test.py`, anything under `tests/`, `conftest.py`) or any non-`.py` file is rejected.
  - **`verify`:** Enforces the **reward-hacking guard** again (defense in depth) before launching the
    sandbox, then applies the canonical unified diff and runs the test suite inside the hardened sandbox (§6).
    A rejected patch (or a patch that fails application or tests) counts as a failed attempt with a clear
    rejection/failure reason fed back to `propose` while `attempt < max_attempts` (fixed `k = 3`). A patch
    that edits an assertion can never produce `verified = True`. Attempt 1 records `pass@1`; succeeding
    within `≤ 3` attempts records `pass@3`.
  - **`report`:** Emits structured findings and only patches that achieved `verified = True` (`patch_applies`,
    `target_test_passes`, and `regressions == 0`).

## 4. LLM provider interface
Defined in `src/patchpilot/providers/base.py`.
| Provider | Where it runs | Use |
|---|---|---|
| `ollama` | Native on Windows host (`http://localhost:11434`). Never in Docker. Default model `qwen2.5-coder:7b` (`PP_OLLAMA_MODEL`). | local, private, GPU |
| `gemini` | Gemini API free tier, plain `httpx` (no vendor SDK). Model name comes from `PP_GEMINI_MODEL` in `.env`; there is no hardcoded default in code. | stronger model, rate-limited; **public/synthetic code only (§9)** |
| `mock` | in-process, deterministic, no network | unit tests and CI (mandatory in CI) |

Contract: `async generate(LLMRequest) -> LLMResponse`; failures raise `ProviderError`.
`LLMRequest` carries `data_class: Literal["private", "public"] = "private"` (defaulting to `"private"` so
unmarked requests fail closed; synthetic/generated eval prompts explicitly pass `data_class="public"`).
Selection: `PP_LLM_PROVIDER` or explicit argument. Dynamic routing (later phase) is a `Router` that
itself implements `LLMProvider`, choosing among the three by task difficulty, quota, failure, and
the data-privacy rule in §9.

**Gemini model names (checked 2026-10-31 against ai.google.dev/gemini-api/docs/pricing).** Text models
listed with a free tier: `gemini-3.8-flash` (chosen as the `.env.example` value), `gemini-3.6-flash`,
`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-2.5-flash-lite`. The 2.5 family may be limited to
projects that already used it. Names and limits change; re-check at phase 1 and whenever calls start
failing. Rate limits (RPM/TPM/RPD) are per project and are not recorded here; read them in AI Studio.

**Reaching host Ollama from a container (tested 2026-10-31).** Nothing in the current design needs this:
FastAPI and the orchestrator run on the host, and sandbox containers have no network. Result of the test:
with Ollama 0.30.7 listening on its default `127.0.0.1:11434` and Docker Desktop (WSL2 backend), a
container started with plain `docker run` reached `http://host.docker.internal:11434/api/version` and got
`{"version":"0.30.7"}`, with no `extra_hosts` entry and no `OLLAMA_HOST` change. This may differ on other
Docker setups. Fallback if it ever fails: set `OLLAMA_HOST=0.0.0.0:11434` on the host and restart Ollama.
**Warning:** that exposes Ollama, which has no authentication, to your LAN; first add a Windows Firewall
inbound rule that limits TCP 11434 to the Docker/WSL subnet (or to localhost) and blocks other sources.
Never expose it to the sandbox network.

## 5. Services in Compose
**Qdrant** (core, no profile): vector store for repository code retrieval. Bound to 127.0.0.1:6333.
Accessed via plain `httpx` against Qdrant's REST API (no extra SDK dependency). Embeddings go through an
`Embedder` protocol: `OllamaEmbedder` (`POST /api/embed`) in live mode and a deterministic hashed-token
`HashEmbedder` in `mock`/CI mode so retrieval tests never call external models. Each case uses an isolated
per-case collection named from the case content hash (`patchpilot_case_<sha256[:16]>`), recreated before
indexing and deleted after the run so retrieval for case B never returns chunks from case A. Pipeline must
degrade gracefully (no retrieval) if Qdrant is down.

**Langfuse v4** (profile `observability` only): 6 containers (web, worker, ClickHouse, Postgres, Redis,
MinIO). This is the minimum official topology; there is no lighter self-hosted variant. Core code never
depends on it; tracing is a no-op if Langfuse is down or unconfigured. It stays an **optional** profile.
To allow non-interactive startup on a fresh volume (`obs-up` → `trace-smoke`), `langfuse-web` is wired
with `LANGFUSE_INIT_*` headless provisioning variables in `docker-compose.yml` and `.env.example`.
The current per-service memory limits (§8) are accepted for now; they are estimates and real usage will be
measured later. **Fallback:** if the local stack proves too heavy, use Langfuse Cloud's free tier instead
(set `LANGFUSE_HOST` and keys in `.env`); that sends traces to a third party, so code in traces must obey the
same public-only rule as §9. All images are pinned (MinIO by digest).

## 6. Sandbox for untrusted patches and tests
Threat model: the PR author, the LLM output, and the repo's test code are all untrusted. They may try
to exfiltrate secrets, reach the network or LAN, escape the container, fork-bomb, fill disk, or hang.

**Execution model.** The host orchestrator launches one fresh container per verification run with
`docker run --rm`, from the image built from `sandbox/Dockerfile` (toolchain only: no secrets, no
repo code, no test wheels/sdists; uses a pure-Python runner entrypoint baked into `sandbox/` with no extra
`apt-get` packages). Containers are labelled with `patchpilot.sandbox=1`, and any leftover container with
this label is removed on orchestrator start. The repo snapshot and patch are streamed in as a tar on stdin and
safely extracted into the tmpfs `/work` by the container entrypoint (rejecting path-traversal, drive-qualified paths,
and symlink escapes; no host bind mounts). Case names and file paths cannot inject extra docker run arguments;
the tar archive is capped at 10,000 members, 32 MB per file, and 200 MB total unpacked size, and patches are capped
at 10 MB. The unified diff is applied in Python and `pytest` runs inside. The container converts JUnit test results
to a capped JSON summary (counts, test names, status, max 500 cases) after a bounded read (at most 1 MB) of the XML;
the host never parses untrusted XML and validates only the JSON summary against a Pydantic schema (`ET.fromstring`
is forbidden in host code). Hostile XML containing DOCTYPE or entity expansion declarations produces a clean error result.
The container is removed afterwards. Runs are **sequential: at most one sandbox at a time**, enforced by a fail-fast
cross-process OS file lock + atomic in-process thread lock that raises `SandboxBusyError` on concurrent attempts.

**Mandatory `docker run` flags / controls:**
| Control | Setting |
|---|---|
| Label | `--label patchpilot.sandbox=1` (all verification and dependency install containers) |
| Network | `--network none` (no egress, no host/LAN access). Dependencies are pre-baked in the image or installed in a separate, network-enabled **install step** (accepted for v1) that runs no repo code: it mounts a cache volume (`patchpilot-deps-<sha256>`) and runs `pip install --only-binary :all: --no-build-isolation --no-input --target /deps` with only the pinned requirements list present (no repo files, no `setup.py`/build hooks), then mounts `/deps:ro` (`PYTHONPATH=/deps`) into the `--network none` verification container. |
| Privileges | `--user 10001:10001 --cap-drop ALL --security-opt no-new-privileges`, default seccomp |
| Filesystem | `--read-only`; writable only `--tmpfs /work:rw,nosuid,size=256m,uid=10001,gid=10001` and `--tmpfs /tmp:rw,noexec,nosuid,size=128m` |
| Resources | `--memory 768m --memory-swap 768m --cpus 1 --pids-limit 256` |
| Time | hard wall-clock timeout (default 120 s) enforced by the orchestrator: `docker kill` then `docker rm -f` |
| Secrets | no `-e` passthrough, no mounted host paths, no Docker socket |
| Output | stdout/stderr size-capped (1 MB); result returned as structured JSON (exit code, capped JSON test summary) |
| Logging | each run records image digest, flags, duration, exit status |

**Rules.** The API/orchestrator process never executes repo code itself. The Docker socket/pipe is
used only by the host orchestrator and is never mounted into any container. Hardened Docker (no network,
read-only filesystem, non-root, dropped capabilities, resource limits) is considered sufficient for v1.

**Known limitation (shared kernel).** Sandbox containers share the Linux kernel of the Docker Desktop
WSL2 VM with every other container (Qdrant, Langfuse). A kernel or runtime vulnerability could let hostile
code escape to that VM, which is not a full VM boundary. v1 accepts this risk and therefore must not be used
on code more hostile than ordinary PRs from untrusted but non-targeted authors; keep Docker Desktop updated.
Upgrade path: gVisor (`runsc`) or a disposable WSL distro/VM.

**Hostile fixtures (acceptance suite; all must be contained):** `network_egress`, `read_host_env`,
`fork_bomb`, `disk_fill`, `infinite_loop`, `memory_balloon`, `write_outside_work`, `symlink_escape`.
- `network_egress`: does not depend on the internet; host starts a local TCP listener on `0.0.0.0:0` and container
  attempts to connect to `host.docker.internal:<port>`. Must fail under `--network none` and succeed when weakened.
- `memory_balloon`: contained only if exit code is 137/247 or `MemoryError` is observed, not any non-zero exit.
- `mutation test suite`: tests each control (network, pids-limit, memory, read-only root, user/caps) against
  deliberately weakened configurations and asserts they report NOT contained.

## 7. Eval dataset plan
Goal: measure (a) whether proposed patches are *correct* and (b) whether routing and tracing behave.
No performance claim is published until the harness has produced it (README policy, AGENTS rule 8).

**Pools and targets**
| Pool | Target size | Source | Used for |
|---|---|---|---|
| `seeded_eval` | **≥ 100** (plan: 120) | script-generated seeded bugs, fixed seeds | evaluation only |
| `seeded_train` | **≥ 300** (plan: 360) | script-generated seeded bugs, **disjoint seeds**, separate generator run | QLoRA distillation / prompt dev only; **never in eval** |
| `realworld_eval` | **20** | bundled curated manifest of merged bug-fix commits from MIT/Apache-2.0/BSD repos | evaluation only |
| `findings_eval` | **50** labelled findings | bundled curated manifest of PR findings with human true/false labels, from MIT/Apache-2.0/BSD repos | review precision/recall |
| `safety` | **8** fixtures | §6 | containment pass/fail |

With 20 real cases and 50 findings, real-world and review metrics will have wide uncertainty; reports must
show the case counts next to any metric.

**Licensing & offline reproducibility.** Real-world cases and findings come from bundled, self-contained
offline manifests extracted from repositories under MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause so
`evals build` runs deterministically without network access. Each case's `meta.json` records `repo`,
`commit`, `license` (SPDX id) and `license_source` (the file/URL checked). `evals verify` fails on a
missing or non-allowlisted license.

Each seeded case = repo snapshot + failing test + known-good fix, produced by a generator over a bug
taxonomy (off-by-one, wrong operator, None handling, missing await, resource leak, mutable default arg,
string-built SQL) across 56 distinct base templates (8 distinct structural templates per taxonomy class).
Generation is deterministic from `(generator_version, seed)`. During `evals verify`, each case runs in its own
fresh `--network none` sandbox container (executing the pre-fix fail check and post-fix pass check inside it,
sequentially, preserving the §6 per-case isolation boundary). To keep subsequent runs fast, `evals verify` caches
each case's verification result keyed by a hash of the case contents and the sandbox image digest
(`sha256(case_content_hash, sandbox_image_digest)`), so only new or changed cases re-run (measured cold runtime:
425.11 s for 480 cases at ~0.89 s/case; warm cached re-run <0.01 s).

**Separation rules (train never leaks into eval)**
- **Template-level partition:** Exactly 25% of templates per taxonomy class (2 templates per class, 14 total)
  are reserved exclusively for `seeded_eval`. The remaining 75% (6 templates per class, 42 total) are reserved
  exclusively for `seeded_train`. The template sets are strictly disjoint (`eval_templates ∩ train_templates == ∅`),
  enforced at runtime by generator rejection and validated by unit tests.
- `seeded_train` and `seeded_eval` are produced by disjoint seed spaces (eval: base_seed + i*37; train: base_seed + 100_000 + i*53)
  and a multi-tier dedupe step: any train case whose content hash, normalized buggy-source hash, reference-diff hash,
  or mutation signature matches an eval case is dropped. `evals verify` fails if any overlap exists.
- `realworld_eval`, `findings_eval` and `safety` are eval-only; they are never used for training or
  prompt tuning.
- **Ground-truth isolation:** Real providers (`ollama`, `gemini`) never cause `patch.diff` or `meta.json`
  to be read by the pipeline. Only the `MockProvider` path (via a dedicated `load_mock_ground_truth`
  function) and test assertions may read `patch.diff` or `meta.json`. No prompt may contain ground-truth
  fix hunks, `patch.diff` content, or a `meta.json` target file hint (unless the filename legitimately
  appears in the failing test output).
- **Reward-hacking guard & Qdrant per-case isolation:** Any patch touching a test file (`test_*.py`,
  `*_test.py`, anything under `tests/`, `conftest.py`) or any non-`.py` file is rejected at both
  normalization time and inside `verify` (defense in depth), counting as a failed attempt with the reason
  fed back to `propose`. Each case indexes into a per-case Qdrant collection named from its content hash
  and deletes the collection after the run.

**Freeze.** The eval split (all eval-only pools) is **frozen before any tuning, prompt iteration or
training starts**: `evals freeze` writes `evals/FROZEN.json` (case ids, per-case content hashes, generator
version, seed ranges, overall sha256). It is committed. `evals verify-frozen` runs in CI and fails on any
change; changing the eval set requires a new dataset version and a DECISIONS.md entry, and invalidates
earlier comparisons.

**Metrics.** Patch applies cleanly; target failing test passes; no previously passing test regresses
(`pass@1` on attempt 1 and `pass@k` at fixed `k = 3` across the bounded retry loop); review
precision/recall on `findings_eval`; sandbox containment; tokens and latency per provider. Reports
include run count, seed, model version and confidence intervals; ≥ 3 runs per configuration because LLMs
are nondeterministic.

**QLoRA distillation handoff (Phase 8).** Training dependencies (`torch`, `transformers`, `peft`,
`bitsandbytes`) are never added to `pyproject.toml`. Instead, `seeded_train` is exported as JSONL for a
standalone Colab notebook that fine-tunes `Qwen2.5-Coder-7B`, exports a quantized GGUF + `Modelfile`, and
registers the model in local Ollama (`ollama create`) so `evals compare` evaluates both baseline and
candidate through `OllamaProvider`.

**Layout.** `evals/datasets/<pool>/<case_id>/{repo/,meta.json}`; results in `evals/results/<run_id>/`
(gitignored; summaries committed and cited by run id). CI runs only a tiny mock-provider eval for
plumbing, never model quality.

## 8. Resource budget
- **Compose total ≤ 6144 MB** via per-service `mem_limit`: Qdrant 512 MB + observability 3840 MB
  (web 1024, worker 768, ClickHouse 1280, Postgres 384, Redis 128, MinIO 256) = **4352 MB**.
- **Sandbox runs are sequential, at most one at a time, and are not counted in the Compose total.**
  Each run is capped at 768 MB RAM / 1 CPU (§6); the orchestrator holds a single-run lock.
- Host processes (FastAPI, orchestrator, Ollama and its VRAM/RAM) are outside Compose. The 8 GB VRAM
  bounds usable local models. These are *limits*, not measurements. With the Compose stack at its
  ceiling, one sandbox and the host processes, total demand against 16 GB host RAM is unmeasured.

## 9. Security, privacy & secrets
- **Never send private code to the Gemini free tier.** Free-tier inputs may be used by Google to improve
  its products. Only repositories explicitly marked `public` (or synthetic/generated eval data, which
  sets `data_class="public"`) may be sent to `gemini`. Anything unmarked defaults to
  `data_class: Literal["private", "public"] = "private"` on `LLMRequest` and may go only to `ollama` or
  `mock`. The Gemini provider and Router reject non-`"public"` requests with `ProviderError`, enforced in
  code and covered by unit tests. Secrets, `.env` contents and credentials never enter any prompt.
- `.env` is gitignored; `.env.example` holds placeholders. Langfuse telemetry disabled.
  Traces may contain code: Langfuse is local-only and not exposed beyond 127.0.0.1.
- All Compose ports bind to 127.0.0.1. No secret ever enters the sandbox.
- **Ground-truth and test-integrity isolation:** Real providers (`ollama`, `gemini`) never read `patch.diff`
  or `meta.json`, and `target_file` is never stored in pipeline state or prompts. Untrusted patches are
  blocked from modifying test files (`test_*.py`, `*_test.py`, `tests/**`, `conftest.py`) or non-`.py` files
  at both normalization time and verification time.

## 10. Definition of done for the scaffold
`.\tasks.ps1 check` and `.\tasks.ps1 config` pass; `docker compose config` validates with and without
`--profile observability`; CI is green using only the mock provider.
