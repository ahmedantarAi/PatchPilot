# PLAN

Each phase ends with a demo and your approval. Phase 0 is done; the rest are not started.
Commands run from the repo root in PowerShell. Commands for phases ≥ 1 describe the **contract the
phase must satisfy**; the modules named in them do not exist yet. A phase is accepted only when its
command prints the expected output and exits 0. Counts shown are planned fixture/test counts, not results.

## Phase 0 — Scaffold (done)
```powershell
.\tasks.ps1 check; .\tasks.ps1 config; docker compose up -d --wait qdrant; curl.exe -s http://127.0.0.1:6333/healthz; docker compose down
```
Expected: `All checks passed!` · `Success: no issues found in 8 source files` · `3 passed` ·
(config prints nothing, exit 0) · `healthz check passed`.

## Phase 1 — Providers (Ollama, Gemini, Mock; privacy guard)
```powershell
uv run --frozen pytest tests/providers -q -m "not live"
$env:PP_LIVE="1"; uv run --frozen pytest tests/providers -q -m live
```
Expected: `20 passed, 2 deselected` (6 shared contract tests × 3 providers + 2 Gemini-specific checks; 2 live tests deselected) then
`2 passed, 20 deselected` (one live Ollama with `qwen2.5-coder:7b`, one live Gemini on a `data_class="public"` synthetic prompt using the model named in
`PP_GEMINI_MODEL` from `.env`; no model name is hardcoded). The 2 Gemini-specific checks verify that a non-`public` request (`data_class="private"`, the default) to `gemini` raises `ProviderError`, and that an empty `PP_GEMINI_MODEL` fails fast.

## Phase 2 — Sandbox runner + hostile fixtures
```powershell
.\tasks.ps1 sandbox-build; uv run --frozen python -m patchpilot.sandbox selftest
```
Expected: eight lines `PASS contained: <fixture>` for `network_egress read_host_env fork_bomb disk_fill
infinite_loop memory_balloon write_outside_work symlink_escape`, then `SANDBOX SELFTEST: 8/8 contained`,
exit 0. Uses a pure-Python runner entrypoint baked into `sandbox/` (no extra `apt-get` packages). Also asserts a second concurrent run is refused via the cross-process + async lock (`SandboxBusyError`): `PASS single-run lock`, and that the network-enabled
install step (`pip install --only-binary :all: --no-build-isolation --no-input --target /deps`, mounted `:ro` into the `--network none` container) runs no repo code: `PASS install step runs no repo code`.

## Phase 3 — Datasets, eval harness, split freeze
```powershell
uv run --frozen python -m patchpilot.evals build --seed 1337
uv run --frozen python -m patchpilot.evals verify
uv run --frozen python -m patchpilot.evals freeze
uv run --frozen python -m patchpilot.evals verify-frozen
```
Expected, in order: `built seeded_eval=120 seeded_train=360 realworld_eval=20 findings_eval=50 safety=8` ·
`OK: every seeded case fails before fix and passes after fix; train/eval overlap = 0; all real-world/findings licenses in {MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause}` ·
`wrote evals/FROZEN.json sha256=<64 hex>` · `OK: frozen eval split unchanged`. `realworld_eval` and `findings_eval` are materialized offline from bundled curated manifests; `evals verify` runs one fresh `--network none` sandbox container per case (pre-fix and post-fix checks inside it, sequentially) and caches each case's verify result keyed by a hash of the case contents and the sandbox image digest so only changed cases re-run (measured runtime: 421.89 s / ~7.0 min cold first run for 480 containers at ~0.89 s/container, <0.01 s warm cached re-run; verified 2026-10-31). No tuning or training starts before
the `freeze` step has been committed.

## Phase 4 — LangGraph pipeline (analyze → propose ⇄ verify → report) + Qdrant retrieval
```powershell
docker compose up -d --wait qdrant
uv run --frozen python -m patchpilot run --provider mock --case seeded_eval/0001
docker compose down
uv run --frozen python -m patchpilot run --provider mock --case seeded_eval/0001
```
Expected (single line of JSON per run, exit 0 both when Qdrant is up and when Qdrant is down):
`{"case": "seeded_eval/0001", "provider": "mock", "patch_applies": true, "target_test_passes": true, "regressions": 0, "verified": true}`
(`mock` uses deterministic `HashEmbedder` vectors for Qdrant REST retrieval when up, degrades gracefully when Qdrant is down, normalizes `SEARCH/REPLACE` or unified diffs via `difflib`, and supports the bounded `verify → propose` retry loop up to `k=3`.)

## Phase 5 — Langfuse tracing
```powershell
.\tasks.ps1 obs-up; uv run --frozen python -m patchpilot trace-smoke; .\tasks.ps1 obs-down
uv run --frozen python -m patchpilot trace-smoke
```
Expected: first run `trace_id=<32 hex> visible=true` (provisioned non-interactively via `LANGFUSE_INIT_*` env vars); after `obs-down`, second run
`tracing disabled: no-op` with exit 0 (core flow works without Langfuse).

## Phase 6 — Model router (Ollama / Gemini / Mock)
```powershell
uv run --frozen pytest tests/test_router.py -q
```
Expected: `9 passed` — covers task-difficulty routing, Gemini quota exhaustion → Ollama fallback, provider
failure fallback, and that private-data requests (`data_class="private"`) are never routed to Gemini.

## Phase 7 — GitHub PR integration
```powershell
uv run --frozen python -m patchpilot review --repo-dir tests/fixtures/pr_demo --dry-run --provider mock
```
Expected: `findings=2 verified_patches=1 comments_posted=0 (dry-run)` for the fixture PR (counts are properties
of the fixture). Live posting to a throwaway test repo is a separate manual step requiring your OK.

## Phase 8 — QLoRA distillation on Colab, tuned-vs-baseline comparison
```powershell
uv run --frozen python -m patchpilot.evals export-train --out evals/results/seeded_train.jsonl
uv run --frozen python -m patchpilot.evals compare --baseline <run_id_a> --candidate <run_id_b>
```
Expected: `export-train` writes `seeded_train` (360 cases) as JSONL for the standalone Colab QLoRA notebook (which exports a quantized GGUF + `Modelfile` registered in local Ollama via `ollama create` so both baseline and candidate run through `OllamaProvider` with no training deps in `pyproject.toml`). `compare` prints header lines `frozen split: OK (sha256=<64 hex>)` and `train/eval overlap: 0`, followed by a metrics table
for both runs with `n_runs>=3` each. Training data comes only from `seeded_train`. Results may be quoted in the
README only with the run ids printed here.
