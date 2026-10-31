# PLAN

Each phase ends with a demo and your approval. Nothing below is started.

| Phase | Deliverable | Exit criteria |
|---|---|---|
| 0 | Scaffold (this commit) | smoke test passes, compose validates, CI green |
| 1 | Providers: real Ollama + Gemini via httpx, retries, rate-limit handling; contract tests shared with Mock | same test suite passes on all three (live tests opt-in) |
| 2 | Sandbox runner + hostile-fixture suite (SPEC §6) | all hostile fixtures contained |
| 3 | Eval harness + seeded-bug dataset generator (SPEC §7) | reproducible baseline run recorded |
| 4 | LangGraph pipeline: analyze → propose → verify → report | end-to-end on seeded set with Mock and one real provider |
| 5 | Langfuse tracing, eval runs as Langfuse datasets | traces visible; tracing no-ops when off |
| 6 | Model router (Ollama/Gemini) | routing evaluated on held-out set |
| 7 | GitHub PR integration (webhook/CLI) | comments on a test repo |
| 8 | QLoRA distillation on Colab, eval of tuned model | compared on held-out test set |
