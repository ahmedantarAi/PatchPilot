# AGENTS.md — rules for AI coding agents working in this repo

1. Read SPEC.md and DECISIONS.md first. Do not contradict them; propose a DECISIONS.md entry instead.
2. Scaffold phase: no feature code until the user approves PLAN.md phase by phase.
3. All LLM calls go through `LLMProvider`. Tests and CI use `MockProvider`; never call Ollama/Gemini in CI.
4. Never execute untrusted patches or repo tests outside the sandbox (SPEC §6). Never mount the Docker socket.
5. Pin every dependency exactly (`==`) and commit `uv.lock`. Pin image tags; justify any exception in DECISIONS.md.
6. Core compose services have no profile; only observability uses `profiles: [observability]`. Keep total `mem_limit` ≤ 6144 MB.
7. Ollama runs natively on the Windows host, not in Docker.
8. No performance/accuracy numbers in README or docs unless produced by a recorded eval run (cite run id).
9. Keep `Makefile` and `tasks.ps1` in sync.
10. Before finishing a change run `.\tasks.ps1 check` and `.\tasks.ps1 config`.
11. Never commit `.env` or secrets.
12. Prompt development is allowed on `seeded_train` only; never read per-case `seeded_eval` results while editing prompts.
