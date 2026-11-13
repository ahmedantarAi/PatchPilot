"""Edge-case unit tests for review/, evals/, distillation/, and build/CI parity."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from patchpilot.config import LangfuseSettings
from patchpilot.distillation.compare import compare_eval_runs, load_run_reports
from patchpilot.evals.generator import build_and_materialize_all
from patchpilot.evals.runner import (
    EvalCaseRunResult,
    EvalIntegrityError,
    EvalRunReport,
    _compute_config_hash,
    run_eval_pool,
    wilson_ci_95,
)
from patchpilot.pipeline.nodes import compute_prompt_version_hash
from patchpilot.review.formatter import extract_diff_suggestions
from patchpilot.review.github import GitHubClient, GitHubError, parse_repo_slug
from patchpilot.review.runner import _load_local_repo_dir, run_review
from patchpilot.sandbox.runner import SandboxResult
from patchpilot.sandbox.tar_and_diff import JUnitReport
from patchpilot.tracing import LLMCallMetric


async def _fast_verify(files: dict[str, str], patch: str) -> SandboxResult:
    return SandboxResult(
        exit_code=0,
        patch_applied=bool(patch.strip()),
        junit=JUnitReport(passed=1, failures=0, errors=0),
    )


def test_compute_config_hash_preserves_default_and_isolates_explicit_run_id() -> None:
    """`_compute_config_hash` keeps legacy hash when `run_id=None` and isolates explicit run_ids."""
    prompt_hash = "ca774b4a6efb8c2384a6f216439068326840bd10ec0543a1032f135bc0f20782"
    default_hash = _compute_config_hash(
        pool="seeded_eval",
        provider="ollama",
        model_name="qwen2.5-coder:7b",
        seed=1,
        prompt_version_hash=prompt_hash,
    )
    assert default_hash == "0589c2787dde0251"
    assert (
        _compute_config_hash(
            pool="seeded_eval",
            provider="ollama",
            model_name="qwen2.5-coder:7b",
            seed=1,
            prompt_version_hash=prompt_hash,
            run_id=None,
        )
        == default_hash
    )

    trackb_hash = _compute_config_hash(
        pool="seeded_eval",
        provider="ollama",
        model_name="qwen2.5-coder:7b",
        seed=1,
        prompt_version_hash=prompt_hash,
        run_id="seeded_eval_ollama_trackb_s1",
    )
    trackc_hash = _compute_config_hash(
        pool="seeded_eval",
        provider="ollama",
        model_name="qwen2.5-coder:7b",
        seed=1,
        prompt_version_hash=prompt_hash,
        run_id="seeded_eval_ollama_trackc_s1",
    )
    assert trackb_hash != default_hash
    assert trackc_hash != default_hash
    assert trackb_hash != trackc_hash


def test_run_eval_pool_isolates_checkpoints_per_run_id_and_resumes_same_run_id(
    tmp_path: Path,
) -> None:
    """Distinct `--run-id` values never share checkpoints; re-running the same `--run-id` does."""
    datasets_dir = tmp_path / "datasets"
    results_dir = tmp_path / "results"
    build_and_materialize_all(
        base_seed=1337,
        datasets_dir=datasets_dir,
        eval_count=2,
        train_count=2,
    )

    executed_by_call: list[tuple[str, str]] = []
    lf_off = LangfuseSettings(host="", public_key="", secret_key="")

    async def _run() -> None:
        # 1. Run with run_id="run_alpha"
        executed_alpha: list[str] = []
        reports_alpha, _ = await run_eval_pool(
            pool="seeded_train",
            provider_name="mock",
            runs=1,
            seed=1,
            limit=2,
            datasets_dir=datasets_dir,
            results_dir=results_dir,
            run_id="run_alpha",
            verify_fn=_fast_verify,
            langfuse_settings=lf_off,
            before_case_hook=lambda cid, _r: executed_alpha.append(cid),
        )
        assert len(executed_alpha) == 2
        assert all(not c.resumed for c in reports_alpha[0].cases)

        # 2. Run with identical pool/provider/seed/limit but distinct run_id="run_beta"
        # Must NOT reuse run_alpha's checkpoints!
        executed_beta: list[str] = []
        reports_beta, _ = await run_eval_pool(
            pool="seeded_train",
            provider_name="mock",
            runs=1,
            seed=1,
            limit=2,
            datasets_dir=datasets_dir,
            results_dir=results_dir,
            run_id="run_beta",
            verify_fn=_fast_verify,
            langfuse_settings=lf_off,
            before_case_hook=lambda cid, _r: executed_beta.append(cid),
        )
        assert executed_beta == executed_alpha
        assert all(not c.resumed for c in reports_beta[0].cases)

        # 3. Re-run with same run_id="run_beta" -> must resume from run_beta's own checkpoints!
        executed_beta_resume: list[str] = []
        reports_beta_resumed, _ = await run_eval_pool(
            pool="seeded_train",
            provider_name="mock",
            runs=1,
            seed=1,
            limit=2,
            datasets_dir=datasets_dir,
            results_dir=results_dir,
            run_id="run_beta",
            verify_fn=_fast_verify,
            langfuse_settings=lf_off,
            before_case_hook=lambda cid, _r: executed_beta_resume.append(cid),
        )
        assert executed_beta_resume == []
        assert all(c.resumed for c in reports_beta_resumed[0].cases)

        ckpt_dirs = [d.name for d in (results_dir / "checkpoints").iterdir() if d.is_dir()]
        assert len(ckpt_dirs) == 2
        executed_by_call.extend(
            [("alpha", str(len(executed_alpha))), ("beta", str(len(executed_beta)))]
        )

    asyncio.run(_run())
    assert len(executed_by_call) == 2


def test_wilson_ci_95_edge_cases() -> None:
    """Verify `wilson_ci_95` handles `n=0`, `k=0`, `k=n`, and out-of-range inputs cleanly."""
    assert wilson_ci_95(0, 0) == (0.0, 0.0)
    assert wilson_ci_95(5, 0) == (0.0, 0.0)
    assert wilson_ci_95(0, -10) == (0.0, 0.0)

    for n in (1, 3, 14, 120):
        lo_zero, hi_zero = wilson_ci_95(0, n)
        assert lo_zero == 0.0
        assert 0.0 < hi_zero <= 1.0

        lo_full, hi_full = wilson_ci_95(n, n)
        assert 0.0 <= lo_full < 1.0
        assert hi_full == 1.0

    # Clamping out-of-bounds k
    assert wilson_ci_95(-5, 10) == wilson_ci_95(0, 10)
    assert wilson_ci_95(25, 10) == wilson_ci_95(10, 10)


def _write_fake_report(
    results_dir: Path,
    run_id: str,
    run_index: int,
    *,
    prompt_hash: str | None = None,
    provider: str = "ollama",
    pass1: int = 10,
    total: int = 12,
) -> Path:
    ph = prompt_hash or compute_prompt_version_hash()
    cases = [
        EvalCaseRunResult(
            case_id=f"{i:04d}",
            pool="seeded_eval",
            taxonomy="off_by_one",
            template="pagination_offset",
            run_index=run_index,
            seed=1,
            provider=provider,
            model_name="qwen2.5-coder:7b",
            prompt_version_hash=ph,
            attempts=1,
            verified=i <= pass1,
            pass_at_1=i <= pass1,
            pass_at_3=i <= pass1,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            llm_calls=[
                LLMCallMetric(
                    node="propose",
                    provider=provider,
                    model="qwen2.5-coder:7b",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    latency_s=0.5,
                    route_target="frontier" if provider in ("gemini", "openrouter") else "local",
                )
            ],
            total_time_s=1.0,
        )
        for i in range(1, total + 1)
    ]
    rep = EvalRunReport(
        run_id=run_id,
        run_index=run_index,
        timestamp="2026-11-13T00:00:00Z",
        pool="seeded_eval",
        provider=provider,
        model_name="qwen2.5-coder:7b",
        seed=1,
        prompt_version_hash=ph,
        total_cases=total,
        pass_at_1_count=pass1,
        pass_at_1_rate=round(pass1 / total, 6),
        pass_at_1_ci_95=wilson_ci_95(pass1, total),
        pass_at_3_count=pass1,
        pass_at_3_rate=round(pass1 / total, 6),
        pass_at_3_ci_95=wilson_ci_95(pass1, total),
        total_prompt_tokens=total * 100,
        total_completion_tokens=total * 50,
        total_tokens=total * 150,
        total_time_s=float(total),
        cases=cases,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{run_id}_run{run_index}.json"
    out.write_text(rep.model_dump_json(indent=2), encoding="utf-8")
    return out


def test_load_run_reports_prefix_boundary_and_compare_guards(tmp_path: Path) -> None:
    """`load_run_reports` isolates exact/prefix matches, ignores non-reports, rejects ambiguity."""
    results_dir = tmp_path / "results"
    for idx in (1, 2, 3):
        _write_fake_report(results_dir, "seeded_eval_ollama_s1_0589c278", idx, pass1=8)
        _write_fake_report(results_dir, "seeded_eval_ollama_s10_deadbeef", idx, pass1=9)
        _write_fake_report(results_dir, "baseline", idx, pass1=7)
        _write_fake_report(results_dir, "baseline_qlora", idx, pass1=11, provider="openrouter")
    # Partial limit run and non-report JSON in the same directory
    _write_fake_report(results_dir, "seeded_eval_ollama_s1_0589c278_limit2", 1, pass1=2, total=2)
    (results_dir / "seeded_eval_ollama_s1_corrupt.json").write_text(
        '{"not": "a report"}', encoding="utf-8"
    )

    # 1. Prefix `seeded_eval_ollama_s1` matches `s1_0589c278` (3 runs), never `s10` or `_limit2`
    s1_reports = load_run_reports("seeded_eval_ollama_s1", results_dir=results_dir)
    assert len(s1_reports) == 3
    assert [r.run_index for r in s1_reports] == [1, 2, 3]
    assert {r.run_id for r in s1_reports} == {"seeded_eval_ollama_s1_0589c278"}

    # 2. Exact `baseline` never pulls in `baseline_qlora`
    base_reports = load_run_reports("baseline", results_dir=results_dir)
    assert len(base_reports) == 3
    assert {r.run_id for r in base_reports} == {"baseline"}

    cand_reports = load_run_reports("baseline_qlora", results_dir=results_dir)
    assert len(cand_reports) == 3
    assert {r.run_id for r in cand_reports} == {"baseline_qlora"}

    # 3. Ambiguous prefix across two distinct run_ids raises EvalIntegrityError
    _write_fake_report(results_dir, "seeded_eval_ollama_s1_ffffffff", 1, pass1=8)
    with pytest.raises(EvalIntegrityError, match="Ambiguous run prefix"):
        load_run_reports("seeded_eval_ollama_s1", results_dir=results_dir)

    # 4. Prompt version hash mismatch between baseline and candidate fails closed
    mismatched_cand = [
        cand_reports[0].model_copy(update={"prompt_version_hash": "1" * 64}),
        cand_reports[1],
        cand_reports[2],
    ]
    with pytest.raises(EvalIntegrityError, match="prompt_version_hash changed"):
        compare_eval_runs(base_reports, mismatched_cand)

    # 5. Valid comparison succeeds and computes frontier cost delta
    cmp_report = compare_eval_runs(base_reports, cand_reports)
    assert cmp_report.baseline.n_runs == 3
    assert cmp_report.candidate.n_runs == 3
    assert cmp_report.candidate.estimated_cost_usd > 0.0
    assert cmp_report.cost_delta_usd > 0.0


def test_github_token_redaction_and_pagination_cycle_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GitHubError` redacts tokens everywhere and pagination stops on cyclic/relative links."""
    env_tok = "ghp_EnvSecretTokenValue999999999"
    monkeypatch.setenv("GITHUB_TOKEN", env_tok)

    # Direct GitHubError construction (e.g. invalid slug containing a token) redacts it
    with pytest.raises(GitHubError) as slug_exc:
        parse_repo_slug(f"https://{env_tok}@github.com/owner/repo")
    assert env_tok not in str(slug_exc.value)
    assert env_tok not in repr(slug_exc.value)
    assert "***REDACTED***" in str(slug_exc.value)

    # Pagination: relative Link header works, cyclic Link header terminates without duplicating
    calls = 0

    def cyclic_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        path = request.url.path
        if path == "/repos/o/r/pulls/1":
            if "diff" in request.headers.get("Accept", ""):
                return httpx.Response(200, text="")
            return httpx.Response(
                200,
                json={"head": {"sha": "h1"}, "base": {"sha": "b1", "repo": {"private": True}}},
            )
        if path == "/repos/o/r/pulls/1/files":
            calls += 1
            page = request.url.params.get("page", "1")
            if page == "1":
                # Relative URL pointing to page 2
                return httpx.Response(
                    200,
                    json=[{"filename": "a.py", "status": "added", "content": "x = 1\n"}],
                    headers={"Link": '</repos/o/r/pulls/1/files?per_page=100&page=2>; rel="next"'},
                )
            # Page 2 cycles back to page 1!
            next_link = '<https://api.github.com/repos/o/r/pulls/1/files?per_page=100>; rel="next"'
            return httpx.Response(
                200,
                json=[{"filename": "b.py", "status": "added", "content": "y = 2\n"}],
                headers={"Link": next_link},
            )
        return httpx.Response(404, json={})

    client = GitHubClient(transport=httpx.MockTransport(cyclic_handler))
    assert env_tok not in repr(client)
    assert env_tok not in str(client)
    ctx = asyncio.run(client.fetch_pull_request("o/r", 1))
    assert calls == 2
    assert set(ctx.files.keys()) == {"a.py", "b.py"}


def test_formatter_diff_edge_cases_dev_null_and_contextless_addition() -> None:
    """Deleted files (`+++ /dev/null`) do not bleed into prior files; contextless additions work."""
    diff_with_deleted_second_file = (
        "--- a/first.py\n"
        "+++ b/first.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-x = 1\n"
        "+x = 2\n"
        " y = 3\n"
        "--- a/deleted.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-dead = 1\n"
        "-dead2 = 2\n"
    )
    sugs = extract_diff_suggestions(diff_with_deleted_second_file)
    assert len(sugs) == 1
    assert sugs[0].file == "first.py"
    assert sugs[0].replacement_code == "x = 2"

    contextless_new_file_diff = (
        "--- /dev/null\n"
        "+++ b/pkg\\new_mod.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def greet() -> str:\n"
        "+    return 'hi'\n"
    )
    new_sugs = extract_diff_suggestions(contextless_new_file_diff)
    assert len(new_sugs) == 1
    assert new_sugs[0].file == "pkg/new_mod.py"
    assert new_sugs[0].start_line == 1
    assert new_sugs[0].end_line == 1
    assert "def greet() -> str:" in new_sugs[0].replacement_code


def test_local_repo_dir_inside_dot_parent_directory_loads_files(tmp_path: Path) -> None:
    """`_load_local_repo_dir` must not skip files when `repo_dir` has a dot-prefixed ancestor."""
    dot_parent_repo = tmp_path / ".work" / "nested_repo"
    dot_parent_repo.mkdir(parents=True)
    (dot_parent_repo / "app.py").write_text(
        "import os\n\ndef f() -> int:\n    return 1\n", encoding="utf-8"
    )
    (dot_parent_repo / ".hidden.py").write_text("secret = 1\n", encoding="utf-8")

    loaded = _load_local_repo_dir(dot_parent_repo)
    assert "app.py" in loaded
    assert ".hidden.py" not in loaded

    res = asyncio.run(
        run_review(
            repo_dir=dot_parent_repo,
            dry_run=True,
            provider_name="mock",
            verify_fn=_fast_verify,
        )
    )
    assert res.findings_count >= 1


def test_makefile_tasks_ps1_and_ci_workflow_parity() -> None:
    """Verify `Makefile` and `tasks.ps1` define the exact same task targets (AGENTS.md Rule 9)."""
    makefile_text = Path("Makefile").read_text(encoding="utf-8")
    tasks_text = Path("tasks.ps1").read_text(encoding="utf-8")
    ci_text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    phony_match = re.search(r"^\.PHONY:\s+(.+)$", makefile_text, flags=re.MULTILINE)
    assert phony_match is not None
    make_targets = phony_match.group(1).strip().split()

    validate_match = re.search(r"\[ValidateSet\(([^)]+)\)\]", tasks_text)
    assert validate_match is not None
    ps_targets = [t.strip().strip("'\"") for t in validate_match.group(1).split(",") if t.strip()]

    assert make_targets == ps_targets
    assert len(make_targets) == 14

    # CI workflow must run ruff, mypy, pytest with mock provider, verify-frozen, and compose config
    assert "uv run --frozen ruff check ." in ci_text
    assert "uv run --frozen ruff format --check ." in ci_text
    assert "uv run --frozen mypy src" in ci_text
    assert "uv run --frozen pytest -q" in ci_text
    assert "python -m patchpilot.evals verify-frozen" in ci_text
    assert "docker compose --profile observability config -q" in ci_text
    assert json.loads(Path("evals/FROZEN.json").read_text(encoding="utf-8"))["overall_sha256"]
