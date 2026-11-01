"""Tests for Phase 5: Langfuse tracing wrapper, no-op degradation, and resumable eval runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from patchpilot.config import LangfuseSettings
from patchpilot.evals.runner import (
    EvalIntegrityError,
    build_eval_summary,
    format_eval_summary,
    run_eval_pool,
    wilson_ci_95,
)
from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import PipelineState
from patchpilot.pipeline.nodes import compute_prompt_version_hash
from patchpilot.providers.mock import MockProvider
from patchpilot.sandbox.runner import JUnitReport, SandboxResult
from patchpilot.tracing import PatchPilotTracer, run_trace_smoke

# ---------------------------------------------------------------------------
# 1. Langfuse tracing wrapper & no-op degradation
# ---------------------------------------------------------------------------


def test_langfuse_settings_redacts_secret_key() -> None:
    cfg = LangfuseSettings(
        _env_file=None,
        host="http://localhost:3000",
        public_key="pk-lf-test-public",
        secret_key="sk-lf-SUPER-SECRET-VALUE",
    )
    assert cfg.is_configured is True
    assert "sk-lf-SUPER-SECRET-VALUE" not in repr(cfg)
    assert "sk-lf-SUPER-SECRET-VALUE" not in str(cfg)
    assert "***REDACTED***" in repr(cfg)


def test_trace_smoke_degrades_to_noop_when_unconfigured_or_down() -> None:
    async def _test() -> None:
        # 1. Unconfigured (CHANGEME / empty)
        unconfigured = LangfuseSettings(
            _env_file=None,
            host="http://localhost:3000",
            public_key="pk-lf-local",
            secret_key="sk-lf-CHANGEME",
        )
        ok1, msg1 = await run_trace_smoke(settings=unconfigured)
        assert ok1 is False
        assert msg1 == "tracing disabled: no-op"

        # 2. Configured but Langfuse is down (HTTP 503 / connection refused)
        def down_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        configured = LangfuseSettings(
            _env_file=None,
            host="http://127.0.0.1:59998",
            public_key="pk-lf-local",
            secret_key="sk-lf-valid-secret",
        )
        ok2, msg2 = await run_trace_smoke(
            settings=configured,
            transport=httpx.MockTransport(down_handler),
        )
        assert ok2 is False
        assert msg2 == "tracing disabled: no-op"

    asyncio.run(_test())


def test_trace_smoke_succeeds_when_langfuse_up() -> None:
    async def _test() -> None:
        ingested_batches: list[dict[str, Any]] = []
        stored_traces: set[str] = set()

        def up_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/public/health":
                return httpx.Response(200, json={"status": "OK"})
            if path == "/api/public/otel/v1/traces" and request.method == "POST":
                body = json.loads(request.content.decode("utf-8"))
                ingested_batches.append(body)
                for rs in body.get("resourceSpans", []):
                    for ss in rs.get("scopeSpans", []):
                        for sp in ss.get("spans", []):
                            tid = sp.get("traceId")
                            if isinstance(tid, str):
                                stored_traces.add(tid)
                return httpx.Response(200, json={"name": "otel-ingestion-job"})
            if path == "/api/public/v2/observations" and request.method == "GET":
                tid = request.url.params.get("traceId", "")
                if tid in stored_traces:
                    return httpx.Response(200, json={"data": [{"traceId": tid}]})
                return httpx.Response(200, json={"data": []})
            if path.startswith("/api/public/traces/") and request.method == "GET":
                tid = path.rsplit("/", 1)[-1]
                if tid in stored_traces:
                    return httpx.Response(200, json={"id": tid})
                return httpx.Response(404, json={"message": "Not found"})
            return httpx.Response(404)

        cfg = LangfuseSettings(
            _env_file=None,
            host="http://localhost:3000",
            public_key="pk-lf-local",
            secret_key="sk-lf-valid-secret",
        )
        ok, msg = await run_trace_smoke(
            settings=cfg,
            transport=httpx.MockTransport(up_handler),
        )
        assert ok is True
        assert msg.startswith("trace_id=")
        assert msg.endswith(" visible=true")
        trace_hex = msg.split("trace_id=", 1)[1].split(" ", 1)[0]
        assert len(trace_hex) == 32
        assert len(ingested_batches) == 1

    asyncio.run(_test())


def test_pipeline_records_node_spans_llm_calls_and_sandbox_runs_and_works_when_langfuse_down() -> (
    None
):
    """One trace per case run, spans for each graph node, each LLM call, and each sandbox run."""

    async def _test() -> None:
        files = {
            "calc.py": "def inc(x: int) -> int:\n    return x - 1\n",
            "test_calc.py": "from calc import inc\ndef test_inc(): assert inc(1) == 2\n",
        }
        good_patch = (
            "<<<<<<< SEARCH:calc.py\n    return x - 1\n=======\n    return x + 1\n>>>>>>> REPLACE"
        )

        def down_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Langfuse down")

        down_cfg = LangfuseSettings(
            _env_file=None,
            host="http://127.0.0.1:59997",
            public_key="pk-lf-local",
            secret_key="sk-lf-valid-secret",
        )
        tracer = PatchPilotTracer(
            settings=down_cfg,
            transport=httpx.MockTransport(down_handler),
        )

        async def mock_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                duration_s=0.12,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        final_state = await run_pipeline(
            PipelineState(case_id="trace_test_case", files=files, data_class="public"),
            provider=MockProvider(canned=good_patch),
            verify_fn=mock_verify,
            tracer=tracer,
        )

        # Pipeline succeeds even when Langfuse is completely down
        assert final_state.verified is True
        assert len(final_state.trace_id) == 32
        assert final_state.prompt_version_hash == compute_prompt_version_hash()

        # Spans recorded for all 4 graph nodes: analyze, propose, verify, report
        node_names = [s.node for s in tracer.node_spans]
        assert node_names == ["analyze", "propose", "verify", "report"]

        # LLM calls recorded for analyze and propose with model, tokens, and latency
        assert len(final_state.llm_calls) == 2
        assert [c.node for c in final_state.llm_calls] == ["analyze", "propose"]
        for call in final_state.llm_calls:
            assert call.model == "mock-1"
            assert call.prompt_tokens > 0
            assert call.completion_tokens > 0
            assert call.total_tokens == call.prompt_tokens + call.completion_tokens
            assert call.latency_s >= 0.0

        # Sandbox run recorded for verify
        assert len(final_state.sandbox_runs) == 1
        assert final_state.sandbox_runs[0].node == "verify"
        assert final_state.sandbox_runs[0].exit_code == 0
        assert final_state.sandbox_runs[0].patch_applied is True

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 2. Wilson confidence intervals & eval integrity (fail closed)
# ---------------------------------------------------------------------------


def test_wilson_ci_95_properties() -> None:
    assert wilson_ci_95(0, 0) == (0.0, 0.0)

    lo_0, hi_0 = wilson_ci_95(0, 10)
    assert lo_0 == 0.0
    assert 0.25 < hi_0 < 0.30

    lo_10, hi_10 = wilson_ci_95(10, 10)
    assert 0.70 < lo_10 < 0.75
    assert hi_10 == 1.0

    lo_half, hi_half = wilson_ci_95(50, 100)
    assert 0.40 < lo_half < 0.41
    assert 0.59 < hi_half < 0.60


def test_eval_runner_fails_closed_if_frozen_split_invalid(tmp_path: Path) -> None:
    """verify-frozen must pass before any seeded_eval run (fail closed)."""

    async def _test() -> None:
        bad_frozen = tmp_path / "BROKEN_FROZEN.json"
        bad_frozen.write_text(
            json.dumps(
                {
                    "generator_version": "1.0.0",
                    "base_seed": 1337,
                    "pools": {},
                    "overall_sha256": "0" * 64,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(EvalIntegrityError, match="verify-frozen failed"):
            await run_eval_pool(
                pool="seeded_eval",
                provider_name="mock",
                runs=1,
                seed=1337,
                limit=1,
                results_dir=tmp_path / "results",
                frozen_path=bad_frozen,
            )

    asyncio.run(_test())


def test_eval_runner_refuses_summary_when_prompt_version_hash_changes(tmp_path: Path) -> None:
    """Runner must refuse to publish a summary if prompt_version_hash changed between runs."""

    async def _test() -> None:
        async def fast_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        reports, summary = await run_eval_pool(
            pool="seeded_train",
            provider_name="mock",
            runs=2,
            seed=1337,
            limit=2,
            results_dir=tmp_path / "results",
            verify_fn=fast_verify,
            langfuse_settings=LangfuseSettings(_env_file=None),
        )
        assert len(reports) == 2
        assert summary.n_runs == 2

        # Mutate the second run's prompt_version_hash to simulate a prompt edit mid-comparison
        tampered_run_2 = reports[1].model_copy(update={"prompt_version_hash": "f" * 64})
        with pytest.raises(
            EvalIntegrityError,
            match="Refusing to publish summary: prompt_version_hash changed",
        ):
            build_eval_summary([reports[0], tampered_run_2])

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 3. Resumable checkpoints per (case, run, config) & Langfuse dataset sync
# ---------------------------------------------------------------------------


def test_eval_runner_resumable_checkpoint_recovers_after_crash(tmp_path: Path) -> None:
    """Crash on case 3 of 4 -> re-run resumes from case 3 without re-running cases 1 and 2."""

    async def _test() -> None:
        executed_cases: list[str] = []

        async def fast_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        def crash_on_third(case_id: str, run_idx: int) -> None:
            executed_cases.append(case_id)
            if len(executed_cases) == 3:
                raise RuntimeError("Simulated crash / network drop on 3rd case")

        results_dir = tmp_path / "results"
        with pytest.raises(RuntimeError, match="Simulated crash"):
            await run_eval_pool(
                pool="seeded_train",
                provider_name="mock",
                runs=1,
                seed=1337,
                limit=4,
                results_dir=results_dir,
                verify_fn=fast_verify,
                langfuse_settings=LangfuseSettings(_env_file=None),
                before_case_hook=crash_on_third,
            )

        # Cases 0001 and 0002 completed before the crash on 0003
        assert executed_cases == ["0001", "0002", "0003"]

        resumed_executions: list[str] = []

        def track_resumed(case_id: str, run_idx: int) -> None:
            resumed_executions.append(case_id)

        reports, summary = await run_eval_pool(
            pool="seeded_train",
            provider_name="mock",
            runs=1,
            seed=1337,
            limit=4,
            results_dir=results_dir,
            verify_fn=fast_verify,
            langfuse_settings=LangfuseSettings(_env_file=None),
            before_case_hook=track_resumed,
        )

        # Only 0003 and 0004 executed on the resumed run; 0001 and 0002 loaded from checkpoint!
        assert resumed_executions == ["0003", "0004"]
        assert len(reports) == 1
        assert len(reports[0].cases) == 4
        assert [c.resumed for c in reports[0].cases] == [True, True, False, False]

        summary_text = format_eval_summary(summary)
        assert "only 14 templates" in summary_text
        assert "95% Wilson CI" in summary_text
        assert "Breakdown by bug class" in summary_text
        assert "Breakdown by template" in summary_text

    asyncio.run(_test())


def test_eval_runner_syncs_langfuse_dataset_when_observability_up(tmp_path: Path) -> None:
    """When observability is up, evals run creates dataset, items, and dataset-run-items."""

    async def _test() -> None:
        datasets_created: list[str] = []
        items_created: list[str] = []
        run_items_created: list[dict[str, Any]] = []

        def lf_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/public/health":
                return httpx.Response(200, json={"status": "OK"})
            if path == "/api/public/otel/v1/traces":
                return httpx.Response(200, json={"name": "otel-ingestion-job"})
            if path in ("/api/public/v2/datasets", "/api/public/datasets"):
                body = json.loads(request.content.decode("utf-8"))
                datasets_created.append(str(body.get("name")))
                return httpx.Response(200, json=body)
            if path == "/api/public/dataset-items":
                body = json.loads(request.content.decode("utf-8"))
                items_created.append(str(body.get("id")))
                return httpx.Response(200, json=body)
            if path == "/api/public/dataset-run-items":
                body = json.loads(request.content.decode("utf-8"))
                run_items_created.append(body)
                return httpx.Response(200, json=body)
            return httpx.Response(404)

        async def fast_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        lf_cfg = LangfuseSettings(
            _env_file=None,
            host="http://localhost:3000",
            public_key="pk-lf-local",
            secret_key="sk-lf-valid",
        )
        reports, _ = await run_eval_pool(
            pool="seeded_train",
            provider_name="mock",
            runs=1,
            seed=1337,
            limit=2,
            results_dir=tmp_path / "results",
            verify_fn=fast_verify,
            langfuse_settings=lf_cfg,
            langfuse_transport=httpx.MockTransport(lf_handler),
        )

        assert reports[0].langfuse_dataset_synced is True
        assert datasets_created == ["seeded_train"]
        assert items_created == ["seeded_train_0001", "seeded_train_0002"]
        assert len(run_items_created) == 2
        assert run_items_created[0]["traceId"] == reports[0].cases[0].trace_id

    asyncio.run(_test())
