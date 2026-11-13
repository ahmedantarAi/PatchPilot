"""Unit tests for FastAPI /healthz, CLI subcommands, Settings/LangfuseSettings, and Tracer."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from patchpilot import __version__
from patchpilot.__main__ import (
    _load_case_files,
    _resolve_case_dir,
    _review_command,
    _run_command,
    _trace_smoke_command,
    load_mock_ground_truth,
    main,
)
from patchpilot.api import app, healthz
from patchpilot.config import LangfuseSettings, Settings, _find_duplicate_keys
from patchpilot.pipeline.models import PipelineState
from patchpilot.providers.base import ProviderError
from patchpilot.retrieval import HashEmbedder, OllamaEmbedder
from patchpilot.tracing import (
    PatchPilotTracer,
    _is_langfuse_container_running,
    _otel_attr,
    run_trace_smoke,
)

# ---------------------------------------------------------------------------
# 1. FastAPI /healthz endpoint tests
# ---------------------------------------------------------------------------


def test_api_healthz_direct_and_asgi_client() -> None:
    """FastAPI `/healthz` returns status ok and version both directly and over ASGI."""
    assert app.title == "PatchPilot"
    assert app.version == __version__
    assert healthz() == {"status": "ok", "version": __version__}

    async def _check_asgi() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/healthz")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok", "version": __version__}

            missing = await client.get("/does-not-exist")
            assert missing.status_code == 404

    asyncio.run(_check_asgi())


# ---------------------------------------------------------------------------
# 2. Settings & LangfuseSettings validation and secret redaction
# ---------------------------------------------------------------------------


def test_settings_redacts_secrets_in_repr_and_str() -> None:
    """`Settings.__repr__` and `__str__` must never leak Gemini or OpenRouter API keys."""
    gemini_secret = "AIzaSy-SUPER-SECRET-GEMINI-KEY-999"
    openrouter_secret = "sk-or-v1-SUPER-SECRET-OPENROUTER-KEY-888"
    cfg = Settings(
        _env_file=None,
        gemini_api_key=gemini_secret,
        gemini_model="gemini-3.6-flash",
        openrouter_api_key=openrouter_secret,
        openrouter_model="google/gemma-4-31b-it:free",
    )
    assert cfg.gemini_api_key == gemini_secret
    assert cfg.openrouter_api_key == openrouter_secret

    for rendered in (repr(cfg), str(cfg), f"{cfg}", f"{cfg!r}"):
        assert gemini_secret not in rendered
        assert openrouter_secret not in rendered
        assert "gemini_api_key='***REDACTED***'" in rendered
        assert "openrouter_api_key='***REDACTED***'" in rendered


def test_settings_graceful_empty_env_vars_and_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Empty PP_GEMINI_MODEL, PP_OPENROUTER_API_KEY, PP_LLM_PROVIDER, PP_FRONTIER_PROVIDER work."""
    monkeypatch.setenv("PP_GEMINI_MODEL", "")
    monkeypatch.setenv("PP_GEMINI_API_KEY", "")
    monkeypatch.setenv("PP_OPENROUTER_API_KEY", "")
    monkeypatch.setenv("PP_LLM_PROVIDER", "   ")
    monkeypatch.setenv("PP_FRONTIER_PROVIDER", "")

    cfg = Settings(_env_file=None)
    assert cfg.gemini_model == ""
    assert cfg.gemini_api_key == ""
    assert cfg.openrouter_api_key == ""
    assert cfg.llm_provider == "mock"
    assert cfg.frontier_provider == "openrouter"

    # Invalid non-empty provider still raises ValidationError
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="not_a_provider")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, frontier_provider="not_a_frontier")


def test_settings_duplicate_env_key_detection_and_env_file_none_isolation(
    tmp_path: Path,
) -> None:
    """Duplicate keys in `.env` raise ValueError, while `_env_file=None` skips disk `.env`."""
    dup_env = tmp_path / "dup.env"
    dup_env.write_text(
        "# comment\n\nPP_OLLAMA_MODEL=qwen2.5-coder:7b\nexport PP_OLLAMA_MODEL=other\n",
        encoding="utf-8",
    )
    assert _find_duplicate_keys(dup_env) == ["PP_OLLAMA_MODEL"]
    assert _find_duplicate_keys(tmp_path / "missing.env") == []

    with pytest.raises(ValueError, match="duplicate variable 'PP_OLLAMA_MODEL'"):
        Settings(_env_file=dup_env)

    # _env_file=None must not inspect any file
    clean = Settings(_env_file=None)
    assert clean.ollama_model != ""


def test_langfuse_settings_redaction_and_edge_cases() -> None:
    """`LangfuseSettings` redacts both `secret_key` and `init_project_secret_key` and validates."""
    sec1 = "sk-lf-PRIMARY-SECRET-111"
    sec2 = "sk-lf-INIT-SECRET-222"
    lf = LangfuseSettings(
        _env_file=None,
        host="  http://localhost:3000/  ",
        public_key="",
        secret_key="",
        init_project_public_key=" pk-lf-init-pub ",
        init_project_secret_key=f" {sec2} ",
    )
    assert lf.effective_public_key == "pk-lf-init-pub"
    assert lf.effective_secret_key == sec2
    assert lf.is_configured is True

    lf_direct = LangfuseSettings(
        _env_file=None,
        host="http://localhost:3000",
        public_key="pk-lf-pub",
        secret_key=sec1,
        init_project_secret_key=sec2,
    )
    for text in (repr(lf), str(lf), repr(lf_direct), str(lf_direct)):
        assert sec1 not in text
        assert sec2 not in text
        assert "***REDACTED***" in text

    # Case-insensitive CHANGEME and blank host checks
    assert (
        LangfuseSettings(
            _env_file=None,
            host="http://localhost:3000",
            public_key="pk-lf-changeme-123",
            secret_key="sk-lf-valid",
        ).is_configured
        is False
    )
    assert (
        LangfuseSettings(
            _env_file=None,
            host="   ",
            public_key="pk-lf-valid",
            secret_key="sk-lf-valid",
        ).is_configured
        is False
    )


# ---------------------------------------------------------------------------
# 3. PatchPilotTracer OTLP/HTTP timeout, malformed response & no-op resilience
# ---------------------------------------------------------------------------


def test_tracer_noop_on_timeout_invalid_url_and_bad_json() -> None:
    """`PatchPilotTracer` never raises on HTTP timeouts, invalid URLs, 5xx, or bad JSON."""

    async def _run() -> None:
        cfg = LangfuseSettings(
            _env_file=None,
            host="http://localhost:3000",
            public_key="pk-lf-test",
            secret_key="sk-lf-secret-do-not-expose",
        )

        def timeout_handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("simulated OTLP timeout", request=req)

        tracer = PatchPilotTracer(
            settings=cfg,
            timeout_s=0.05,
            transport=httpx.MockTransport(timeout_handler),
        )
        assert "sk-lf-secret-do-not-expose" not in repr(tracer)
        assert "sk-lf-secret-do-not-expose" not in str(tracer)

        assert await tracer.check_health(wait_s=0.0) is False
        tracer.start_trace(case_id="case_timeout")
        sid, pstart, istart = tracer.start_node_span("analyze", {"files": 2})
        llm_metric = tracer.record_llm_call(
            node="analyze",
            model="qwen2.5-coder:7b",
            provider="ollama",
            prompt_tokens=-5,
            completion_tokens=-3,
            latency_s=-1.0,
            parent_span_id=sid,
            route_target="local",
            decision_reason="fallback",
            fallback_used=True,
            warning_flag="frontier_fallback:timeout",
            estimated_cost_usd=-0.5,
        )
        assert llm_metric.prompt_tokens == 0
        assert llm_metric.completion_tokens == 0
        assert llm_metric.total_tokens == 0
        assert llm_metric.latency_s == 0.0
        assert llm_metric.estimated_cost_usd == 0.0

        tracer.end_node_span(sid, "analyze", pstart, istart, {"findings": 0})
        sbx_metric = tracer.record_sandbox_run(
            node="verify",
            exit_code=1,
            duration_s=-0.5,
            patch_applied=False,
            passed=-1,
            failures=-2,
            errors=-3,
        )
        assert sbx_metric.duration_s == 0.0
        assert (sbx_metric.passed, sbx_metric.failures, sbx_metric.errors) == (0, 0, 0)

        assert (
            await tracer.finish_trace(
                case_id="case_timeout",
                verified=False,
                pass_at_1=False,
                pass_at_k=False,
                attempts=1,
                metadata={"pool": "seeded_train"},
            )
            is False
        )
        assert (
            await tracer.verify_trace_visible(
                timeout_s=0.05,
                poll_interval_s=0.01,
            )
            is False
        )
        assert await tracer.ensure_dataset("seeded_train") is False
        assert (
            await tracer.upsert_dataset_item(
                dataset_name="seeded_train",
                item_id="item_1",
                input_data={},
                expected_output={},
            )
            is False
        )
        assert (
            await tracer.create_dataset_run_item(
                run_name="run_1",
                run_description="desc",
                item_id="item_1",
                trace_id=tracer.trace_id,
            )
            is False
        )

        # Malformed JSON / 500 response handler
        def bad_json_handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/api/public/otel/v1/traces":
                return httpx.Response(500, text="Internal Server Error")
            return httpx.Response(200, content=b"{not-valid-json")

        tracer_bad = PatchPilotTracer(
            settings=cfg,
            timeout_s=0.05,
            transport=httpx.MockTransport(bad_json_handler),
        )
        tracer_bad.record_llm_call(
            node="propose",
            model="m",
            provider="mock",
            prompt_tokens=1,
            completion_tokens=1,
            latency_s=0.01,
        )
        assert await tracer_bad.flush_events() is False
        assert await tracer_bad.verify_trace_visible(timeout_s=0.05, poll_interval_s=0.01) is False

        # Disabled tracer no-ops immediately
        disabled = PatchPilotTracer(settings=cfg, enabled=False)
        assert disabled.is_configured() is False
        assert await disabled.check_health() is False
        assert await disabled.flush_events() is False
        assert await disabled.verify_trace_visible(timeout_s=0.01) is False
        assert await disabled.ensure_dataset("d") is False
        assert (
            await disabled.upsert_dataset_item(
                dataset_name="d", item_id="i", input_data={}, expected_output={}
            )
            is False
        )
        assert (
            await disabled.create_dataset_run_item(
                run_name="r", run_description="d", item_id="i", trace_id="t"
            )
            is False
        )

    asyncio.run(_run())


def test_tracer_dataset_v1_fallback_and_trace_endpoint_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback from v2 datasets to v1 datasets and trace visibility via `/traces/<id>`."""

    async def _run() -> None:
        cfg = LangfuseSettings(
            _env_file=None,
            host="http://localhost:3000",
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
        )

        def fallback_handler(req: httpx.Request) -> httpx.Response:
            path = req.url.path
            if path == "/api/public/v2/datasets":
                return httpx.Response(404)
            if path == "/api/public/datasets":
                return httpx.Response(409)  # Already exists -> True
            if path == "/api/public/v2/observations":
                return httpx.Response(200, json={"data": []})
            if path.startswith("/api/public/traces/"):
                tid = path.rsplit("/", 1)[-1]
                return httpx.Response(200, json={"id": tid})
            if path == "/api/public/health":
                return httpx.Response(200)
            if path == "/api/public/otel/v1/traces":
                return httpx.Response(503)  # Ingestion failed -> run_trace_smoke degrades to no-op
            return httpx.Response(404)

        tracer = PatchPilotTracer(settings=cfg, transport=httpx.MockTransport(fallback_handler))
        assert await tracer.ensure_dataset("pool_x") is True
        assert await tracer.verify_trace_visible(timeout_s=0.2, poll_interval_s=0.01) is True

        ok, msg = await run_trace_smoke(
            settings=cfg,
            transport=httpx.MockTransport(fallback_handler),
        )
        assert ok is False
        assert msg == "tracing disabled: no-op"

    asyncio.run(_run())

    # _otel_attr type coverage
    assert _otel_attr("b", True) == {"key": "b", "value": {"boolValue": True}}
    assert _otel_attr("i", 42) == {"key": "i", "value": {"intValue": "42"}}
    assert _otel_attr("f", 1.5) == {"key": "f", "value": {"doubleValue": 1.5}}
    assert _otel_attr("s", "hi") == {"key": "s", "value": {"stringValue": "hi"}}

    # _is_langfuse_container_running handles OSError cleanly
    def raise_oserror(*args: Any, **kwargs: Any) -> Any:
        raise OSError("docker not found")

    monkeypatch.setattr(subprocess, "run", raise_oserror)
    assert _is_langfuse_container_running() is False


# ---------------------------------------------------------------------------
# 4. CLI subcommands (`run`, `review`, `trace-smoke`) & helper edge cases
# ---------------------------------------------------------------------------


def test_cli_resolve_and_load_case_files_edge_cases(tmp_path: Path) -> None:
    """`_resolve_case_dir`, `_load_case_files`, and `load_mock_ground_truth` guard edge cases."""
    for bad in ("", "   ", ".", "..", "nonexistent_case_id_999999"):
        with pytest.raises(FileNotFoundError):
            _resolve_case_dir(bad)

    assert _resolve_case_dir("0001").is_dir()
    assert _resolve_case_dir("seeded_eval/0001").is_dir()

    # Missing repo/ directory raises FileNotFoundError
    empty_case = tmp_path / "case_no_repo"
    empty_case.mkdir()
    with pytest.raises(FileNotFoundError, match="repo directory not found"):
        _load_case_files(empty_case)

    # Binary .pyc and __pycache__ files are skipped while UTF-8 .py files load cleanly
    case_dir = tmp_path / "case_utf8"
    repo_dir = case_dir / "repo"
    pycache = repo_dir / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "mod.cpython-311.pyc").write_bytes(b"\xff\xfe\xfd\x00binary")
    (repo_dir / "broken.bin").write_bytes(b"\xff\xfe\xfd\x80invalid_utf8")
    (repo_dir / "mod.py").write_text("# café Ω\nx = 1\n", encoding="utf-8")

    loaded = _load_case_files(case_dir)
    assert list(loaded.keys()) == ["mod.py"]
    assert "café Ω" in loaded["mod.py"]

    # Ground-truth loader handles missing files, null patch_diff, corrupt JSON, and valid meta.json
    assert load_mock_ground_truth(case_dir) == ""
    (case_dir / "meta.json").write_text('{"patch_diff": null}', encoding="utf-8")
    assert load_mock_ground_truth(case_dir) == ""
    (case_dir / "meta.json").write_text("{corrupt json", encoding="utf-8")
    assert load_mock_ground_truth(case_dir) == ""
    (case_dir / "meta.json").write_text('{"patch_diff": "--- a/mod.py\\n"}', encoding="utf-8")
    assert load_mock_ground_truth(case_dir) == "--- a/mod.py\n"
    (case_dir / "patch.diff").write_text("--- a/direct.py\n", encoding="utf-8")
    assert load_mock_ground_truth(case_dir) == "--- a/direct.py\n"


def test_cli_subcommands_run_trace_smoke_and_review(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify `main()` and subcommand handlers for `run`, `trace-smoke`, and `review`."""
    captured_embedders: list[Any] = []

    async def fake_run_pipeline(
        state: PipelineState,
        *,
        provider: Any,
        sandbox_runner: Any = None,
        qdrant: Any = None,
        embedder: Any = None,
        **kwargs: Any,
    ) -> PipelineState:
        captured_embedders.append(embedder)
        return state.model_copy(
            update={
                "patch_applies": True,
                "target_test_passes": True,
                "regressions": 0,
                "verified": True,
            }
        )

    monkeypatch.setattr("patchpilot.__main__.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        "patchpilot.__main__.SandboxRunner",
        lambda: object(),
    )

    # 1. `run` with mock provider uses HashEmbedder and exits 0
    with pytest.raises(SystemExit) as exc_run:
        main(["run", "--case", "seeded_eval/0001", "--provider", "mock"])
    assert exc_run.value.code == 0
    out_run = capsys.readouterr().out.strip()
    parsed = json.loads(out_run)
    assert parsed["case"] == "seeded_eval/0001"
    assert parsed["provider"] == "mock"
    assert parsed["verified"] is True
    assert isinstance(captured_embedders[-1], HashEmbedder)

    # 2. `run` with router provider uses OllamaEmbedder
    rc_router = asyncio.run(_run_command("seeded_eval/0001", "router"))
    assert rc_router == 0
    assert isinstance(captured_embedders[-1], OllamaEmbedder)
    capsys.readouterr()

    # 3. `run` with missing case prints ERROR to stderr and exits 1
    with pytest.raises(SystemExit) as exc_bad_run:
        main(["run", "--case", "nonexistent_case_xyz", "--provider", "mock"])
    assert exc_bad_run.value.code == 1
    err_run = capsys.readouterr().err
    assert "ERROR:" in err_run

    # 4. `trace-smoke` subcommand prints smoke result and exits 0
    async def fake_smoke(provider_name: str = "mock") -> tuple[bool, str]:
        return False, f"tracing disabled: no-op ({provider_name})"

    monkeypatch.setattr("patchpilot.__main__.run_trace_smoke", fake_smoke)
    with pytest.raises(SystemExit) as exc_smoke:
        main(["trace-smoke", "--provider", "mock"])
    assert exc_smoke.value.code == 0
    assert "tracing disabled: no-op (mock)" in capsys.readouterr().out

    # 5. `review` subcommand handles both success and ProviderError/ValueError cleanly
    with pytest.raises(SystemExit) as exc_rev_err:
        main(["review", "--provider", "mock"])  # neither --repo-dir nor --repo/--pr
    assert exc_rev_err.value.code == 1
    assert "ERROR:" in capsys.readouterr().err

    async def failing_review(**kwargs: Any) -> Any:
        raise ProviderError("gemini key missing")

    monkeypatch.setattr("patchpilot.review.runner.run_review", failing_review)
    rc_rev_prov = asyncio.run(
        _review_command(
            repo_dir=str(tmp_path),
            repo=None,
            pr_number=None,
            dry_run=True,
            provider_name="gemini",
            report_out=None,
        )
    )
    assert rc_rev_prov == 1
    assert "ERROR: gemini key missing" in capsys.readouterr().err

    assert asyncio.run(_trace_smoke_command("mock")) == 0
