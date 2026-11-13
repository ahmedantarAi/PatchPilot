"""Tests for AST symbol chunking, traceback extraction, and budgeted context slicing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import Finding, PipelineState
from patchpilot.pipeline.nodes import (
    DEFAULT_MAX_TOKENS,
    create_propose_node,
    create_verify_node,
    extract_traceback_frames,
    run_llm_diff_analysis,
    run_ruff_check,
)
from patchpilot.providers.base import LLMRequest, LLMResponse
from patchpilot.providers.mock import MockProvider
from patchpilot.retrieval import HashEmbedder, QdrantStore, extract_ast_chunks
from patchpilot.router.features import _extract_embedded_files, extract_task_features
from patchpilot.sandbox.runner import JUnitReport, SandboxResult


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashembedder_uses_word_tokenization() -> None:
    """HashEmbedder lowercases and tokenizes identifiers/punctuation via re.findall(r'\\w+')."""

    async def _test() -> None:
        embedder = HashEmbedder(dim=64)
        vecs = await embedder.embed(
            [
                "def compute_total(user_profile):",
                "COMPUTE_TOTAL USER_PROFILE",
                "completely_unrelated_xyz",
            ]
        )
        assert _cosine(vecs[0], vecs[1]) > 0.6
        assert _cosine(vecs[0], vecs[2]) < 0.3

    asyncio.run(_test())


def test_extract_ast_chunks_nodes_and_syntax_fallback() -> None:
    """extract_ast_chunks splits module header, ClassDef/methods, FunctionDef, AsyncFunctionDef."""
    src = (
        "import os\n"
        "from typing import Any\n"
        "\n"
        "def top_sync(x: int) -> int:\n"
        "    return x + 1\n"
        "\n"
        "async def top_async(y: str) -> str:\n"
        "    return y.strip()\n"
        "\n"
        "class Worker:\n"
        '    """Worker docstring."""\n'
        "    rate = 2\n"
        "\n"
        "    def run(self, v: int) -> int:\n"
        "        return v * self.rate\n"
        "\n"
        "    async def arun(self, v: int) -> int:\n"
        "        return v * self.rate\n"
    )
    chunks = extract_ast_chunks("./pkg/worker.py", src)
    symbols = [c["symbol"] for c in chunks]
    assert symbols == [
        "<module>",
        "top_sync",
        "top_async",
        "Worker",
        "Worker.run",
        "Worker.arun",
    ]
    for ch in chunks:
        assert set(ch) == {"file", "symbol", "start_line", "end_line", "text"}
        assert ch["file"] == "pkg/worker.py"
        assert 1 <= ch["start_line"] <= ch["end_line"]
        assert ch["text"].strip() != ""

    # SyntaxError falls back gracefully to line windows
    broken_lines = [f"x_{i} = {i}" for i in range(75)]
    broken_lines[10] = "def broken(:"
    fallback_chunks = extract_ast_chunks("broken.py", "\n".join(broken_lines))
    assert len(fallback_chunks) >= 2
    assert fallback_chunks[0]["start_line"] == 1
    assert fallback_chunks[0]["end_line"] == 60
    assert fallback_chunks[-1]["end_line"] == 75
    for ch in fallback_chunks:
        assert set(ch) == {"file", "symbol", "start_line", "end_line", "text"}


def test_qdrant_index_case_files_stores_ast_chunk_metadata() -> None:
    """QdrantStore.index_case_files indexes AST chunks with line/symbol metadata, skipping tests."""

    async def _test() -> None:
        upserted: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/healthz":
                return httpx.Response(200, text="ok")
            if path.startswith("/collections/") and path.endswith("/points"):
                body = json.loads(request.content.decode("utf-8"))
                upserted.extend(body.get("points", []))
                return httpx.Response(200, json={"result": {"status": "completed"}})
            if path.startswith("/collections/"):
                return httpx.Response(200, json={"result": True})
            return httpx.Response(404)

        store = QdrantStore(transport=httpx.MockTransport(handler))
        embedder = HashEmbedder(dim=64)
        files = {
            "pkg/calc.py": (
                "import math\n\ndef fn_one(x: int) -> int:\n    return x\n\n"
                "def fn_two(y: int) -> int:\n    return y * 2\n"
            ),
            "tests/test_calc.py": "def test_fn():\n    assert True\n",
        }
        col = await store.index_case_files(files, embedder)
        assert col is not None
        assert len(upserted) == 3  # <module>, fn_one, fn_two (tests excluded)
        payloads = [pt["payload"] for pt in upserted]
        assert [p["symbol"] for p in payloads] == ["<module>", "fn_one", "fn_two"]
        for p in payloads:
            assert set(p) == {"file", "symbol", "start_line", "end_line", "text"}
            assert p["file"] == "pkg/calc.py"

    asyncio.run(_test())


def test_extract_traceback_frames_innermost_first_and_analyze_fallback() -> None:
    """extract_traceback_frames orders innermost frames first and analyze uses them."""
    files = {
        "pkg/alpha.py": "def alpha():\n    return 1\n",
        "pkg/beta.py": (
            "def outer(val):\n    return inner(val)\n\ndef inner(val):\n    return val.strip()\n"
        ),
        "tests/test_beta.py": (
            "from pkg.beta import outer\n\ndef test_outer():\n    assert outer(None) == 'x'\n"
        ),
    }
    pytest_output = (
        "___________________________ test_outer ___________________________\n"
        "def test_outer():\n"
        ">       assert outer(None) == 'x'\n\n"
        "/work/tests/test_beta.py:4:\n"
        "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n"
        "/work/pkg/beta.py:2: in outer\n"
        "    return inner(val)\n"
        "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n"
        "/work/pkg/beta.py:5: AttributeError\n"
        "FAILED tests/test_beta.py::test_outer - AttributeError: 'NoneType'\n"
    )
    frames = extract_traceback_frames(pytest_output, files)
    assert frames[0] == ("pkg/beta.py", 5, "inner")
    assert frames[1] == ("pkg/beta.py", 2, "outer")
    assert frames[2] == ("tests/test_beta.py", 4, "test_outer")

    # When LLM output has no FINDING: line and no filename, analyze uses innermost source frame
    async def _run() -> list[Finding]:
        provider = MockProvider(canned="There is an attribute error on None value.")
        return await run_llm_diff_analysis(
            files=files,
            provider=provider,
            data_class="public",
            findings=[],
            failing_test_output=pytest_output,
        )

    findings = asyncio.run(_run())
    assert len(findings) == 1
    assert findings[0].file == "pkg/beta.py"
    assert findings[0].line == 5


def test_run_ruff_check_skips_test_files() -> None:
    """run_ruff_check only lints modifiable non-test .py files."""
    files = {
        "clean_src.py": "def ok(x: int) -> int:\n    return x + 1\n",
        "test_dirty.py": "import unused_mod\ndef test_ok():\n    x = 1\n    assert True\n",
        "tests/helper.py": "import os\n",
        "conftest.py": "import sys\n",
    }
    findings = run_ruff_check(files)
    assert findings == []


def test_budgeted_slicing_on_large_multifile_repo_without_duplication() -> None:
    """Large multi-file repos are sliced to imports + hit functions without duplicate dumps."""
    big_lines_a = ["import math", ""]
    for i in range(1, 35):
        big_lines_a.extend(
            [
                f"def filler_a_{i}(x: int) -> int:",
                f"    val = x + {i}",
                "    return val",
                "",
            ]
        )
    big_lines_a.extend(
        [
            "def target_buggy_fn(name: str | None) -> str:",
            "    return name.strip()",
            "",
        ]
    )
    big_src_a = "\n".join(big_lines_a)
    hit_line_a = len(big_lines_a) - 1  # line inside target_buggy_fn

    big_lines_b = ["import json", ""]
    for i in range(1, 35):
        big_lines_b.extend(
            [
                f"def filler_b_{i}(y: int) -> int:",
                f"    return y * {i}",
                "",
            ]
        )
    big_lines_b.extend(
        [
            "def retrieved_helper(raw: str) -> dict:",
            "    return json.loads(raw)",
            "",
        ]
    )
    big_src_b = "\n".join(big_lines_b)
    ret_end = len(big_lines_b) - 1
    ret_start = ret_end - 1

    test_lines = ["from pkg.mod_a import target_buggy_fn", ""]
    for i in range(1, 25):
        test_lines.extend([f"def test_unrelated_{i}():", f"    assert {i} == {i}", ""])
    test_lines.extend(
        [
            "def test_target_none():",
            "    assert target_buggy_fn(None) == ''",
            "",
        ]
    )

    files = {
        "pkg/mod_a.py": big_src_a,
        "pkg/mod_b.py": big_src_b,
        "tests/test_mod.py": "\n".join(test_lines),
    }
    failing_output = (
        "___________________________ test_target_none ___________________________\n"
        f"pkg/mod_a.py:{hit_line_a}: in target_buggy_fn\n"
        "E   AttributeError: 'NoneType' object has no attribute 'strip'\n"
        "FAILED tests/test_mod.py::test_target_none - AttributeError\n"
    )
    retrieved = [
        f"--- Retrieved: pkg/mod_b.py::retrieved_helper (lines {ret_start}-{ret_end}) ---\n"
        "def retrieved_helper(raw: str) -> dict:\n"
        "    return json.loads(raw)"
    ]

    seen_requests: list[LLMRequest] = []

    class _CaptureProvider(MockProvider):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            seen_requests.append(request)
            return await super().generate(request)

    async def _run() -> None:
        provider = _CaptureProvider(canned="FINDING: pkg/mod_a.py:1: none: bug")
        await run_llm_diff_analysis(
            files=files,
            provider=provider,
            data_class="public",
            findings=[],
            failing_test_output=failing_output,
            retrieved_context=retrieved,
        )
        propose = create_propose_node(provider)
        await propose(
            PipelineState(
                case_id="large_multi",
                files=files,
                failing_test_output=failing_output,
                findings=[
                    Finding(
                        file="pkg/mod_a.py",
                        line=hit_line_a,
                        category="none_handling",
                        explanation="missing None check",
                    )
                ],
                retrieved_context=retrieved,
                data_class="public",
            )
        )

    asyncio.run(_run())
    assert len(seen_requests) == 2

    for req in seen_requests:
        assert req.max_tokens == DEFAULT_MAX_TOKENS
        prompt = req.messages[-1].content
        # Unrelated filler functions and unrelated passing tests are omitted
        assert "def filler_a_20" not in prompt
        assert "def filler_b_20" not in prompt
        assert "def test_unrelated_15" not in prompt
        # Hit source function, retrieved function, and failing test function are present ONCE
        assert prompt.count("def target_buggy_fn(") == 1
        assert prompt.count("def retrieved_helper(") == 1
        assert prompt.count("def test_target_none():") == 1
        assert "# [lines " in prompt
        # Header compatibility for router/features.py
        embedded = _extract_embedded_files(prompt)
        assert "pkg/mod_a.py" in embedded
        assert "pkg/mod_b.py" in embedded
        feats = extract_task_features(req)
        assert feats.non_test_file_count >= 2
        assert feats.ast_node_count > 0


def test_verify_node_preserves_result_error_alongside_pytest_output() -> None:
    """verify_node preserves result.error (e.g. ReadTimeout) alongside pytest stdout/stderr."""

    async def _test() -> None:
        files = {
            "calc.py": "def f(x: int) -> int:\n    return x + 1\n",
            "test_calc.py": "from calc import f\ndef test_f(): assert f(1) == 2\n",
        }
        patch = (
            "<<<<<<< SEARCH:calc.py\n    return x + 1\n=======\n    return x + 2\n>>>>>>> REPLACE"
        )

        async def timeout_verify(fs: dict[str, str], p: str) -> SandboxResult:
            return SandboxResult(
                exit_code=124,
                patch_applied=True,
                timed_out=True,
                stdout="FAILED test_calc.py::test_f - assert 3 == 2",
                stderr="Container killed after timeout",
                error="ReadTimeout: sandbox timed out waiting for pytest",
                junit=JUnitReport(passed=0, failures=1, errors=0),
            )

        verify_node = create_verify_node(verify_fn=timeout_verify)
        state = PipelineState(
            case_id="timeout_preserve",
            files=files,
            patch_diff=(
                "--- a/calc.py\n+++ b/calc.py\n"
                "@@ -1,2 +1,2 @@\n def f(x: int) -> int:\n"
                "-    return x + 1\n+    return x + 2\n"
            ),
            data_class="public",
        )
        out = await verify_node(state)
        assert out["verified"] is False
        assert out["error"] == "ReadTimeout: sandbox timed out waiting for pytest"
        assert "ReadTimeout: sandbox timed out waiting for pytest" in out["feedback"]
        assert "FAILED test_calc.py::test_f - assert 3 == 2" in out["feedback"]
        assert "Container killed after timeout" in out["feedback"]

        # Also verify end-to-end pipeline preserves error in final state on timeout
        final_state = await run_pipeline(
            PipelineState(case_id="timeout_e2e", files=files, max_attempts=1, data_class="public"),
            provider=MockProvider(canned=patch),
            verify_fn=timeout_verify,
        )
        assert final_state.verified is False
        assert final_state.error == "ReadTimeout: sandbox timed out waiting for pytest"
        assert "ReadTimeout" in final_state.feedback
        assert "FAILED test_calc.py::test_f" in final_state.feedback

    asyncio.run(_test())
