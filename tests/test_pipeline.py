"""Tests for Phase 4: LangGraph pipeline, normalization, retry loop, retrieval, and eval."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import PipelineState
from patchpilot.pipeline.normalization import (
    normalize_patch,
)
from patchpilot.providers.base import LLMRequest
from patchpilot.providers.gemini import GeminiProvider
from patchpilot.providers.mock import MockProvider
from patchpilot.providers.ollama import OllamaProvider
from patchpilot.retrieval import HashEmbedder, QdrantStore
from patchpilot.sandbox.runner import JUnitReport, SandboxResult, SandboxRunner

# ---------------------------------------------------------------------------
# 1. SEARCH/REPLACE and unified diff normalization edge cases
# ---------------------------------------------------------------------------


def test_normalization_search_replace_standard() -> None:
    files = {
        "mod.py": "def calc(x: int) -> int:\n    res = x * 2\n    return res + 1\n",
    }
    raw = (
        "<<<<<<< SEARCH\n"
        "    res = x * 2\n"
        "    return res + 1\n"
        "=======\n"
        "    res = x * 3\n"
        "    return res\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(raw, files)
    assert diff.startswith("--- a/mod.py\n+++ b/mod.py\n")
    assert "-    res = x * 2" in diff
    assert "+    res = x * 3" in diff

    # Idempotence: re-normalizing the unified diff produces identical result
    assert normalize_patch(diff, files) == diff


def test_normalization_with_file_header_comment() -> None:
    files = {
        "src/util.py": "def run():\n    return False\n",
    }
    raw = (
        "# src/util.py\n<<<<<<< SEARCH\n    return False\n=======\n    return True\n>>>>>>> REPLACE"
    )
    diff = normalize_patch(raw, files)
    assert "--- a/src/util.py\n+++ b/src/util.py\n" in diff
    assert "+    return True" in diff


def test_normalization_with_search_delimiter_file() -> None:
    files = {
        "pkg/core.py": "ALPHA = 10\nBETA = 20\n",
    }
    raw = "<<<<<<< SEARCH:pkg/core.py\nALPHA = 10\n=======\nALPHA = 100\n>>>>>>> REPLACE"
    diff = normalize_patch(raw, files)
    assert "--- a/pkg/core.py" in diff
    assert "+ALPHA = 100" in diff


def test_normalization_raw_unified_diff_and_fences() -> None:
    files = {
        "foo.py": "a = 1\nb = 2\n",
    }
    raw = "```diff\n--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,2 @@\n a = 1\n-b = 2\n+b = 42\n```"
    diff = normalize_patch(raw, files)
    assert "--- a/foo.py\n+++ b/foo.py\n" in diff
    assert "+b = 42" in diff


def test_normalization_multiple_blocks() -> None:
    files = {
        "f1.py": "x = 1\ny = 2\n",
        "f2.py": "z = 3\nw = 4\n",
    }
    raw = (
        "<<<<<<< SEARCH:f1.py\n"
        "x = 1\n"
        "=======\n"
        "x = 10\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "<<<<<<< SEARCH:f2.py\n"
        "w = 4\n"
        "=======\n"
        "w = 40\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(raw, files)
    assert "--- a/f1.py" in diff
    assert "+x = 10" in diff
    assert "--- a/f2.py" in diff
    assert "+w = 40" in diff


def test_normalization_error_cases() -> None:
    files = {"a.py": "hello = 1\n"}

    # Empty patch
    assert normalize_patch("", files) == ""

    # Invalid text
    with pytest.raises(ValueError, match="neither a SEARCH/REPLACE block nor a unified diff"):
        normalize_patch("just some random text without markers", files)

    # Search block not found in file
    missing_sr = "<<<<<<< SEARCH:a.py\nnot_present = 999\n=======\nfixed = 1\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="Search block not found"):
        normalize_patch(missing_sr, files)

    # Missing replace delimiter
    broken_sr = "<<<<<<< SEARCH:a.py\nhello = 1\n=======\nhello = 2\n"
    with pytest.raises(ValueError, match="neither a SEARCH/REPLACE block nor a unified diff"):
        normalize_patch(broken_sr, files)


# ---------------------------------------------------------------------------
# 2. Embedder protocol & Qdrant retrieval / down degradation
# ---------------------------------------------------------------------------


def test_hashembedder_determinism_and_properties() -> None:
    async def _test() -> None:
        embedder = HashEmbedder(dim=64)
        text1 = "def paginate_inventory(items: list[str], page: int):"
        text2 = "def other_function(x: int):"

        vecs1 = await embedder.embed([text1, text2])
        vecs2 = await embedder.embed([text1, text2])

        assert len(vecs1) == 2
        assert len(vecs1[0]) == 64
        # Deterministic
        assert vecs1 == vecs2
        # Sensitive to different content
        assert vecs1[0] != vecs1[1]

        # Unit norm
        norm_sq = sum(x * x for x in vecs1[0])
        assert abs(norm_sq**0.5 - 1.0) < 1e-4

    asyncio.run(_test())


def test_qdrant_down_graceful_degradation() -> None:
    """When Qdrant is down or unreachable, the store degrades gracefully without crashing."""

    async def _test() -> None:
        down_store = QdrantStore(base_url="http://127.0.0.1:59999", timeout_s=0.2)
        assert await down_store.is_healthy() is False
        assert await down_store.ensure_collection("test_col") is False
        assert await down_store.upsert("test_col", [{"id": 1, "vector": [0.1] * 64}]) is False
        assert await down_store.search("test_col", [0.1] * 64) == []
        assert await down_store.delete_collection("test_col") is False

    asyncio.run(_test())


def test_pipeline_runs_with_qdrant_down() -> None:
    """Pipeline executes end-to-end even when Qdrant is completely down."""

    async def _test() -> None:
        files = {
            "mod.py": "def f(): return 1\n",
            "test_mod.py": "from mod import f\ndef test_f(): assert f() == 2\n",
        }
        good_patch = (
            "--- a/mod.py\n+++ b/mod.py\n@@ -1,1 +1,1 @@\n-def f(): return 1\n+def f(): return 2\n"
        )
        provider = MockProvider(canned=good_patch)
        embedder = HashEmbedder(dim=64)
        down_qdrant = QdrantStore(base_url="http://127.0.0.1:59999", timeout_s=0.2)

        async def mock_verify(fs: dict[str, str], p: str) -> SandboxResult:
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        initial_state = PipelineState(
            case_id="down_test",
            files=files,
            target_file="mod.py",
            data_class="public",
        )
        final_state = await run_pipeline(
            initial_state,
            provider=provider,
            verify_fn=mock_verify,
            qdrant=down_qdrant,
            embedder=embedder,
        )
        assert final_state.verified is True
        assert final_state.patch_applies is True
        assert final_state.target_test_passes is True
        assert final_state.retrieved_context == []

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 3. Retry loop, feedback, and retry cap (k = 3)
# ---------------------------------------------------------------------------


def test_retry_loop_exercises_feedback_and_succeeds_at_attempt_2() -> None:
    """Attempt 1 fails -> failing test feedback fed back to propose -> attempt 2 succeeds."""

    async def _test() -> None:
        files = {
            "math_mod.py": "def add(a: int, b: int) -> int:\n    return a - b\n",
            "test_math_mod.py": "from math_mod import add\ndef test_add(): assert add(2, 3) == 5\n",
        }
        wrong_patch = "<<<<<<< SEARCH\n    return a - b\n=======\n    return a * b\n>>>>>>> REPLACE"
        good_patch = "<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"

        received_feedback: list[str] = []

        def scripted_mock(req: LLMRequest) -> str:
            prompt_text = req.messages[0].content
            if "PREVIOUS ATTEMPT FAILED" in prompt_text:
                received_feedback.append(prompt_text)
                return good_patch
            return wrong_patch

        provider = MockProvider(canned=scripted_mock)

        async def dynamic_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            if "+    return a + b" in patch:
                return SandboxResult(
                    exit_code=0,
                    patch_applied=True,
                    junit=JUnitReport(passed=1, failures=0, errors=0),
                )
            return SandboxResult(
                exit_code=1,
                patch_applied=True,
                stdout="FAILED test_math_mod.py::test_add - assert 6 == 5",
                stderr="AssertionError",
                junit=JUnitReport(passed=0, failures=1, errors=0),
            )

        initial_state = PipelineState(
            case_id="retry_demo",
            files=files,
            target_file="math_mod.py",
            data_class="public",
        )
        final_state = await run_pipeline(
            initial_state,
            provider=provider,
            verify_fn=dynamic_verify,
        )

        assert final_state.attempt == 2
        assert final_state.pass_at_1 is False
        assert final_state.pass_at_k is True
        assert final_state.verified is True
        assert len(received_feedback) == 1
        assert "FAILED test_math_mod.py::test_add - assert 6 == 5" in received_feedback[0]
        assert final_state.reported_patch != ""

    asyncio.run(_test())


def test_retry_cap_stops_at_max_k_attempts() -> None:
    """When all attempts fail, retry terminates at attempt k = 3 with verified=False."""

    async def _test() -> None:
        files = {
            "bug.py": "def bad(): return 0\n",
        }
        always_wrong = (
            "<<<<<<< SEARCH\ndef bad(): return 0\n=======\ndef bad(): return -1\n>>>>>>> REPLACE"
        )
        provider = MockProvider(canned=always_wrong)

        call_count = 0

        async def fail_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            nonlocal call_count
            call_count += 1
            return SandboxResult(
                exit_code=1,
                patch_applied=True,
                stdout=f"FAILED test run {call_count}",
                junit=JUnitReport(passed=0, failures=1, errors=0),
            )

        initial_state = PipelineState(
            case_id="cap_demo",
            files=files,
            target_file="bug.py",
            max_attempts=3,
            data_class="public",
        )
        final_state = await run_pipeline(
            initial_state,
            provider=provider,
            verify_fn=fail_verify,
        )

        assert final_state.attempt == 3
        assert call_count == 3
        assert final_state.verified is False
        assert final_state.pass_at_1 is False
        assert final_state.pass_at_k is False
        # Only verified patches are reported!
        assert final_state.reported_patch == ""

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 4. Privacy guard: non-public requests never reach Gemini
# ---------------------------------------------------------------------------


def test_privacy_guard_enforced_in_pipeline() -> None:
    async def _test() -> None:
        gemini = GeminiProvider(api_key="mock-key", default_model="gemini-3.8-flash")
        files = {"secret.py": "API_KEY = 'secret'\n"}

        initial_state = PipelineState(
            case_id="private_demo",
            files=files,
            target_file="secret.py",
            data_class="private",  # default private!
        )
        # Gemini rejects private data_class with ProviderError
        final_state = await run_pipeline(
            initial_state,
            provider=gemini,
            verify_fn=lambda fs, p: SandboxResult(exit_code=1, patch_applied=False),
        )
        assert final_state.verified is False
        assert "ProviderError" in str(final_state.error) or "private" in str(final_state.error)

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 5. End-to-end on 10 seeded_eval cases with MockProvider
# ---------------------------------------------------------------------------


def _load_seeded_case(case_num: int) -> tuple[dict[str, str], str, str]:
    case_id = f"{case_num:04d}"
    case_dir = Path("evals/datasets/seeded_eval") / case_id
    repo_dir = case_dir / "repo"

    files: dict[str, str] = {}
    for p in repo_dir.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(repo_dir)).replace("\\", "/")
            files[rel] = p.read_text(encoding="utf-8")

    known_patch = (case_dir / "patch.diff").read_text(encoding="utf-8")

    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    target_file = ""
    for line in meta.get("patch_diff", "").splitlines():
        if line.startswith("--- a/"):
            target_file = line[6:].strip()
            break
    if not target_file:
        py_files = [f for f in files if f.endswith(".py") and not Path(f).name.startswith("test_")]
        target_file = py_files[0] if py_files else ""

    return files, target_file, known_patch


def test_end_to_end_10_seeded_eval_cases_mock() -> None:
    """Scripted Mock provider on 10 seeded cases exercising pass@1, pass@3, and retry cap."""

    async def _test() -> None:
        results: list[PipelineState] = []

        for i in range(1, 11):
            files, target_file, known_patch = _load_seeded_case(i)
            case_id = f"seeded_eval/{i:04d}"

            # Canonical version of the known fix
            canonical_known = normalize_patch(known_patch, files, default_file=target_file)

            # A deliberately incorrect patch (wrong modification)
            target_lines = files[target_file].splitlines(keepends=True)
            first_line = target_lines[0]
            wrong_patch = (
                "<<<<<<< SEARCH\n"
                + first_line
                + "=======\n"
                + first_line
                + "# wrong mutation\n"
                + ">>>>>>> REPLACE\n"
            )

            # Cases 1-7: known-good on attempt 1 (pass@1 = True, pass@3 = True)
            # Cases 8-9: wrong on attempt 1, known-good on attempt 2 (pass@1 = False, pass@3 = True)
            # Case 10: wrong on all attempts (pass@1 = False, pass@3 = False, cap exercised)
            def make_responder(case_idx: int, kp: str, wp: str):
                def responder(req: LLMRequest) -> str:
                    prompt_text = req.messages[0].content
                    if case_idx <= 7:
                        return kp
                    elif case_idx in (8, 9):
                        if "PREVIOUS ATTEMPT FAILED" in prompt_text:
                            return kp
                        return wp
                    else:
                        return wp

                return responder

            provider = MockProvider(canned=make_responder(i, known_patch, wrong_patch))

            def make_case_verifier(expected_diff: str):
                async def case_verifier(fs: dict[str, str], patch: str) -> SandboxResult:
                    if patch == expected_diff:
                        return SandboxResult(
                            exit_code=0,
                            patch_applied=True,
                            junit=JUnitReport(passed=1, failures=0, errors=0),
                        )
                    return SandboxResult(
                        exit_code=1,
                        patch_applied=True,
                        stdout="FAILED test - assert error in seeded case",
                        junit=JUnitReport(passed=0, failures=1, errors=0),
                    )

                return case_verifier

            initial_state = PipelineState(
                case_id=case_id,
                files=files,
                target_file=target_file,
                data_class="public",
            )
            final_state = await run_pipeline(
                initial_state,
                provider=provider,
                verify_fn=make_case_verifier(canonical_known),
            )
            results.append(final_state)

        pass_at_1_count = sum(1 for r in results if r.pass_at_1)
        pass_at_3_count = sum(1 for r in results if r.pass_at_k)

        assert pass_at_1_count == 7, f"Expected 7 pass@1, got {pass_at_1_count}"
        assert pass_at_3_count == 9, f"Expected 9 pass@3, got {pass_at_3_count}"

        # Verify that case 10 was rejected and not reported
        assert results[9].verified is False
        assert results[9].reported_patch == ""

        # Verify that cases 1-9 were verified and reported
        for r in results[:9]:
            assert r.verified is True
            assert r.reported_patch != ""

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 6. Docker container end-to-end integration test (marked `docker`)
# ---------------------------------------------------------------------------


@pytest.mark.docker
def test_pipeline_docker_real_container_verification() -> None:
    """Run real SandboxRunner container inside Docker on seeded_eval/0001."""

    async def _test() -> None:
        files, target_file, known_patch = _load_seeded_case(1)
        runner = SandboxRunner()

        provider = MockProvider(canned=known_patch)
        initial_state = PipelineState(
            case_id="seeded_eval/0001",
            files=files,
            target_file=target_file,
            data_class="public",
        )
        final_state = await run_pipeline(
            initial_state,
            provider=provider,
            sandbox_runner=runner,
        )

        assert final_state.patch_applies is True
        assert final_state.target_test_passes is True
        assert final_state.regressions == 0
        assert final_state.verified is True
        assert final_state.pass_at_1 is True
        assert final_state.pass_at_k is True
        assert len(runner.run_logs) >= 1

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 7. Optional live Ollama run on 5 seeded cases (marked `live`)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_ollama_qwen_5_seeded_cases() -> None:
    """Live run on 5 seeded_eval cases with host Ollama qwen2.5-coder:7b."""

    async def _test() -> None:
        # Check if host Ollama is reachable
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get("http://127.0.0.1:11434/api/version")
                if res.status_code != 200:
                    pytest.skip("Host Ollama is not responding with HTTP 200")
        except Exception as exc:
            pytest.skip(f"Host Ollama is unreachable: {exc}")

        runner = SandboxRunner()
        provider = OllamaProvider(
            base_url="http://127.0.0.1:11434",
            default_model="qwen2.5-coder:7b",
            timeout_s=60.0,
        )

        results: list[PipelineState] = []
        for i in range(1, 6):
            files, target_file, _ = _load_seeded_case(i)
            case_id = f"seeded_eval/{i:04d}"
            initial_state = PipelineState(
                case_id=case_id,
                files=files,
                target_file=target_file,
                data_class="public",
            )
            final_state = await run_pipeline(
                initial_state,
                provider=provider,
                sandbox_runner=runner,
            )
            results.append(final_state)

        pass1 = sum(1 for r in results if r.pass_at_1)
        pass3 = sum(1 for r in results if r.pass_at_k)

        print("\n[LIVE EVAL RESULTS - qwen2.5-coder:7b on 5 seeded_eval cases]")
        for r in results:
            print(
                f"Case {r.case_id}: verified={r.verified}, pass@1={r.pass_at_1}, "
                f"pass@3={r.pass_at_k}, attempts={r.attempt}"
            )
        print(f"Overall: pass@1={pass1}/5 ({pass1 / 5:.1%}), pass@3={pass3}/5 ({pass3 / 5:.1%})")

    asyncio.run(_test())
