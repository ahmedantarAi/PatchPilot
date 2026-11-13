"""Tests for Phase 4: LangGraph pipeline, ground-truth isolation, reward-hacking, and Qdrant."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from patchpilot.__main__ import _load_case_files, load_mock_ground_truth
from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import PipelineState
from patchpilot.pipeline.nodes import create_verify_node
from patchpilot.pipeline.normalization import (
    check_forbidden_patch_path,
    is_modifiable_py_file,
    is_test_file,
    normalize_patch,
)
from patchpilot.providers.base import LLMRequest, LLMResponse
from patchpilot.providers.gemini import GeminiProvider
from patchpilot.providers.mock import MockProvider
from patchpilot.providers.ollama import OllamaProvider
from patchpilot.retrieval import HashEmbedder, QdrantStore, case_collection_name
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
# 2. Reward-hacking guard (normalization + verify defense in depth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "test_foo.py",
        "pkg/test_bar.py",
        "foo_test.py",
        "pkg/bar_test.py",
        "tests/helper.py",
        "tests/unit/check.py",
        "pkg/tests/fixture.py",
        "conftest.py",
        "pkg/conftest.py",
        "README.md",
        "pyproject.toml",
        "data/config.json",
    ],
)
def test_reward_hacking_guard_rejects_forbidden_paths_at_normalization(
    forbidden_path: str,
) -> None:
    files = {
        "app.py": "def compute() -> int:\n    return 1\n",
        forbidden_path: "ORIGINAL_LINE = 1\n",
    }

    # 1. Explicit SEARCH/REPLACE targeting forbidden_path
    sr_explicit = (
        f"<<<<<<< SEARCH:{forbidden_path}\n"
        "ORIGINAL_LINE = 1\n"
        "=======\n"
        "ORIGINAL_LINE = 2\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Reward-hacking guard"):
        normalize_patch(sr_explicit, files)

    # 2. Implicit SEARCH/REPLACE matching content in forbidden_path
    sr_implicit = "<<<<<<< SEARCH\nORIGINAL_LINE = 1\n=======\nORIGINAL_LINE = 2\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="Reward-hacking guard"):
        normalize_patch(sr_implicit, files)

    # 3. Unified diff targeting forbidden_path
    udiff = (
        f"--- a/{forbidden_path}\n"
        f"+++ b/{forbidden_path}\n"
        "@@ -1,1 +1,1 @@\n"
        "-ORIGINAL_LINE = 1\n"
        "+ORIGINAL_LINE = 2\n"
    )
    with pytest.raises(ValueError, match="Reward-hacking guard"):
        normalize_patch(udiff, files)

    assert check_forbidden_patch_path(forbidden_path) is not None
    if forbidden_path.endswith(".py"):
        assert is_test_file(forbidden_path) is True
    assert is_modifiable_py_file(forbidden_path) is False


def test_reward_hacking_guard_defense_in_depth_inside_verify() -> None:
    """Even if a test-editing diff bypasses propose, verify rejects it before sandbox execution."""

    async def _test() -> None:
        sandbox_called = False

        async def should_not_be_called(fs: dict[str, str], patch: str) -> SandboxResult:
            nonlocal sandbox_called
            sandbox_called = True
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        verify_node = create_verify_node(verify_fn=should_not_be_called)

        bad_targets = (
            "test_calc.py",
            "calc_test.py",
            "tests/util.py",
            "conftest.py",
            "cfg.ini",
        )
        for bad_target in bad_targets:
            malicious_diff = (
                f"--- a/{bad_target}\n"
                f"+++ b/{bad_target}\n"
                "@@ -1,1 +1,1 @@\n"
                "-assert calc(2) == 4\n"
                "+assert True\n"
            )
            state = PipelineState(
                case_id="verify_guard_test",
                files={"calc.py": "def calc(x): return x\n", bad_target: "assert calc(2) == 4\n"},
                patch_diff=malicious_diff,
                attempt=0,
                max_attempts=3,
            )
            out = await verify_node(state)
            assert sandbox_called is False
            assert out["verified"] is False
            assert out["patch_applies"] is False
            assert out["target_test_passes"] is False
            assert out["attempt"] == 1
            assert "reward-hacking guard" in out["feedback"].lower()

    asyncio.run(_test())


def test_patch_editing_assertion_never_produces_verified_true_and_feeds_back_reason() -> None:
    """A patch editing an assertion never produces verified=True and counts as a failed attempt."""

    async def _test() -> None:
        files = {
            "calc.py": "def double(x: int) -> int:\n    return x + 1\n",
            "test_calc.py": (
                "from calc import double\ndef test_double():\n    assert double(3) == 6\n"
            ),
        }
        assertion_hack = (
            "<<<<<<< SEARCH:test_calc.py\n"
            "    assert double(3) == 6\n"
            "=======\n"
            "    assert double(3) == 4\n"
            ">>>>>>> REPLACE"
        )

        # Case 1: Model only proposes assertion edits across all 3 attempts -> verified is False
        provider_always_hack = MockProvider(canned=assertion_hack)

        async def permissive_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            # Even if sandbox would return exit_code=0, assertion edit must never reach it or verify
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        state1 = await run_pipeline(
            PipelineState(case_id="hack_always", files=files, data_class="public"),
            provider=provider_always_hack,
            verify_fn=permissive_verify,
        )
        assert state1.verified is False
        assert state1.pass_at_1 is False
        assert state1.pass_at_k is False
        assert state1.attempt == 3
        assert state1.reported_patch == ""
        assert "reward-hacking guard" in state1.feedback.lower()

        # Case 2: Attempt 1 edits test assertion -> rejected -> Attempt 2 fixes calc.py
        legit_fix = (
            "<<<<<<< SEARCH:calc.py\n    return x + 1\n=======\n    return x * 2\n>>>>>>> REPLACE"
        )
        propose_prompts: list[str] = []

        def recover_after_rejection(req: LLMRequest) -> str:
            text = req.messages[0].content
            if "autonomous automated code repair agent" in text:
                propose_prompts.append(text)
                if "PREVIOUS ATTEMPT FAILED" in text and "Reward-hacking guard" in text:
                    return legit_fix
                return assertion_hack
            return "FINDING: calc.py:2: wrong_operator: uses + 1 instead of * 2"

        state2 = await run_pipeline(
            PipelineState(case_id="hack_then_fix", files=files, data_class="public"),
            provider=MockProvider(canned=recover_after_rejection),
            verify_fn=permissive_verify,
        )
        assert state2.attempt == 2
        assert state2.pass_at_1 is False
        assert state2.pass_at_k is True
        assert state2.verified is True
        assert len(propose_prompts) == 2
        assert "Reward-hacking guard" in propose_prompts[1]

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 3. Ground-truth isolation & multi-file localization
# ---------------------------------------------------------------------------


def test_target_file_removed_from_state_and_ground_truth_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PipelineState has no target_file, and _load_case_files never reads patch.diff/meta.json."""
    assert "target_file" not in PipelineState.model_fields

    orig_read_text = Path.read_text
    accessed_files: list[str] = []

    def spied_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        accessed_files.append(self.name)
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spied_read_text)

    case_dir = Path("evals/datasets/seeded_eval/0001")
    repo_files = _load_case_files(case_dir)
    assert len(repo_files) >= 2
    assert "patch.diff" not in accessed_files
    assert "meta.json" not in accessed_files


def test_multi_file_repo_model_chooses_file_without_target_file() -> None:
    """In a multi-file repository, the model localizes the fault and chooses the file itself."""

    async def _test() -> None:
        files = {
            "pkg/alpha.py": "def alpha_helper(x: int) -> int:\n    return x + 10\n",
            "pkg/beta.py": "def compute_total(a: int, b: int) -> int:\n    return a - b\n",
            "tests/test_beta.py": (
                "from pkg.beta import compute_total\n"
                "def test_total():\n"
                "    assert compute_total(7, 3) == 10\n"
            ),
        }

        def model_responder(req: LLMRequest) -> str:
            prompt = req.messages[0].content
            assert "Target file" not in prompt
            assert "Primary target file" not in prompt
            if "static analysis and fault localization" in prompt:
                assert "pkg/alpha.py" in prompt
                assert "pkg/beta.py" in prompt
                return "FINDING: pkg/beta.py:2: wrong_operator: subtracts instead of adding"
            return (
                "<<<<<<< SEARCH:pkg/beta.py\n"
                "    return a - b\n"
                "=======\n"
                "    return a + b\n"
                ">>>>>>> REPLACE"
            )

        async def verify_multi(fs: dict[str, str], patch: str) -> SandboxResult:
            assert "--- a/pkg/beta.py" in patch
            assert "pkg/alpha.py" not in patch
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        final_state = await run_pipeline(
            PipelineState(case_id="multi_file_case", files=files, data_class="public"),
            provider=MockProvider(canned=model_responder),
            verify_fn=verify_multi,
        )
        assert final_state.verified is True
        assert "--- a/pkg/beta.py" in final_state.reported_patch

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 4. Leak regression test on seeded_eval with recording fake provider
# ---------------------------------------------------------------------------


class RecordingFakeProvider:
    """Recording fake provider that captures every prompt sent by the pipeline."""

    name = "recording_fake"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        full_prompt = "\n".join(m.content for m in request.messages)
        self.prompts.append(full_prompt)
        # Return an unverified syntax-valid edit on non-test code so retry prompts are also captured
        for line in full_prompt.splitlines():
            if line.startswith("def ") and ":" in line:
                return LLMResponse(
                    text=(
                        "<<<<<<< SEARCH\n"
                        f"{line}\n"
                        "=======\n"
                        f"{line}\n"
                        "    # attempted repair\n"
                        ">>>>>>> REPLACE"
                    ),
                    model="recording-fake",
                    provider=self.name,
                )
        return LLMResponse(text="MOCK_RESPONSE", model="recording-fake", provider=self.name)


def test_leak_regression_seeded_eval_no_ground_truth_in_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run pipeline on seeded_eval with recording provider and assert zero ground-truth leakage."""

    async def _test() -> None:
        case_dir = Path("evals/datasets/seeded_eval/0001")

        # Read ground truth ONLY in test code before blocking access
        gt_patch_diff = (case_dir / "patch.diff").read_text(encoding="utf-8")
        meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        gt_file = ""
        for line in str(meta.get("patch_diff", "")).splitlines():
            if line.startswith("--- a/"):
                gt_file = line[6:].strip()
                break
        assert gt_file != ""

        fix_hunk_lines = [
            line[1:].strip()
            for line in gt_patch_diff.splitlines()
            if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
        ]
        assert len(fix_hunk_lines) >= 1

        # Spy on Path.read_text to guarantee pipeline never reads meta.json or patch.diff
        orig_read_text = Path.read_text

        def guarded_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
            if self.name in ("patch.diff", "meta.json"):
                raise AssertionError(f"Pipeline illegally read ground-truth file: {self}")
            return orig_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", guarded_read_text)

        files = _load_case_files(case_dir)
        recorder = RecordingFakeProvider()

        async def fake_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            return SandboxResult(
                exit_code=1,
                patch_applied=True,
                stdout=f"FAILED test_{gt_file}::test_case - AssertionError",
                junit=JUnitReport(passed=0, failures=1, errors=0),
            )

        final_state = await run_pipeline(
            PipelineState(
                case_id="seeded_eval/0001",
                files=files,
                data_class="public",
                max_attempts=2,
            ),
            provider=recorder,
            verify_fn=fake_verify,
        )

        # Restore Path.read_text
        monkeypatch.setattr(Path, "read_text", orig_read_text)

        assert len(recorder.prompts) >= 2
        failing_test_output = final_state.failing_test_output

        for prompt in recorder.prompts:
            # Ground-truth file name from meta.json must not appear unless in failing test output
            if gt_file not in failing_test_output:
                assert gt_file not in prompt
            # No target_file prompt markers may exist
            assert "Target file:" not in prompt
            assert "Primary target file:" not in prompt
            # No fix hunk text may appear in any prompt
            for hunk_line in fix_hunk_lines:
                assert hunk_line not in prompt
            # No content of patch.diff may appear in any prompt
            assert gt_patch_diff.strip() not in prompt

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 5. Embedder protocol & Qdrant per-case isolation / down degradation
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
        assert vecs1 == vecs2
        assert vecs1[0] != vecs1[1]

        norm_sq = sum(x * x for x in vecs1[0])
        assert abs(norm_sq**0.5 - 1.0) < 1e-4

    asyncio.run(_test())


def test_qdrant_per_case_isolation_case_b_never_returns_case_a() -> None:
    """Retrieval for Case B uses its own content-hash collection and never returns Case A chunks."""

    async def _test() -> None:
        collections: dict[str, list[dict[str, Any]]] = {}
        deleted_collections: list[str] = []

        def mock_qdrant_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            method = request.method
            if path == "/healthz":
                return httpx.Response(200, text="healthz check passed")

            if path.startswith("/collections/"):
                rest = path[len("/collections/") :]
                parts = rest.split("/")
                col_name = parts[0]

                if len(parts) == 1:
                    if method == "GET":
                        if col_name in collections:
                            return httpx.Response(200, json={"result": {"status": "green"}})
                        return httpx.Response(404, json={"status": {"error": "Not found"}})
                    if method == "PUT":
                        collections[col_name] = []
                        return httpx.Response(200, json={"result": True})
                    if method == "DELETE":
                        collections.pop(col_name, None)
                        deleted_collections.append(col_name)
                        return httpx.Response(200, json={"result": True})

                elif len(parts) == 2 and parts[1] == "points":
                    if method == "PUT":
                        body = json.loads(request.content.decode("utf-8"))
                        pts = body.get("points", [])
                        collections.setdefault(col_name, []).extend(pts)
                        return httpx.Response(200, json={"result": {"status": "completed"}})

                elif len(parts) == 3 and parts[1] == "points" and parts[2] == "search":
                    if method == "POST":
                        pts = collections.get(col_name, [])
                        return httpx.Response(
                            200,
                            json={
                                "result": [
                                    {"id": p["id"], "score": 0.99, "payload": p["payload"]}
                                    for p in pts
                                ]
                            },
                        )

            return httpx.Response(404)

        transport = httpx.MockTransport(mock_qdrant_handler)
        qdrant = QdrantStore(base_url="http://127.0.0.1:6333", transport=transport)
        embedder = HashEmbedder(dim=64)

        files_a = {
            "alpha_mod.py": "def alpha_calc():\n    return 'CASE_A_EXCLUSIVE_SENTINEL_CHUNK'\n",
            "test_alpha_mod.py": (
                "from alpha_mod import alpha_calc\ndef test_a(): assert alpha_calc() == 'ok'\n"
            ),
        }
        files_b = {
            "beta_mod.py": "def beta_calc():\n    return 'CASE_B_EXCLUSIVE_SENTINEL_CHUNK'\n",
            "test_beta_mod.py": (
                "from beta_mod import beta_calc\ndef test_b(): assert beta_calc() == 'ok'\n"
            ),
        }

        col_a = case_collection_name(files_a)
        col_b = case_collection_name(files_b)
        assert col_a != col_b

        # Pre-populate Case A collection in Qdrant to verify that even if Case A exists,
        # Case B queries only col_b and never receives Case A chunks.
        await qdrant.index_case_files(files_a, embedder)
        assert col_a in collections

        patch_a = (
            "<<<<<<< SEARCH:alpha_mod.py\n"
            "    return 'CASE_A_EXCLUSIVE_SENTINEL_CHUNK'\n"
            "=======\n"
            "    return 'ok'\n"
            ">>>>>>> REPLACE"
        )
        patch_b = (
            "<<<<<<< SEARCH:beta_mod.py\n"
            "    return 'CASE_B_EXCLUSIVE_SENTINEL_CHUNK'\n"
            "=======\n"
            "    return 'ok'\n"
            ">>>>>>> REPLACE"
        )

        async def ok_verify(fs: dict[str, str], p: str) -> SandboxResult:
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        state_a = await run_pipeline(
            PipelineState(case_id="case_a", files=files_a, data_class="public"),
            provider=MockProvider(canned=patch_a),
            verify_fn=ok_verify,
            qdrant=qdrant,
            embedder=embedder,
        )
        assert any("CASE_A_EXCLUSIVE_SENTINEL_CHUNK" in c for c in state_a.retrieved_context)
        assert col_a in deleted_collections
        assert col_a not in collections

        # Even if a stale Case A collection is re-inserted before Case B runs, Case B searches col_b
        await qdrant.index_case_files(files_a, embedder)
        assert col_a in collections

        prompts_b: list[str] = []

        def provider_b_fn(req: LLMRequest) -> str:
            prompts_b.append(req.messages[0].content)
            return patch_b

        state_b = await run_pipeline(
            PipelineState(case_id="case_b", files=files_b, data_class="public"),
            provider=MockProvider(canned=provider_b_fn),
            verify_fn=ok_verify,
            qdrant=qdrant,
            embedder=embedder,
        )

        assert any("CASE_B_EXCLUSIVE_SENTINEL_CHUNK" in c for c in state_b.retrieved_context)
        assert all("CASE_A_EXCLUSIVE_SENTINEL_CHUNK" not in c for c in state_b.retrieved_context)
        assert all("CASE_A_EXCLUSIVE_SENTINEL_CHUNK" not in p for p in prompts_b)
        assert col_b in deleted_collections
        assert col_b not in collections

    asyncio.run(_test())


def test_qdrant_down_graceful_degradation() -> None:
    """When Qdrant is down or unreachable, the store degrades gracefully without crashing."""

    async def _test() -> None:
        down_store = QdrantStore(base_url="http://127.0.0.1:59999", timeout_s=0.2)
        assert await down_store.is_healthy() is False
        assert await down_store.ensure_collection("test_col") is False
        assert await down_store.recreate_collection("test_col") is False
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
# 6. Retry loop, feedback, and retry cap (k = 3)
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
        assert final_state.reported_patch == ""

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 7. Privacy guard: non-public requests never reach Gemini
# ---------------------------------------------------------------------------


def test_privacy_guard_enforced_in_pipeline() -> None:
    async def _test() -> None:
        gemini = GeminiProvider(api_key="mock-key", default_model="gemini-3.8-flash")
        files = {"secret.py": "API_KEY = 'secret'\n"}

        initial_state = PipelineState(
            case_id="private_demo",
            files=files,
            data_class="private",
        )
        final_state = await run_pipeline(
            initial_state,
            provider=gemini,
            verify_fn=lambda fs, p: SandboxResult(exit_code=1, patch_applied=False),
        )
        assert final_state.verified is False
        assert "ProviderError" in str(final_state.error) or "private" in str(final_state.error)

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 8. End-to-end on 10 seeded_eval cases with MockProvider
# ---------------------------------------------------------------------------


def test_end_to_end_10_seeded_eval_cases_mock() -> None:
    """Scripted Mock provider on 10 seeded cases exercising pass@1, pass@3, and retry cap."""

    async def _test() -> None:
        results: list[PipelineState] = []

        for i in range(1, 11):
            case_dir = Path("evals/datasets/seeded_eval") / f"{i:04d}"
            files = _load_case_files(case_dir)
            known_patch = load_mock_ground_truth(case_dir)
            case_id = f"seeded_eval/{i:04d}"

            canonical_known = normalize_patch(known_patch, files)

            non_test_py = [f for f in sorted(files) if is_modifiable_py_file(f)]
            first_line = files[non_test_py[0]].splitlines(keepends=True)[0]
            wrong_patch = (
                "<<<<<<< SEARCH\n"
                + first_line
                + "=======\n"
                + first_line
                + "# wrong mutation\n"
                + ">>>>>>> REPLACE\n"
            )

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

        assert results[9].verified is False
        assert results[9].reported_patch == ""

        for r in results[:9]:
            assert r.verified is True
            assert r.reported_patch != ""

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 9. Docker container end-to-end integration test (marked `docker`)
# ---------------------------------------------------------------------------


@pytest.mark.docker
def test_pipeline_docker_real_container_verification() -> None:
    """Run real SandboxRunner container inside Docker on seeded_eval/0001."""

    async def _test() -> None:
        case_dir = Path("evals/datasets/seeded_eval/0001")
        files = _load_case_files(case_dir)
        known_patch = load_mock_ground_truth(case_dir)
        runner = SandboxRunner()

        provider = MockProvider(canned=known_patch)
        initial_state = PipelineState(
            case_id="seeded_eval/0001",
            files=files,
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
# 10. Optional live Ollama smoke check on 5 seeded cases (marked `live`)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_ollama_qwen_5_seeded_cases() -> None:
    """Live smoke test (n=5) on 5 seeded_eval cases with host Ollama qwen2.5-coder:7b."""

    async def _test() -> None:
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
            case_dir = Path("evals/datasets/seeded_eval") / f"{i:04d}"
            # Strictly load only repo files; never read patch.diff or meta.json
            files = _load_case_files(case_dir)
            case_id = f"seeded_eval/{i:04d}"
            initial_state = PipelineState(
                case_id=case_id,
                files=files,
                data_class="public",
                provider="ollama",
            )
            final_state = await run_pipeline(
                initial_state,
                provider=provider,
                sandbox_runner=runner,
            )
            results.append(final_state)

        pass1 = sum(1 for r in results if r.pass_at_1)
        pass3 = sum(1 for r in results if r.pass_at_k)

        print("\n[LIVE SMOKE TEST (n=5, not a formal eval result) - qwen2.5-coder:7b]")
        for r in results:
            print(
                f"Case {r.case_id}: verified={r.verified}, pass@1={r.pass_at_1}, "
                f"pass@3={r.pass_at_k}, attempts={r.attempt}"
            )
        print(
            f"Smoke test (n=5): pass@1={pass1}/5 ({pass1 / 5:.1%}), "
            f"pass@3={pass3}/5 ({pass3 / 5:.1%})"
        )

    asyncio.run(_test())


def test_strip_markdown_code_fences_ignores_blank_lines_outside_blocks() -> None:
    from patchpilot.pipeline.normalization import _strip_markdown_code_fences as strip

    body = "--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n-x = 1\n+x = 2\n\n y = 3\n"
    # plain fence, unchanged behaviour
    assert strip(f"```diff\n{body}```") == body
    # leading / trailing blank lines outside the fence
    assert strip(f"\n\n```diff\n{body}```\n\n\n") == body
    # blank lines inside the block are preserved verbatim
    inner = "<<<<<<< SEARCH\na = 1\n\n\nb = 2\n=======\na = 2\n\n\nb = 2\n>>>>>>> REPLACE\n"
    assert strip(f"\n```\n{inner}```\n") == inner
    # blank lines between consecutive blocks do not corrupt content
    two = f"```diff\n{body}```\n\n\n```diff\n{body}```\n"
    assert strip(two) == body + body
    # unterminated fence, and unfenced text, are handled
    assert strip(f"\n```python\n{body}") == body
    assert strip("no fence here\n\n") == "no fence here\n\n"


def test_prompts_do_not_inject_identical_files_twice() -> None:
    from patchpilot.pipeline.nodes import create_propose_node, run_llm_diff_analysis

    seen: list[str] = []

    class Capture(MockProvider):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            seen.append(request.messages[-1].content)
            return await super().generate(request)

    src = "def add(a, b):\n    return a - b\n"
    test_src = "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    files = {
        "calc.py": src,
        "./calc.py": src,  # alias of the same file
        "tests/test_calc.py": test_src,
    }
    failing = f"tests/test_calc.py:\n{test_src}"

    asyncio.run(
        run_llm_diff_analysis(
            files, Capture(canned="[]"), "private", [], failing_test_output=failing
        )
    )
    prompt = seen[-1]
    assert prompt.count("def add(a, b):") == 1
    assert prompt.count("def test_add():") == 1

    node = create_propose_node(Capture(canned="no patch"))
    asyncio.run(
        node(
            PipelineState(
                case_id="c", files=files, failing_test_output=failing, data_class="private"
            )
        )
    )
    prompt = seen[-1]
    assert prompt.count("def add(a, b):") == 1
    assert prompt.count("def test_add():") == 1


def test_prompt_version_hash_unchanged_by_dedup_and_dependency_cleanup() -> None:
    from patchpilot.pipeline.nodes import compute_prompt_version_hash

    assert compute_prompt_version_hash() == (
        "ca774b4a6efb8c2384a6f216439068326840bd10ec0543a1032f135bc0f20782"
    )
