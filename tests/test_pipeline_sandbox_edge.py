"""Edge-case unit tests for pipeline normalization, Qdrant retrieval, and sandbox runner."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import PipelineState
from patchpilot.pipeline.nodes import compute_prompt_version_hash
from patchpilot.pipeline.normalization import (
    _apply_unified_diff_to_mapping,
    check_forbidden_patch_path,
    is_modifiable_py_file,
    is_test_file,
    normalize_patch,
    normalize_search_replace,
    normalize_unified_diff,
    validate_patch_targets,
)
from patchpilot.providers.base import ProviderError
from patchpilot.providers.mock import MockProvider
from patchpilot.retrieval import (
    HashEmbedder,
    OllamaEmbedder,
    QdrantStore,
    case_collection_name,
)
from patchpilot.sandbox import runner as host_runner_mod
from patchpilot.sandbox.runner import (
    SandboxLock,
    SandboxResult,
    SandboxRunner,
)
from patchpilot.sandbox.tar_and_diff import (
    MAX_OUTPUT_BYTES,
    MAX_PATCH_BYTES,
    MAX_TAR_FILE_BYTES,
    MAX_TAR_MEMBERS,
    MAX_TAR_TOTAL_BYTES,
    JUnitReport,
    TarSafetyError,
    apply_unified_diff,
    parse_unified_diff,
    read_stream_capped,
    safe_extract_tar,
    validate_safe_relative_path,
)


def _load_container_runner_module() -> ModuleType:
    runner_path = Path(__file__).resolve().parent.parent / "sandbox" / "runner.py"
    spec = importlib.util.spec_from_file_location("sandbox_container_runner_edge", runner_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_container_runner_edge"] = mod
    spec.loader.exec_module(mod)
    return mod


CONTAINER_RUNNER = _load_container_runner_module()


# ---------------------------------------------------------------------------
# 1. SandboxRunner.__init__ lock-aware cleanup_on_start
# ---------------------------------------------------------------------------


def test_sandbox_runner_cleanup_on_start_runs_when_lock_is_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_file = tmp_path / "sandbox.lock"
    cleanup_calls: list[str] = []

    def fake_cleanup(label: str = "patchpilot.sandbox=1") -> list[str]:
        # Verify the in-process lock is held while cleanup_leftover_containers executes
        assert host_runner_mod._IN_PROCESS_LOCK.locked()
        cleanup_calls.append(label)
        return []

    monkeypatch.setattr(host_runner_mod, "cleanup_leftover_containers", fake_cleanup)
    _ = SandboxRunner(lock_path=lock_file, cleanup_on_start=True)
    assert len(cleanup_calls) == 1
    assert not host_runner_mod._IN_PROCESS_LOCK.locked()


def test_sandbox_runner_cleanup_on_start_skips_when_in_process_lock_held(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_file = tmp_path / "sandbox.lock"
    cleanup_calls: list[str] = []

    def fake_cleanup(label: str = "patchpilot.sandbox=1") -> list[str]:
        cleanup_calls.append(label)
        return []

    monkeypatch.setattr(host_runner_mod, "cleanup_leftover_containers", fake_cleanup)

    with SandboxLock(lock_file):
        # Must catch SandboxBusyError as a clean no-op and never kill active containers
        _ = SandboxRunner(lock_path=lock_file, cleanup_on_start=True)
        assert cleanup_calls == []


def test_sandbox_runner_cleanup_on_start_skips_when_cross_process_lock_held(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_file = tmp_path / "cross-proc.lock"
    cleanup_calls: list[str] = []

    def fake_cleanup(label: str = "patchpilot.sandbox=1") -> list[str]:
        cleanup_calls.append(label)
        return []

    monkeypatch.setattr(host_runner_mod, "cleanup_leftover_containers", fake_cleanup)

    src_root = str(Path(__file__).resolve().parent.parent / "src")
    child_code = (
        "import sys, time\n"
        f"sys.path.insert(0, {src_root!r})\n"
        "from pathlib import Path\n"
        "from patchpilot.sandbox.runner import SandboxLock\n"
        "with SandboxLock(Path(sys.argv[1])):\n"
        "    sys.stdout.write('LOCKED\\n')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(5.0)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(lock_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "LOCKED"
        _ = SandboxRunner(lock_path=lock_file, cleanup_on_start=True)
        assert cleanup_calls == []
        assert not host_runner_mod._IN_PROCESS_LOCK.locked()
    finally:
        child.kill()
        child.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# 2. Synchronization between sandbox/runner.py and tar_and_diff.py
# ---------------------------------------------------------------------------


def test_container_runner_and_tar_and_diff_synchronization(tmp_path: Path) -> None:
    assert CONTAINER_RUNNER.MAX_OUTPUT_BYTES == MAX_OUTPUT_BYTES
    assert CONTAINER_RUNNER.MAX_PATCH_BYTES == MAX_PATCH_BYTES
    assert CONTAINER_RUNNER.MAX_TAR_FILE_BYTES == MAX_TAR_FILE_BYTES
    assert CONTAINER_RUNNER.MAX_TAR_TOTAL_BYTES == MAX_TAR_TOTAL_BYTES
    assert CONTAINER_RUNNER.MAX_TAR_MEMBERS == MAX_TAR_MEMBERS

    # Invalid UTF-8 in __patchpilot__/patch.diff rejected with TarSafetyError in both
    buf_bad_patch = io.BytesIO()
    with tarfile.open(fileobj=buf_bad_patch, mode="w") as tar:
        bad_bytes = b"\xff\xfe\xfd"
        info = tarfile.TarInfo("__patchpilot__/patch.diff")
        info.size = len(bad_bytes)
        tar.addfile(info, io.BytesIO(bad_bytes))

    with pytest.raises(TarSafetyError, match="not valid UTF-8"):
        safe_extract_tar(io.BytesIO(buf_bad_patch.getvalue()), tmp_path / "w1")
    with pytest.raises(CONTAINER_RUNNER.TarSafetyError, match="not valid UTF-8"):
        CONTAINER_RUNNER.safe_extract_tar(io.BytesIO(buf_bad_patch.getvalue()), tmp_path / "w2")

    # Malformed JSON in __patchpilot__/meta.json rejected with TarSafetyError in both
    buf_bad_meta = io.BytesIO()
    with tarfile.open(fileobj=buf_bad_meta, mode="w") as tar:
        bad_json = b"{not-valid-json"
        info = tarfile.TarInfo("__patchpilot__/meta.json")
        info.size = len(bad_json)
        tar.addfile(info, io.BytesIO(bad_json))

    with pytest.raises(TarSafetyError, match="not valid UTF-8 JSON"):
        safe_extract_tar(io.BytesIO(buf_bad_meta.getvalue()), tmp_path / "w3")
    with pytest.raises(CONTAINER_RUNNER.TarSafetyError, match="not valid UTF-8 JSON"):
        CONTAINER_RUNNER.safe_extract_tar(io.BytesIO(buf_bad_meta.getvalue()), tmp_path / "w4")

    # Identical behavior across core functions in both modules
    sample_diff = "--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n+y = 3\n"
    for idx, (val_fn, parse_fn, apply_fn, read_fn) in enumerate(
        [
            (
                validate_safe_relative_path,
                parse_unified_diff,
                apply_unified_diff,
                read_stream_capped,
            ),
            (
                CONTAINER_RUNNER.validate_safe_relative_path,
                CONTAINER_RUNNER.parse_unified_diff,
                CONTAINER_RUNNER.apply_unified_diff,
                CONTAINER_RUNNER.read_stream_capped,
            ),
        ]
    ):
        sub = tmp_path / f"sync_{idx}"
        sub.mkdir()
        p = val_fn("m.py", sub)
        p.write_text("x = 1\ny = 2\n", encoding="utf-8")
        parsed = parse_fn(sample_diff, sub)
        assert len(parsed) == 1
        assert apply_fn(sample_diff, sub) == 1
        assert p.read_text(encoding="utf-8") == "x = 1\ny = 3\n"
        out, trunc = read_fn(io.BytesIO(b"hello world"), 5)
        assert out == "hello" and trunc is True


def test_container_runner_install_deps_handles_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStdin:
        def __init__(self, raw: bytes) -> None:
            self.buffer = io.BytesIO(raw)

    for bad_payload in (b"{malformed", b'{"unclosed": [1, 2'):
        out_buf = io.StringIO()
        monkeypatch.setattr(CONTAINER_RUNNER.sys, "stdin", FakeStdin(bad_payload))
        monkeypatch.setattr(CONTAINER_RUNNER.sys, "stdout", out_buf)
        rc = CONTAINER_RUNNER.run_install_deps()
        assert rc == 2
        parsed = json.loads(out_buf.getvalue().strip())
        assert parsed["exit_code"] == 2
        assert "Invalid JSON" in parsed["stderr"]


# ---------------------------------------------------------------------------
# 3. 4-tier SEARCH/REPLACE normalization edge cases
# ---------------------------------------------------------------------------


def test_empty_search_block_rejected_and_never_matches_blank_line() -> None:
    files = {"mod.py": "def f():\n    x = 1\n\n    return x\n"}

    for empty_search in (
        "<<<<<<< SEARCH:mod.py\n=======\n    x = 2\n>>>>>>> REPLACE",
        "<<<<<<< SEARCH:mod.py\n   \n\n=======\n    x = 2\n>>>>>>> REPLACE",
        "<<<<<<< SEARCH\n\n=======\n    x = 2\n>>>>>>> REPLACE",
    ):
        with pytest.raises(ValueError, match="Empty SEARCH block"):
            normalize_patch(empty_search, files)


def test_empty_replace_block_deletes_matched_lines() -> None:
    files = {"mod.py": ("def compute(x: int) -> int:\n    debug_temp = x * 99\n    return x + 1\n")}
    raw = "<<<<<<< SEARCH:mod.py\n    debug_temp = x * 99\n=======\n>>>>>>> REPLACE"
    diff = normalize_patch(raw, files)
    assert "-    debug_temp = x * 99\n" in diff
    assert "+    debug_temp" not in diff


def test_multi_block_same_file_sequential_and_inter_block_prose() -> None:
    files = {
        "calc.py": (
            "def step_one(x: int) -> int:\n"
            "    return x - 1\n"
            "\n"
            "def step_two(y: int) -> int:\n"
            "    return y * 2\n"
        )
    }
    # Two fenced blocks with prose between them, plus a sequential edit on step_one
    raw = (
        "```python\n"
        "<<<<<<< SEARCH:calc.py\n"
        "def step_one(x: int) -> int:\n"
        "    return x - 1\n"
        "=======\n"
        "def step_one(x: int) -> int:\n"
        "    val = x + 1\n"
        "    return val\n"
        ">>>>>>> REPLACE\n"
        "```\n"
        "Now update step_two and refine step_one:\n"
        "```python\n"
        "<<<<<<< SEARCH:calc.py\n"
        "    val = x + 1\n"
        "    return val\n"
        "=======\n"
        "    val = x + 10\n"
        "    return val\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH:calc.py\n"
        "def step_two(y: int) -> int:\n"
        "    return y * 2\n"
        "=======\n"
        "def step_two(y: int) -> int:\n"
        "    return y + 2\n"
        ">>>>>>> REPLACE\n"
        "```"
    )
    diff = normalize_patch(raw, files)
    assert "+    val = x + 10\n" in diff
    assert "+    return y + 2\n" in diff


def test_mixed_search_replace_and_unified_diff_output() -> None:
    files = {
        "a.py": "def fa() -> int:\n    return 1\n",
        "b.py": "def fb() -> int:\n    return 2\n",
        "test_a.py": "def test_a():\n    assert True\n",
    }

    # 1. Mixed SEARCH/REPLACE for a.py + unified diff for b.py
    mixed_valid = (
        "<<<<<<< SEARCH:a.py\n"
        "    return 1\n"
        "=======\n"
        "    return 10\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def fb() -> int:\n"
        "-    return 2\n"
        "+    return 20\n"
    )
    diff = normalize_patch(mixed_valid, files)
    assert "--- a/a.py\n+++ b/a.py\n" in diff
    assert "+    return 10\n" in diff
    assert "--- a/b.py\n+++ b/b.py\n" in diff
    assert "+    return 20\n" in diff

    # 2. Mixed SEARCH/REPLACE + redundant unified diff of the exact same edit on a.py
    mixed_redundant = (
        "<<<<<<< SEARCH:a.py\n"
        "    return 1\n"
        "=======\n"
        "    return 10\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def fa() -> int:\n"
        "-    return 1\n"
        "+    return 10\n"
    )
    diff_red = normalize_patch(mixed_redundant, files)
    assert diff_red.count("--- a/a.py") == 1
    assert "+    return 10\n" in diff_red

    # 3. Mixed SEARCH/REPLACE on a.py + unified diff touching test_a.py -> rejected by guard!
    mixed_forbidden = (
        "<<<<<<< SEARCH:a.py\n"
        "    return 1\n"
        "=======\n"
        "    return 10\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "--- a/test_a.py\n"
        "+++ b/test_a.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def test_a():\n"
        "-    assert True\n"
        "+    assert False\n"
    )
    with pytest.raises(ValueError, match="Reward-hacking guard"):
        normalize_patch(mixed_forbidden, files)

    # 4. Nested/unclosed SEARCH delimiter inside a block -> rejected cleanly
    malformed_nested = (
        "<<<<<<< SEARCH:a.py\n"
        "    return 1\n"
        "<<<<<<< SEARCH:b.py\n"
        "    return 2\n"
        "=======\n"
        "    return 20\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Malformed SEARCH/REPLACE block"):
        normalize_patch(malformed_nested, files)


def test_windows_crlf_line_endings_in_patch_and_files() -> None:
    # Repository files with Windows \r\n line endings (including a test file)
    files_crlf = {
        "pkg/service.py": "def greet(name: str) -> str:\r\n    return 'hi ' + name\r\n",
        "tests/test_service.py": "def test_greet():\r\n    assert True\r\n",
    }

    # 1. SEARCH/REPLACE patch with \r\n delimiters and body preserves file's CRLF endings
    sr_crlf = (
        "<<<<<<< SEARCH:pkg/service.py\r\n"
        "    return 'hi ' + name\r\n"
        "=======\r\n"
        "    return f'hello {name}'\r\n"
        ">>>>>>> REPLACE\r\n"
    )
    diff1 = normalize_search_replace(sr_crlf, files_crlf)
    assert "--- a/pkg/service.py\n+++ b/pkg/service.py\n" in diff1
    assert "-    return 'hi ' + name\r\n" in diff1
    assert "+    return f'hello {name}'\r\n" in diff1
    assert "test_service.py" not in diff1
    assert normalize_patch(sr_crlf, files_crlf) == diff1

    # 2. Unified diff normalization against CRLF files (must not flag untouched test_service.py!)
    diff2 = normalize_unified_diff(diff1, files_crlf)
    assert "--- a/pkg/service.py\n+++ b/pkg/service.py\n" in diff2
    assert "-    return 'hi ' + name\n" in diff2
    assert "+    return f'hello {name}'\n" in diff2
    assert "test_service.py" not in diff2


def test_tab_indentation_across_tiers() -> None:
    # 1. Tab-indented file matched by dedented tab-indented SEARCH/REPLACE block (Tier 3)
    tab_file = (
        "class Worker:\n"
        "\tdef compute(self, x: int) -> int:\n"
        "\t\tif x < 0:\n"
        "\t\t\treturn 0\n"
        "\t\treturn x - 1\n"
    )
    files = {"worker.py": tab_file}

    sr_tabs = (
        "<<<<<<< SEARCH:worker.py\n"
        "if x < 0:\n"
        "\treturn 0\n"
        "return x - 1\n"
        "=======\n"
        "if x <= 0:\n"
        "\treturn -1\n"
        "return x + 1\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(sr_tabs, files)
    assert "-\t\tif x < 0:\n" in diff
    assert "+\t\tif x <= 0:\n" in diff
    assert "+\t\t\treturn -1\n" in diff
    assert "+\t\treturn x + 1\n" in diff

    # Verify resulting patched Python code parses without TabError
    patched_map = _apply_unified_diff_to_mapping(diff, files)
    ast.parse(patched_map["worker.py"])

    # 2. Per D-47, Tier 3 preserves tabs/spaces strictly and never converts spaces <-> tabs
    sr_spaces_vs_tabs = (
        "<<<<<<< SEARCH:worker.py\n"
        "if x < 0:\n"
        "    return 0\n"
        "=======\n"
        "if x <= 0:\n"
        "    return -1\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Search block not found"):
        normalize_patch(sr_spaces_vs_tabs, files)

    space_files = {
        "space_mod.py": (
            "def check(val: int) -> int:\n    if val > 0:\n        return val - 1\n    return 0\n"
        )
    }
    sr_tabs_vs_spaces = (
        "<<<<<<< SEARCH:space_mod.py\n"
        "\tif val > 0:\n"
        "\t\treturn val - 1\n"
        "=======\n"
        "\tif val >= 0:\n"
        "\t\treturn val + 1\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Search block not found"):
        normalize_patch(sr_tabs_vs_spaces, space_files)


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "Test_upper.py",
        "pkg/ CALC_TEST.PY".replace(" ", ""),
        "Tests/sub/helper.py",
        "Conftest.py",
        "src/../tests/sneaky.py",
        "src/../test_sneaky.py",
        '"a/test_quoted.py"',
        "Makefile",
        ".env",
        "config.yaml",
        "../outside.py",
        "/etc/Absolute.py",
        "C:/drive_path.py",
    ],
)
def test_check_forbidden_patch_path_edge_cases(forbidden_path: str) -> None:
    reason = check_forbidden_patch_path(forbidden_path)
    assert reason is not None
    assert "Reward-hacking guard" in reason
    assert is_modifiable_py_file(forbidden_path) is False

    with pytest.raises(ValueError, match="Reward-hacking guard"):
        validate_patch_targets(
            f"--- a/{forbidden_path}\n+++ b/{forbidden_path}\n@@ -1 +1 @@\n-x\n+y\n"
        )


def test_is_test_file_case_and_normpath_variants() -> None:
    assert is_test_file("Test_Alpha.py") is True
    assert is_test_file("alpha_TEST.py") is True
    assert is_test_file("pkg/Tests/unit.py") is True
    assert is_test_file("src/../tests/unit.py") is True
    assert is_test_file("pkg/Conftest.py") is True
    assert is_test_file("pkg/testing_utils.py") is False


# ---------------------------------------------------------------------------
# 4. Qdrant per-case collection cleanup in graph.py & retrieval.py resilience
# ---------------------------------------------------------------------------


def test_graph_guarantees_qdrant_cleanup_when_pipeline_raises() -> None:
    async def _test() -> None:
        created: list[str] = []
        deleted: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/healthz":
                return httpx.Response(200, text="ok")
            if path.startswith("/collections/"):
                col = path.split("/")[2]
                if request.method == "PUT" and path.count("/") == 2:
                    created.append(col)
                    return httpx.Response(200, json={"result": True})
                if request.method == "DELETE" and path.count("/") == 2:
                    deleted.append(col)
                    return httpx.Response(200, json={"result": True})
                if request.method == "PUT" and path.endswith("/points"):
                    return httpx.Response(200, json={"result": {"status": "completed"}})
                if request.method == "POST" and path.endswith("/points/search"):
                    return httpx.Response(200, json={"result": []})
            return httpx.Response(200, json={"result": True})

        qdrant = QdrantStore(transport=httpx.MockTransport(handler))
        embedder = HashEmbedder(dim=64)
        files = {"app.py": "def run() -> int:\n    return 1\n"}
        expected_col = case_collection_name(files)

        async def exploding_verify(fs: dict[str, str], patch: str) -> SandboxResult:
            raise RuntimeError("Simulated unexpected crash inside verify")

        good_patch = "<<<<<<< SEARCH:app.py\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE"
        with pytest.raises(RuntimeError, match="Simulated unexpected crash"):
            await run_pipeline(
                PipelineState(case_id="crash_cleanup", files=files, data_class="public"),
                provider=MockProvider(canned=good_patch),
                verify_fn=exploding_verify,
                qdrant=qdrant,
                embedder=embedder,
            )

        assert expected_col in created
        assert expected_col in deleted

    asyncio.run(_test())


def test_ollama_embedder_malformed_json_and_unreachable_handling() -> None:
    async def _test() -> None:
        # 1. Unreachable / transport error
        def transport_err(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        emb_down = OllamaEmbedder(transport=httpx.MockTransport(transport_err))
        with pytest.raises(ProviderError, match="transport error"):
            await emb_down.embed(["hello"])

        # 2. HTTP 500 error
        emb_500 = OllamaEmbedder(
            transport=httpx.MockTransport(lambda r: httpx.Response(500, text="Internal Error"))
        )
        with pytest.raises(ProviderError, match="HTTP 500"):
            await emb_500.embed(["hello"])

        # 3. Invalid JSON body
        emb_bad_json = OllamaEmbedder(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text="{not-json"))
        )
        with pytest.raises(ProviderError, match="invalid JSON"):
            await emb_bad_json.embed(["hello"])

        # 4. Non-dict JSON payload (e.g. list or null)
        emb_list_json = OllamaEmbedder(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[1, 2, 3]))
        )
        with pytest.raises(ProviderError, match="not an object"):
            await emb_list_json.embed(["hello"])

        # 5. Malformed embedding vector elements (empty list, string, bool, count mismatch)
        bad_payloads: list[dict[str, object]] = [
            {"embeddings": [[]]},
            {"embeddings": [["not_a_float"]]},
            {"embeddings": [[True, False]]},
            {"embeddings": [[0.1, 0.2], [0.3, 0.4]]},  # 2 vectors for 1 input text
        ]

        def _make_transport(body: dict[str, object]) -> httpx.MockTransport:
            return httpx.MockTransport(lambda req: httpx.Response(200, json=body))

        for bad_body in bad_payloads:
            emb_bad_vec = OllamaEmbedder(transport=_make_transport(bad_body))
            with pytest.raises(ProviderError):
                await emb_bad_vec.embed(["single text"])

    asyncio.run(_test())


def test_qdrant_store_graceful_degradation_on_malformed_json_and_embedder_error() -> None:
    async def _test() -> None:
        # 1. Qdrant search returns HTTP 200 with invalid JSON or non-dict JSON -> returns []
        q_bad_json = QdrantStore(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<<<not json>>>"))
        )
        assert await q_bad_json.search("col", [0.1, 0.2]) == []

        q_non_dict = QdrantStore(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=["unexpected"]))
        )
        assert await q_non_dict.search("col", [0.1, 0.2]) == []

        # 2. index_case_files degrades gracefully (returns None) when OllamaEmbedder fails
        q_healthy = QdrantStore(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"result": True}))
        )
        failing_embedder = OllamaEmbedder(
            transport=httpx.MockTransport(lambda r: httpx.Response(503, text="Ollama overloaded"))
        )
        col_res = await q_healthy.index_case_files(
            {"mod.py": "def f():\n    return 1\n"},
            failing_embedder,
        )
        assert col_res is None

        # 3. Full pipeline succeeds when Qdrant returns malformed JSON and OllamaEmbedder fails
        files = {
            "mod.py": "def f() -> int:\n    return 1\n",
            "test_mod.py": "from mod import f\ndef test_f():\n    assert f() == 2\n",
        }
        good_patch = "<<<<<<< SEARCH:mod.py\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE"

        async def ok_verify(fs: dict[str, str], p: str) -> SandboxResult:
            return SandboxResult(
                exit_code=0,
                patch_applied=True,
                junit=JUnitReport(passed=1, failures=0, errors=0),
            )

        state = await run_pipeline(
            PipelineState(case_id="degraded_retrieval", files=files, data_class="public"),
            provider=MockProvider(canned=good_patch),
            verify_fn=ok_verify,
            qdrant=q_bad_json,
            embedder=failing_embedder,
        )
        assert state.verified is True
        assert state.retrieved_context == []

    asyncio.run(_test())


def test_prompt_version_hash_invariant_preserved() -> None:
    assert compute_prompt_version_hash() == (
        "ca774b4a6efb8c2384a6f216439068326840bd10ec0543a1032f135bc0f20782"
    )
