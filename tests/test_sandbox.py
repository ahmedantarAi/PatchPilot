"""Unit tests (no Docker) and Docker containment tests (@pytest.mark.docker) for Phase 2."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

from patchpilot.sandbox import (
    HOSTILE_FIXTURES,
    MAX_OUTPUT_BYTES,
    HostileFixture,
    PatchApplyError,
    SandboxBusyError,
    SandboxLock,
    SandboxRunner,
    TarSafetyError,
    apply_unified_diff,
    build_docker_install_cmd,
    build_docker_run_cmd,
    build_repo_tar,
    cleanup_leftover_containers,
    compute_deps_volume_name,
    is_container_present,
    list_leftover_containers,
    parse_junit_xml,
    read_stream_capped,
    run_single_fixture,
    safe_extract_tar,
    validate_deps_volume_name,
    validate_requirements,
    validate_safe_relative_path,
    verify_install_step_no_repo_code,
    verify_single_run_lock,
)
from patchpilot.sandbox.runner import _minimal_docker_client_env


def _load_container_runner_module() -> ModuleType:
    runner_path = Path(__file__).resolve().parent.parent / "sandbox" / "runner.py"
    spec = importlib.util.spec_from_file_location("sandbox_container_runner", runner_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_container_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


CONTAINER_RUNNER = _load_container_runner_module()


# ---------------------------------------------------------------------------
# 1. Unit tests — Tar safety (runs without Docker)
# ---------------------------------------------------------------------------


def test_tar_extract_clean_repo_and_patch(tmp_path: Path) -> None:
    tar_bytes = build_repo_tar(
        files={
            "pkg/__init__.py": "",
            "pkg/math_ops.py": "def add(a: int, b: int) -> int:\n    return a - b\n",
            "tests/test_math.py": (
                "from pkg.math_ops import add\ndef test_add():\n    assert add(2, 3) == 5\n"
            ),
        },
        patch_diff=(
            "--- a/pkg/math_ops.py\n"
            "+++ b/pkg/math_ops.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a: int, b: int) -> int:\n"
            "-    return a - b\n"
            "+    return a + b\n"
        ),
        pytest_args=["-k", "test_add"],
    )

    for extractor in (safe_extract_tar, CONTAINER_RUNNER.safe_extract_tar):
        work_dir = tmp_path / f"work_{extractor.__module__}"
        work_dir.mkdir()
        extracted = extractor(io.BytesIO(tar_bytes), work_dir)
        assert extracted.files_extracted == 3
        assert "return a + b" in extracted.patch_diff
        assert extracted.pytest_args == ["-k", "test_add"]
        assert (work_dir / "pkg" / "math_ops.py").is_file()
        assert not (work_dir / "__patchpilot__").exists()


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escape.py",
        "sub/../../escape.py",
        "..\\escape.py",
        "/etc/passwd",
        "\\Windows\\win.ini",
        "C:/evil.py",
        "valid/../../../outside.txt",
    ],
)
def test_tar_rejects_path_traversal_and_absolute_paths(tmp_path: Path, bad_name: str) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = b"pwned"
        info = tarfile.TarInfo(name=bad_name)
        info.type = tarfile.REGTYPE
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    raw_tar = buf.getvalue()

    with pytest.raises(TarSafetyError):
        safe_extract_tar(io.BytesIO(raw_tar), tmp_path)

    with pytest.raises(CONTAINER_RUNNER.TarSafetyError):
        CONTAINER_RUNNER.safe_extract_tar(io.BytesIO(raw_tar), tmp_path)


def test_validate_safe_relative_path_rejects_null_and_empty(tmp_path: Path) -> None:
    for bad in ("", "null\x00byte.py", ".", "./"):
        with pytest.raises(TarSafetyError):
            validate_safe_relative_path(bad, tmp_path)
        with pytest.raises(CONTAINER_RUNNER.TarSafetyError):
            CONTAINER_RUNNER.validate_safe_relative_path(bad, tmp_path)


@pytest.mark.parametrize(
    ("link_type", "link_target"),
    [
        (tarfile.SYMTYPE, "/etc/passwd"),
        (tarfile.SYMTYPE, "../../etc/shadow"),
        (tarfile.SYMTYPE, "local_file.txt"),
        (tarfile.LNKTYPE, "/etc/passwd"),
        (tarfile.FIFOTYPE, ""),
    ],
)
def test_tar_rejects_symlinks_hardlinks_and_special_entries(
    tmp_path: Path,
    link_type: bytes,
    link_target: str,
) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="evil_entry")
        info.type = link_type
        info.linkname = link_target
        info.size = 0
        tar.addfile(info)
    raw_tar = buf.getvalue()

    with pytest.raises(TarSafetyError):
        safe_extract_tar(io.BytesIO(raw_tar), tmp_path)

    with pytest.raises(CONTAINER_RUNNER.TarSafetyError):
        CONTAINER_RUNNER.safe_extract_tar(io.BytesIO(raw_tar), tmp_path)


def test_build_repo_tar_packs_symlinks_manifest_without_host_deref(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_x.py").write_text("def test_x(): pass\n", encoding="utf-8")
    (repo / "__symlinks__.json").write_text(
        '{"leak": "/nonexistent/host/secret/path"}',
        encoding="utf-8",
    )
    tar_bytes = build_repo_tar(repo)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(TarSafetyError, match="Symlink/hardlink rejected"):
        safe_extract_tar(io.BytesIO(tar_bytes), work_dir)


# ---------------------------------------------------------------------------
# 2. Unit tests — Unified diff application in pure Python (runs without Docker)
# ---------------------------------------------------------------------------


def test_apply_unified_diff_modify_create_and_delete(tmp_path: Path) -> None:
    for applicator in (apply_unified_diff, CONTAINER_RUNNER.apply_unified_diff):
        work = tmp_path / f"diff_{applicator.__module__}"
        work.mkdir()
        (work / "calc.py").write_text(
            "def add(x: int, y: int) -> int:\n"
            "    return x - y\n"
            "\n"
            "def mul(x: int, y: int) -> int:\n"
            "    return x + y\n",
            encoding="utf-8",
        )
        (work / "obsolete.py").write_text("OLD = True\n", encoding="utf-8")

        diff = (
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,5 +1,5 @@\n"
            " def add(x: int, y: int) -> int:\n"
            "-    return x - y\n"
            "+    return x + y\n"
            " \n"
            " def mul(x: int, y: int) -> int:\n"
            "-    return x + y\n"
            "+    return x * y\n"
            "--- /dev/null\n"
            "+++ b/sub/helper.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def greet() -> str:\n"
            "+    return 'hi'\n"
            "--- a/obsolete.py\n"
            "+++ /dev/null\n"
            "@@ -1,1 +0,0 @@\n"
            "-OLD = True\n"
        )

        count = applicator(diff, work)
        assert count == 3
        assert (work / "calc.py").read_text(encoding="utf-8") == (
            "def add(x: int, y: int) -> int:\n"
            "    return x + y\n"
            "\n"
            "def mul(x: int, y: int) -> int:\n"
            "    return x * y\n"
        )
        assert (work / "sub" / "helper.py").read_text(encoding="utf-8") == (
            "def greet() -> str:\n    return 'hi'\n"
        )
        assert not (work / "obsolete.py").exists()


def test_apply_unified_diff_rejects_context_mismatch_and_traversal(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    original = "VALUE = 1\n"
    target.write_text(original, encoding="utf-8")

    bad_context_diff = "--- a/app.py\n+++ b/app.py\n@@ -1,1 +1,1 @@\n-VALUE = 999\n+VALUE = 2\n"
    with pytest.raises(PatchApplyError, match="Context mismatch"):
        apply_unified_diff(bad_context_diff, tmp_path)
    assert target.read_text(encoding="utf-8") == original

    traversal_diff = "--- /dev/null\n+++ b/../../escaped.py\n@@ -0,0 +1,1 @@\n+PWNED = True\n"
    with pytest.raises(PatchApplyError, match="Unsafe path"):
        apply_unified_diff(traversal_diff, tmp_path)


# ---------------------------------------------------------------------------
# 3. Unit tests — Single-run lock (in-process + cross-process, no Docker)
# ---------------------------------------------------------------------------


def test_single_run_lock_in_process_and_cross_process(tmp_path: Path) -> None:
    lock_file = tmp_path / "sandbox.lock"
    verify_single_run_lock(lock_file)

    # Verify lock can be re-acquired cleanly after release
    with SandboxLock(lock_file):
        with pytest.raises(SandboxBusyError):
            with SandboxLock(lock_file):
                pass

    async def _acquire_async() -> None:
        async with SandboxLock(lock_file):
            with pytest.raises(SandboxBusyError):
                async with SandboxLock(lock_file):
                    pass

    asyncio.run(_acquire_async())


# ---------------------------------------------------------------------------
# 4. Unit tests — 1 MB output cap, JUnit XML parser, Docker flags, and logging
# ---------------------------------------------------------------------------


def test_stream_output_capped_at_1mb() -> None:
    oversized = b"A" * (MAX_OUTPUT_BYTES + 500_000)
    text, truncated = read_stream_capped(io.BytesIO(oversized), MAX_OUTPUT_BYTES)
    assert truncated is True
    assert len(text.encode("utf-8")) == MAX_OUTPUT_BYTES

    exact = b"B" * MAX_OUTPUT_BYTES
    text2, truncated2 = read_stream_capped(io.BytesIO(exact), MAX_OUTPUT_BYTES)
    assert truncated2 is False
    assert len(text2) == MAX_OUTPUT_BYTES


def test_parse_junit_xml_structured_counts() -> None:
    sample_xml = """<?xml version="1.0" encoding="utf-8"?>
    <testsuites>
      <testsuite name="pytest" tests="3" errors="0" failures="1" skipped="1">
        <testcase classname="test_mod" name="test_pass" time="0.01" />
        <testcase classname="test_mod" name="test_fail" time="0.02">
          <failure message="assert 1 == 2">AssertionError: assert 1 == 2</failure>
        </testcase>
        <testcase classname="test_mod" name="test_skip" time="0.00">
          <skipped message="skipped reason" />
        </testcase>
      </testsuite>
    </testsuites>
    """
    report = parse_junit_xml(sample_xml)
    assert report.tests == 3
    assert report.passed == 1
    assert report.failures == 1
    assert report.errors == 0
    assert report.skipped == 1
    assert len(report.cases) == 3
    assert report.cases[0].status == "passed"
    assert report.cases[1].status == "failed"
    assert report.cases[2].status == "skipped"


def test_docker_run_cmd_enforces_all_spec_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PP_GEMINI_API_KEY", "SUPER_SECRET_KEY")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-secret")

    cmd = build_docker_run_cmd("patchpilot-sbx-test123456")
    assert cmd[:4] == ["docker", "run", "--rm", "-i"]
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--user" in cmd and cmd[cmd.index("--user") + 1] == "10001:10001"
    assert "--cap-drop" in cmd and cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in cmd and cmd[cmd.index("--security-opt") + 1] == "no-new-privileges"
    assert "--read-only" in cmd
    assert "/work:rw,nosuid,size=256m,uid=10001,gid=10001" in cmd
    assert "/tmp:rw,noexec,nosuid,size=128m" in cmd
    assert "--memory" in cmd and cmd[cmd.index("--memory") + 1] == "768m"
    assert "--memory-swap" in cmd and cmd[cmd.index("--memory-swap") + 1] == "768m"
    assert "--cpus" in cmd and cmd[cmd.index("--cpus") + 1] == "1"
    assert "--pids-limit" in cmd and cmd[cmd.index("--pids-limit") + 1] == "256"
    assert "-e" not in cmd
    assert "-v" not in cmd
    assert "--privileged" not in cmd
    assert all("docker.sock" not in part for part in cmd)

    vol = compute_deps_volume_name(["pytest==9.2.0"])
    cmd_with_deps = build_docker_run_cmd("patchpilot-sbx-test123456", deps_volume=vol)
    assert f"{vol}:/deps:ro" in cmd_with_deps
    assert "PYTHONPATH=/deps" in cmd_with_deps

    install_cmd = build_docker_install_cmd("patchpilot-sbx-deps1234", vol)
    assert f"{vol}:/deps:rw" in install_cmd
    assert install_cmd[-1] == "--install-deps"

    client_env = _minimal_docker_client_env()
    assert "PP_GEMINI_API_KEY" not in client_env
    assert "LANGFUSE_SECRET_KEY" not in client_env
    assert "SUPER_SECRET_KEY" not in str(client_env)


def test_validate_requirements_and_volume_names() -> None:
    assert validate_requirements(["six==1.17.0", " # comment ", "pydantic==2.14.2"]) == [
        "six==1.17.0",
        "pydantic==2.14.2",
    ]
    for invalid_req in (
        "unpinned_pkg",
        "pkg>=1.0",
        "-e .",
        "--extra-index-url https://evil.example/simple",
        "git+https://github.com/org/repo.git",
        "../local_pkg",
        "/tmp/wheel.whl",
        "pkg @ https://evil.example/pkg.whl",
    ):
        with pytest.raises(ValueError):
            validate_requirements([invalid_req])

    for invalid_vol in ("/host/path", "C:\\Windows", "patchpilot-deps-../evil", "other-vol"):
        with pytest.raises(ValueError):
            validate_deps_volume_name(invalid_vol)


# ---------------------------------------------------------------------------
# 5. Docker integration tests (marked `docker`, deselected by default / CI)
# ---------------------------------------------------------------------------


@pytest.mark.docker
@pytest.mark.parametrize("fixture", HOSTILE_FIXTURES, ids=[f.name for f in HOSTILE_FIXTURES])
def test_hostile_fixture_contained_in_docker(fixture: HostileFixture) -> None:
    runner = SandboxRunner()
    result = run_single_fixture(runner, fixture)
    assert fixture.check_contained(result), (
        f"Fixture {fixture.name} was not contained: "
        f"exit_code={result.exit_code}, timed_out={result.timed_out}, error={result.error}"
    )
    assert not is_container_present(result.container_name)
    assert len(runner.run_logs) == 1
    log_entry = runner.run_logs[0]
    assert log_entry.image_digest.startswith("sha256:")
    assert log_entry.duration_s > 0.0
    assert log_entry.exit_status == result.exit_code
    assert log_entry.limits["memory"] == "768m"
    assert list_leftover_containers() == []


@pytest.mark.docker
def test_docker_clean_patch_and_two_step_deps() -> None:
    cleanup_leftover_containers()
    runner = SandboxRunner()
    res = runner.run_sync(
        files={
            "buggy.py": "def inc(x: int) -> int:\n    return x - 1\n",
            "test_buggy.py": "from buggy import inc\ndef test_inc():\n    assert inc(4) == 5\n",
        },
        patch_diff=(
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def inc(x: int) -> int:\n"
            "-    return x - 1\n"
            "+    return x + 1\n"
        ),
    )
    assert res.exit_code == 0
    assert res.patch_applied is True
    assert res.junit.passed == 1
    assert res.junit.failures == 0
    assert not is_container_present(res.container_name)

    verify_install_step_no_repo_code(runner)
    assert list_leftover_containers() == []
    assert os.environ.get("PP_SANDBOX_LEAK") is None
    proc = subprocess.run(
        ["docker", "volume", "ls", "--filter", "name=patchpilot-deps-", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.stdout.strip() == ""
