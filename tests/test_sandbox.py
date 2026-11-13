"""Unit tests (no Docker) and Docker containment tests (@pytest.mark.docker) for Phase 2."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

from patchpilot.sandbox import (
    HOSTILE_FIXTURES,
    MAX_OUTPUT_BYTES,
    MAX_PATCH_BYTES,
    MAX_TAR_FILE_BYTES,
    MAX_TAR_MEMBERS,
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
    read_stream_capped,
    run_single_fixture,
    safe_extract_tar,
    validate_deps_volume_name,
    validate_junit_summary,
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
# 1. Unit tests â€” Tar safety (runs without Docker)
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
# 2. Unit tests â€” Unified diff application in pure Python (runs without Docker)
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
# 3. Unit tests â€” Single-run lock (in-process + cross-process, no Docker)
# ---------------------------------------------------------------------------


def test_single_run_lock_in_process_and_cross_process(tmp_path: Path) -> None:
    lock_file = tmp_path / "sandbox.lock"
    verify_single_run_lock(lock_file)

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
# 4. Unit tests â€” Capped output, JSON summary, injection prevention, limits
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


def test_container_parses_junit_to_json_and_host_validates() -> None:
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
    summary = CONTAINER_RUNNER.parse_junit_xml(sample_xml)
    assert summary["tests"] == 3
    assert summary["passed"] == 1
    assert summary["failures"] == 1
    assert summary["errors"] == 0
    assert summary["skipped"] == 1
    assert len(summary["cases"]) == 3
    assert summary["cases"][0]["status"] == "passed"
    assert summary["cases"][1]["status"] == "failed"
    assert summary["cases"][2]["status"] == "skipped"

    # Host validates only the JSON summary (Fix 1)
    report = validate_junit_summary(summary)
    assert report.tests == 3
    assert report.passed == 1
    assert report.failures == 1
    assert report.cases[0].status == "passed"


def test_hostile_xml_with_doctype_entity_expansion_rejected() -> None:
    # Fix 1: Hostile XML with DOCTYPE/entity expansion must produce clean error
    hostile_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!DOCTYPE bomb [\n"
        '  <!ENTITY a "1234567890">\n'
        '  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">\n'
        '  <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">\n'
        ']><testsuites><testsuite name="pwn"><testcase name="&c;"/></testsuite></testsuites>'
    )
    summary = CONTAINER_RUNNER.parse_junit_xml(hostile_xml)
    assert "error" in summary or summary["tests"] == 0
    report = validate_junit_summary(summary)
    assert report.tests == 0


def test_docker_run_cmd_enforces_all_spec_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PP_GEMINI_API_KEY", "SUPER_SECRET_KEY")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-secret")

    cmd = build_docker_run_cmd("patchpilot-sbx-test123456")
    assert cmd[:4] == ["docker", "run", "--rm", "-i"]
    assert "--label" in cmd and cmd[cmd.index("--label") + 1] == "patchpilot.sandbox=1"
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
    assert "--label" in install_cmd
    lbl_idx = install_cmd.index("--label")
    assert install_cmd[lbl_idx + 1] == "patchpilot.sandbox=1"
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


def test_injection_resistance_case_names_and_file_paths() -> None:
    # Fix 9: Case names or file paths cannot inject extra docker run arguments
    malicious_case_name = "test_fn; rm -rf /; --privileged --volume /:/host"
    malicious_path = "src/evil$(reboot);touch pwned/mod.py"
    tar_bytes = build_repo_tar(
        files={malicious_path: "X = 1\n"},
        pytest_args=["-k", malicious_case_name],
    )
    cmd = build_docker_run_cmd("patchpilot-sbx-testsafe")
    assert not any(malicious_case_name in arg for arg in cmd)
    assert not any(malicious_path in arg for arg in cmd)
    assert "--privileged" not in cmd

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        extracted = safe_extract_tar(io.BytesIO(tar_bytes), work_dir)
        assert extracted.pytest_args == ["-k", malicious_case_name]
        assert (work_dir / malicious_path).is_file()


def test_oversize_inputs_rejected_cleanly(tmp_path: Path) -> None:
    # Fix 9: Documented maximum size and file count; oversize input rejected cleanly
    oversize_patch = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-" + ("X" * (MAX_PATCH_BYTES + 1024))
    with pytest.raises(PatchApplyError, match="Patch exceeds maximum size"):
        apply_unified_diff(oversize_patch, tmp_path)

    with pytest.raises(ValueError, match="Patch diff size exceeds limit"):
        build_repo_tar(patch_diff=oversize_patch)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        ti = tarfile.TarInfo(name="bigfile.bin")
        ti.size = MAX_TAR_FILE_BYTES + 1024
        tar.addfile(ti)
    with pytest.raises(TarSafetyError, match="exceeds per-file limit"):
        safe_extract_tar(io.BytesIO(buf.getvalue()), tmp_path)

    buf2 = io.BytesIO()
    with tarfile.open(fileobj=buf2, mode="w") as tar:
        for i in range(MAX_TAR_MEMBERS + 5):
            ti = tarfile.TarInfo(name=f"file_{i}.txt")
            ti.size = 0
            tar.addfile(ti)
    with pytest.raises(TarSafetyError, match="exceeds 10000 entries"):
        safe_extract_tar(io.BytesIO(buf2.getvalue()), tmp_path)


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


@pytest.mark.docker
def test_docker_orchestrator_start_removes_leftover_containers() -> None:
    # Fix 8: On orchestrator start, remove any leftover container with label patchpilot.sandbox=1
    dummy_name = "patchpilot-sbx-leftover-test"
    subprocess.run(["docker", "rm", "-f", dummy_name], capture_output=True, check=False)
    subprocess.run(
        [
            "docker",
            "create",
            "--name",
            dummy_name,
            "--label",
            "patchpilot.sandbox=1",
            "patchpilot-sandbox:0.0.0",
        ],
        check=True,
        capture_output=True,
    )
    assert is_container_present(dummy_name)

    # Initializing SandboxRunner triggers cleanup_leftover_containers()
    _ = SandboxRunner(cleanup_on_start=True)
    assert not is_container_present(dummy_name)
    assert dummy_name not in list_leftover_containers()


@pytest.mark.docker
def test_docker_hostile_xml_from_sandboxed_code() -> None:
    # Fix 1: Sandboxed code writing hostile XML with DOCTYPE/entity expansion produces clean error
    runner = SandboxRunner()
    res = runner.run_sync(
        files={
            "conftest.py": (
                "import pytest\n"
                "from pathlib import Path\n\n"
                "def pytest_unconfigure(config):\n"
                "    Path('/tmp/patchpilot_junit.xml').write_text(\n"
                "        '<!DOCTYPE bomb [<!ENTITY a \"123\">]><testsuites/>',\n"
                "        encoding='utf-8',\n"
                "    )\n"
            ),
            "test_sample.py": "def test_ok(): pass\n",
        }
    )
    assert res.error == "Hostile XML: DOCTYPE/ENTITY declarations rejected"
    assert res.exit_code != 0
    assert res.junit.tests == 0


# ---------------------------------------------------------------------------
# 6. Docker Mutation Test Suite (marked `docker`) (Fix 3)
# ---------------------------------------------------------------------------


@pytest.mark.docker
def test_mutation_network() -> None:
    # Weakened network: --network bridge instead of --network none
    fixture = next(f for f in HOSTILE_FIXTURES if f.name == "network_egress")
    weak_runner = SandboxRunner(override_limits={"network": "bridge"})
    res = run_single_fixture(weak_runner, fixture)
    assert not fixture.check_contained(res), (
        "network_egress fixture reported CONTAINED under weakened network config (--network bridge)"
    )


@pytest.mark.docker
def test_mutation_pids_limit() -> None:
    # Weakened pids: --pids-limit 1024 instead of 256
    fixture = next(f for f in HOSTILE_FIXTURES if f.name == "fork_bomb")
    weak_runner = SandboxRunner(override_limits={"pids_limit": 1024})
    res = run_single_fixture(weak_runner, fixture)
    assert not fixture.check_contained(res), (
        "fork_bomb fixture reported CONTAINED under weakened pids-limit config (1024)"
    )


@pytest.mark.docker
def test_mutation_memory() -> None:
    # Weakened memory: 2048m instead of 768m
    fixture = next(f for f in HOSTILE_FIXTURES if f.name == "memory_balloon")
    weak_runner = SandboxRunner(override_limits={"memory": "2048m", "memory_swap": "2048m"})
    res = run_single_fixture(weak_runner, fixture)
    assert not fixture.check_contained(res), (
        "memory_balloon fixture reported CONTAINED under weakened memory config (2048m)"
    )


@pytest.mark.docker
def test_mutation_read_only_root() -> None:
    # Weakened filesystem: read_only=False instead of True
    fixture = next(f for f in HOSTILE_FIXTURES if f.name == "write_outside_work")
    weak_runner = SandboxRunner(override_limits={"read_only": False})
    res = run_single_fixture(weak_runner, fixture)
    assert not fixture.check_contained(res), (
        "write_outside_work fixture reported CONTAINED under weakened rootfs config (read-write)"
    )


@pytest.mark.docker
def test_mutation_user_caps() -> None:
    # Weakened user/caps: user="0:0" and caps not dropped
    fixture = next(f for f in HOSTILE_FIXTURES if f.name == "read_host_env")
    weak_runner = SandboxRunner(override_limits={"user": "0:0", "cap_drop": None})
    res = run_single_fixture(weak_runner, fixture)
    assert not fixture.check_contained(res), (
        "read_host_env fixture reported CONTAINED under weakened user/caps config (root/caps)"
    )


# ---------------------------------------------------------------------------
# Regression tests: dotfiles, case-insensitive DTD guard, verify/image slicing
# ---------------------------------------------------------------------------


def test_tar_roundtrip_preserves_dotfiles(tmp_path: Path) -> None:
    files = {
        ".gitignore": "*.pyc\n",
        ".flake8": "[flake8]\n",
        ".env.example": "A=1\n",
        ".hidden/file.py": "x = 1\n",
        "pkg/mod.py": "y = 2\n",
    }
    tar_bytes = build_repo_tar(files=files)
    for extractor in (safe_extract_tar, CONTAINER_RUNNER.safe_extract_tar):
        work = tmp_path / f"w_{extractor.__module__}"
        work.mkdir()
        extracted = extractor(io.BytesIO(tar_bytes), work)
        assert extracted.files_extracted == len(files)
        for rel, content in files.items():
            assert (work / rel).read_text() == content
        assert not (work / "gitignore").exists()
        assert not (work / "env.example").exists()

    # "./"-prefixed dotfile members are also extracted intact.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = b"x"
        info = tarfile.TarInfo("./.gitignore")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    for extractor in (safe_extract_tar, CONTAINER_RUNNER.safe_extract_tar):
        work = tmp_path / f"d_{extractor.__module__}"
        work.mkdir()
        extractor(io.BytesIO(buf.getvalue()), work)
        assert (work / ".gitignore").is_file()


@pytest.mark.parametrize("bad_name", ["../.gitignore", ".hidden/../../x", "./../x"])
def test_tar_dotfile_fix_still_rejects_traversal(tmp_path: Path, bad_name: str) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(bad_name)
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    for extractor, exc in (
        (safe_extract_tar, TarSafetyError),
        (CONTAINER_RUNNER.safe_extract_tar, CONTAINER_RUNNER.TarSafetyError),
    ):
        work = tmp_path / f"t_{extractor.__module__}"
        work.mkdir(exist_ok=True)
        with pytest.raises(exc):
            extractor(io.BytesIO(buf.getvalue()), work)


def test_tar_dotfile_fix_still_rejects_symlinks(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(".link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(TarSafetyError):
        safe_extract_tar(io.BytesIO(buf.getvalue()), tmp_path)


@pytest.mark.parametrize(
    "decl",
    [
        "<!doctype bomb [<!entity a 'x'>]>",
        "<!DoCtYpE bomb>",
        "<!DOCTYPE bomb>",
        "<!entity a 'x'>",
        "<!EnTiTy a 'x'>",
    ],
)
def test_junit_xml_guard_is_case_insensitive(decl: str) -> None:
    xml = f'<?xml version="1.0"?>\n{decl}<testsuites><testsuite tests="1"/></testsuites>'
    summary = CONTAINER_RUNNER.parse_junit_xml(xml)
    assert "Hostile XML" in str(summary.get("error"))
    assert summary["tests"] == 0


@pytest.mark.parametrize("verify", [False, True])
def test_run_sync_separates_image_from_container_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verify: bool
) -> None:
    from patchpilot.sandbox import runner as runner_mod

    captured: list[list[str]] = []

    class FakeProc:
        def __init__(self, cmd: list[str], **_: object) -> None:
            captured.append(list(cmd))
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(runner_mod.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(runner_mod, "get_image_digest", lambda image: "sha256:test")
    monkeypatch.setattr(runner_mod, "is_container_present", lambda name: False)
    runner = runner_mod.SandboxRunner(
        image="patchpilot-sandbox:test",
        lock_path=tmp_path / "lock",
        cleanup_on_start=False,
    )
    runner.run_sync(files={"a.py": "x = 1\n"}, verify=verify)

    cmd = captured[0]
    assert cmd[-2:] == ["patchpilot-sandbox:test", "--verify"] if verify else True
    assert cmd[-1] == ("--verify" if verify else "patchpilot-sandbox:test")
    log = runner.run_logs[-1]
    assert "patchpilot-sandbox:test" not in log.flags
    assert "--verify" not in log.flags
    assert log.flags[-1] == "768m" or "--pids-limit" in log.flags
    for required in ("--network", "none", "--read-only", "--cap-drop", "ALL", "256", "768m"):
        assert required in log.flags
