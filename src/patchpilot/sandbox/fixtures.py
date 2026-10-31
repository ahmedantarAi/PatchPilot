"""Hostile-fixture containment suite and Phase 2 sandbox selftest."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from patchpilot.sandbox.runner import (
    SandboxBusyError,
    SandboxLock,
    SandboxResult,
    SandboxRunner,
    cleanup_leftover_containers,
    is_container_present,
    list_leftover_containers,
    remove_deps_volume,
)


@dataclass(frozen=True)
class HostileFixture:
    name: str
    description: str
    files: dict[str, str]
    symlinks: dict[str, str] = field(default_factory=dict)
    patch_diff: str = ""
    timeout_s: float = 20.0
    check_contained: Callable[[SandboxResult], bool] = lambda r: r.exit_code != 0


NETWORK_EGRESS_CODE = """\
import socket
import urllib.request


def test_network_egress_blocked() -> None:
    targets = [
        "http://1.1.1.1",
        "http://example.com",
        "http://host.docker.internal:11434/api/version",
    ]
    connected = False
    for url in targets:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status > 0:
                    connected = True
        except Exception:
            pass

    try:
        sock = socket.create_connection(("8.8.8.8", 53), timeout=1.5)
        sock.close()
        connected = True
    except Exception:
        pass

    # Attack succeeds only if outbound egress worked; must fail in sandbox.
    assert connected, "Outbound network egress was completely blocked"
"""

READ_HOST_ENV_CODE = """\
import os
from pathlib import Path


def test_read_host_files_and_env() -> None:
    leaked: list[str] = []

    for key, val in os.environ.items():
        if key.startswith(("PP_", "LANGFUSE_", "GEMINI", "OPENAI", "AWS_", "GITHUB_")):
            leaked.append(f"env:{key}={val}")
        if key in ("USERPROFILE", "APPDATA", "LOCALAPPDATA"):
            leaked.append(f"host_env:{key}")

    proc_env_path = Path("/proc/1/environ")
    if proc_env_path.exists():
        try:
            raw = proc_env_path.read_bytes().decode("utf-8", errors="ignore")
            for entry in raw.split("\\x00"):
                if entry.startswith(("PP_", "LANGFUSE_", "GEMINI", "USERPROFILE=")):
                    leaked.append(f"proc1:{entry}")
        except OSError:
            pass

    for sensitive_path in (
        "/etc/shadow",
        "/root/.bashrc",
        "/var/run/docker.sock",
        "/mnt/c/Windows/win.ini",
        "/host_mnt/c/Windows/win.ini",
    ):
        p = Path(sensitive_path)
        try:
            if p.exists() and p.read_bytes():
                leaked.append(f"file:{sensitive_path}")
        except OSError:
            pass

    # Attack succeeds only if host secrets/files were readable; must fail in sandbox.
    assert leaked, "No host secrets, /proc/1/environ secrets, or host files were readable"
"""

FORK_BOMB_CODE = """\
import os
import signal
import time


def test_fork_bomb_contained() -> None:
    pids: list[int] = []
    fork_blocked = False
    try:
        for _ in range(400):
            try:
                pid = os.fork()
            except OSError:
                fork_blocked = True
                break
            if pid == 0:
                time.sleep(10.0)
                os._exit(0)
            else:
                pids.append(pid)
    finally:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        for pid in pids:
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass

    # Attack succeeds only if all 400 forks succeeded without hitting pids-limit 256.
    assert not fork_blocked and len(pids) >= 400, (
        f"Fork bomb contained at {len(pids)} processes by pids-limit"
    )
"""

DISK_FILL_CODE = """\
import os
import subprocess
from pathlib import Path


def test_disk_fill_and_tmp_noexec_contained() -> None:
    work_target = Path("/work/bigfile.bin")
    tmp_target = Path("/tmp/bigfile.bin")
    work_filled_all = False
    tmp_filled_all = False
    tmp_exec_worked = False

    try:
        # /work tmpfs is 256 MB; attempting to write 300 MB with dd must fail with ENOSPC
        proc_work = subprocess.run(
            ["dd", "if=/dev/zero", f"of={work_target}", "bs=1M", "count=300"],
            capture_output=True,
            text=True,
            check=False,
        )
        work_filled_all = proc_work.returncode == 0
    finally:
        if work_target.exists():
            work_target.unlink()

    try:
        # /tmp tmpfs is 128 MB; attempting to write 160 MB with dd must fail with ENOSPC
        proc_tmp = subprocess.run(
            ["dd", "if=/dev/zero", f"of={tmp_target}", "bs=1M", "count=160"],
            capture_output=True,
            text=True,
            check=False,
        )
        tmp_filled_all = proc_tmp.returncode == 0
    finally:
        if tmp_target.exists():
            tmp_target.unlink()

    # /tmp is mounted noexec; executing a script from /tmp must fail with PermissionError
    script = Path("/tmp/exploit.sh")
    try:
        script.write_text("#!/bin/sh\\necho pwned\\n", encoding="utf-8")
        script.chmod(0o755)
        try:
            os.execv(str(script), [str(script)])
            tmp_exec_worked = True
        except OSError:
            tmp_exec_worked = False
    finally:
        if script.exists():
            script.unlink()

    assert work_filled_all or tmp_filled_all or tmp_exec_worked, (
        "Disk fill with dd and /tmp execution were blocked by tmpfs limits and noexec"
    )
"""

INFINITE_LOOP_CODE = """\
import signal
import time


def test_infinite_loop_killed_by_timeout() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.05)
"""

MEMORY_BALLOON_CODE = """\
def test_memory_balloon_contained() -> None:
    # Sandbox memory + swap limit is 768 MB; attempt to allocate and touch 1024 MB.
    chunks: list[bytearray] = []
    chunk_size = 16 * 1024 * 1024  # 16 MB
    for _ in range(64):  # 64 * 16 MB = 1024 MB
        buf = bytearray(chunk_size)
        for offset in range(0, chunk_size, 4096):
            buf[offset] = 1
        chunks.append(buf)
    assert len(chunks) == 64
"""

WRITE_OUTSIDE_WORK_CODE = """\
from pathlib import Path


def test_write_outside_work_blocked() -> None:
    targets = [
        Path("/evil.txt"),
        Path("/etc/evil.conf"),
        Path("/usr/local/bin/evil"),
        Path("/home/runner/evil.txt"),
        Path("/var/tmp/evil.txt"),
    ]
    wrote_any = False
    for target in targets:
        try:
            target.write_text("pwned", encoding="utf-8")
            wrote_any = True
        except OSError:
            pass

    assert wrote_any, "All writes outside /work and /tmp were blocked by read-only rootfs"
"""

SYMLINK_ESCAPE_CODE = """\
from pathlib import Path


def test_symlink_escape_from_repo() -> None:
    escaped = Path("/work/escape_passwd")
    assert escaped.exists() and "root:" in escaped.read_text(encoding="utf-8")
"""

HOSTILE_FIXTURES: tuple[HostileFixture, ...] = (
    HostileFixture(
        name="network_egress",
        description="Attempts outbound HTTP and raw socket connections under --network none",
        files={"test_network_egress.py": NETWORK_EGRESS_CODE},
        check_contained=lambda r: r.exit_code != 0 and not r.timed_out and r.junit.failures >= 1,
    ),
    HostileFixture(
        name="read_host_env",
        description="Attempts to read host files, /etc/shadow, /proc/1/environ, and secrets",
        files={"test_read_host_env.py": READ_HOST_ENV_CODE},
        check_contained=lambda r: r.exit_code != 0 and not r.timed_out and r.junit.failures >= 1,
    ),
    HostileFixture(
        name="fork_bomb",
        description="Attempts to fork processes past --pids-limit 256",
        files={"test_fork_bomb.py": FORK_BOMB_CODE},
        check_contained=lambda r: r.exit_code != 0 and not r.timed_out,
    ),
    HostileFixture(
        name="disk_fill",
        description="Attempts to overfill /work and /tmp with dd and execute from noexec /tmp",
        files={"test_disk_fill.py": DISK_FILL_CODE},
        check_contained=lambda r: r.exit_code != 0 and not r.timed_out and r.junit.failures >= 1,
    ),
    HostileFixture(
        name="infinite_loop",
        description="Spins indefinitely ignoring SIGTERM; must be killed by wall-clock timeout",
        files={"test_infinite_loop.py": INFINITE_LOOP_CODE},
        timeout_s=2.5,
        check_contained=lambda r: r.timed_out and r.exit_code != 0,
    ),
    HostileFixture(
        name="memory_balloon",
        description="Allocates > 768 MB physical RAM with no swap; must fail with OOM/MemoryError",
        files={"test_memory_balloon.py": MEMORY_BALLOON_CODE},
        check_contained=lambda r: r.exit_code != 0,
    ),
    HostileFixture(
        name="write_outside_work",
        description="Attempts to write to /, /etc, /usr, /home, /var outside /work and /tmp",
        files={"test_write_outside_work.py": WRITE_OUTSIDE_WORK_CODE},
        check_contained=lambda r: r.exit_code != 0 and not r.timed_out and r.junit.failures >= 1,
    ),
    HostileFixture(
        name="symlink_escape",
        description="Copied repo contains symlinks escaping /work to /etc/passwd and /etc/shadow",
        files={"test_symlink_escape.py": SYMLINK_ESCAPE_CODE},
        symlinks={
            "escape_passwd": "/etc/passwd",
            "sub/escape_rel": "../../etc/shadow",
        },
        check_contained=lambda r: (
            r.exit_code != 0
            and not r.patch_applied
            and r.error is not None
            and "TarSafetyError" in r.error
        ),
    ),
)


def run_single_fixture(runner: SandboxRunner, fixture: HostileFixture) -> SandboxResult:
    """Execute one hostile fixture inside the sandbox and verify container cleanup."""
    if fixture.symlinks:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            for rel_path, content in fixture.files.items():
                dest = repo_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            import json

            (repo_dir / "__symlinks__.json").write_text(
                json.dumps(fixture.symlinks),
                encoding="utf-8",
            )
            result = runner.run_sync(
                repo_dir,
                patch_diff=fixture.patch_diff,
                timeout_s=fixture.timeout_s,
            )
    else:
        result = runner.run_sync(
            files=fixture.files,
            patch_diff=fixture.patch_diff,
            timeout_s=fixture.timeout_s,
        )

    if is_container_present(result.container_name):
        raise AssertionError(f"Sandbox container {result.container_name} was not removed!")
    return result


def verify_single_run_lock(lock_path: Path | None = None) -> None:
    """Verify both in-process async lock and cross-process OS file lock raise SandboxBusyError."""
    with tempfile.TemporaryDirectory() as tmp:
        effective_lock = lock_path or (Path(tmp) / "test-sandbox.lock")
        runner = SandboxRunner(lock_path=effective_lock)

        # 1. In-process contention: hold lock and attempt async run + second sync lock
        with SandboxLock(effective_lock):
            try:
                with SandboxLock(effective_lock):
                    raise AssertionError("Expected SandboxBusyError on second sync lock acquire")
            except SandboxBusyError:
                pass

            try:
                asyncio.run(runner.run(files={"test_noop.py": "def test_ok(): pass\n"}))
                raise AssertionError("Expected SandboxBusyError on concurrent async run")
            except SandboxBusyError:
                pass

        # 2. Cross-process contention: hold OS file lock in a child process
        src_root = str(Path(__file__).resolve().parent.parent.parent)
        child_code = (
            "import sys, time\n"
            f"sys.path.insert(0, {src_root!r})\n"
            "from pathlib import Path\n"
            "from patchpilot.sandbox.runner import SandboxLock\n"
            "lock_file = Path(sys.argv[1])\n"
            "with SandboxLock(lock_file):\n"
            "    sys.stdout.write('LOCKED\\n')\n"
            "    sys.stdout.flush()\n"
            "    time.sleep(5.0)\n"
        )
        child = subprocess.Popen(
            [sys.executable, "-c", child_code, str(effective_lock)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout is not None
            ready_line = child.stdout.readline().strip()
            if ready_line != "LOCKED":
                stderr_out = child.stderr.read() if child.stderr else ""
                raise AssertionError(f"Child failed to acquire lock: {stderr_out}")

            try:
                with SandboxLock(effective_lock):
                    raise AssertionError(
                        "Expected SandboxBusyError when cross-process file lock is held"
                    )
            except SandboxBusyError:
                pass
        finally:
            child.kill()
            child.wait(timeout=5.0)


def verify_install_step_no_repo_code(runner: SandboxRunner) -> None:
    """Verify D-24 two-step dependency install runs no repo code and mounts /deps read-only."""
    # Step 1: Verify sdist with setup.py is rejected by --only-binary :all:
    sdist_res = runner.install_dependencies_sync(
        ["malicious-sdist==0.1.0"],
        find_links="/usr/local/share/patchpilot-wheels",
        network="none",
    )
    try:
        if sdist_res.exit_code == 0:
            raise AssertionError("Expected sdist without binary wheel to be rejected")
        for required_flag in ("--only-binary", ":all:", "--no-build-isolation", "--no-input"):
            if required_flag not in sdist_res.cmd:
                raise AssertionError(f"Missing required flag {required_flag} in {sdist_res.cmd}")
    finally:
        remove_deps_volume(sdist_res.volume_name)

    # Step 2: Install binary wheel into /deps volume with hostile setup.py in the repo on host
    wheel_res = runner.install_dependencies_sync(
        ["patchpilot-dummy-dep==0.1.0"],
        find_links="/usr/local/share/patchpilot-wheels",
        network="none",
    )
    try:
        if wheel_res.exit_code != 0:
            raise AssertionError(
                f"Wheel install failed ({wheel_res.exit_code}): {wheel_res.stderr}"
            )

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            # Hostile setup.py in repo root — must never be present during the install step
            (repo_dir / "setup.py").write_text(
                "import pathlib\n"
                "pathlib.Path('/deps/repo_setup_ran').write_text('pwned')\n"
                "raise SystemExit('setup.py in repo was executed!')\n",
                encoding="utf-8",
            )
            (repo_dir / "test_deps_verify.py").write_text(
                "from pathlib import Path\n"
                "import pytest\n"
                "import patchpilot_dummy_dep\n\n"
                "def test_wheel_available_and_deps_ro() -> None:\n"
                "    assert patchpilot_dummy_dep.VALUE == 'installed_from_wheel'\n"
                "    assert not Path('/deps/repo_setup_ran').exists()\n"
                "    assert not Path('/deps/pwned_by_sdist').exists()\n"
                "    with pytest.raises(OSError):\n"
                "        Path('/deps/tamper.py').write_text('evil', encoding='utf-8')\n",
                encoding="utf-8",
            )
            verify_res = runner.run_sync(repo_dir, deps_volume=wheel_res.volume_name)
            if verify_res.exit_code != 0 or verify_res.junit.passed != 1:
                raise AssertionError(
                    f"Verification with /deps:ro failed: exit={verify_res.exit_code}, "
                    f"stdout={verify_res.stdout}, stderr={verify_res.stderr}"
                )
    finally:
        remove_deps_volume(wheel_res.volume_name)


def run_selftest() -> int:
    """Run the full Phase 2 sandbox selftest and print the required acceptance lines."""
    cleanup_leftover_containers()
    runner = SandboxRunner()
    contained_count = 0

    for fixture in HOSTILE_FIXTURES:
        result = run_single_fixture(runner, fixture)
        if not fixture.check_contained(result):
            sys.stdout.write(
                f"FAIL uncontained: {fixture.name} "
                f"(exit_code={result.exit_code}, timed_out={result.timed_out}, "
                f"error={result.error})\n"
            )
            sys.stdout.flush()
            return 1
        contained_count += 1
        sys.stdout.write(f"PASS contained: {fixture.name}\n")
        sys.stdout.flush()

    sys.stdout.write(f"SANDBOX SELFTEST: {contained_count}/{len(HOSTILE_FIXTURES)} contained\n")
    sys.stdout.flush()

    verify_single_run_lock()
    sys.stdout.write("PASS single-run lock\n")
    sys.stdout.flush()

    verify_install_step_no_repo_code(runner)
    sys.stdout.write("PASS install step runs no repo code\n")
    sys.stdout.flush()

    leftovers = list_leftover_containers()
    if leftovers:
        cleanup_leftover_containers()
        sys.stdout.write(f"FAIL leftover sandbox containers found: {leftovers}\n")
        sys.stdout.flush()
        return 1

    return 0
