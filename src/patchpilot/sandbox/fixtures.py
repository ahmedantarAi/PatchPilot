"""Hostile-fixture containment suite and Phase 2 sandbox selftest."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import threading
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


class LocalTcpEchoServer:
    """Ephemeral host-side TCP listener for offline, mutation-safe network egress tests."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", 0))
        self.port = int(self.sock.getsockname()[1])
        self.sock.listen(5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
                try:
                    conn.sendall(b"HELLO_FROM_HOST\n")
                finally:
                    conn.close()
            except (TimeoutError, OSError):
                continue

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)


@dataclass(frozen=True)
class HostileFixture:
    name: str
    description: str
    files: dict[str, str]
    symlinks: dict[str, str] = field(default_factory=dict)
    patch_diff: str = ""
    timeout_s: float = 20.0
    check_contained: Callable[[SandboxResult], bool] = lambda r: r.exit_code != 0


NETWORK_EGRESS_TEMPLATE = """\
import socket


def test_network_egress_blocked() -> None:
    connected = False
    try:
        sock = socket.create_connection(("host.docker.internal", {port}), timeout=2.0)
        try:
            data = sock.recv(1024)
            if data:
                connected = True
        finally:
            sock.close()
    except Exception:
        connected = False

    # Attack succeeds only if outbound connection succeeded; must fail in sandbox.
    assert connected, "Outbound network connection to host was completely blocked"
"""

NETWORK_EGRESS_CODE = NETWORK_EGRESS_TEMPLATE.format(port=11434)

READ_HOST_ENV_CODE = """\
import os
from pathlib import Path


def test_read_host_files_and_env() -> None:
    leaked: list[str] = []

    # Check for root privilege
    if os.getuid() == 0 or os.geteuid() == 0:
        leaked.append(f"uid:{os.getuid()}")

    # Check for retained capabilities
    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for s_line in status_path.read_text(encoding="utf-8").splitlines():
                if s_line.startswith(("CapPrm:", "CapEff:", "CapBnd:")):
                    cap_val = s_line.split(":", 1)[1].strip()
                    if cap_val != "0000000000000000":
                        leaked.append(f"caps:{s_line}")
        except OSError:
            pass

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

    # Attack succeeds only if host secrets/files or root/caps were available; must fail in sandbox.
    assert leaked, (
        "No host secrets, /proc/1/environ secrets, host files, or root/caps were available"
    )
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


def check_memory_balloon_contained(r: SandboxResult) -> bool:
    """Contained only if exit code is 137/247 or MemoryError is observed (Fix 4)."""
    is_oom_exit = r.exit_code in (137, 247)
    has_mem_err = "MemoryError" in (r.stderr or "") or "MemoryError" in (r.stdout or "")
    if r.junit:
        for tc in r.junit.cases:
            if "MemoryError" in (tc.message or ""):
                has_mem_err = True
    return is_oom_exit or has_mem_err


HOSTILE_FIXTURES: tuple[HostileFixture, ...] = (
    HostileFixture(
        name="network_egress",
        description="Attempts outbound TCP connection to host under --network none",
        files={"test_network_egress.py": NETWORK_EGRESS_CODE},
        check_contained=lambda r: r.exit_code != 0 and not r.timed_out and r.junit.failures >= 1,
    ),
    HostileFixture(
        name="read_host_env",
        description="Attempts to read host files, /etc/shadow, secrets, and root/caps",
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
        description="Allocates > 768 MB physical RAM; must fail with 137/247 or MemoryError",
        files={"test_memory_balloon.py": MEMORY_BALLOON_CODE},
        check_contained=check_memory_balloon_contained,
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
    server: LocalTcpEchoServer | None = None
    files_to_run = dict(fixture.files)

    if fixture.name == "network_egress":
        server = LocalTcpEchoServer()
        files_to_run["test_network_egress.py"] = NETWORK_EGRESS_TEMPLATE.format(port=server.port)

    try:
        if fixture.symlinks:
            with tempfile.TemporaryDirectory() as tmp:
                repo_dir = Path(tmp)
                for rel_path, content in files_to_run.items():
                    dest = repo_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")

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
                files=files_to_run,
                patch_diff=fixture.patch_diff,
                timeout_s=fixture.timeout_s,
            )

        if is_container_present(result.container_name):
            raise AssertionError(f"Sandbox container {result.container_name} was not removed!")
        return result
    finally:
        if server is not None:
            server.close()


def verify_single_run_lock(lock_path: Path | None = None) -> None:
    """Verify both in-process async lock and cross-process OS file lock raise SandboxBusyError."""
    with tempfile.TemporaryDirectory() as tmp:
        effective_lock = lock_path or (Path(tmp) / "test-sandbox.lock")
        runner = SandboxRunner(lock_path=effective_lock, cleanup_on_start=False)

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


def build_test_wheels_and_sdists(dest_dir: Path) -> tuple[Path, Path]:
    """Build test wheel and malicious sdist in a temporary directory for install tests."""
    import base64
    import io
    import tarfile
    import zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    whl_path = dest_dir / "patchpilot_dummy_dep-0.1.0-py3-none-any.whl"
    mod = b'VALUE = "installed_from_wheel"\n'
    meta = b"Metadata-Version: 2.1\nName: patchpilot-dummy-dep\nVersion: 0.1.0\n"
    wheel = b"Wheel-Version: 1.0\nGenerator: patchpilot\nRoot-Is-Purelib: true\nTag: py3-none-any\n"

    def h(b: bytes) -> str:
        return (
            "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b"=").decode()
        )

    rec = (
        f"patchpilot_dummy_dep/__init__.py,{h(mod)},{len(mod)}\n"
        f"patchpilot_dummy_dep-0.1.0.dist-info/METADATA,{h(meta)},{len(meta)}\n"
        f"patchpilot_dummy_dep-0.1.0.dist-info/WHEEL,{h(wheel)},{len(wheel)}\n"
        f"patchpilot_dummy_dep-0.1.0.dist-info/RECORD,,\n"
    ).encode()

    with zipfile.ZipFile(whl_path, "w") as z:
        z.writestr("patchpilot_dummy_dep/__init__.py", mod)
        z.writestr("patchpilot_dummy_dep-0.1.0.dist-info/METADATA", meta)
        z.writestr("patchpilot_dummy_dep-0.1.0.dist-info/WHEEL", wheel)
        z.writestr("patchpilot_dummy_dep-0.1.0.dist-info/RECORD", rec)

    sdist_path = dest_dir / "malicious_sdist-0.1.0.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as t:
        setup_py = (
            b'import pathlib; pathlib.Path("/deps/pwned_by_sdist").write_text("pwned"); '
            b'raise SystemExit("setup.py ran")\n'
        )
        ti = tarfile.TarInfo("malicious_sdist-0.1.0/setup.py")
        ti.size = len(setup_py)
        t.addfile(ti, io.BytesIO(setup_py))

        pkg_info = b"Metadata-Version: 2.1\nName: malicious-sdist\nVersion: 0.1.0\n"
        ti2 = tarfile.TarInfo("malicious_sdist-0.1.0/PKG-INFO")
        ti2.size = len(pkg_info)
        t.addfile(ti2, io.BytesIO(pkg_info))

    return whl_path, sdist_path


def verify_install_step_no_repo_code(runner: SandboxRunner) -> None:
    """Verify D-24 two-step dependency install runs no repo code and mounts /deps read-only."""
    with tempfile.TemporaryDirectory() as wheels_tmp:
        wheels_dir = Path(wheels_tmp)
        build_test_wheels_and_sdists(wheels_dir)

        # Step 1: Verify sdist with setup.py is rejected by --only-binary :all:
        sdist_res = runner.install_dependencies_sync(
            ["malicious-sdist==0.1.0"],
            find_links_host_dir=wheels_dir,
            network="none",
        )
        try:
            if sdist_res.exit_code == 0:
                raise AssertionError("Expected sdist without binary wheel to be rejected")
            for required_flag in ("--only-binary", ":all:", "--no-build-isolation", "--no-input"):
                if required_flag not in sdist_res.cmd:
                    raise AssertionError(
                        f"Missing required flag {required_flag} in {sdist_res.cmd}"
                    )
        finally:
            remove_deps_volume(sdist_res.volume_name)

        # Step 2: Install binary wheel into /deps volume with hostile setup.py in the repo on host
        wheel_res = runner.install_dependencies_sync(
            ["patchpilot-dummy-dep==0.1.0"],
            find_links_host_dir=wheels_dir,
            network="none",
        )
        try:
            if wheel_res.exit_code != 0:
                raise AssertionError(
                    f"Wheel install failed ({wheel_res.exit_code}): {wheel_res.stderr}"
                )

            with tempfile.TemporaryDirectory() as tmp:
                repo_dir = Path(tmp)
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
