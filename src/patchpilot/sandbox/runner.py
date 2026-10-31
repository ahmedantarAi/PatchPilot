"""Host-side sandbox orchestrator launching hardened single-use Docker containers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from pydantic import BaseModel, Field

from patchpilot.sandbox.tar_and_diff import (
    MAX_OUTPUT_BYTES,
    JUnitReport,
    build_repo_tar,
    parse_junit_xml,
    read_stream_capped,
)

logger = logging.getLogger("patchpilot.sandbox")

SANDBOX_IMAGE = "patchpilot-sandbox:0.0.0"
SANDBOX_LABEL = "patchpilot.sandbox=true"
SANDBOX_CONTAINER_PREFIX = "patchpilot-sbx-"
SANDBOX_DEPS_VOLUME_PREFIX = "patchpilot-deps-"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_LOCK_PATH = Path(tempfile.gettempdir()) / "patchpilot-sandbox.lock"

DEPS_VOLUME_RE = re.compile(r"^patchpilot-deps-[a-f0-9]{12,64}$")
PINNED_REQ_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==[A-Za-z0-9.*+!-]+$"
)

SANDBOX_LIMITS: dict[str, object] = {
    "network": "none",
    "user": "10001:10001",
    "cap_drop": "ALL",
    "security_opt": "no-new-privileges",
    "read_only": True,
    "tmpfs_work": "/work:rw,nosuid,size=256m,uid=10001,gid=10001",
    "tmpfs_tmp": "/tmp:rw,noexec,nosuid,size=128m",
    "memory": "768m",
    "memory_swap": "768m",
    "cpus": "1",
    "pids_limit": 256,
}

_IN_PROCESS_LOCK = threading.Lock()


class SandboxError(RuntimeError):
    """Base error for sandbox orchestration failures."""


class SandboxBusyError(SandboxError):
    """Raised when a second sandbox run is attempted while one is already active."""


class SandboxRunLog(BaseModel):
    container_name: str
    image: str
    image_digest: str
    limits: dict[str, object]
    flags: list[str]
    duration_s: float
    exit_status: int
    timed_out: bool = False


class SandboxResult(BaseModel):
    exit_code: int
    patch_applied: bool
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    junit_xml: str = ""
    junit: JUnitReport = Field(default_factory=JUnitReport)
    error: str | None = None
    container_name: str = ""
    image_digest: str = ""
    duration_s: float = 0.0
    limits: dict[str, object] = Field(default_factory=lambda: dict(SANDBOX_LIMITS))


class InstallStepResult(BaseModel):
    volume_name: str
    exit_code: int
    cmd: list[str]
    stdout: str
    stderr: str
    duration_s: float


def _lock_file_nonblocking(fh: BinaryIO) -> None:
    fh.seek(0, os.SEEK_END)
    if fh.tell() == 0:
        fh.write(b"\x00")
        fh.flush()
    fh.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(fh: BinaryIO) -> None:
    fh.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class SandboxLock:
    """Fail-fast in-process + cross-process OS file lock raising `SandboxBusyError`."""

    def __init__(self, lock_path: Path = DEFAULT_LOCK_PATH) -> None:
        self.lock_path = lock_path
        self._fh: BinaryIO | None = None
        self._acquired_in_proc = False

    def acquire(self) -> None:
        if not _IN_PROCESS_LOCK.acquire(blocking=False):
            raise SandboxBusyError("Sandbox is busy: in-process lock is already held")
        self._acquired_in_proc = True
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(self.lock_path, "a+b")  # noqa: SIM115
            try:
                _lock_file_nonblocking(fh)
            except OSError as exc:
                fh.close()
                raise SandboxBusyError(
                    f"Sandbox is busy: cross-process file lock {self.lock_path} is held"
                ) from exc
            self._fh = fh
        except BaseException:
            self._acquired_in_proc = False
            _IN_PROCESS_LOCK.release()
            raise

    def release(self) -> None:
        if self._fh is not None:
            try:
                _unlock_file(self._fh)
            except OSError:
                pass
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        if self._acquired_in_proc:
            self._acquired_in_proc = False
            _IN_PROCESS_LOCK.release()

    def __enter__(self) -> SandboxLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    async def __aenter__(self) -> SandboxLock:
        self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def validate_requirements(requirements: Sequence[str]) -> list[str]:
    """Validate that requirements are strictly pinned specifiers (no paths, URLs, or flags)."""
    cleaned: list[str] = []
    for raw in requirements:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            raise ValueError(f"Pip option flags are forbidden in requirements: {line!r}")
        if "/" in line or "\\" in line or ".." in line or ":" in line or "@" in line:
            raise ValueError(f"Paths and URLs are forbidden in requirements: {line!r}")
        if not PINNED_REQ_RE.match(line):
            raise ValueError(f"Requirement must be an exact 'pkg==version' pin: {line!r}")
        cleaned.append(line)
    return cleaned


def compute_deps_volume_name(requirements: Sequence[str]) -> str:
    """Return a deterministic named volume `patchpilot-deps-<sha256>` for a requirements list."""
    validated = sorted(set(validate_requirements(requirements)))
    canonical = "\n".join(validated).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:16]
    return f"{SANDBOX_DEPS_VOLUME_PREFIX}{digest}"


def validate_deps_volume_name(volume_name: str) -> str:
    if not DEPS_VOLUME_RE.match(volume_name):
        raise ValueError(
            f"Invalid deps volume name {volume_name!r}; must match {DEPS_VOLUME_RE.pattern}"
        )
    return volume_name


def build_docker_run_cmd(
    container_name: str,
    *,
    image: str = SANDBOX_IMAGE,
    deps_volume: str | None = None,
) -> list[str]:
    """Build the hardened `docker run` command with all mandatory SPEC §6 controls."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        container_name,
        "--label",
        SANDBOX_LABEL,
        "--network",
        "none",
        "--user",
        "10001:10001",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/work:rw,nosuid,size=256m,uid=10001,gid=10001",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=128m",
        "--memory",
        "768m",
        "--memory-swap",
        "768m",
        "--cpus",
        "1",
        "--pids-limit",
        "256",
    ]
    if deps_volume is not None:
        validate_deps_volume_name(deps_volume)
        cmd.extend(["-v", f"{deps_volume}:/deps:ro", "-e", "PYTHONPATH=/deps"])
    cmd.append(image)
    return cmd


def build_docker_install_cmd(
    container_name: str,
    volume_name: str,
    *,
    image: str = SANDBOX_IMAGE,
    network: str = "bridge",
) -> list[str]:
    """Build the `docker run` command for the isolated two-step dependency installer."""
    validate_deps_volume_name(volume_name)
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        container_name,
        "--label",
        SANDBOX_LABEL,
        "--network",
        network,
        "--user",
        "10001:10001",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/work:rw,nosuid,size=256m,uid=10001,gid=10001",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=128m",
        "--memory",
        "768m",
        "--memory-swap",
        "768m",
        "--cpus",
        "1",
        "--pids-limit",
        "256",
        "-v",
        f"{volume_name}:/deps:rw",
        image,
        "--install-deps",
    ]


def _minimal_docker_client_env() -> dict[str, str]:
    """Return a minimal environment for the host `docker` CLI without any project secrets."""
    allowed_keys = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "DOCKER_HOST",
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
    )
    return {k: os.environ[k] for k in allowed_keys if k in os.environ}


def get_image_digest(image: str = SANDBOX_IMAGE) -> str:
    """Return the image sha256 ID/digest for logging."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            timeout=10.0,
            env=_minimal_docker_client_env(),
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def force_kill_and_remove_container(container_name: str) -> None:
    """Kill and remove a sandbox container by name, ignoring errors if already gone."""
    env = _minimal_docker_client_env()
    for subcmd in (["docker", "kill", container_name], ["docker", "rm", "-f", container_name]):
        try:
            subprocess.run(
                subcmd,
                capture_output=True,
                text=True,
                timeout=15.0,
                env=env,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass


def is_container_present(container_name: str) -> bool:
    """Check whether a container with `container_name` exists in `docker ps -a`."""
    proc = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name=^/{container_name}$",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        timeout=10.0,
        env=_minimal_docker_client_env(),
        check=False,
    )
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return container_name in names


def list_leftover_containers() -> list[str]:
    """List any remaining sandbox containers matching the sandbox label or name prefix."""
    env = _minimal_docker_client_env()
    found: set[str] = set()
    for flt in (f"label={SANDBOX_LABEL}", f"name={SANDBOX_CONTAINER_PREFIX}"):
        proc = subprocess.run(
            ["docker", "ps", "-a", "--filter", flt, "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10.0,
            env=env,
            check=False,
        )
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            if stripped:
                found.add(stripped)
    return sorted(found)


def cleanup_leftover_containers() -> list[str]:
    """Force-remove any leftover sandbox containers and return the names removed."""
    leftovers = list_leftover_containers()
    for name in leftovers:
        force_kill_and_remove_container(name)
    return leftovers


def remove_deps_volume(volume_name: str) -> None:
    """Remove a `patchpilot-deps-<sha256>` named volume."""
    validate_deps_volume_name(volume_name)
    subprocess.run(
        ["docker", "volume", "rm", "-f", volume_name],
        capture_output=True,
        text=True,
        timeout=15.0,
        env=_minimal_docker_client_env(),
        check=False,
    )


class SandboxRunner:
    """Host orchestrator for running untrusted patches and tests inside a hardened container."""

    def __init__(
        self,
        *,
        image: str = SANDBOX_IMAGE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        lock_path: Path = DEFAULT_LOCK_PATH,
    ) -> None:
        self.image = image
        self.timeout_s = timeout_s
        self.lock_path = lock_path
        self.run_logs: list[SandboxRunLog] = []

    def install_dependencies_sync(
        self,
        requirements: Sequence[str],
        *,
        find_links: str | None = None,
        network: str = "bridge",
        timeout_s: float | None = None,
    ) -> InstallStepResult:
        """Populate a `patchpilot-deps-<sha256>` volume with binary wheels and no repo files."""
        validated = validate_requirements(requirements)
        volume_name = compute_deps_volume_name(validated)
        effective_timeout = timeout_s if timeout_s is not None else self.timeout_s
        container_name = f"{SANDBOX_CONTAINER_PREFIX}deps-{uuid.uuid4().hex[:12]}"
        cmd = build_docker_install_cmd(
            container_name,
            volume_name,
            image=self.image,
            network=network,
        )
        stdin_payload = json.dumps(
            {
                "requirements": validated,
                "find_links": find_links,
            }
        ).encode("utf-8")

        with SandboxLock(self.lock_path):
            t0 = time.monotonic()
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_minimal_docker_client_env(),
            )
            try:
                assert proc.stdin is not None
                proc.stdin.write(stdin_payload)
                proc.stdin.close()

                stdout_box: list[tuple[str, bool]] = [("", False)]
                stderr_box: list[tuple[str, bool]] = [("", False)]

                def _read_out() -> None:
                    assert proc.stdout is not None
                    stdout_box[0] = read_stream_capped(proc.stdout, MAX_OUTPUT_BYTES)

                def _read_err() -> None:
                    assert proc.stderr is not None
                    stderr_box[0] = read_stream_capped(proc.stderr, MAX_OUTPUT_BYTES)

                t_out = threading.Thread(target=_read_out, daemon=True)
                t_err = threading.Thread(target=_read_err, daemon=True)
                t_out.start()
                t_err.start()

                try:
                    exit_code = proc.wait(timeout=effective_timeout)
                except subprocess.TimeoutExpired as exc:
                    force_kill_and_remove_container(container_name)
                    proc.kill()
                    raise SandboxError(
                        f"Dependency install timed out after {effective_timeout}s"
                    ) from exc

                t_out.join(timeout=5.0)
                t_err.join(timeout=5.0)
                duration_s = round(time.monotonic() - t0, 4)

                stdout_raw, _ = stdout_box[0]
                stderr_raw, _ = stderr_box[0]

                pip_cmd: list[str] = []
                pip_stdout = stdout_raw
                pip_stderr = stderr_raw
                for line in reversed(stdout_raw.splitlines()):
                    if line.strip().startswith("{"):
                        try:
                            parsed = json.loads(line)
                            if isinstance(parsed, dict):
                                pip_cmd = list(parsed.get("cmd", []))
                                pip_stdout = str(parsed.get("stdout", ""))
                                pip_stderr = str(parsed.get("stderr", ""))
                                exit_code = int(parsed.get("exit_code", exit_code))
                                break
                        except ValueError:
                            pass

                return InstallStepResult(
                    volume_name=volume_name,
                    exit_code=exit_code,
                    cmd=pip_cmd,
                    stdout=pip_stdout,
                    stderr=pip_stderr,
                    duration_s=duration_s,
                )
            finally:
                if is_container_present(container_name):
                    force_kill_and_remove_container(container_name)

    def run_sync(
        self,
        repo_dir: Path | None = None,
        *,
        files: Mapping[str, str | bytes] | None = None,
        symlinks: Mapping[str, str] | None = None,
        patch_diff: str = "",
        pytest_args: list[str] | None = None,
        deps_volume: str | None = None,
        timeout_s: float | None = None,
        tar_bytes: bytes | None = None,
    ) -> SandboxResult:
        """Run a single verification container synchronously under the single-run lock."""
        effective_timeout = timeout_s if timeout_s is not None else self.timeout_s
        payload_bytes = (
            tar_bytes
            if tar_bytes is not None
            else build_repo_tar(
                repo_dir,
                files=files,
                symlinks=symlinks,
                patch_diff=patch_diff,
                pytest_args=pytest_args,
            )
        )
        container_name = f"{SANDBOX_CONTAINER_PREFIX}{uuid.uuid4().hex[:12]}"
        cmd = build_docker_run_cmd(
            container_name,
            image=self.image,
            deps_volume=deps_volume,
        )
        flags = cmd[2:-1]

        with SandboxLock(self.lock_path):
            image_digest = get_image_digest(self.image)
            limits = {**SANDBOX_LIMITS, "timeout_s": effective_timeout}
            t0 = time.monotonic()
            timed_out = False
            exit_code = -1
            stdout_text = ""
            stderr_text = ""
            stdout_truncated = False
            stderr_truncated = False

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_minimal_docker_client_env(),
            )
            try:
                stdout_box: list[tuple[str, bool]] = [("", False)]
                stderr_box: list[tuple[str, bool]] = [("", False)]

                def _read_out() -> None:
                    assert proc.stdout is not None
                    stdout_box[0] = read_stream_capped(proc.stdout, MAX_OUTPUT_BYTES)

                def _read_err() -> None:
                    assert proc.stderr is not None
                    stderr_box[0] = read_stream_capped(proc.stderr, MAX_OUTPUT_BYTES)

                t_out = threading.Thread(target=_read_out, daemon=True)
                t_err = threading.Thread(target=_read_err, daemon=True)
                t_out.start()
                t_err.start()

                assert proc.stdin is not None
                try:
                    proc.stdin.write(payload_bytes)
                except BrokenPipeError:
                    pass
                finally:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass

                try:
                    exit_code = proc.wait(timeout=effective_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    force_kill_and_remove_container(container_name)
                    proc.kill()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
                    exit_code = 124

                t_out.join(timeout=5.0)
                t_err.join(timeout=5.0)
                stdout_text, stdout_truncated = stdout_box[0]
                stderr_text, stderr_truncated = stderr_box[0]
            finally:
                if is_container_present(container_name):
                    force_kill_and_remove_container(container_name)
                duration_s = round(time.monotonic() - t0, 4)
                parsed_result = _parse_container_output(
                    stdout_text=stdout_text,
                    stderr_text=stderr_text,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                    container_exit_code=exit_code,
                    timed_out=timed_out,
                    container_name=container_name,
                    image_digest=image_digest,
                    duration_s=duration_s,
                    limits=limits,
                )
                run_log = SandboxRunLog(
                    container_name=container_name,
                    image=self.image,
                    image_digest=image_digest,
                    limits=limits,
                    flags=flags,
                    duration_s=duration_s,
                    exit_status=parsed_result.exit_code,
                    timed_out=timed_out,
                )
                self.run_logs.append(run_log)
                logger.info(
                    "sandbox_run image_digest=%s duration_s=%.4f exit_status=%d "
                    "timed_out=%s limits=%s",
                    image_digest,
                    duration_s,
                    parsed_result.exit_code,
                    timed_out,
                    json.dumps(limits, sort_keys=True),
                )

            return parsed_result

    async def run(
        self,
        repo_dir: Path | None = None,
        *,
        files: Mapping[str, str | bytes] | None = None,
        symlinks: Mapping[str, str] | None = None,
        patch_diff: str = "",
        pytest_args: list[str] | None = None,
        deps_volume: str | None = None,
        timeout_s: float | None = None,
        tar_bytes: bytes | None = None,
    ) -> SandboxResult:
        """Async wrapper around `run_sync` that fails fast with `SandboxBusyError` if locked."""
        if _IN_PROCESS_LOCK.locked():
            raise SandboxBusyError("Sandbox is busy: in-process lock is already held")
        return await asyncio.to_thread(
            self.run_sync,
            repo_dir,
            files=files,
            symlinks=symlinks,
            patch_diff=patch_diff,
            pytest_args=pytest_args,
            deps_volume=deps_volume,
            timeout_s=timeout_s,
            tar_bytes=tar_bytes,
        )


def _parse_container_output(
    *,
    stdout_text: str,
    stderr_text: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
    container_exit_code: int,
    timed_out: bool,
    container_name: str,
    image_digest: str,
    duration_s: float,
    limits: dict[str, object],
) -> SandboxResult:
    if timed_out:
        return SandboxResult(
            exit_code=124,
            patch_applied=False,
            timed_out=True,
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            junit_xml="",
            junit=JUnitReport(),
            error="Sandbox wall-clock timeout exceeded; container killed and removed",
            container_name=container_name,
            image_digest=image_digest,
            duration_s=duration_s,
            limits=limits,
        )

    for line in reversed(stdout_text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(payload, dict) or "exit_code" not in payload:
            continue

        raw_junit = payload.get("junit")
        junit_xml = str(payload.get("junit_xml", ""))[:MAX_OUTPUT_BYTES]
        if isinstance(raw_junit, dict):
            junit_report = JUnitReport.model_validate(raw_junit)
        else:
            junit_report = parse_junit_xml(junit_xml)

        inner_stdout = str(payload.get("stdout", ""))[:MAX_OUTPUT_BYTES]
        inner_stderr = str(payload.get("stderr", ""))[:MAX_OUTPUT_BYTES]
        if stderr_text and stderr_text not in inner_stderr:
            combined_stderr = (inner_stderr + ("\n" if inner_stderr else "") + stderr_text)[
                :MAX_OUTPUT_BYTES
            ]
        else:
            combined_stderr = inner_stderr

        raw_err = payload.get("error")
        return SandboxResult(
            exit_code=int(payload.get("exit_code", container_exit_code)),
            patch_applied=bool(payload.get("patch_applied", False)),
            timed_out=bool(payload.get("timed_out", False)),
            stdout=inner_stdout,
            stderr=combined_stderr,
            stdout_truncated=bool(payload.get("stdout_truncated", False)) or stdout_truncated,
            stderr_truncated=bool(payload.get("stderr_truncated", False)) or stderr_truncated,
            junit_xml=junit_xml,
            junit=junit_report,
            error=str(raw_err) if raw_err is not None else None,
            container_name=container_name,
            image_digest=image_digest,
            duration_s=duration_s,
            limits=limits,
        )

    effective_exit = container_exit_code if container_exit_code != 0 else 1
    return SandboxResult(
        exit_code=effective_exit,
        patch_applied=False,
        timed_out=False,
        stdout=stdout_text,
        stderr=stderr_text,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        junit_xml="",
        junit=JUnitReport(),
        error=f"Container exited ({container_exit_code}) without structured JSON output",
        container_name=container_name,
        image_digest=image_digest,
        duration_s=duration_s,
        limits=limits,
    )
