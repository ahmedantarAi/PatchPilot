"""Pure-Python entrypoint running inside the hardened sandbox container.

Unpacks a tar stream from stdin into /work (rejecting path traversal, symlinks,
hardlinks, and special files), applies a unified diff in pure Python, executes
pytest with capped output, and prints a single JSON result object to stdout.

Also supports `--install-deps` mode for the two-step dependency installer, which
runs `pip install --only-binary :all: --no-build-isolation --no-input --target /deps`
with no repository files present.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

MAX_OUTPUT_BYTES = 1_048_576  # 1 MB cap per SPEC §6
MAX_PATCH_BYTES = 10 * 1024 * 1024  # 10 MB max patch size
MAX_TAR_FILE_BYTES = 32 * 1024 * 1024  # 32 MB per single file
MAX_TAR_TOTAL_BYTES = 200 * 1024 * 1024  # 200 MB total unpacked
MAX_TAR_MEMBERS = 10_000

MAX_JUNIT_CASES = 500
MAX_JUNIT_STR_LEN = 1024
MAX_JUNIT_MSG_LEN = 4096

META_PREFIX = "__patchpilot__/"
PATCH_MEMBER_NAME = "__patchpilot__/patch.diff"
META_MEMBER_NAME = "__patchpilot__/meta.json"
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class TarSafetyError(ValueError):
    """Raised when a tar archive contains unsafe paths, links, or special entries."""


class PatchApplyError(ValueError):
    """Raised when a unified diff is malformed, unsafe, or fails to apply cleanly."""


@dataclass
class ExtractedPayload:
    patch_diff: str = ""
    pytest_args: list[str] = field(default_factory=list)
    files_extracted: int = 0
    verify_mode: bool = False


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class FilePatch:
    old_path: str
    new_path: str
    hunks: list[Hunk]


def validate_safe_relative_path(raw_path: str, work_dir: Path) -> Path:
    """Validate that `raw_path` is a safe relative path strictly inside `work_dir`."""
    if not raw_path or "\x00" in raw_path:
        raise TarSafetyError(f"Invalid empty or null-byte path: {raw_path!r}")

    normalized_slashes = raw_path.replace("\\", "/")
    if normalized_slashes.startswith("/"):
        raise TarSafetyError(f"Absolute path rejected: {raw_path!r}")

    first_segment = normalized_slashes.split("/", 1)[0]
    if ":" in first_segment:
        raise TarSafetyError(f"Drive-qualified path rejected: {raw_path!r}")

    parts = [p for p in normalized_slashes.split("/") if p not in ("", ".")]
    if not parts:
        raise TarSafetyError(f"Path resolves to root directory: {raw_path!r}")
    if any(part == ".." for part in parts):
        raise TarSafetyError(f"Path traversal ('..') rejected: {raw_path!r}")

    norm_rel = posixpath.normpath("/".join(parts))
    if norm_rel.startswith("..") or norm_rel.startswith("/"):
        raise TarSafetyError(f"Normalized path escapes work_dir: {raw_path!r}")

    work_resolved = work_dir.resolve()
    candidate = work_dir.joinpath(*parts)

    current = work_dir
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise TarSafetyError(f"Path traverses symlink at {current}: {raw_path!r}")

    resolved = candidate.resolve()
    if resolved != work_resolved and not resolved.is_relative_to(work_resolved):
        raise TarSafetyError(f"Resolved path escapes work_dir: {raw_path!r}")

    return candidate


def safe_extract_tar(stream: BinaryIO, work_dir: Path) -> ExtractedPayload:
    """Safely unpack a tar stream into `work_dir`, rejecting links and traversal."""
    payload = ExtractedPayload()
    total_bytes = 0
    member_count = 0

    try:
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            for member in tar:
                member_count += 1
                if member_count > MAX_TAR_MEMBERS:
                    raise TarSafetyError(f"Tar archive exceeds {MAX_TAR_MEMBERS} entries")

                name = member.name
                if name in (".", "./"):
                    continue

                if member.issym() or member.islnk():
                    raise TarSafetyError(
                        f"Symlink/hardlink rejected in tar: {name!r} -> {member.linkname!r}"
                    )

                if not (member.isreg() or member.isdir()):
                    raise TarSafetyError(f"Unsupported tar entry type {member.type!r} for {name!r}")

                if member.size < 0 or member.size > MAX_TAR_FILE_BYTES:
                    raise TarSafetyError(
                        f"Tar entry {name!r} size {member.size} exceeds per-file limit"
                    )

                total_bytes += member.size
                if total_bytes > MAX_TAR_TOTAL_BYTES:
                    raise TarSafetyError(
                        f"Tar archive exceeds total unpacked limit ({MAX_TAR_TOTAL_BYTES} bytes)"
                    )

                normalized_name = name.replace("\\", "/").lstrip("./")
                if normalized_name == PATCH_MEMBER_NAME:
                    if not member.isreg():
                        raise TarSafetyError("Patch metadata entry must be a regular file")
                    if member.size > MAX_PATCH_BYTES:
                        raise TarSafetyError(
                            f"Patch size {member.size} exceeds {MAX_PATCH_BYTES} limit"
                        )
                    fobj = tar.extractfile(member)
                    if fobj is not None:
                        data = fobj.read(MAX_PATCH_BYTES + 1)
                        if len(data) > MAX_PATCH_BYTES:
                            raise TarSafetyError("Patch exceeded maximum size limit")
                        payload.patch_diff = data.decode("utf-8")
                    continue

                if normalized_name == META_MEMBER_NAME:
                    if not member.isreg():
                        raise TarSafetyError("Meta entry must be a regular file")
                    fobj = tar.extractfile(member)
                    if fobj is not None:
                        meta_raw = json.loads(fobj.read(MAX_TAR_FILE_BYTES).decode("utf-8"))
                        if isinstance(meta_raw, dict):
                            args = meta_raw.get("pytest_args", [])
                            if isinstance(args, list) and all(isinstance(a, str) for a in args):
                                payload.pytest_args = list(args)
                            if bool(meta_raw.get("verify", False)):
                                payload.verify_mode = True
                    continue

                if normalized_name.startswith(META_PREFIX):
                    continue

                dest = validate_safe_relative_path(name, work_dir)
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    dest.chmod(0o755)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        raise TarSafetyError(f"Could not read regular file entry {name!r}")
                    data = fobj.read(MAX_TAR_FILE_BYTES + 1)
                    if len(data) > MAX_TAR_FILE_BYTES:
                        raise TarSafetyError(f"File {name!r} exceeded size limit while reading")
                    dest.write_bytes(data)
                    dest.chmod(0o644)
                    payload.files_extracted += 1
    except tarfile.TarError as exc:
        raise TarSafetyError(f"Invalid or corrupt tar stream: {exc}") from exc

    return payload


def _clean_diff_path(header_path: str, work_dir: Path) -> str:
    """Strip tab timestamps and leading a/ or b/ prefixes from a unified diff path."""
    path_part = header_path.split("\t", 1)[0].strip()
    if path_part == "/dev/null":
        return "/dev/null"
    if (path_part.startswith("a/") or path_part.startswith("b/")) and len(path_part) > 2:
        stripped = path_part[2:]
        try:
            literal_candidate = validate_safe_relative_path(path_part, work_dir)
            if literal_candidate.exists():
                return path_part
        except TarSafetyError:
            pass
        return stripped
    return path_part


def parse_unified_diff(diff_text: str, work_dir: Path) -> list[FilePatch]:
    """Parse a unified diff string into structured `FilePatch` objects."""
    if not diff_text.strip():
        return []
    if len(diff_text.encode("utf-8")) > MAX_PATCH_BYTES:
        raise PatchApplyError(f"Patch exceeds maximum size of {MAX_PATCH_BYTES} bytes")

    raw_lines = diff_text.splitlines()
    idx = 0
    n = len(raw_lines)
    patches: list[FilePatch] = []

    while idx < n:
        line = raw_lines[idx]
        if not line.startswith("--- "):
            idx += 1
            continue

        if idx + 1 >= n or not raw_lines[idx + 1].startswith("+++ "):
            raise PatchApplyError(f"Missing '+++' header after line {idx + 1}: {line!r}")

        old_path = _clean_diff_path(line[4:], work_dir)
        new_path = _clean_diff_path(raw_lines[idx + 1][4:], work_dir)
        idx += 2

        hunks: list[Hunk] = []
        while idx < n and raw_lines[idx].startswith("@@ "):
            header_line = raw_lines[idx]
            match = HUNK_HEADER_RE.match(header_line)
            if match is None:
                raise PatchApplyError(f"Malformed hunk header at line {idx + 1}: {header_line!r}")

            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) is not None else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) is not None else 1
            idx += 1

            hunk_lines: list[str] = []
            seen_old = 0
            seen_new = 0
            while idx < n:
                hline = raw_lines[idx]
                if hline.startswith("--- ") or hline.startswith("@@ "):
                    if seen_old == old_count and seen_new == new_count:
                        break
                if hline.startswith("diff --git "):
                    break
                if hline == r"\ No newline at end of file":
                    hunk_lines.append(hline)
                    idx += 1
                    continue
                if hline == "" and seen_old < old_count and seen_new < new_count:
                    hunk_lines.append(" ")
                    seen_old += 1
                    seen_new += 1
                    idx += 1
                    continue
                if not hline:
                    if seen_old == old_count and seen_new == new_count:
                        idx += 1
                        break
                    raise PatchApplyError(f"Unexpected empty line inside hunk at line {idx + 1}")

                prefix = hline[0]
                if prefix == " ":
                    seen_old += 1
                    seen_new += 1
                    hunk_lines.append(hline)
                elif prefix == "-":
                    seen_old += 1
                    hunk_lines.append(hline)
                elif prefix == "+":
                    seen_new += 1
                    hunk_lines.append(hline)
                else:
                    if seen_old == old_count and seen_new == new_count:
                        break
                    raise PatchApplyError(
                        f"Invalid hunk line prefix {prefix!r} at line {idx + 1}: {hline!r}"
                    )
                idx += 1

            if seen_old != old_count or seen_new != new_count:
                raise PatchApplyError(
                    f"Hunk line count mismatch in {new_path!r}: "
                    f"expected (-{old_count}, +{new_count}), got (-{seen_old}, +{seen_new})"
                )

            hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=hunk_lines,
                )
            )

        if not hunks:
            raise PatchApplyError(f"File patch for {new_path!r} contains no hunks")

        patches.append(FilePatch(old_path=old_path, new_path=new_path, hunks=hunks))

    if not patches:
        raise PatchApplyError("Unified diff contains no valid file headers or hunks")

    return patches


def apply_unified_diff(diff_text: str, work_dir: Path) -> int:
    """Apply a unified diff inside `work_dir` atomically. Returns number of patched files."""
    if not diff_text.strip():
        return 0
    if len(diff_text.encode("utf-8")) > MAX_PATCH_BYTES:
        raise PatchApplyError(f"Patch exceeds maximum size of {MAX_PATCH_BYTES} bytes")

    try:
        patches = parse_unified_diff(diff_text, work_dir)
    except TarSafetyError as exc:
        raise PatchApplyError(str(exc)) from exc

    planned_writes: dict[Path, str] = {}
    planned_deletes: set[Path] = set()

    for fpatch in patches:
        is_new_file = fpatch.old_path == "/dev/null"
        is_delete_file = fpatch.new_path == "/dev/null"

        if is_new_file and is_delete_file:
            raise PatchApplyError("Patch cannot have both old and new path as /dev/null")

        try:
            old_target = (
                None if is_new_file else validate_safe_relative_path(fpatch.old_path, work_dir)
            )
            new_target = (
                None if is_delete_file else validate_safe_relative_path(fpatch.new_path, work_dir)
            )
        except TarSafetyError as exc:
            raise PatchApplyError(f"Unsafe path in diff header: {exc}") from exc

        if is_new_file:
            orig_lines: list[str] = []
            had_trailing_newline = True
        else:
            assert old_target is not None
            if old_target in planned_writes:
                raw_content = planned_writes[old_target]
            elif old_target.is_file() and not old_target.is_symlink():
                raw_content = old_target.read_text(encoding="utf-8")
            else:
                raise PatchApplyError(f"Target file to patch does not exist: {fpatch.old_path!r}")
            had_trailing_newline = raw_content.endswith("\n") or raw_content == ""
            orig_lines = raw_content.splitlines()

        result_lines: list[str] = []
        src_idx = 0
        no_newline_at_eof = not had_trailing_newline

        for hunk in fpatch.hunks:
            hunk_start_idx = 0 if hunk.old_start == 0 else hunk.old_start - 1
            if hunk_start_idx < src_idx or hunk_start_idx > len(orig_lines):
                raise PatchApplyError(
                    f"Hunk start {hunk.old_start} out of range for {fpatch.old_path!r} "
                    f"(current line {src_idx + 1}, total lines {len(orig_lines)})"
                )

            result_lines.extend(orig_lines[src_idx:hunk_start_idx])
            src_idx = hunk_start_idx
            last_op = ""

            for hline in hunk.lines:
                if hline == r"\ No newline at end of file":
                    if last_op in ("+", " "):
                        no_newline_at_eof = True
                    continue

                prefix = hline[0]
                content = hline[1:]
                last_op = prefix

                if prefix in (" ", "-"):
                    if src_idx >= len(orig_lines):
                        raise PatchApplyError(
                            f"Hunk exceeds end of file {fpatch.old_path!r} at line {src_idx + 1}"
                        )
                    actual = orig_lines[src_idx]
                    if actual != content:
                        raise PatchApplyError(
                            f"Context mismatch in {fpatch.old_path!r} at line {src_idx + 1}: "
                            f"expected {content!r}, found {actual!r}"
                        )
                    src_idx += 1
                    if prefix == " ":
                        result_lines.append(content)
                        no_newline_at_eof = False
                elif prefix == "+":
                    result_lines.append(content)
                    no_newline_at_eof = False

        result_lines.extend(orig_lines[src_idx:])

        if is_delete_file:
            if result_lines:
                raise PatchApplyError(
                    f"File deletion patch for {fpatch.old_path!r} left {len(result_lines)} lines"
                )
            assert old_target is not None
            planned_deletes.add(old_target)
            planned_writes.pop(old_target, None)
        else:
            assert new_target is not None
            if not result_lines:
                new_text = ""
            else:
                new_text = "\n".join(result_lines)
                if not no_newline_at_eof:
                    new_text += "\n"
            planned_writes[new_target] = new_text
            planned_deletes.discard(new_target)
            if old_target is not None and old_target != new_target:
                planned_deletes.add(old_target)

    for del_path in planned_deletes:
        if del_path.exists():
            del_path.unlink()

    for write_path, content in planned_writes.items():
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(content, encoding="utf-8")
        write_path.chmod(0o644)

    return len(patches)


def read_stream_capped(stream: BinaryIO, cap_bytes: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    """Read from `stream` until EOF, keeping at most `cap_bytes` and draining the rest."""
    chunks: list[bytes] = []
    kept = 0
    truncated = False
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        if kept < cap_bytes:
            remaining = cap_bytes - kept
            if len(chunk) <= remaining:
                chunks.append(chunk)
                kept += len(chunk)
            else:
                chunks.append(chunk[:remaining])
                kept = cap_bytes
                truncated = True
        else:
            truncated = True
    return b"".join(chunks).decode("utf-8", errors="replace"), truncated


def parse_junit_xml(xml_text: str) -> dict[str, object]:
    """Parse pytest JUnit XML into a small structured summary dictionary with caps.

    Safely handles or rejects hostile XML (e.g. DOCTYPE/entity expansion declarations).
    """
    summary: dict[str, object] = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "passed": 0,
        "cases": [],
    }
    if not xml_text.strip():
        return summary

    # Reject DOCTYPE / ENTITY declarations which are never legitimate in pytest JUnit XML
    if "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:
        summary["error"] = "Hostile XML: DOCTYPE/ENTITY declarations rejected"
        return summary

    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:
        summary["error"] = f"XMLParseError: {exc}"
        return summary

    cases_list: list[dict[str, object]] = []
    tests_total = 0
    failures_total = 0
    errors_total = 0
    skipped_total = 0

    for tc in root.findall(".//testcase"):
        tests_total += 1
        failure_el = tc.find("failure")
        error_el = tc.find("error")
        skipped_el = tc.find("skipped")

        status = "passed"
        message: str | None = None
        if failure_el is not None:
            status = "failed"
            failures_total += 1
            msg_raw = failure_el.attrib.get("message") or failure_el.text or ""
            message = msg_raw[:MAX_JUNIT_MSG_LEN]
        elif error_el is not None:
            status = "error"
            errors_total += 1
            msg_raw = error_el.attrib.get("message") or error_el.text or ""
            message = msg_raw[:MAX_JUNIT_MSG_LEN]
        elif skipped_el is not None:
            status = "skipped"
            skipped_total += 1
            msg_raw = skipped_el.attrib.get("message") or skipped_el.text or ""
            message = msg_raw[:MAX_JUNIT_MSG_LEN]

        if len(cases_list) < MAX_JUNIT_CASES:
            classname = (tc.attrib.get("classname", "") or "")[:MAX_JUNIT_STR_LEN]
            name = (tc.attrib.get("name", "") or "")[:MAX_JUNIT_STR_LEN]
            try:
                time_s = float(tc.attrib.get("time", "0") or "0")
            except ValueError:
                time_s = 0.0

            cases_list.append(
                {
                    "classname": classname,
                    "name": name,
                    "time": time_s,
                    "status": status,
                    "message": message,
                }
            )

    passed_total = max(0, tests_total - failures_total - errors_total - skipped_total)
    summary["tests"] = tests_total
    summary["failures"] = failures_total
    summary["errors"] = errors_total
    summary["skipped"] = skipped_total
    summary["passed"] = passed_total
    summary["cases"] = cases_list
    return summary


def run_pytest_in_work(
    work_dir: Path,
    extra_args: list[str] | None = None,
    cap_bytes: int = MAX_OUTPUT_BYTES,
) -> dict[str, object]:
    """Run pytest inside `work_dir` with capped stdout/stderr and bounded JUnit XML output."""
    junit_dir = Path(tempfile.gettempdir())
    junit_path = junit_dir / "patchpilot_junit.xml"
    if junit_path.exists():
        try:
            junit_path.unlink()
        except OSError:
            pass

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--disable-warnings",
        f"--junitxml={junit_path}",
        "-o",
        f"cache_dir={junit_dir / 'pytest_cache'}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(work_dir),
        "TMPDIR": str(junit_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    deps_dir = Path("/deps")
    existing_py_path = os.environ.get("PYTHONPATH", "")
    if existing_py_path:
        env["PYTHONPATH"] = existing_py_path
    elif deps_dir.is_dir():
        env["PYTHONPATH"] = str(deps_dir)

    proc = subprocess.Popen(
        cmd,
        cwd=str(work_dir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    stdout_box: list[tuple[str, bool]] = [("", False)]
    stderr_box: list[tuple[str, bool]] = [("", False)]

    def _read_out() -> None:
        assert proc.stdout is not None
        stdout_box[0] = read_stream_capped(proc.stdout, cap_bytes)

    def _read_err() -> None:
        assert proc.stderr is not None
        stderr_box[0] = read_stream_capped(proc.stderr, cap_bytes)

    t_out = threading.Thread(target=_read_out, daemon=True)
    t_err = threading.Thread(target=_read_err, daemon=True)
    t_out.start()
    t_err.start()

    exit_code = proc.wait()
    if exit_code < 0:
        exit_code = 128 + abs(exit_code)
    t_out.join(timeout=5.0)
    t_err.join(timeout=5.0)

    stdout_text, stdout_truncated = stdout_box[0]
    stderr_text, stderr_truncated = stderr_box[0]

    # Bounded read of the JUnit file (at most cap_bytes directly)
    junit_xml = ""
    if junit_path.is_file():
        try:
            with open(junit_path, "rb") as f:
                raw_xml = f.read(cap_bytes)
            junit_xml = raw_xml.decode("utf-8", errors="replace")
        except OSError:
            junit_xml = ""

    junit_summary = parse_junit_xml(junit_xml)
    err = junit_summary.get("error")
    effective_exit = int(exit_code)
    if err and effective_exit == 0:
        effective_exit = 1

    combined_err = (stderr_text + ("\n" if stderr_text else "") + str(err)) if err else stderr_text

    return {
        "exit_code": effective_exit,
        "patch_applied": True,
        "timed_out": False,
        "stdout": stdout_text,
        "stderr": combined_err,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "junit_xml": junit_xml,
        "junit": junit_summary,
        "error": str(err) if err else None,
    }


def run_install_deps() -> int:
    """Install pinned binary wheels into /deps with no repository files present."""
    raw_input = sys.stdin.buffer.read(MAX_TAR_FILE_BYTES).decode("utf-8", errors="replace")
    payload = (
        json.loads(raw_input)
        if raw_input.strip().startswith("{")
        else {
            "requirements": [
                line.strip()
                for line in raw_input.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        }
    )
    requirements: list[str] = payload.get("requirements", [])
    find_links: str | None = payload.get("find_links")

    deps_dir = Path("/deps")
    tmp_dir = Path(tempfile.gettempdir())
    req_file = tmp_dir / "requirements.txt"
    req_file.write_text("\n".join(requirements) + ("\n" if requirements else ""), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--only-binary",
        ":all:",
        "--no-build-isolation",
        "--no-input",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--target",
        str(deps_dir),
    ]
    if find_links:
        cmd.extend(["--no-index", "--find-links", find_links])
    if requirements:
        cmd.extend(["-r", str(req_file)])

    if not requirements:
        result_obj = {
            "exit_code": 0,
            "cmd": cmd,
            "stdout": "No requirements to install",
            "stderr": "",
        }
        sys.stdout.write(json.dumps(result_obj) + "\n")
        sys.stdout.flush()
        return 0

    proc = subprocess.run(
        cmd,
        cwd=str(tmp_dir),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    result_obj = {
        "exit_code": proc.returncode,
        "cmd": cmd,
        "stdout": proc.stdout[:MAX_OUTPUT_BYTES],
        "stderr": proc.stderr[:MAX_OUTPUT_BYTES],
    }
    sys.stdout.write(json.dumps(result_obj) + "\n")
    sys.stdout.flush()
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if "--install-deps" in args:
        return run_install_deps()

    work_dir = Path("/work")
    try:
        extracted = safe_extract_tar(sys.stdin.buffer, work_dir)
    except TarSafetyError as exc:
        err_result = {
            "exit_code": 2,
            "patch_applied": False,
            "timed_out": False,
            "stdout": "",
            "stderr": str(exc),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "junit_xml": "",
            "junit": parse_junit_xml(""),
            "error": f"TarSafetyError: {exc}",
        }
        sys.stdout.write(json.dumps(err_result) + "\n")
        sys.stdout.flush()
        return 2

    if work_dir.is_symlink() or not stat.S_ISDIR(work_dir.stat().st_mode):
        err_result = {
            "exit_code": 2,
            "patch_applied": False,
            "timed_out": False,
            "stdout": "",
            "stderr": "/work is not a valid directory",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "junit_xml": "",
            "junit": parse_junit_xml(""),
            "error": "WorkDirError: /work is not a valid directory",
        }
        sys.stdout.write(json.dumps(err_result) + "\n")
        sys.stdout.flush()
        return 2

    verify_mode = ("--verify" in args) or extracted.verify_mode
    if verify_mode:
        # Pre-fix check: tests must run and fail
        pre_result = run_pytest_in_work(work_dir, extra_args=extracted.pytest_args)
        pre_failed = bool(
            int(pre_result.get("exit_code", 0)) != 0
            and (
                int(pre_result.get("junit", {}).get("failures", 0)) > 0
                or int(pre_result.get("junit", {}).get("errors", 0)) > 0
            )
        )

        patch_applied = False
        patch_err = None
        if extracted.patch_diff.strip():
            try:
                apply_unified_diff(extracted.patch_diff, work_dir)
                patch_applied = True
            except PatchApplyError as exc:
                patch_err = str(exc)
        else:
            patch_err = "No patch diff provided for verification"

        post_result = None
        post_passed = False
        if patch_applied:
            post_result = run_pytest_in_work(work_dir, extra_args=extracted.pytest_args)
            post_passed = bool(
                int(post_result.get("exit_code", 1)) == 0
                and int(post_result.get("junit", {}).get("failures", 1)) == 0
                and int(post_result.get("junit", {}).get("errors", 1)) == 0
                and int(post_result.get("junit", {}).get("passed", 0)) > 0
            )

        verified = bool(pre_failed and patch_applied and post_passed)
        active_res = post_result if post_result is not None else pre_result
        combined_result = {
            "verify_mode": True,
            "verified": verified,
            "pre_fix_failed": pre_failed,
            "patch_applied": patch_applied,
            "post_fix_passed": post_passed,
            "exit_code": 0 if verified else 1,
            "pre_fix": pre_result,
            "post_fix": post_result,
            "error": patch_err if not patch_applied else None,
            "stdout": active_res.get("stdout", ""),
            "stderr": active_res.get("stderr", ""),
            "stdout_truncated": bool(active_res.get("stdout_truncated", False)),
            "stderr_truncated": bool(active_res.get("stderr_truncated", False)),
            "junit_xml": active_res.get("junit_xml", ""),
            "junit": active_res.get("junit", parse_junit_xml("")),
            "timed_out": False,
        }
        sys.stdout.write(json.dumps(combined_result) + "\n")
        sys.stdout.flush()
        return 0 if verified else 1

    try:
        if extracted.patch_diff.strip():
            apply_unified_diff(extracted.patch_diff, work_dir)
    except PatchApplyError as exc:
        err_result = {
            "exit_code": 3,
            "patch_applied": False,
            "timed_out": False,
            "stdout": "",
            "stderr": str(exc),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "junit_xml": "",
            "junit": parse_junit_xml(""),
            "error": f"PatchApplyError: {exc}",
        }
        sys.stdout.write(json.dumps(err_result) + "\n")
        sys.stdout.flush()
        return 3

    result = run_pytest_in_work(work_dir, extra_args=extracted.pytest_args)
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return int(result["exit_code"])  # type: ignore[call-overload]


if __name__ == "__main__":
    raise SystemExit(main())
