"""Tar builder, safe tar extractor, pure-Python unified diff applicator, and output capper."""

from __future__ import annotations

import io
import json
import os
import posixpath
import re
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from pydantic import BaseModel, Field

MAX_OUTPUT_BYTES = 1_048_576  # 1 MB cap per SPEC §6
MAX_PATCH_BYTES = 10 * 1024 * 1024  # 10 MB maximum patch size
MAX_TAR_FILE_BYTES = 32 * 1024 * 1024  # 32 MB per single file
MAX_TAR_TOTAL_BYTES = 200 * 1024 * 1024  # 200 MB total unpacked
MAX_TAR_MEMBERS = 10_000

MAX_JUNIT_CASES = 500
MAX_JUNIT_STR_LEN = 1024
MAX_JUNIT_MSG_LEN = 4096

META_PREFIX = "__patchpilot__/"
PATCH_MEMBER_NAME = "__patchpilot__/patch.diff"
META_MEMBER_NAME = "__patchpilot__/meta.json"
SYMLINKS_MANIFEST_NAME = "__symlinks__.json"
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class TarSafetyError(ValueError):
    """Raised when a tar archive contains unsafe paths, links, or special entries."""


class PatchApplyError(ValueError):
    """Raised when a unified diff is malformed, unsafe, or fails to apply cleanly."""


class JUnitCase(BaseModel):
    classname: str = Field(default="", max_length=MAX_JUNIT_STR_LEN)
    name: str = Field(default="", max_length=MAX_JUNIT_STR_LEN)
    time: float = 0.0
    status: str = "passed"
    message: str | None = Field(default=None, max_length=MAX_JUNIT_MSG_LEN)


class JUnitReport(BaseModel):
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    passed: int = 0
    cases: list[JUnitCase] = Field(default_factory=list, max_length=MAX_JUNIT_CASES)


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


def _add_bytes_to_tar(
    tar: tarfile.TarFile,
    arcname: str,
    data: bytes,
    mode: int = 0o644,
) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.type = tarfile.REGTYPE
    info.size = len(data)
    info.mode = mode
    info.uid = 10001
    info.gid = 10001
    info.uname = "runner"
    info.gname = "runner"
    tar.addfile(info, io.BytesIO(data))


def _add_symlink_to_tar(tar: tarfile.TarFile, arcname: str, linkname: str) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.type = tarfile.SYMTYPE
    info.linkname = linkname
    info.size = 0
    info.mode = 0o777
    info.uid = 10001
    info.gid = 10001
    info.uname = "runner"
    info.gname = "runner"
    tar.addfile(info)


def _walk_repo_no_deref(
    root_dir: Path,
    current_dir: Path,
    tar: tarfile.TarFile,
) -> None:
    entries = sorted(os.scandir(current_dir), key=lambda e: e.name)
    for entry in entries:
        entry_path = Path(entry.path)
        rel_posix = entry_path.relative_to(root_dir).as_posix()
        if entry.is_symlink():
            link_target = os.readlink(entry_path)
            _add_symlink_to_tar(tar, rel_posix, link_target)
        elif entry.is_dir(follow_symlinks=False):
            if entry.name in (".git", "__pycache__", ".pytest_cache"):
                continue
            _walk_repo_no_deref(root_dir, entry_path, tar)
        elif entry.is_file(follow_symlinks=False):
            if rel_posix == SYMLINKS_MANIFEST_NAME:
                manifest_raw = json.loads(entry_path.read_text(encoding="utf-8"))
                if isinstance(manifest_raw, dict):
                    for sym_name, sym_target in sorted(manifest_raw.items()):
                        if isinstance(sym_name, str) and isinstance(sym_target, str):
                            _add_symlink_to_tar(tar, sym_name, sym_target)
                continue
            _add_bytes_to_tar(tar, rel_posix, entry_path.read_bytes())


def build_repo_tar(
    repo_dir: Path | None = None,
    *,
    files: Mapping[str, str | bytes] | None = None,
    symlinks: Mapping[str, str] | None = None,
    patch_diff: str = "",
    pytest_args: list[str] | None = None,
    verify: bool = False,
) -> bytes:
    """Build an in-memory tar archive for sandbox stdin without dereferencing host symlinks."""
    if patch_diff and len(patch_diff.encode("utf-8")) > MAX_PATCH_BYTES:
        raise ValueError(f"Patch diff size exceeds limit of {MAX_PATCH_BYTES} bytes")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        if repo_dir is not None:
            if not repo_dir.is_dir() or repo_dir.is_symlink():
                raise ValueError(f"repo_dir must be a real directory: {repo_dir}")
            _walk_repo_no_deref(repo_dir, repo_dir, tar)

        if files:
            for arcname, content in sorted(files.items()):
                raw_bytes = content.encode("utf-8") if isinstance(content, str) else content
                _add_bytes_to_tar(tar, arcname, raw_bytes)

        if symlinks:
            for arcname, target in sorted(symlinks.items()):
                _add_symlink_to_tar(tar, arcname, target)

        if patch_diff:
            _add_bytes_to_tar(tar, PATCH_MEMBER_NAME, patch_diff.encode("utf-8"))

        meta_obj: dict[str, object] = {}
        if pytest_args:
            meta_obj["pytest_args"] = list(pytest_args)
        if verify:
            meta_obj["verify"] = True
        if meta_obj:
            meta_bytes = json.dumps(meta_obj).encode("utf-8")
            _add_bytes_to_tar(tar, META_MEMBER_NAME, meta_bytes)

    return buf.getvalue()


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


def safe_extract_tar(stream: IO[bytes], work_dir: Path) -> ExtractedPayload:
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


def read_stream_capped(stream: IO[bytes], cap_bytes: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
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


def cap_bytes_output(data: bytes, cap_bytes: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    """Cap raw bytes at `cap_bytes` and decode as UTF-8."""
    if len(data) <= cap_bytes:
        return data.decode("utf-8", errors="replace"), False
    return data[:cap_bytes].decode("utf-8", errors="replace"), True


def validate_junit_summary(summary_data: object) -> JUnitReport:
    """Validate a container-produced JSON summary against the Pydantic schema with size limits."""
    if isinstance(summary_data, JUnitReport):
        return summary_data
    if isinstance(summary_data, dict):
        try:
            raw_bytes = json.dumps(summary_data).encode("utf-8")
            if len(raw_bytes) <= MAX_OUTPUT_BYTES:
                return JUnitReport.model_validate(summary_data)
        except Exception:
            pass
    return JUnitReport()
