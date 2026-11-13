"""Normalization of SEARCH/REPLACE blocks and raw unified diffs with reward-hacking guard."""

from __future__ import annotations

import difflib
import re
import tempfile
from pathlib import Path, PurePosixPath

from patchpilot.sandbox.tar_and_diff import PatchApplyError, apply_unified_diff

SEARCH_REPLACE_PATTERN = re.compile(
    r"(?:^[ \t]*(?:#+|//+|\*+|File:)?\s*([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+)\s*\n)?"
    r"^[ \t]*<{7}\s*SEARCH(?::\s*([a-zA-Z0-9_./\\-]+))?(?:[ \t]+([a-zA-Z0-9_./\\-]+))?[ \t]*\n"
    r"(.*?)"
    r"^[ \t]*={7}[ \t]*\n"
    r"(.*?)"
    r"^[ \t]*>{7}\s*REPLACE[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def _clean_rel_path(raw_path: str) -> str:
    """Normalize a diff or file header path to a clean POSIX relative path."""
    cleaned = raw_path.split("\t", 1)[0].replace("\\", "/").strip()
    if cleaned == "/dev/null":
        return "/dev/null"
    if cleaned.startswith("a/") or cleaned.startswith("b/"):
        cleaned = cleaned[2:]
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def is_test_file(rel_path: str) -> bool:
    """Return True if `rel_path` is a test file (test_*.py, *_test.py, tests/**, conftest.py)."""
    cleaned = _clean_rel_path(rel_path)
    if not cleaned or cleaned == "/dev/null":
        return False
    posix = PurePosixPath(cleaned)
    name = posix.name
    parts = posix.parts
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or "tests" in parts[:-1]
    )


def is_modifiable_py_file(rel_path: str) -> bool:
    """Return True if `rel_path` is a non-test Python source file eligible for modification."""
    cleaned = _clean_rel_path(rel_path)
    if not cleaned or cleaned == "/dev/null":
        return False
    return cleaned.endswith(".py") and not is_test_file(cleaned)


def check_forbidden_patch_path(rel_path: str) -> str | None:
    """Return a clear rejection reason if `rel_path` is a test file or non-.py file, else None."""
    cleaned = _clean_rel_path(rel_path)
    if not cleaned or cleaned == "/dev/null":
        return None
    if is_test_file(cleaned):
        return f"Reward-hacking guard: patch is not allowed to touch test file {cleaned!r}"
    if not cleaned.endswith(".py"):
        return f"Reward-hacking guard: patch is not allowed to touch non-.py file {cleaned!r}"
    return None


def validate_patch_targets(diff_text: str) -> None:
    """Validate that a unified diff only touches allowed non-test .py files."""
    if not diff_text.strip():
        return

    lines = diff_text.splitlines()
    found_headers = 0
    for idx, line in enumerate(lines):
        if line.startswith("--- "):
            old_raw = line[4:]
            old_clean = _clean_rel_path(old_raw)
            reason = check_forbidden_patch_path(old_clean)
            if reason is not None:
                raise ValueError(reason)
            if idx + 1 < len(lines) and lines[idx + 1].startswith("+++ "):
                new_raw = lines[idx + 1][4:]
                new_clean = _clean_rel_path(new_raw)
                reason_new = check_forbidden_patch_path(new_clean)
                if reason_new is not None:
                    raise ValueError(reason_new)
                if old_clean == "/dev/null" and new_clean == "/dev/null":
                    raise ValueError("Invalid unified diff: both old and new paths are /dev/null")
                found_headers += 1
        elif line.startswith("+++ "):
            new_clean = _clean_rel_path(line[4:])
            reason_new = check_forbidden_patch_path(new_clean)
            if reason_new is not None:
                raise ValueError(reason_new)

    if found_headers == 0:
        raise ValueError("Unified diff contains no valid file headers")


def _strip_markdown_code_fences(text: str) -> str:
    """Strip markdown ``` fences (```diff, ```python, ...) around the payload if present.

    Blank lines outside fenced blocks (leading, trailing, or between consecutive blocks) are
    ignored; blank lines inside a block are preserved verbatim. Consecutive fenced blocks are
    concatenated. Text without a leading fence is returned unchanged.
    """
    lines = text.splitlines(keepends=True)
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or not lines[idx].strip().startswith("```"):
        return text

    out: list[str] = []
    while idx < len(lines) and lines[idx].strip().startswith("```"):
        idx += 1  # skip opening fence
        while idx < len(lines) and lines[idx].strip() != "```":
            out.append(lines[idx])
            idx += 1
        idx += 1  # skip closing fence (no-op if unterminated)
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
    return "".join(out)


def _content_contains_block(content: str, search_text: str) -> bool:
    if search_text in content:
        return True
    return search_text.replace("\r\n", "\n") in content.replace("\r\n", "\n")


def _find_target_file(
    search_text: str,
    explicit_file: str | None,
    files: dict[str, str],
) -> str | None:
    """Resolve the target file for a search/replace block without any ground-truth hint."""
    if explicit_file:
        cleaned = _clean_rel_path(explicit_file)
        # If the explicit file is forbidden (test file or non-.py), return it immediately
        # so the caller's reward-hacking guard raises a clear rejection error.
        if check_forbidden_patch_path(cleaned) is not None:
            return cleaned
        if cleaned in files:
            return cleaned
        for f in files:
            if f.replace("\\", "/") == cleaned or f.replace("\\", "/").endswith(f"/{cleaned}"):
                return f
        return cleaned

    # Search for files containing search_text
    matches = [f for f, content in files.items() if _content_contains_block(content, search_text)]
    if len(matches) == 1:
        return matches[0]

    # Prefer modifiable non-test python files containing search_text
    non_test_matches = [f for f in matches if is_modifiable_py_file(f)]
    if len(non_test_matches) == 1:
        return non_test_matches[0]
    if len(non_test_matches) > 1:
        return sorted(non_test_matches)[0]

    # If search_text matched only test or non-.py files, return the first match so
    # the reward-hacking guard rejects it with a clear message.
    if matches:
        return matches[0]

    non_test_files = [f for f in files if is_modifiable_py_file(f)]
    if len(non_test_files) == 1:
        return non_test_files[0]

    return None


def normalize_search_replace(
    raw_text: str,
    files: dict[str, str],
) -> str:
    """Extract SEARCH/REPLACE blocks, enforce guard, and generate canonical unified diff."""
    text = _strip_markdown_code_fences(raw_text)
    matches = list(SEARCH_REPLACE_PATTERN.finditer(text))
    if not matches:
        raise ValueError("No valid SEARCH/REPLACE blocks found in text")

    working_files = dict(files)
    changed_files: set[str] = set()

    for m in matches:
        file_pre, file_col, file_spc, search_block, replace_block = m.groups()
        explicit_file = file_col or file_spc or file_pre
        if explicit_file:
            explicit_reason = check_forbidden_patch_path(explicit_file)
            if explicit_reason is not None:
                raise ValueError(explicit_reason)

        target = _find_target_file(search_block, explicit_file, working_files)
        if not target:
            raise ValueError(
                f"Could not identify target file for search block: {search_block[:50]!r}"
            )

        target_reason = check_forbidden_patch_path(target)
        if target_reason is not None:
            raise ValueError(target_reason)

        if target not in working_files:
            raise ValueError(f"Target file {target!r} not found in repository")

        orig_content = working_files[target]
        if search_block not in orig_content:
            norm_orig = orig_content.replace("\r\n", "\n")
            norm_search = search_block.replace("\r\n", "\n")
            if norm_search in norm_orig:
                working_files[target] = norm_orig.replace(
                    norm_search, replace_block.replace("\r\n", "\n"), 1
                )
                changed_files.add(target)
                continue
            raise ValueError(f"Search block not found in {target}: {search_block[:50]!r}")

        working_files[target] = orig_content.replace(search_block, replace_block, 1)
        changed_files.add(target)

    canonical_diff = ""
    for path in sorted(changed_files):
        reason = check_forbidden_patch_path(path)
        if reason is not None:
            raise ValueError(reason)
        orig_lines = files[path].splitlines(keepends=True)
        new_lines = working_files[path].splitlines(keepends=True)
        rel_posix = path.replace("\\", "/")
        diff_lines = list(
            difflib.unified_diff(
                orig_lines,
                new_lines,
                fromfile=f"a/{rel_posix}",
                tofile=f"b/{rel_posix}",
            )
        )
        canonical_diff += "".join(diff_lines)

    if canonical_diff.strip():
        validate_patch_targets(canonical_diff)

    return canonical_diff


def normalize_unified_diff(
    diff_text: str,
    files: dict[str, str],
) -> str:
    """Normalize a raw unified diff, enforcing reward-hacking guard and re-diffing via difflib."""
    text = _strip_markdown_code_fences(diff_text)
    if not text.strip():
        return ""

    # Enforce reward-hacking guard on diff headers before applying
    validate_patch_targets(text)

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for rel_path, content in files.items():
            dest = work_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        try:
            apply_unified_diff(text, work_dir)
        except PatchApplyError as exc:
            raise ValueError(f"Failed to apply unified diff: {exc}") from exc

        all_paths = set(files.keys())
        for disk_file in work_dir.rglob("*"):
            if disk_file.is_file():
                rel_disk = disk_file.relative_to(work_dir).as_posix()
                all_paths.add(rel_disk)

        canonical_diff = ""
        for rel_path in sorted(all_paths):
            orig_lines = files[rel_path].splitlines(keepends=True) if rel_path in files else []
            new_file = work_dir / rel_path
            if new_file.is_file():
                new_lines = new_file.read_text(encoding="utf-8").splitlines(keepends=True)
            else:
                new_lines = []
            if orig_lines != new_lines:
                reason = check_forbidden_patch_path(rel_path)
                if reason is not None:
                    raise ValueError(reason)
                rel_posix = rel_path.replace("\\", "/")
                diff_lines = list(
                    difflib.unified_diff(
                        orig_lines,
                        new_lines,
                        fromfile=f"a/{rel_posix}",
                        tofile=f"b/{rel_posix}",
                    )
                )
                canonical_diff += "".join(diff_lines)

        if canonical_diff.strip():
            validate_patch_targets(canonical_diff)

        return canonical_diff


def normalize_patch(
    raw_patch: str,
    files: dict[str, str],
) -> str:
    """Normalize either SEARCH/REPLACE blocks or a raw unified diff into canonical unified diff."""
    text = _strip_markdown_code_fences(raw_patch)
    if not text.strip():
        return ""

    if "<<<<<<< SEARCH" in text and ">>>>>>> REPLACE" in text:
        return normalize_search_replace(text, files)

    if "--- " in text and "+++ " in text:
        return normalize_unified_diff(text, files)

    raise ValueError("Patch text is neither a SEARCH/REPLACE block nor a unified diff")
