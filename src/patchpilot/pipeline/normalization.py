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
    """Strip outermost markdown ```diff or ```python code fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = text.splitlines(keepends=True)
        start_idx = 0
        end_idx = len(lines)
        if lines[0].strip().startswith("```"):
            start_idx = 1
        if len(lines) > 1 and lines[-1].strip() == "```":
            end_idx = len(lines) - 1
        return "".join(lines[start_idx:end_idx])
    return text


def _strip_inline_comment(line: str) -> str:
    """Strip trailing inline comments (# ...) not inside quotes."""
    in_single = False
    in_double = False
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i].rstrip()
    return line.rstrip()


def _strip_if_elif(line: str) -> str | None:
    """If line starts with if or elif, return the condition part, else None."""
    s = line.strip()
    if s.startswith("if "):
        return s[3:].strip()
    if s.startswith("elif "):
        return s[5:].strip()
    return None


def _normalize_line_tokens(line: str) -> str:
    """Normalize line by stripping comments and collapsing whitespace."""
    no_comment = _strip_inline_comment(line)
    return re.sub(r"\s+", " ", no_comment).strip()


def _lines_match_tier2(orig_l: str, search_l: str, allow_if_elif: bool) -> bool:
    o_raw = orig_l.rstrip("\r\n")
    s_raw = search_l.rstrip("\r\n")
    if o_raw == s_raw:
        return True

    o_norm = _normalize_line_tokens(o_raw)
    s_norm = _normalize_line_tokens(s_raw)
    if o_norm == s_norm:
        return True

    if allow_if_elif:
        o_cond = _strip_if_elif(o_norm)
        s_cond = _strip_if_elif(s_norm)
        if o_cond is not None and s_cond is not None and o_cond == s_cond:
            return True

    return False


def _tier2_find_unique_window(orig_lines: list[str], search_lines: list[str]) -> int | None:
    """Find unique matching start index of search_lines in orig_lines, or None."""
    if not orig_lines or not search_lines:
        return None
    M = len(search_lines)
    candidate_indices: list[int] = []
    for i in range(len(orig_lines) - M + 1):
        window = orig_lines[i : i + M]
        match = True
        if_elif_used = 0
        for o_l, s_l in zip(window, search_lines, strict=False):
            if _lines_match_tier2(o_l, s_l, allow_if_elif=False):
                continue
            if M > 1 and if_elif_used == 0 and _lines_match_tier2(o_l, s_l, allow_if_elif=True):
                if_elif_used += 1
                continue
            match = False
            break
        if match:
            candidate_indices.append(i)
    if len(candidate_indices) == 1:
        return candidate_indices[0]
    return None


def _tier2_has_unique_match(content: str, search_text: str) -> bool:
    """Check if search_text uniquely anchors to content under Tier 2 normalization."""
    if not content.strip() or not search_text.strip():
        return False
    orig_lines = content.splitlines(keepends=True)
    search_lines = search_text.splitlines(keepends=True)
    return _tier2_find_unique_window(orig_lines, search_lines) is not None


def _tier2_fallback_replace(
    orig_text: str,
    search_block: str,
    replace_block: str,
) -> str | None:
    """Tier 2 fallback matcher: resilient to whitespace, inline comments, and if/elif drift."""
    if not orig_text.strip() or not search_block.strip():
        return None

    newline = "\r\n" if "\r\n" in orig_text else "\n"
    orig_lines = orig_text.splitlines(keepends=True)
    search_lines = search_block.splitlines(keepends=True)
    replace_lines = replace_block.splitlines(keepends=True)

    idx = _tier2_find_unique_window(orig_lines, search_lines)
    if idx is None:
        return None

    M = len(search_lines)
    matched_window = orig_lines[idx : idx + M]

    orig_first = ""
    for line in matched_window:
        if line.strip():
            orig_first = line
            break

    search_first = ""
    for line in search_lines:
        if line.strip():
            search_first = line
            break

    orig_indent = len(orig_first) - len(orig_first.lstrip())
    search_indent = len(search_first) - len(search_first.lstrip())
    indent_delta = orig_indent - search_indent

    adjusted_replace: list[str] = []
    for r_line in replace_lines:
        r_raw = r_line.rstrip("\r\n")
        if not r_raw.strip():
            adjusted_replace.append(newline)
            continue

        if indent_delta > 0:
            adj = (" " * indent_delta) + r_raw
        elif indent_delta < 0:
            dedent = min(-indent_delta, len(r_raw) - len(r_raw.lstrip()))
            adj = r_raw[dedent:]
        else:
            adj = r_raw

        adjusted_replace.append(adj + newline)

    # If the original matched window had no trailing newline on its last line
    if idx + M == len(orig_lines) and orig_lines and not orig_lines[-1].endswith(("\r", "\n")):
        if adjusted_replace:
            adjusted_replace[-1] = adjusted_replace[-1].rstrip("\r\n")

    new_lines = orig_lines[:idx] + adjusted_replace + orig_lines[idx + M :]
    return "".join(new_lines)


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

    # Search for files containing search_text (Tier 1 exact, then Tier 2 resilient)
    matches = [f for f, content in files.items() if _content_contains_block(content, search_text)]
    if not matches:
        matches = [
            f for f, content in files.items() if _tier2_has_unique_match(content, search_text)
        ]

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

            # Tier 2 fallback resilient match
            tier2_res = _tier2_fallback_replace(orig_content, search_block, replace_block)
            if tier2_res is not None:
                working_files[target] = tier2_res
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
