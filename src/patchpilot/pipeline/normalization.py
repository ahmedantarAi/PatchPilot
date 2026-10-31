"""Normalization of SEARCH/REPLACE blocks and raw unified diffs via difflib."""

from __future__ import annotations

import difflib
import re
import tempfile
from pathlib import Path

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


def _find_target_file(
    search_text: str,
    explicit_file: str | None,
    files: dict[str, str],
    default_file: str | None = None,
) -> str | None:
    """Resolve the target file for a search/replace block."""
    if explicit_file:
        cleaned = explicit_file.replace("\\", "/").strip().lstrip("./")
        if cleaned.startswith("a/") or cleaned.startswith("b/"):
            cleaned = cleaned[2:]
        if cleaned in files:
            return cleaned
        for f in files:
            if f.endswith(cleaned) or f.replace("\\", "/").endswith(cleaned):
                return f

    if default_file and default_file in files:
        if search_text in files[default_file]:
            return default_file

    # Search for files containing search_text
    matches = [f for f, content in files.items() if search_text in content]
    if len(matches) == 1:
        return matches[0]

    # Non-test python files containing search_text
    non_test_matches = [
        f
        for f in matches
        if not Path(f).name.startswith("test_") and not Path(f).name.endswith("_test.py")
    ]
    if len(non_test_matches) == 1:
        return non_test_matches[0]

    if default_file and default_file in files:
        return default_file

    non_test_files = [
        f for f in files if f.endswith(".py") and not Path(f).name.startswith("test_")
    ]
    if len(non_test_files) == 1:
        return non_test_files[0]

    return None


def normalize_search_replace(
    raw_text: str,
    files: dict[str, str],
    default_file: str | None = None,
) -> str:
    """Extract SEARCH/REPLACE blocks, apply in memory, and generate canonical unified diff."""
    text = _strip_markdown_code_fences(raw_text)
    matches = list(SEARCH_REPLACE_PATTERN.finditer(text))
    if not matches:
        raise ValueError("No valid SEARCH/REPLACE blocks found in text")

    working_files = dict(files)
    changed_files: set[str] = set()

    for m in matches:
        file_pre, file_col, file_spc, search_block, replace_block = m.groups()
        explicit_file = file_col or file_spc or file_pre
        target = _find_target_file(search_block, explicit_file, working_files, default_file)
        if not target:
            raise ValueError(
                f"Could not identify target file for search block: {search_block[:50]!r}"
            )

        orig_content = working_files[target]
        if search_block not in orig_content:
            # Try with normalized newlines
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

    # Generate canonical unified diff using difflib
    canonical_diff = ""
    for path in sorted(changed_files):
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

    return canonical_diff


def normalize_unified_diff(
    diff_text: str,
    files: dict[str, str],
) -> str:
    """Normalize a raw unified diff by applying it in memory and re-diffing via difflib."""
    text = _strip_markdown_code_fences(diff_text)
    if not text.strip():
        return ""

    # Apply diff to a temporary directory containing the files
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

        canonical_diff = ""
        for rel_path in sorted(files):
            orig_lines = files[rel_path].splitlines(keepends=True)
            new_file = work_dir / rel_path
            if new_file.is_file():
                new_lines = new_file.read_text(encoding="utf-8").splitlines(keepends=True)
            else:
                new_lines = []
            if orig_lines != new_lines:
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

        return canonical_diff


def normalize_patch(
    raw_patch: str,
    files: dict[str, str],
    default_file: str | None = None,
) -> str:
    """Normalize either SEARCH/REPLACE blocks or a raw unified diff into canonical unified diff."""
    text = _strip_markdown_code_fences(raw_patch)
    if not text.strip():
        return ""

    if "<<<<<<< SEARCH" in text and ">>>>>>> REPLACE" in text:
        return normalize_search_replace(text, files, default_file=default_file)

    if "--- " in text and "+++ " in text:
        return normalize_unified_diff(text, files)

    raise ValueError("Patch text is neither a SEARCH/REPLACE block nor a unified diff")
