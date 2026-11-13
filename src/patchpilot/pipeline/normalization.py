"""Normalization of SEARCH/REPLACE blocks and raw unified diffs with reward-hacking guard."""

from __future__ import annotations

import ast
import difflib
import posixpath
import re
import tempfile
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from patchpilot.sandbox.tar_and_diff import PatchApplyError, apply_unified_diff

SEARCH_REPLACE_PATTERN = re.compile(
    r"(?:^[ \t]*(?:#+|//+|\*+|File:)?[ \t]*"
    r"([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+)[ \t]*\r?\n)?"
    r"^[ \t]*<{7}[ \t]*SEARCH(?::[ \t]*([a-zA-Z0-9_./\\-]+))?"
    r"(?:[ \t]+([a-zA-Z0-9_./\\-]+))?[ \t]*\r?\n"
    r"(.*?)"
    r"^[ \t]*={7}[ \t]*\r?\n"
    r"(.*?)"
    r"^[ \t]*>{7}[ \t]*REPLACE[ \t]*\r?$",
    re.MULTILINE | re.DOTALL,
)

_SEARCH_HEADER_RE = re.compile(r"^[ \t]*<{7}[ \t]*SEARCH\b", re.MULTILINE)
_REPLACE_FOOTER_RE = re.compile(r"^[ \t]*>{7}[ \t]*REPLACE\b", re.MULTILINE)


def _clean_rel_path(raw_path: str) -> str:
    """Normalize a diff or file header path to a clean POSIX relative path."""
    cleaned = raw_path.split("\t", 1)[0].strip().strip("\"'").replace("\\", "/").strip()
    if cleaned == "/dev/null":
        return "/dev/null"
    if cleaned.startswith("a/") or cleaned.startswith("b/"):
        cleaned = cleaned[2:]
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned and cleaned != "/dev/null" and not cleaned.startswith("/"):
        normed = posixpath.normpath(cleaned)
        if normed != ".":
            cleaned = normed
    return cleaned


def is_test_file(rel_path: str) -> bool:
    """Return True if `rel_path` is a test file (test_*.py, *_test.py, tests/**, conftest.py)."""
    cleaned = _clean_rel_path(rel_path)
    if not cleaned or cleaned == "/dev/null":
        return False
    posix = PurePosixPath(cleaned)
    name = posix.name.lower()
    parts_lower = [p.lower() for p in posix.parts[:-1]]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or "tests" in parts_lower
    )


def is_modifiable_py_file(rel_path: str) -> bool:
    """Return True if `rel_path` is a non-test Python source file eligible for modification."""
    cleaned = _clean_rel_path(rel_path)
    if not cleaned or cleaned == "/dev/null":
        return False
    if cleaned.startswith("/") or ":" in cleaned.split("/", 1)[0] or ".." in cleaned.split("/"):
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
    if cleaned.startswith("/") or ":" in cleaned.split("/", 1)[0] or ".." in cleaned.split("/"):
        return f"Reward-hacking guard: patch is not allowed to touch unsafe path {cleaned!r}"
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


def _has_patch_markers(block_lines: list[str]) -> bool:
    """Return True if fenced block lines contain SEARCH/REPLACE or unified diff headers."""
    block_text = "".join(block_lines)
    if _SEARCH_HEADER_RE.search(block_text):
        return True
    has_old = any(ln.startswith("--- ") for ln in block_lines)
    has_new = any(ln.startswith("+++ ") for ln in block_lines)
    return has_old and has_new


def _strip_markdown_code_fences(text: str) -> str:
    """Strip markdown ``` fences (```diff, ```python, ...) around the payload if present.

    Blank lines outside fenced blocks (leading, trailing, or between consecutive blocks) are
    ignored; blank lines inside a block are preserved verbatim. Consecutive fenced blocks are
    concatenated. When a short prose preamble or inter-block prose accompanies fenced blocks,
    any fenced blocks containing SEARCH/REPLACE or unified diff headers are extracted. Text
    without matching fenced blocks is returned unchanged.
    """
    lines = text.splitlines(keepends=True)
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return text

    if lines[idx].strip().startswith("```"):
        out: list[str] = []
        while idx < len(lines) and lines[idx].strip().startswith("```"):
            idx += 1  # skip opening fence
            while idx < len(lines) and lines[idx].strip() != "```":
                out.append(lines[idx])
                idx += 1
            idx += 1  # skip closing fence (no-op if unterminated)
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
        while idx < len(lines):
            if lines[idx].strip().startswith("```"):
                idx += 1
                extra_lines: list[str] = []
                while idx < len(lines) and lines[idx].strip() != "```":
                    extra_lines.append(lines[idx])
                    idx += 1
                if idx < len(lines) and lines[idx].strip() == "```":
                    idx += 1
                if _has_patch_markers(extra_lines):
                    if out and not out[-1].endswith(("\n", "\r")):
                        out.append("\n")
                    out.extend(extra_lines)
            else:
                idx += 1
        return "".join(out)

    # Prose preamble before fenced blocks: extract fenced blocks with patch markers.
    extracted_blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("```"):
            i += 1
            block_lines: list[str] = []
            while i < len(lines) and lines[i].strip() != "```":
                block_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() == "```":
                i += 1
            if _has_patch_markers(block_lines):
                block_str = "".join(block_lines)
                if extracted_blocks and not extracted_blocks[-1].endswith(("\n", "\r")):
                    extracted_blocks.append("\n")
                extracted_blocks.append(block_str)
        else:
            i += 1

    if extracted_blocks:
        return "".join(extracted_blocks)
    return text


_IF_STMT_RE = re.compile(r"^if(?:\s|\(|:|$)")
_ELIF_STMT_RE = re.compile(r"^elif(?:\s|\(|:|$)")


def _split_lines_preserve(text: str) -> list[str]:
    """Split text only on \\r\\n or \\n, preserving line endings and not splitting on \\v/\\f."""
    if not text:
        return []
    lines: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\r":
            if i + 1 < n and text[i + 1] == "\n":
                lines.append(text[start : i + 2])
                i += 2
            else:
                lines.append(text[start : i + 1])
                i += 1
            start = i
        elif ch == "\n":
            lines.append(text[start : i + 1])
            i += 1
            start = i
        else:
            i += 1
    if start < n:
        lines.append(text[start:])
    return lines


def _dominant_eol(text: str) -> str:
    """Return '\\r\\n' if text uses CRLF exclusively, else '\\n'."""
    if "\r\n" in text and "\n" not in text.replace("\r\n", ""):
        return "\r\n"
    return "\n"


def _leading_ws(line: str) -> str:
    """Return the leading spaces/tabs prefix of a line."""
    stripped = line.lstrip(" \t")
    return line[: len(line) - len(stripped)]


def _build_replacement_lines(
    replace_block: str,
    eol: str,
    preserve_no_final_eol: bool,
) -> list[str]:
    """Format raw replace_block lines using `eol`."""
    if not replace_block:
        return []
    raw_lines = _split_lines_preserve(replace_block.replace("\r\n", "\n"))
    out = [ln.rstrip("\r\n") + eol for ln in raw_lines]
    if preserve_no_final_eol and out:
        out[-1] = out[-1].rstrip("\r\n")
    return out


def _reindent_replace_block(
    replace_block: str,
    target_prefix: str,
    eol: str,
    preserve_no_final_eol: bool,
) -> list[str]:
    """Dedent replace_block and re-indent non-empty lines with exact `target_prefix`."""
    if not replace_block:
        return []
    dedented = textwrap.dedent(replace_block.replace("\r\n", "\n"))
    raw_lines = _split_lines_preserve(dedented)
    out: list[str] = []
    for ln in raw_lines:
        body = ln.rstrip("\r\n")
        if body.strip():
            out.append(f"{target_prefix}{body}{eol}")
        else:
            out.append(eol)
    if preserve_no_final_eol and out:
        out[-1] = out[-1].rstrip("\r\n")
    return out


def _scan_uniform_prefix_hits(
    orig_raw: list[str],
    s_ded_lines: list[str],
) -> list[tuple[int, int, str]]:
    """Scan `orig_raw` for windows matching `s_ded_lines` with a uniform whitespace prefix."""
    k = len(s_ded_lines)
    if k == 0 or k > len(orig_raw):
        return []

    hits: list[tuple[int, int, str]] = []
    for i in range(len(orig_raw) - k + 1):
        prefix: str | None = None
        ok = True
        for w_ln, s_ln in zip(orig_raw[i : i + k], s_ded_lines, strict=True):
            if not s_ln.strip():
                if w_ln.strip():
                    ok = False
                    break
                continue
            w_r = w_ln.rstrip()
            s_r = s_ln.rstrip()
            if not w_r.endswith(s_r):
                ok = False
                break
            cand_prefix = w_r[: len(w_r) - len(s_r)]
            if any(ch not in (" ", "\t") for ch in cand_prefix):
                ok = False
                break
            if prefix is None:
                prefix = cand_prefix
            elif cand_prefix != prefix:
                ok = False
                break
        if ok and prefix is not None:
            hits.append((i, i + k, prefix))
    return hits


def _has_if_elif_mismatch(w_stripped: list[str], s_stripped: list[str]) -> bool:
    """Return True if any aligned replacement pairs an `if` line with an `elif` line."""
    matcher = difflib.SequenceMatcher(None, w_stripped, s_stripped)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        w_part = w_stripped[i1:i2]
        s_part = s_stripped[j1:j2]
        w_has_if = any(_IF_STMT_RE.match(ln) for ln in w_part)
        w_has_elif = any(_ELIF_STMT_RE.match(ln) for ln in w_part)
        s_has_if = any(_IF_STMT_RE.match(ln) for ln in s_part)
        s_has_elif = any(_ELIF_STMT_RE.match(ln) for ln in s_part)
        if (w_has_if and s_has_elif) or (w_has_elif and s_has_if):
            return True
    return False


@dataclass(frozen=True)
class _TierScanResult:
    """Outcome of scanning a single matching tier against file content."""

    status: str  # "unique", "ambiguous", "none", "syntax_error"
    new_content: str = ""
    match_lines: tuple[int, ...] = ()
    detail: str = ""


def _try_tier1_exact(
    orig_content: str,
    search_block: str,
    replace_block: str,
) -> _TierScanResult:
    """Tier 1: Line-aligned exact or \\r\\n-normalized match with unique-match enforcement."""
    if not search_block or not search_block.strip():
        return _TierScanResult(status="none")

    orig_lines = _split_lines_preserve(orig_content)
    search_lines = _split_lines_preserve(search_block)
    k = len(search_lines)
    if k == 0 or k > len(orig_lines):
        return _TierScanResult(status="none")

    exact_hits = [
        i for i in range(len(orig_lines) - k + 1) if orig_lines[i : i + k] == search_lines
    ]
    if len(exact_hits) == 1:
        start = exact_hits[0]
        end = start + k
        rep_lines = _split_lines_preserve(replace_block)
        new_content = "".join(orig_lines[:start] + rep_lines + orig_lines[end:])
        return _TierScanResult(status="unique", new_content=new_content)
    if len(exact_hits) > 1:
        return _TierScanResult(
            status="ambiguous",
            match_lines=tuple(i + 1 for i in exact_hits),
            detail=f"matched {len(exact_hits)} locations",
        )

    orig_raw = [ln.rstrip("\r\n") for ln in orig_lines]
    search_raw = [ln.rstrip("\r\n") for ln in search_lines]
    norm_hits = [i for i in range(len(orig_raw) - k + 1) if orig_raw[i : i + k] == search_raw]
    if len(norm_hits) == 1:
        start = norm_hits[0]
        end = start + k
        eol = _dominant_eol(orig_content)
        preserve_no_final_eol = end == len(orig_lines) and not orig_lines[-1].endswith(("\n", "\r"))
        rep_lines = _build_replacement_lines(replace_block, eol, preserve_no_final_eol)
        new_content = "".join(orig_lines[:start] + rep_lines + orig_lines[end:])
        return _TierScanResult(status="unique", new_content=new_content)
    if len(norm_hits) > 1:
        return _TierScanResult(
            status="ambiguous",
            match_lines=tuple(i + 1 for i in norm_hits),
            detail=f"matched {len(norm_hits)} locations (CRLF-normalized)",
        )
    return _TierScanResult(status="none")


def _try_tier2_rstrip(
    orig_content: str,
    search_block: str,
    replace_block: str,
) -> _TierScanResult:
    """Tier 2: Line-rstrip() match preserving exact leading indentation; require count == 1."""
    if not search_block or not search_block.strip():
        return _TierScanResult(status="none")

    orig_lines = _split_lines_preserve(orig_content)
    orig_raw = [ln.rstrip("\r\n") for ln in orig_lines]
    search_raw = [ln.rstrip("\r\n") for ln in _split_lines_preserve(search_block)]
    k = len(search_raw)
    if k == 0 or k > len(orig_raw):
        return _TierScanResult(status="none")

    s_rstrip = [ln.rstrip() for ln in search_raw]
    o_rstrip = [ln.rstrip() for ln in orig_raw]
    hits = [i for i in range(len(orig_raw) - k + 1) if o_rstrip[i : i + k] == s_rstrip]
    if not hits:
        return _TierScanResult(status="none")
    if len(hits) > 1:
        return _TierScanResult(
            status="ambiguous",
            match_lines=tuple(i + 1 for i in hits),
            detail=f"matched {len(hits)} locations (Tier-2 rstrip)",
        )

    start = hits[0]
    end = start + k
    eol = _dominant_eol(orig_content)
    preserve_no_final_eol = end == len(orig_lines) and not orig_lines[-1].endswith(("\n", "\r"))
    rep_lines = _build_replacement_lines(replace_block, eol, preserve_no_final_eol)
    new_content = "".join(orig_lines[:start] + rep_lines + orig_lines[end:])
    return _TierScanResult(status="unique", new_content=new_content)


def _try_tier3_uniform_indent(
    orig_content: str,
    search_block: str,
    replace_block: str,
) -> _TierScanResult:
    """Tier 3: Uniform indentation-delta match for >= 2 non-empty lines, preserving tabs/spaces."""
    if not search_block or not search_block.strip():
        return _TierScanResult(status="none")

    orig_lines = _split_lines_preserve(orig_content)
    orig_raw = [ln.rstrip("\r\n") for ln in orig_lines]
    search_raw = [ln.rstrip("\r\n") for ln in _split_lines_preserve(search_block)]
    if sum(1 for ln in search_raw if ln.strip()) < 2:
        return _TierScanResult(status="none")

    dedented_search = textwrap.dedent(search_block.replace("\r\n", "\n"))
    s_ded_lines = [ln.rstrip("\r\n") for ln in _split_lines_preserve(dedented_search)]
    k = len(s_ded_lines)
    if k == 0 or k > len(orig_raw):
        return _TierScanResult(status="none")

    hits = _scan_uniform_prefix_hits(orig_raw, s_ded_lines)

    if not hits:
        return _TierScanResult(status="none")
    if len(hits) > 1:
        return _TierScanResult(
            status="ambiguous",
            match_lines=tuple(h[0] + 1 for h in hits),
            detail=f"matched {len(hits)} locations (Tier-3 uniform indent)",
        )

    start, end, target_prefix = hits[0]
    eol = _dominant_eol(orig_content)
    preserve_no_final_eol = end == len(orig_lines) and not orig_lines[-1].endswith(("\n", "\r"))
    rep_lines = _reindent_replace_block(replace_block, target_prefix, eol, preserve_no_final_eol)
    new_content = "".join(orig_lines[:start] + rep_lines + orig_lines[end:])
    return _TierScanResult(status="unique", new_content=new_content)


def _try_tier4_ast_fuzzy(
    orig_content: str,
    search_block: str,
    replace_block: str,
    target: str,
    validate_ast: bool = True,
) -> _TierScanResult:
    """Tier 4: AST-anchored fuzzy fallback (>=3 non-empty lines, ratio>=0.85, margin>=0.10)."""
    if not search_block or not search_block.strip():
        return _TierScanResult(status="none")

    orig_lines = _split_lines_preserve(orig_content)
    orig_raw = [ln.rstrip("\r\n") for ln in orig_lines]
    search_raw = [ln.rstrip("\r\n") for ln in _split_lines_preserve(search_block)]
    if sum(1 for ln in search_raw if ln.strip()) < 3:
        return _TierScanResult(status="none")

    first_anchor = search_raw[0].strip()
    last_anchor = search_raw[-1].strip()
    if not first_anchor or not last_anchor:
        return _TierScanResult(status="none")

    k = len(search_raw)
    slack = max(2, k // 3)
    min_win = max(3, k - slack)
    max_win = k + slack

    dedented_search = textwrap.dedent(search_block.replace("\r\n", "\n"))
    s_ded_lines = [ln.rstrip("\r\n").rstrip() for ln in _split_lines_preserve(dedented_search)]
    s_ded_text = "\n".join(s_ded_lines)
    s_stripped = [ln.strip() for ln in search_raw]

    candidates: list[tuple[float, int, int]] = []
    for i in range(len(orig_raw)):
        if orig_raw[i].strip() != first_anchor:
            continue
        j_start = i + min_win - 1
        j_end = min(len(orig_raw), i + max_win)
        for j in range(j_start, j_end):
            if orig_raw[j].strip() != last_anchor:
                continue
            w_slice = orig_raw[i : j + 1]
            if sum(1 for ln in w_slice if ln.strip()) < 3:
                continue
            w_stripped = [ln.strip() for ln in w_slice]
            if _has_if_elif_mismatch(w_stripped, s_stripped):
                continue
            w_ded_raw = textwrap.dedent("\n".join(w_slice))
            w_ded_lines = [ln.rstrip() for ln in _split_lines_preserve(w_ded_raw)]
            w_ded_text = "\n".join(w_ded_lines)
            ratio = difflib.SequenceMatcher(None, w_ded_text, s_ded_text).ratio()
            candidates.append((ratio, i, j + 1))

    if not candidates:
        return _TierScanResult(status="none")

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_ratio, best_start, best_end = candidates[0]
    if best_ratio < 0.85:
        return _TierScanResult(status="none")

    runner_up_ratio = candidates[1][0] if len(candidates) > 1 else 0.0
    if best_ratio - runner_up_ratio < 0.10:
        top_lines = tuple(c[1] + 1 for c in candidates[:3] if c[0] >= 0.85 or c == candidates[1])
        return _TierScanResult(
            status="ambiguous",
            match_lines=top_lines,
            detail=(
                f"Tier-4 fuzzy match ambiguous (best ratio {best_ratio:.2f} vs "
                f"runner-up {runner_up_ratio:.2f}, margin < 0.10)"
            ),
        )

    eol = _dominant_eol(orig_content)
    preserve_no_final_eol = best_end == len(orig_lines) and not orig_lines[-1].endswith(
        ("\n", "\r")
    )
    orig_ws = _leading_ws(orig_raw[best_start])
    search_ws = _leading_ws(search_raw[0])
    if orig_ws == search_ws:
        rep_lines = _build_replacement_lines(replace_block, eol, preserve_no_final_eol)
    else:
        s_ded0_ws = _leading_ws(s_ded_lines[0]) if s_ded_lines else ""
        if s_ded0_ws and orig_ws.endswith(s_ded0_ws):
            target_prefix = orig_ws[: len(orig_ws) - len(s_ded0_ws)]
        else:
            target_prefix = orig_ws
        rep_lines = _reindent_replace_block(
            replace_block, target_prefix, eol, preserve_no_final_eol
        )

    new_content = "".join(orig_lines[:best_start] + rep_lines + orig_lines[best_end:])
    if validate_ast and target.endswith(".py"):
        try:
            ast.parse(new_content, filename=target)
        except SyntaxError as exc:
            return _TierScanResult(
                status="syntax_error",
                match_lines=(best_start + 1,),
                detail=(
                    f"Tier-4 fuzzy match in {target} (lines {best_start + 1}-{best_end}) "
                    f"produced invalid Python syntax: {exc.msg} at line {exc.lineno}"
                ),
            )

    return _TierScanResult(status="unique", new_content=new_content)


def _best_candidate_window(content: str, search_block: str) -> tuple[float, int, int, list[str]]:
    """Return (score, start_idx, end_idx, orig_raw) for the closest window in `content`."""
    orig_raw = [ln.rstrip("\r\n") for ln in _split_lines_preserve(content)]
    search_raw = [ln.rstrip("\r\n") for ln in _split_lines_preserve(search_block)]
    n = len(orig_raw)
    if n == 0:
        return (0.0, 0, 0, orig_raw)

    k = max(len(search_raw), 1)
    if k > 25 and n > 25:
        matcher = difflib.SequenceMatcher(
            None,
            [ln.strip() for ln in orig_raw],
            [ln.strip() for ln in search_raw],
        )
        for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
            if tag != "equal":
                start = max(0, i1 - 2)
                end = min(n, max(i2 + 2, start + 18))
                return (matcher.ratio(), start, end, orig_raw)
        return (1.0, 0, min(n, 20), orig_raw)

    if n <= k:
        end = min(n, 25)
        score = difflib.SequenceMatcher(
            None,
            "\n".join(ln.strip() for ln in orig_raw[:end]),
            "\n".join(ln.strip() for ln in search_raw),
        ).ratio()
        return (score, 0, end, orig_raw)

    s_norm = "\n".join(ln.strip() for ln in search_raw)
    s_tokens = set(re.findall(r"\w+", s_norm))
    first_s = search_raw[0].strip() if search_raw else ""

    indices = list(range(n - k + 1))
    if len(indices) > 300:
        scored_indices: list[tuple[float, int]] = []
        for idx in indices:
            win = orig_raw[idx : idx + k]
            w_text = " ".join(ln.strip() for ln in win)
            w_tokens = set(re.findall(r"\w+", w_text))
            overlap = len(s_tokens & w_tokens) / max(len(s_tokens | w_tokens), 1)
            bonus = 0.2 if first_s and win[0].strip() == first_s else 0.0
            scored_indices.append((overlap + bonus, idx))
        scored_indices.sort(reverse=True)
        indices = [idx for _, idx in scored_indices[:30]]

    sm = difflib.SequenceMatcher(None, "", s_norm)
    best_score = -1.0
    best_idx = 0
    for idx in indices:
        w_norm = "\n".join(ln.strip() for ln in orig_raw[idx : idx + k])
        sm.set_seq1(w_norm)
        score = sm.ratio()
        if score > best_score:
            best_score = score
            best_idx = idx

    ctx_start = max(0, best_idx - 1)
    ctx_end = min(n, best_idx + k + 1)
    if ctx_end - ctx_start > 25:
        ctx_end = ctx_start + 25
    return (max(best_score, 0.0), ctx_start, ctx_end, orig_raw)


def _format_candidate_feedback(content: str, search_block: str, target: str) -> str:
    """Format a closest-candidate line range snippet with 1-based line numbers."""
    _score, start, end, orig_raw = _best_candidate_window(content, search_block)
    if not orig_raw:
        return f"(target file {target} is empty)"
    snippet = "\n".join(f"{idx + 1:>4}: {orig_raw[idx]}" for idx in range(start, end))
    return f"Closest candidate lines {start + 1}-{end} in {target}:\n{snippet}"


def _file_matches_tier(content: str, search_text: str, tier: int, target: str) -> bool:
    """Return True if `search_text` matches (uniquely or ambiguously) in `content` at `tier`."""
    if tier == 1:
        res = _try_tier1_exact(content, search_text, search_text)
    elif tier == 2:
        res = _try_tier2_rstrip(content, search_text, search_text)
    elif tier == 3:
        res = _try_tier3_uniform_indent(content, search_text, search_text)
    else:
        res = _try_tier4_ast_fuzzy(
            content, search_text, search_text, target=target, validate_ast=False
        )
    return res.status in ("unique", "ambiguous")


def _dedupe_match_keys(paths: list[str]) -> list[str]:
    """Deduplicate path aliases (e.g. 'a.py' vs './a.py') while preserving order."""
    seen: dict[str, str] = {}
    for p in paths:
        cleaned = _clean_rel_path(p)
        seen.setdefault(cleaned, p)
    return list(seen.values())


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
        suffix_matches = _dedupe_match_keys(
            [
                f
                for f in files
                if _clean_rel_path(f) == cleaned or _clean_rel_path(f).endswith(f"/{cleaned}")
            ]
        )
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            for tier in (1, 2, 3, 4):
                tiered = [
                    f
                    for f in suffix_matches
                    if _file_matches_tier(files[f], search_text, tier, target=f)
                ]
                if len(tiered) == 1:
                    return tiered[0]
                if len(tiered) > 1:
                    raise ValueError(
                        f"Ambiguous target filename {explicit_file!r}: matches multiple files "
                        f"({', '.join(sorted(tiered))}); specify the full relative path."
                    )
            raise ValueError(
                f"Ambiguous target filename {explicit_file!r}: matches multiple files "
                f"({', '.join(sorted(suffix_matches))}); specify the full relative path."
            )
        return cleaned

    # Search across tiers in priority order (Tier 1 -> Tier 2 -> Tier 3 -> Tier 4)
    for tier in (1, 2, 3, 4):
        matches = [
            f for f, content in files.items() if _file_matches_tier(content, search_text, tier, f)
        ]
        if not matches:
            continue
        unique_matches = _dedupe_match_keys(matches)
        if len(unique_matches) == 1:
            return unique_matches[0]

        non_test_matches = _dedupe_match_keys([f for f in matches if is_modifiable_py_file(f)])
        if len(non_test_matches) == 1:
            return non_test_matches[0]
        if len(non_test_matches) > 1:
            raise ValueError(
                "Ambiguous SEARCH block: matches multiple files "
                f"({', '.join(sorted(non_test_matches))}); specify a target filename in "
                "'<<<<<<< SEARCH:<file>'."
            )

        # Matched only test or non-.py files: return first match so reward-hacking guard rejects it.
        return unique_matches[0]

    non_test_files = _dedupe_match_keys([f for f in files if is_modifiable_py_file(f)])
    if len(non_test_files) == 1:
        return non_test_files[0]

    return None


def _apply_search_replace_block(
    orig_content: str,
    search_block: str,
    replace_block: str,
    target: str,
) -> str:
    """Apply a single SEARCH/REPLACE block via 4-tier matching with unique-match enforcement."""
    if not search_block or not search_block.strip():
        raise ValueError(
            f"Empty SEARCH block in {target}: SEARCH block must contain at least one "
            "non-whitespace line"
        )

    tier_fns: tuple[Callable[[], _TierScanResult], ...] = (
        lambda: _try_tier1_exact(orig_content, search_block, replace_block),
        lambda: _try_tier2_rstrip(orig_content, search_block, replace_block),
        lambda: _try_tier3_uniform_indent(orig_content, search_block, replace_block),
        lambda: _try_tier4_ast_fuzzy(
            orig_content, search_block, replace_block, target=target, validate_ast=True
        ),
    )

    for run_tier in tier_fns:
        res = run_tier()
        if res.status == "unique":
            return res.new_content
        if res.status == "ambiguous":
            locs = ", ".join(str(ln) for ln in res.match_lines)
            feedback = _format_candidate_feedback(orig_content, search_block, target)
            raise ValueError(
                f"Ambiguous SEARCH block in {target}: {res.detail} (lines {locs}); "
                f"include more surrounding context lines to match uniquely.\n{feedback}"
            )
        if res.status == "syntax_error":
            raise ValueError(res.detail)

    feedback = _format_candidate_feedback(orig_content, search_block, target)
    raise ValueError(f"Search block not found in {target}: {search_block[:50]!r}\n{feedback}")


def _apply_unified_diff_to_mapping(
    diff_text: str,
    base_files: dict[str, str],
) -> dict[str, str]:
    """Apply `diff_text` to a copy of `base_files` in a temporary directory and return result."""
    validate_patch_targets(diff_text)
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        key_by_clean: dict[str, str] = {}
        for rel_path, content in base_files.items():
            clean_rel = _clean_rel_path(rel_path)
            if not clean_rel or clean_rel == "/dev/null":
                continue
            key_by_clean.setdefault(clean_rel, rel_path)
            dest = work_dir / clean_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8")

        try:
            apply_unified_diff(diff_text, work_dir)
        except PatchApplyError as exc:
            raise ValueError(f"Failed to apply unified diff: {exc}") from exc

        updated: dict[str, str] = {}
        for disk_file in work_dir.rglob("*"):
            if disk_file.is_file():
                rel_disk = disk_file.relative_to(work_dir).as_posix()
                orig_key = key_by_clean.get(rel_disk, rel_disk)
                updated[orig_key] = disk_file.read_text(encoding="utf-8")
        return updated


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

        if _SEARCH_HEADER_RE.search(search_block) or _SEARCH_HEADER_RE.search(replace_block):
            raise ValueError("Malformed SEARCH/REPLACE block: nested or unclosed delimiter")
        if _REPLACE_FOOTER_RE.search(search_block) or _REPLACE_FOOTER_RE.search(replace_block):
            raise ValueError("Malformed SEARCH/REPLACE block: nested or unclosed delimiter")

        if not search_block or not search_block.strip():
            raise ValueError(
                "Empty SEARCH block: SEARCH block must contain at least one non-whitespace line"
            )

        target = _find_target_file(search_block, explicit_file, working_files)
        if not target:
            non_test = _dedupe_match_keys([f for f in working_files if is_modifiable_py_file(f)])
            best_feedback = ""
            if non_test:
                scored = [
                    (_best_candidate_window(working_files[f], search_block)[0], f) for f in non_test
                ]
                scored.sort(reverse=True)
                best_file = scored[0][1]
                best_feedback = "\n" + _format_candidate_feedback(
                    working_files[best_file], search_block, best_file
                )
            raise ValueError(
                f"Could not identify target file for search block: {search_block[:50]!r}"
                f"{best_feedback}"
            )

        target_reason = check_forbidden_patch_path(target)
        if target_reason is not None:
            raise ValueError(target_reason)

        if target not in working_files:
            raise ValueError(f"Target file {target!r} not found in repository")

        orig_content = working_files[target]
        new_target_content = _apply_search_replace_block(
            orig_content, search_block, replace_block, target=target
        )
        working_files[target] = new_target_content
        # Keep any path aliases (e.g. './mod.py' vs 'mod.py') in sync in working_files
        clean_target = _clean_rel_path(target)
        for alias_key in list(working_files):
            if alias_key != target and _clean_rel_path(alias_key) == clean_target:
                working_files[alias_key] = new_target_content
        changed_files.add(target)

    # Check if text outside matched SEARCH/REPLACE blocks also contains a unified diff
    remaining_chunks: list[str] = []
    last_end = 0
    for m in matches:
        remaining_chunks.append(text[last_end : m.start()])
        last_end = m.end()
    remaining_chunks.append(text[last_end:])
    remaining_text = "\n".join(remaining_chunks)

    if _SEARCH_HEADER_RE.search(remaining_text) or _REPLACE_FOOTER_RE.search(remaining_text):
        raise ValueError("Malformed SEARCH/REPLACE block: unmatched delimiter in output")

    if re.search(r"^--- \S+", remaining_text, re.MULTILINE) and re.search(
        r"^\+\+\+ \S+", remaining_text, re.MULTILINE
    ):
        validate_patch_targets(remaining_text)
        if re.search(r"^@@ -\d+", remaining_text, re.MULTILINE):
            try:
                updated_from_diff = _apply_unified_diff_to_mapping(remaining_text, working_files)
                for k, v in updated_from_diff.items():
                    orig_norm = files.get(k, "").replace("\r\n", "\n").replace("\r", "\n")
                    if v.replace("\r\n", "\n").replace("\r", "\n") != orig_norm:
                        working_files[k] = v
                        changed_files.add(k)
            except ValueError:
                # Check if the unified diff was a redundant representation of the same change
                updated_from_orig = _apply_unified_diff_to_mapping(remaining_text, files)
                for k, v in updated_from_orig.items():
                    orig_norm = files.get(k, "").replace("\r\n", "\n").replace("\r", "\n")
                    if (
                        v.replace("\r\n", "\n").replace("\r", "\n") != orig_norm
                        and k not in changed_files
                    ):
                        working_files[k] = v
                        changed_files.add(k)

    canonical_diff = ""
    seen_clean_paths: set[str] = set()
    for path in sorted(changed_files):
        reason = check_forbidden_patch_path(path)
        if reason is not None:
            raise ValueError(reason)
        rel_posix = _clean_rel_path(path)
        if rel_posix in seen_clean_paths:
            continue
        seen_clean_paths.add(rel_posix)
        orig_lines = files[path].splitlines(keepends=True) if path in files else []
        new_lines = working_files[path].splitlines(keepends=True)
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

    updated_files = _apply_unified_diff_to_mapping(text, files)
    all_paths = set(files.keys()) | set(updated_files.keys())

    canonical_diff = ""
    seen_clean_paths: set[str] = set()
    for rel_path in sorted(all_paths):
        rel_posix = _clean_rel_path(rel_path)
        if not rel_posix or rel_posix in seen_clean_paths:
            continue
        orig_lines = (
            files[rel_path].replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
            if rel_path in files
            else []
        )
        new_lines = (
            updated_files[rel_path]
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .splitlines(keepends=True)
            if rel_path in updated_files
            else []
        )
        if orig_lines != new_lines:
            reason = check_forbidden_patch_path(rel_path)
            if reason is not None:
                raise ValueError(reason)
            seen_clean_paths.add(rel_posix)
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

    if SEARCH_REPLACE_PATTERN.search(text):
        return normalize_search_replace(text, files)

    if "--- " in text and "+++ " in text:
        return normalize_unified_diff(text, files)

    raise ValueError("Patch text is neither a SEARCH/REPLACE block nor a unified diff")
