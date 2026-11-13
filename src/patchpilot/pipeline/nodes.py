"""LangGraph pipeline nodes: analyze, propose, verify, report."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from patchpilot.pipeline.models import Finding, PipelineState
from patchpilot.pipeline.normalization import (
    is_modifiable_py_file,
    is_test_file,
    normalize_patch,
    validate_patch_targets,
)
from patchpilot.providers.base import LLMProvider, LLMRequest, Message, ProviderError
from patchpilot.retrieval import Embedder, QdrantStore, case_collection_name
from patchpilot.sandbox.runner import SandboxResult, SandboxRunner
from patchpilot.tracing import PatchPilotTracer

logger = logging.getLogger("patchpilot.pipeline")

VerifyCallable = Callable[[dict[str, str], str], Awaitable[SandboxResult]]

DEFAULT_MAX_TOKENS = 2048
LARGE_FILE_LINE_THRESHOLD = 250
MAX_TEST_OUTPUT_CHARS = 4000
MAX_SOURCE_BUDGET_LINES = 250

_PY_TRACEBACK_FRAME_RE = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+([^\s,]+))?')
_PYTEST_FRAME_RE = re.compile(
    r"^([^\s:][^:\n]*\.py):(\d+):(?:\s+in\s+([^\s:]+))?",
    re.MULTILINE,
)
_PYTEST_FAILURE_LINE_RE = re.compile(
    r"^(?:FAILED|ERROR)\s+([^\s:]+)(?:::([^\s]+))?",
    re.MULTILINE,
)
_PYTEST_SECTION_HEADER_RE = re.compile(
    r"^_{4,}\s+([^\s_].*?)\s+_{4,}$",
    re.MULTILINE,
)
_RETRIEVED_HEADER_RE = re.compile(
    r"^---\s+Retrieved:\s+([^:\n]+?)(?:::([^\s(]+))?\s+\(lines\s+(\d+)-(\d+)\)\s+---\n",
    re.MULTILINE,
)

ANALYZE_PROMPT_TEMPLATE: tuple[str, ...] = (
    "You are PatchPilot static analysis and fault localization engine.",
    (
        "Localize the fault across the repository's non-test Python files using "
        "the failing test output and file contents."
    ),
    "For multi-file repositories, choose the non-test .py file that contains the defect.",
    (
        "Inspect the failing test assertions and source comments carefully to identify: "
        "(1) exact SQL parameter placeholder syntax (e.g., '?' vs '%s') and return types "
        "(plain str, tuple), (2) dictionary keys present with explicit None values (where "
        "d.get(k, default) still returns None, so the code must check "
        "'if val is None: val = default'), and (3) which module-level caller function omits "
        "resource cleanup, uses the wrong operator, or has a boundary bug."
    ),
    "Emit findings as lines in format: FINDING: <file>:<line>: <category>: <explanation>",
)

PROPOSE_PROMPT_TEMPLATE: tuple[str, ...] = (
    "You are PatchPilot, an autonomous automated code repair agent.",
    "Fix the bug in the repository's non-test .py file(s) so that all tests pass.",
    (
        "Never modify test files (test_*.py, *_test.py, anything under tests/, "
        "conftest.py) or non-.py files."
    ),
    "Output ONLY raw SEARCH/REPLACE blocks (no ``` markdown code fences) using these exact lines:",
    "<<<<<<< SEARCH:<filename.py>",
    "<exact verbatim lines copied from the file>",
    "=======",
    "<replacement lines>",
    ">>>>>>> REPLACE",
    "CRITICAL RULES:",
    (
        "1. Do NOT use markdown backticks (```) anywhere. Start with literally 7 '<' characters: "
        "'<<<<<<< SEARCH:<filename.py>', then '=======', then '>>>>>>> REPLACE'."
    ),
    (
        '2. Replacing the ENTIRE file from line 1 (starting with the module docstring \'"""...\') '
        "to the last line is the safest way to avoid indentation or missing-comment mismatches "
        "in SEARCH. Otherwise, copy the exact lines from the file with identical leading spaces "
        "and comments (top-level 'def ' functions have 0 leading spaces; lines inside a function "
        "have 4 leading spaces)."
    ),
    (
        "3. Never import third-party libraries (such as psycopg2). For SQL parameterization, "
        "return a plain Python str query using the '?' placeholder and a parameter tuple "
        "(e.g., (val,)), matching the test assertions."
    ),
    (
        "4. NEVER fix a NoneType error by only changing 'd.get(k)' to 'd.get(k, default)', because "
        "in Python '{k: None}.get(k, default)' STILL returns None! Always write an explicit check: "
        "'if val is None: val = default' (or 'd.get(k) or default') before using the value."
    ),
    (
        "5. For resource leaks, ensure the caller function explicitly closes/cleans up/releases "
        "the resource (or wraps usage in a 'with' statement) before returning."
    ),
)


def compute_prompt_version_hash() -> str:
    """Compute a deterministic SHA-256 hash of the active pipeline prompt templates."""
    hasher = hashlib.sha256()
    hasher.update(b"analyze:\n")
    hasher.update("\n".join(ANALYZE_PROMPT_TEMPLATE).encode("utf-8"))
    hasher.update(b"\npropose:\n")
    hasher.update("\n".join(PROPOSE_PROMPT_TEMPLATE).encode("utf-8"))
    return hasher.hexdigest()


def _dedupe_files(files: dict[str, str]) -> dict[str, str]:
    """Collapse path aliases (e.g. `./a.py` vs `a.py`, backslashes) so a file is injected once."""
    seen: dict[str, str] = {}
    for path in sorted(files):
        key = path.replace("\\", "/")
        while key.startswith("./"):
            key = key[2:]
        seen.setdefault(key, path)
    return {key: files[orig] for key, orig in seen.items()}


def _clean_rel_path(path: str) -> str:
    norm = path.replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _resolve_repo_path(raw_path: str, files: Mapping[str, str]) -> str | None:
    """Resolve a path from pytest/traceback output against repository file keys."""
    norm = raw_path.replace("\\", "/").strip()
    if norm.startswith("/work/"):
        norm = norm[len("/work/") :]
    elif norm.startswith("work/"):
        norm = norm[len("work/") :]
    while norm.startswith("./"):
        norm = norm[2:]

    if norm in files:
        return norm

    for candidate in sorted(files, key=len, reverse=True):
        if norm.endswith(f"/{candidate}"):
            return candidate

    base = norm.rsplit("/", 1)[-1]
    base_matches = [c for c in sorted(files) if c.rsplit("/", 1)[-1] == base]
    if len(base_matches) == 1:
        return base_matches[0]
    return None


def _node_line_span(node: ast.stmt) -> tuple[int, int]:
    """Return 1-based `(start_line, end_line)` including decorators."""
    decorators = getattr(node, "decorator_list", [])
    start_line = min([node.lineno] + [d.lineno for d in decorators])
    end_line = getattr(node, "end_lineno", None) or node.lineno
    return (start_line, end_line)


def _enclosing_symbol_at_line(content: str, line_no: int) -> str:
    """Return enclosing `Class.method` or `func` name at 1-based `line_no`, or `''`."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            s, e = _node_line_span(node)
            if s <= line_no <= e:
                return node.name
        elif isinstance(node, ast.ClassDef):
            cs, ce = _node_line_span(node)
            if cs <= line_no <= ce:
                for child in node.body:
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                        ms, me = _node_line_span(child)
                        if ms <= line_no <= me:
                            return f"{node.name}.{child.name}"
                return node.name
    return ""


def extract_traceback_frames(
    output: str,
    files: Mapping[str, str],
) -> list[tuple[str, int, str]]:
    """Extract innermost-first `(file, line_no, func_name)` frames intersecting `files`."""
    if not output or not files:
        return []

    deduped = _dedupe_files(dict(files))
    # Split into per-failure / per-traceback sections so innermost frames are ordered first
    split_positions: list[int] = [0]
    for m in re.finditer(
        r"(?:^Traceback \(most recent call last\):|^_{4,}\s+\S.*?\s+_{4,}$)",
        output,
        re.MULTILINE,
    ):
        if m.start() > 0:
            split_positions.append(m.start())
    split_positions.append(len(output))

    sections = [
        output[split_positions[i] : split_positions[i + 1]] for i in range(len(split_positions) - 1)
    ]

    frames: list[tuple[str, int, str]] = []
    seen: dict[tuple[str, int], int] = {}

    for section in sections:
        raw_hits: list[tuple[int, str, int, str]] = []
        for m in _PY_TRACEBACK_FRAME_RE.finditer(section):
            raw_hits.append((m.start(), m.group(1), int(m.group(2)), m.group(3) or ""))
        for m in _PYTEST_FRAME_RE.finditer(section):
            raw_hits.append((m.start(), m.group(1), int(m.group(2)), m.group(3) or ""))

        raw_hits.sort(key=lambda item: item[0])
        for _, raw_path, line_no, raw_func in reversed(raw_hits):
            resolved = _resolve_repo_path(raw_path, deduped)
            if resolved is None:
                continue
            func = raw_func.strip()
            if not func or func == "<module>":
                inferred = _enclosing_symbol_at_line(deduped[resolved], line_no)
                if inferred:
                    func = inferred
            key = (resolved, line_no)
            if key in seen:
                prev_idx = seen[key]
                if not frames[prev_idx][2] and func:
                    frames[prev_idx] = (resolved, line_no, func)
                continue
            seen[key] = len(frames)
            frames.append((resolved, line_no, func))

    return frames


def summarize_pytest_output(output: str, max_chars: int = MAX_TEST_OUTPUT_CHARS) -> str:
    """Return a concise pytest failure summary when `output` exceeds `max_chars`."""
    stripped = output.strip()
    if not stripped or len(stripped) <= max_chars:
        return output

    lines = stripped.splitlines()
    selected: list[str] = []
    seen_idx: set[int] = set()

    def _add_idx(idx: int) -> None:
        if 0 <= idx < len(lines) and idx not in seen_idx:
            seen_idx.add(idx)
            selected.append(lines[idx])

    # Always keep FAILED/ERROR lines, assertion/exception lines, section headers, and frames
    for idx, line in enumerate(lines):
        s = line.strip()
        if (
            s.startswith(("FAILED ", "ERROR ", "E   ", "> "))
            or _PYTEST_SECTION_HEADER_RE.match(s)
            or _PYTEST_FRAME_RE.match(s)
            or _PY_TRACEBACK_FRAME_RE.search(s)
            or "short test summary info" in s.lower()
        ):
            _add_idx(idx)

    # Keep the final summary lines (e.g. "2 failed, 10 passed in 0.42s")
    for idx in range(max(0, len(lines) - 5), len(lines)):
        _add_idx(idx)

    if not selected:
        head_budget = max(200, max_chars - 60)
        return f"{stripped[:head_budget]}\n... [pytest output truncated]"

    summary = "\n".join(selected)
    if len(summary) <= max_chars:
        return summary

    tail_len = min(800, max_chars // 3)
    head_len = max(200, max_chars - tail_len - 40)
    return f"{summary[:head_len]}\n... [pytest output truncated] ...\n{summary[-tail_len:]}"


def _extract_failing_test_targets(
    failing_test_output: str,
    files: Mapping[str, str],
    traceback_frames: list[tuple[str, int, str]],
) -> tuple[dict[str, set[str]], dict[str, set[int]], set[str]]:
    """Return `(file->func_names, file->line_nos, global_func_names)` from test output."""
    by_file_funcs: dict[str, set[str]] = {}
    by_file_lines: dict[str, set[int]] = {}
    global_funcs: set[str] = set()

    for f_path, ln, func_name in traceback_frames:
        if is_test_file(f_path):
            by_file_lines.setdefault(f_path, set()).add(ln)
            if func_name:
                by_file_funcs.setdefault(f_path, set()).add(func_name.split(".")[-1])

    for m in _PYTEST_FAILURE_LINE_RE.finditer(failing_test_output):
        raw_file = m.group(1)
        node_id = m.group(2) or ""
        resolved = _resolve_repo_path(raw_file, files)
        func = node_id.split("::")[-1].split("[")[0] if node_id else ""
        if resolved is not None:
            by_file_funcs.setdefault(resolved, set())
            if func:
                by_file_funcs[resolved].add(func)
        elif func:
            global_funcs.add(func)

    for m in _PYTEST_SECTION_HEADER_RE.finditer(failing_test_output):
        name = m.group(1).strip().split(".")[-1].split("[")[0]
        if name.startswith("test_") or name:
            global_funcs.add(name)

    return by_file_funcs, by_file_lines, global_funcs


def _render_intervals_with_line_numbers(
    lines: list[str],
    intervals: list[tuple[int, int, str]],
    *,
    max_lines: int,
) -> str:
    """Render sorted `(start, end, label)` 1-based inclusive slices with line-number comments."""
    if not intervals:
        return ""

    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int, str]] = []
    for s, e, label in sorted_iv:
        s_clamped = max(1, s)
        e_clamped = min(len(lines), e)
        if s_clamped > e_clamped:
            continue
        if merged and s_clamped <= merged[-1][1] + 2:
            ps, pe, plabel = merged[-1]
            new_label = plabel if plabel == label else f"{plabel}, {label}"
            merged[-1] = (ps, max(pe, e_clamped), new_label)
        else:
            merged.append((s_clamped, e_clamped, label))

    out_parts: list[str] = []
    used_lines = 0
    prev_end = 0
    for s, e, label in merged:
        span = e - s + 1
        if used_lines > 0 and used_lines + span > max_lines:
            remaining = max_lines - used_lines
            if remaining < 5:
                break
            e = s + remaining - 1
            span = e - s + 1
        if prev_end > 0 and s > prev_end + 1:
            out_parts.append(f"# ... (omitted lines {prev_end + 1}-{s - 1})")
        tag = f"# [lines {s}-{e}: {label}]" if label else f"# [lines {s}-{e}]"
        out_parts.append(tag)
        out_parts.extend(lines[s - 1 : e])
        used_lines += span
        prev_end = e
        if used_lines >= max_lines:
            break

    if prev_end < len(lines):
        out_parts.append(f"# ... (omitted lines {prev_end + 1}-{len(lines)})")
    return "\n".join(out_parts)


def _slice_python_file_by_hits(
    content: str,
    *,
    hit_lines: set[int],
    hit_symbols: set[str],
    hit_ranges: list[tuple[int, int]],
    max_lines: int = 180,
    include_module_header: bool = True,
) -> str:
    """Slice a Python file to module imports/class headers + AST nodes matching hits."""
    lines = content.splitlines()
    if not lines:
        return ""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        intervals: list[tuple[int, int, str]] = []
        for ln in sorted(hit_lines):
            intervals.append((max(1, ln - 15), min(len(lines), ln + 15), f"line {ln}"))
        for rs, re_ in hit_ranges:
            intervals.append((rs, re_, f"retrieved {rs}-{re_}"))
        if not intervals:
            intervals.append((1, min(len(lines), max_lines), "head"))
        return _render_intervals_with_line_numbers(lines, intervals, max_lines=max_lines)

    top_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    ]

    selected_intervals: list[tuple[int, int, str]] = []
    if include_module_header:
        first_def_line = _node_line_span(top_defs[0])[0] if top_defs else len(lines) + 1
        header_end = min(first_def_line - 1, 40)
        if header_end >= 1 and "\n".join(lines[:header_end]).strip():
            selected_intervals.append((1, header_end, "<module>"))

    def _is_hit(start: int, end: int, sym: str) -> bool:
        bare = sym.split(".")[-1]
        if sym in hit_symbols or bare in hit_symbols:
            return True
        if any(start <= ln <= end for ln in hit_lines):
            return True
        return any(not (re_ < start or rs > end) for rs, re_ in hit_ranges)

    for node in top_defs:
        n_start, n_end = _node_line_span(node)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if _is_hit(n_start, n_end, node.name):
                selected_intervals.append((n_start, n_end, node.name))
        else:
            sub_defs = [
                c
                for c in node.body
                if isinstance(c, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            hit_subs: list[tuple[int, int, str]] = []
            for sub in sub_defs:
                s_start, s_end = _node_line_span(sub)
                full_sym = f"{node.name}.{sub.name}"
                if _is_hit(s_start, s_end, full_sym):
                    hit_subs.append((s_start, s_end, full_sym))

            if hit_subs:
                first_sub_start = _node_line_span(sub_defs[0])[0]
                cls_hdr_end = min(first_sub_start - 1, n_start + 15)
                if cls_hdr_end >= n_start:
                    selected_intervals.append((n_start, cls_hdr_end, node.name))
                selected_intervals.extend(hit_subs)
            elif _is_hit(n_start, n_end, node.name):
                if not sub_defs:
                    selected_intervals.append((n_start, n_end, node.name))
                else:
                    # Include class header + first method/init so AST stays valid
                    first_sub = sub_defs[0]
                    init_sub = next(
                        (s for s in sub_defs if getattr(s, "name", "") == "__init__"),
                        first_sub,
                    )
                    i_start, i_end = _node_line_span(init_sub)
                    first_sub_start = _node_line_span(sub_defs[0])[0]
                    cls_hdr_end = min(first_sub_start - 1, n_start + 15)
                    if cls_hdr_end >= n_start:
                        selected_intervals.append((n_start, cls_hdr_end, node.name))
                    selected_intervals.append((i_start, i_end, f"{node.name}.{init_sub.name}"))

    # Module-level lines outside top_defs that were directly hit
    for ln in sorted(hit_lines):
        if not any(s <= ln <= e for s, e, _ in selected_intervals):
            selected_intervals.append(
                (max(1, ln - 10), min(len(lines), ln + 10), f"module_line_{ln}")
            )

    # Fallback if only module header (or nothing) was selected
    has_def_slice = any(lbl != "<module>" for _, _, lbl in selected_intervals)
    if not has_def_slice:
        for node in top_defs[:3]:
            s, e = _node_line_span(node)
            selected_intervals.append((s, e, node.name))
        if not top_defs and not selected_intervals:
            selected_intervals.append((1, min(len(lines), max_lines), "<module>"))

    return _render_intervals_with_line_numbers(lines, selected_intervals, max_lines=max_lines)


def _slice_test_files(
    files: Mapping[str, str],
    failing_test_output: str,
    traceback_frames: list[tuple[str, int, str]],
) -> dict[str, str]:
    """Select concise failing test functions/slices instead of dumping full test files."""
    test_paths = [
        p
        for p in sorted(files)
        if is_test_file(p) and f"{p}:\n{files[p]}" not in failing_test_output
    ]
    if not test_paths:
        return {}

    by_file_funcs, by_file_lines, global_funcs = _extract_failing_test_targets(
        failing_test_output, files, traceback_frames
    )

    sliced: dict[str, str] = {}
    for p in test_paths:
        content = files[p]
        lines = content.splitlines()
        hit_syms = set(by_file_funcs.get(p, set()))
        hit_lines = set(by_file_lines.get(p, set()))

        # Apply global test names (e.g., `____ test_foo ____`) if defined in this file
        for g_sym in global_funcs:
            if g_sym and f"def {g_sym}" in content:
                hit_syms.add(g_sym)

        if hit_syms or hit_lines:
            sliced_text = _slice_python_file_by_hits(
                content,
                hit_lines=hit_lines,
                hit_symbols=hit_syms,
                hit_ranges=[],
                max_lines=100,
                include_module_header=True,
            )
            if sliced_text:
                sliced[p] = sliced_text
        elif p in by_file_funcs or (not by_file_funcs and not by_file_lines and len(lines) <= 60):
            sliced[p] = content
        elif len(test_paths) == 1:
            if len(lines) <= 60:
                sliced[p] = content
            else:
                sliced[p] = _render_intervals_with_line_numbers(
                    lines, [(1, 60, "test_head")], max_lines=60
                )

    return sliced


def _parse_retrieved_chunks(
    retrieved_context: list[str] | None,
    files: Mapping[str, str],
) -> tuple[dict[str, list[tuple[int, int, str]]], list[str]]:
    """Parse retrieved Qdrant chunks into per-file line ranges plus unmapped chunks."""
    by_file: dict[str, list[tuple[int, int, str]]] = {}
    unmapped: list[str] = []
    if not retrieved_context:
        return by_file, unmapped

    for ctx in retrieved_context:
        m = _RETRIEVED_HEADER_RE.match(ctx)
        if m:
            raw_file = m.group(1).strip()
            sym = (m.group(2) or "").strip()
            s_line = int(m.group(3))
            e_line = int(m.group(4))
            resolved = _resolve_repo_path(raw_file, files)
            if resolved is not None:
                by_file.setdefault(resolved, []).append((s_line, e_line, sym))
                continue

        matched_file: str | None = None
        ctx_stripped = ctx.strip()
        if ctx_stripped:
            for p in sorted(files):
                if is_modifiable_py_file(p) and ctx_stripped in files[p]:
                    matched_file = p
                    break
        if matched_file is not None:
            start_idx = files[matched_file].find(ctx_stripped)
            start_line = files[matched_file][:start_idx].count("\n") + 1
            end_line = start_line + max(0, ctx_stripped.count("\n"))
            by_file.setdefault(matched_file, []).append((start_line, end_line, ""))
        else:
            unmapped.append(ctx)

    return by_file, unmapped


def _collect_test_referenced_hits(
    non_test_py_files: list[str],
    failing_test_output: str,
    sliced_tests: Mapping[str, str],
) -> tuple[dict[str, set[str]], set[str]]:
    """Infer referenced source files and symbols from imports/mentions in failing tests."""
    mod_to_file: dict[str, str] = {}
    for p in non_test_py_files:
        no_ext = p[:-3] if p.endswith(".py") else p
        dotted = no_ext.replace("/", ".")
        mod_to_file[dotted] = p
        if dotted.endswith(".__init__"):
            mod_to_file[dotted[: -len(".__init__")]] = p
        leaf = dotted.rsplit(".", 1)[-1]
        mod_to_file.setdefault(leaf, p)

    ref_symbols: dict[str, set[str]] = {}
    ref_files: set[str] = set()

    combined_text = f"{failing_test_output}\n" + "\n".join(sliced_tests.values())
    for p in non_test_py_files:
        if p in combined_text:
            ref_files.add(p)

    for test_src in sliced_tests.values():
        try:
            tree = ast.parse(test_src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                target_file = mod_to_file.get(node.module) or mod_to_file.get(
                    node.module.rsplit(".", 1)[-1]
                )
                if target_file:
                    ref_files.add(target_file)
                    for alias in node.names:
                        if alias.name != "*":
                            ref_symbols.setdefault(target_file, set()).add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    target_file = mod_to_file.get(alias.name) or mod_to_file.get(
                        alias.name.rsplit(".", 1)[-1]
                    )
                    if target_file:
                        ref_files.add(target_file)

    return ref_symbols, ref_files


def _select_source_context(
    files: Mapping[str, str],
    failing_test_output: str,
    findings: list[Finding],
    traceback_frames: list[tuple[str, int, str]],
    retrieved_context: list[str] | None,
    sliced_tests: Mapping[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Build non-duplicative source file contents (full for small single-file, sliced for large)."""
    non_test_py_files = [p for p in sorted(files) if is_modifiable_py_file(p)]
    if not non_test_py_files:
        return ({}, list(retrieved_context or []))

    retrieved_by_file, unmapped_ctx = _parse_retrieved_chunks(retrieved_context, files)

    total_lines = sum(len(files[p].splitlines()) for p in non_test_py_files)
    needs_slicing = (len(non_test_py_files) > 1 and total_lines > 80) or any(
        len(files[p].splitlines()) > LARGE_FILE_LINE_THRESHOLD for p in non_test_py_files
    )

    if not needs_slicing:
        full_sources = {p: files[p] for p in non_test_py_files}
        # Avoid re-injecting retrieved chunks whose code is already present verbatim in full_sources
        deduped_unmapped = [
            ctx
            for ctx in unmapped_ctx
            if not any(ctx.strip() and ctx.strip() in src for src in full_sources.values())
        ]
        return (full_sources, deduped_unmapped)

    tb_lines: dict[str, set[int]] = {}
    tb_syms: dict[str, set[str]] = {}
    priority_order: list[str] = []

    def _prioritize(path: str) -> None:
        if path in files and is_modifiable_py_file(path) and path not in priority_order:
            priority_order.append(path)

    for f_path, ln, func_name in traceback_frames:
        if is_modifiable_py_file(f_path) and f_path in files:
            _prioritize(f_path)
            tb_lines.setdefault(f_path, set()).add(ln)
            if func_name:
                tb_syms.setdefault(f_path, set()).add(func_name)

    finding_lines: dict[str, set[int]] = {}
    for fnd in findings:
        resolved = _resolve_repo_path(fnd.file, files)
        if resolved is not None and is_modifiable_py_file(resolved):
            _prioritize(resolved)
            if fnd.line > 1:
                finding_lines.setdefault(resolved, set()).add(fnd.line)

    for r_path in retrieved_by_file:
        _prioritize(r_path)

    test_ref_syms, test_ref_files = _collect_test_referenced_hits(
        non_test_py_files, failing_test_output, sliced_tests
    )
    for t_path in sorted(test_ref_files):
        _prioritize(t_path)

    if not priority_order:
        priority_order = list(non_test_py_files)

    rendered_sources: dict[str, str] = {}
    remaining_budget = MAX_SOURCE_BUDGET_LINES
    for path in priority_order:
        if remaining_budget < 15 and rendered_sources:
            break
        content = files[path]
        line_count = len(content.splitlines())
        hit_lines = set(tb_lines.get(path, set())) | set(finding_lines.get(path, set()))
        hit_syms = set(tb_syms.get(path, set())) | set(test_ref_syms.get(path, set()))
        hit_ranges = [(s, e) for s, e, _ in retrieved_by_file.get(path, [])]
        for _, _, sym in retrieved_by_file.get(path, []):
            if sym and sym != "<module>":
                hit_syms.add(sym)

        per_file_cap = min(remaining_budget, 180)
        if (
            line_count <= 80
            and line_count <= remaining_budget
            and (hit_lines or hit_syms or hit_ranges)
        ):
            rendered_sources[path] = content
            remaining_budget -= line_count
        else:
            sliced_txt = _slice_python_file_by_hits(
                content,
                hit_lines=hit_lines,
                hit_symbols=hit_syms,
                hit_ranges=hit_ranges,
                max_lines=max(30, per_file_cap),
                include_module_header=True,
            )
            if sliced_txt:
                rendered_sources[path] = sliced_txt
                remaining_budget -= len(sliced_txt.splitlines())

    return (rendered_sources, unmapped_ctx)


def run_ast_parse(files: dict[str, str]) -> list[Finding]:
    """Parse all Python files with ast.parse to detect syntax errors."""
    findings: list[Finding] = []
    for rel_path, content in files.items():
        if not rel_path.endswith(".py"):
            continue
        try:
            ast.parse(content, filename=rel_path)
        except SyntaxError as exc:
            findings.append(
                Finding(
                    file=rel_path.replace("\\", "/"),
                    line=exc.lineno or 1,
                    category="syntax_error",
                    explanation=f"SyntaxError: {exc.msg}",
                    fixable=True,
                )
            )
    return findings


def run_ruff_check(files: dict[str, str]) -> list[Finding]:
    """Run ruff check --output-format json --isolated without loading target repo config."""
    findings: list[Finding] = []
    py_files = {p: c for p, c in files.items() if is_modifiable_py_file(p)}
    if not py_files:
        return findings

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for rel_path, content in py_files.items():
            dest = work_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--output-format",
                    "json",
                    "--isolated",
                    str(work_dir),
                ],
                capture_output=True,
                text=True,
                timeout=15.0,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("Ruff check execution failed: %s", exc)
            return findings

        if not proc.stdout.strip():
            return findings

        try:
            data = json.loads(proc.stdout)
        except ValueError:
            return findings

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                abs_filename = Path(str(item.get("filename", "")))
                try:
                    rel_file = str(abs_filename.relative_to(work_dir)).replace("\\", "/")
                except ValueError:
                    rel_file = abs_filename.name
                loc = item.get("location", {})
                line = int(loc.get("row", 1)) if isinstance(loc, dict) else 1
                code = str(item.get("code", "lint"))
                message = str(item.get("message", ""))
                fixable = bool(item.get("fix"))
                findings.append(
                    Finding(
                        file=rel_file,
                        line=line,
                        category=code,
                        explanation=message,
                        fixable=fixable,
                    )
                )

    return findings


async def run_llm_diff_analysis(
    files: dict[str, str],
    provider: LLMProvider,
    data_class: Any,
    findings: list[Finding],
    failing_test_output: str = "",
    retrieved_context: list[str] | None = None,
    tracer: PatchPilotTracer | None = None,
    parent_span_id: str | None = None,
) -> list[Finding]:
    """Perform LLM fault localization and inspection without ground-truth file hints."""
    files = _dedupe_files(files)
    non_test_py_files = [p for p in sorted(files) if is_modifiable_py_file(p)]
    traceback_frames = extract_traceback_frames(failing_test_output, files)
    source_tb_frames = [f for f in traceback_frames if is_modifiable_py_file(f[0])]
    concise_test_output = summarize_pytest_output(failing_test_output)
    sliced_tests = _slice_test_files(files, failing_test_output, traceback_frames)
    rendered_sources, unmapped_ctx = _select_source_context(
        files=files,
        failing_test_output=failing_test_output,
        findings=findings,
        traceback_frames=traceback_frames,
        retrieved_context=retrieved_context,
        sliced_tests=sliced_tests,
    )

    prompt_lines = list(ANALYZE_PROMPT_TEMPLATE)
    if non_test_py_files:
        prompt_lines.append("Repository non-test .py files:")
        for p in non_test_py_files:
            prompt_lines.append(f"- {p}")
    if concise_test_output:
        prompt_lines.append("Failing test output:")
        prompt_lines.append(concise_test_output)
    if sliced_tests:
        prompt_lines.append("Failing test content:")
        for p, content in sorted(sliced_tests.items()):
            prompt_lines.append(f"--- Test File: {p} ---")
            prompt_lines.append(content)
    if findings:
        prompt_lines.append("Initial static analysis findings:")
        for f in findings:
            prompt_lines.append(f"- {f.file}:{f.line} [{f.category}] {f.explanation}")
    if unmapped_ctx:
        prompt_lines.append("Retrieved context:")
        for ctx in unmapped_ctx:
            prompt_lines.append(ctx)

    prompt_lines.append("Repository non-test .py file contents:")
    for path, content in rendered_sources.items():
        prompt_lines.append(f"--- File: {path} ---")
        prompt_lines.append(content)

    req = LLMRequest(
        messages=[Message(role="user", content="\n".join(prompt_lines))],
        temperature=0.0,
        max_tokens=DEFAULT_MAX_TOKENS,
        data_class=data_class,
    )
    llm_findings: list[Finding] = []
    t0 = time.perf_counter()
    try:
        resp = await provider.generate(req)
        latency_s = time.perf_counter() - t0
        if tracer is not None:
            last_dec = getattr(provider, "last_decision", None)
            tracer.record_llm_call(
                node="analyze",
                model=resp.model,
                provider=resp.provider,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                latency_s=latency_s,
                parent_span_id=parent_span_id,
                route_target=getattr(last_dec, "route_target", ""),
                decision_reason=getattr(last_dec, "decision_reason", ""),
                fallback_used=bool(getattr(last_dec, "fallback_used", False)),
                warning_flag=getattr(last_dec, "warning_flag", ""),
                estimated_cost_usd=float(getattr(last_dec, "estimated_cost_usd", 0.0)),
            )
        for line in resp.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("FINDING:"):
                body = stripped[len("FINDING:") :].strip()
                parts = [p.strip() for p in body.split(":", 3)]
                if len(parts) >= 4:
                    try:
                        llm_findings.append(
                            Finding(
                                file=parts[0],
                                line=int(parts[1]),
                                category=parts[2],
                                explanation=parts[3],
                                fixable=True,
                            )
                        )
                    except ValueError:
                        pass
        if not llm_findings and resp.text.strip() and resp.text != "MOCK_RESPONSE":
            chosen_file = "unknown"
            chosen_line = 1
            for candidate in non_test_py_files:
                if candidate in resp.text:
                    chosen_file = candidate
                    for tb_file, tb_line, _ in source_tb_frames:
                        if tb_file == candidate:
                            chosen_line = tb_line
                            break
                    break
            if chosen_file == "unknown":
                if source_tb_frames:
                    chosen_file, chosen_line, _ = source_tb_frames[0]
                elif non_test_py_files:
                    chosen_file = non_test_py_files[0]
            llm_findings.append(
                Finding(
                    file=chosen_file,
                    line=chosen_line,
                    category="llm_diff_inspection",
                    explanation=resp.text.strip().splitlines()[0][:200],
                    fixable=True,
                )
            )
    except ProviderError as exc:
        logger.warning("LLM diff analysis failed: %s", exc)

    return llm_findings


def create_analyze_node(
    provider: LLMProvider,
    qdrant: QdrantStore | None = None,
    embedder: Embedder | None = None,
    sandbox_runner: SandboxRunner | None = None,
    tracer: PatchPilotTracer | None = None,
) -> Callable[[PipelineState], Awaitable[dict[str, Any]]]:
    """Create the analyze node pure function over PipelineState."""

    async def analyze_node(state: PipelineState) -> dict[str, Any]:
        span_id: str | None = None
        perf_start = 0.0
        iso_start = ""
        if tracer is not None:
            span_id, perf_start, iso_start = tracer.start_node_span(
                "analyze",
                {"case_id": state.case_id, "file_count": len(state.files)},
            )

        # 1. ast.parse
        ast_findings = run_ast_parse(state.files)

        # 2. ruff check --output-format json --isolated
        ruff_findings = run_ruff_check(state.files)

        all_findings = ast_findings + ruff_findings

        # 3. Failing test output (never running untrusted code on the host per SPEC §6)
        failing_test_output = state.failing_test_output
        if not failing_test_output and sandbox_runner is not None:
            try:
                t_sb = time.perf_counter()
                init_res = await sandbox_runner.run(
                    files=state.files,
                    patch_diff="",
                    pytest_args=["-q"],
                )
                sb_dur = time.perf_counter() - t_sb
                raw_out = f"{init_res.stdout}\n{init_res.stderr}".strip()
                if init_res.error and init_res.error not in raw_out:
                    raw_out = f"{raw_out}\nError: {init_res.error}".strip()
                failing_test_output = summarize_pytest_output(raw_out)
                if tracer is not None:
                    tracer.record_sandbox_run(
                        node="analyze",
                        exit_code=init_res.exit_code,
                        duration_s=init_res.duration_s or sb_dur,
                        patch_applied=init_res.patch_applied,
                        passed=init_res.junit.passed,
                        failures=init_res.junit.failures,
                        errors=init_res.junit.errors,
                        container_name=init_res.container_name,
                        image_digest=init_res.image_digest,
                        parent_span_id=span_id,
                    )
            except Exception as exc:
                logger.info("Initial sandbox test run skipped or failed in analyze: %s", exc)

        if not failing_test_output:
            test_snippets: list[str] = []
            for p, c in sorted(state.files.items()):
                if is_test_file(p):
                    c_lines = c.splitlines()
                    if len(c_lines) <= 100:
                        test_snippets.append(f"{p}:\n{c}")
                    else:
                        test_snippets.append(f"{p}:\n" + "\n".join(c_lines[:100]))
            failing_test_output = "\n".join(test_snippets)

        # 4. Retrieval from Qdrant per-case collection if available
        retrieved_context: list[str] = []
        if qdrant is not None and embedder is not None:
            try:
                healthy = await qdrant.is_healthy()
                if healthy:
                    col_name = case_collection_name(state.files)
                    non_test_list = "\n".join(
                        p for p in sorted(state.files) if is_modifiable_py_file(p)
                    )
                    findings_text = "\n".join(
                        f"{f.file}:{f.category}:{f.explanation}" for f in all_findings
                    )
                    concise_test = summarize_pytest_output(failing_test_output, max_chars=1500)
                    query_text = f"{non_test_list}\n{concise_test}\n{findings_text}".strip()
                    if query_text:
                        query_vecs = await embedder.embed([query_text])
                        if query_vecs:
                            hits = await qdrant.search(col_name, query_vecs[0], limit=3)
                            for hit in hits:
                                payload = hit.get("payload")
                                if isinstance(payload, dict) and "text" in payload:
                                    hit_file = str(payload.get("file", "")).strip()
                                    hit_sym = str(payload.get("symbol", "")).strip()
                                    s_line = payload.get("start_line")
                                    e_line = payload.get("end_line")
                                    text = str(payload["text"])
                                    if (
                                        hit_file
                                        and isinstance(s_line, int)
                                        and isinstance(e_line, int)
                                    ):
                                        sym_part = f"::{hit_sym}" if hit_sym else ""
                                        retrieved_context.append(
                                            f"--- Retrieved: {hit_file}{sym_part} "
                                            f"(lines {s_line}-{e_line}) ---\n{text}"
                                        )
                                    else:
                                        retrieved_context.append(text)
            except Exception as exc:
                logger.info("Qdrant retrieval degraded gracefully during analyze: %s", exc)

        # 5. LLM fault localization and diff analysis
        llm_findings = await run_llm_diff_analysis(
            files=state.files,
            provider=provider,
            data_class=state.data_class,
            findings=all_findings,
            failing_test_output=failing_test_output,
            retrieved_context=retrieved_context,
            tracer=tracer,
            parent_span_id=span_id,
        )
        combined_findings = all_findings + llm_findings

        if tracer is not None and span_id is not None:
            tracer.end_node_span(
                span_id,
                "analyze",
                perf_start,
                iso_start,
                {"findings_count": len(combined_findings)},
            )

        return {
            "findings": combined_findings,
            "retrieved_context": retrieved_context,
            "failing_test_output": failing_test_output,
        }

    return analyze_node


def create_propose_node(
    provider: LLMProvider,
    tracer: PatchPilotTracer | None = None,
) -> Callable[[PipelineState], Awaitable[dict[str, Any]]]:
    """Create the propose node pure function over PipelineState."""

    async def propose_node(state: PipelineState) -> dict[str, Any]:
        span_id: str | None = None
        perf_start = 0.0
        iso_start = ""
        if tracer is not None:
            span_id, perf_start, iso_start = tracer.start_node_span(
                "propose",
                {"attempt": state.attempt + 1},
            )

        prompt_files = _dedupe_files(state.files)
        non_test_py_files = [p for p in sorted(prompt_files) if is_modifiable_py_file(p)]
        tb_frames = extract_traceback_frames(state.failing_test_output, prompt_files)
        if state.feedback:
            fb_frames = extract_traceback_frames(state.feedback, prompt_files)
            tb_frames = fb_frames + [f for f in tb_frames if f not in fb_frames]

        concise_test_output = summarize_pytest_output(state.failing_test_output)
        sliced_tests = _slice_test_files(prompt_files, state.failing_test_output, tb_frames)
        rendered_sources, unmapped_ctx = _select_source_context(
            files=prompt_files,
            failing_test_output=state.failing_test_output,
            findings=state.findings,
            traceback_frames=tb_frames,
            retrieved_context=state.retrieved_context,
            sliced_tests=sliced_tests,
        )

        prompt_parts = list(PROPOSE_PROMPT_TEMPLATE)
        if non_test_py_files:
            prompt_parts.append("Candidate non-test .py files:")
            for p in non_test_py_files:
                prompt_parts.append(f"- {p}")
        if concise_test_output:
            prompt_parts.append("Failing test output:")
            prompt_parts.append(concise_test_output)
        if state.findings:
            prompt_parts.append("Identified findings:")
            for f in state.findings:
                prompt_parts.append(f"- {f.file}:{f.line} [{f.category}] {f.explanation}")
        if unmapped_ctx:
            prompt_parts.append("Retrieved repository context:")
            for ctx in unmapped_ctx:
                prompt_parts.append(ctx)
        if state.feedback:
            prompt_parts.append("PREVIOUS ATTEMPT FAILED WITH TEST OUTPUT / ERROR:")
            prompt_parts.append(summarize_pytest_output(state.feedback))
            prompt_parts.append("Please analyze the failure and adjust your fix accordingly.")

        prompt_parts.append("Repository files:")
        for path, content in rendered_sources.items():
            prompt_parts.append(f"--- File: {path} ---")
            prompt_parts.append(content)
        for path, content in sorted(sliced_tests.items()):
            prompt_parts.append(f"--- Test File: {path} ---")
            prompt_parts.append(content)

        req = LLMRequest(
            messages=[Message(role="user", content="\n".join(prompt_parts))],
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS,
            data_class=state.data_class,
        )

        t0 = time.perf_counter()
        try:
            resp = await provider.generate(req)
            latency_s = time.perf_counter() - t0
            raw_patch = resp.text
            if tracer is not None:
                last_dec = getattr(provider, "last_decision", None)
                tracer.record_llm_call(
                    node="propose",
                    model=resp.model,
                    provider=resp.provider,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    latency_s=latency_s,
                    parent_span_id=span_id,
                    route_target=getattr(last_dec, "route_target", ""),
                    decision_reason=getattr(last_dec, "decision_reason", ""),
                    fallback_used=bool(getattr(last_dec, "fallback_used", False)),
                    warning_flag=getattr(last_dec, "warning_flag", ""),
                    estimated_cost_usd=float(getattr(last_dec, "estimated_cost_usd", 0.0)),
                )
        except ProviderError as exc:
            logger.warning("LLM generation failed in propose: %s", exc)
            if tracer is not None and span_id is not None:
                tracer.end_node_span(span_id, "propose", perf_start, iso_start, {"error": str(exc)})
            return {
                "raw_patch": "",
                "patch_diff": "",
                "error": f"LLM error: {exc}",
            }

        try:
            canonical_diff = normalize_patch(raw_patch, state.files)
            if tracer is not None and span_id is not None:
                tracer.end_node_span(
                    span_id,
                    "propose",
                    perf_start,
                    iso_start,
                    {"normalized": bool(canonical_diff.strip())},
                )
            return {
                "raw_patch": raw_patch,
                "patch_diff": canonical_diff,
                "error": None,
            }
        except Exception as exc:
            logger.info("Patch normalization failed: %s", exc)
            if tracer is not None and span_id is not None:
                tracer.end_node_span(span_id, "propose", perf_start, iso_start, {"error": str(exc)})
            return {
                "raw_patch": raw_patch,
                "patch_diff": "",
                "error": f"Normalization error: {exc}",
            }

    return propose_node


def create_verify_node(
    sandbox_runner: SandboxRunner | None = None,
    verify_fn: VerifyCallable | None = None,
    tracer: PatchPilotTracer | None = None,
) -> Callable[[PipelineState], Awaitable[dict[str, Any]]]:
    """Create the verify node pure function over PipelineState."""

    async def verify_node(state: PipelineState) -> dict[str, Any]:
        attempt = state.attempt + 1
        span_id: str | None = None
        perf_start = 0.0
        iso_start = ""
        if tracer is not None:
            span_id, perf_start, iso_start = tracer.start_node_span(
                "verify",
                {"attempt": attempt},
            )

        if not state.patch_diff.strip():
            reason = state.error or "No patch proposed or patch could not be normalized."
            if tracer is not None and span_id is not None:
                tracer.end_node_span(
                    span_id,
                    "verify",
                    perf_start,
                    iso_start,
                    {"verified": False, "reason": reason},
                )
            return {
                "attempt": attempt,
                "patch_applies": False,
                "target_test_passes": False,
                "regressions": 0,
                "verified": False,
                "pass_at_1": False if attempt == 1 else state.pass_at_1,
                "pass_at_k": False if attempt >= state.max_attempts else state.pass_at_k,
                "feedback": f"Attempt {attempt} failed before sandbox execution: {reason}",
            }

        # Defense in depth: enforce reward-hacking guard again inside verify before sandbox run
        try:
            validate_patch_targets(state.patch_diff)
        except ValueError as exc:
            if tracer is not None and span_id is not None:
                tracer.end_node_span(
                    span_id,
                    "verify",
                    perf_start,
                    iso_start,
                    {"verified": False, "rejected": str(exc)},
                )
            return {
                "attempt": attempt,
                "patch_applies": False,
                "target_test_passes": False,
                "regressions": 0,
                "verified": False,
                "pass_at_1": False if attempt == 1 else state.pass_at_1,
                "pass_at_k": False if attempt >= state.max_attempts else state.pass_at_k,
                "feedback": f"Attempt {attempt} rejected by verify reward-hacking guard: {exc}",
            }

        # Run verification in sandbox
        t_sb = time.perf_counter()
        if verify_fn is not None:
            result = await verify_fn(state.files, state.patch_diff)
        else:
            runner = sandbox_runner or SandboxRunner()
            result = await runner.run(
                files=state.files,
                patch_diff=state.patch_diff,
                pytest_args=["-q"],
            )
        sb_dur = time.perf_counter() - t_sb

        if tracer is not None:
            tracer.record_sandbox_run(
                node="verify",
                exit_code=result.exit_code,
                duration_s=result.duration_s or sb_dur,
                patch_applied=result.patch_applied,
                passed=result.junit.passed,
                failures=result.junit.failures,
                errors=result.junit.errors,
                container_name=result.container_name,
                image_digest=result.image_digest,
                parent_span_id=span_id,
            )

        patch_applies = bool(result.patch_applied)
        tests_passed = (
            result.exit_code == 0
            and result.junit.failures == 0
            and result.junit.errors == 0
            and result.junit.passed > 0
            and not result.timed_out
            and not result.error
        )
        regressions = result.junit.failures + result.junit.errors if not tests_passed else 0
        verified = bool(patch_applies and tests_passed and regressions == 0)

        pass_at_1 = state.pass_at_1
        if attempt == 1:
            pass_at_1 = verified

        pass_at_k = state.pass_at_k or verified
        if attempt >= state.max_attempts and not verified:
            pass_at_k = False

        error_msg = result.error
        if result.timed_out and not error_msg:
            error_msg = "Sandbox execution timed out"

        feedback = ""
        if not verified:
            concise_stdout = summarize_pytest_output(result.stdout, max_chars=2500)
            concise_stderr = summarize_pytest_output(result.stderr, max_chars=1500)
            feedback_lines = [
                f"Attempt {attempt} verification failed:",
                f"Patch applied: {patch_applies}",
                f"Exit code: {result.exit_code}",
            ]
            if error_msg:
                feedback_lines.append(f"Error: {error_msg}")
            if concise_stdout:
                feedback_lines.append(f"STDOUT:\n{concise_stdout}")
            if concise_stderr:
                feedback_lines.append(f"STDERR:\n{concise_stderr}")
            if not error_msg and not concise_stdout and not concise_stderr:
                feedback_lines.append("Error: None")
            feedback = "\n".join(feedback_lines)

        if tracer is not None and span_id is not None:
            tracer.end_node_span(
                span_id,
                "verify",
                perf_start,
                iso_start,
                {
                    "verified": verified,
                    "patch_applies": patch_applies,
                    "target_test_passes": tests_passed,
                },
            )

        return {
            "attempt": attempt,
            "patch_applies": patch_applies,
            "target_test_passes": tests_passed,
            "regressions": regressions,
            "verified": verified,
            "pass_at_1": pass_at_1,
            "pass_at_k": pass_at_k,
            "feedback": feedback,
            "error": error_msg if not verified else None,
        }

    return verify_node


def should_retry(state: PipelineState) -> Literal["propose", "report"]:
    """Conditional edge from verify: loop back to propose if attempt < max_attempts."""
    if state.verified:
        return "report"
    if state.attempt < state.max_attempts:
        return "propose"
    return "report"


def create_report_node(
    tracer: PatchPilotTracer | None = None,
) -> Callable[[PipelineState], Awaitable[dict[str, Any]]]:
    """Create the report node pure function over PipelineState."""

    async def _report_node(state: PipelineState) -> dict[str, Any]:
        span_id: str | None = None
        perf_start = 0.0
        iso_start = ""
        if tracer is not None:
            span_id, perf_start, iso_start = tracer.start_node_span(
                "report",
                {"verified": state.verified},
            )

        reported = state.patch_diff if state.verified else ""

        if tracer is not None and span_id is not None:
            tracer.end_node_span(
                span_id,
                "report",
                perf_start,
                iso_start,
                {"reported": bool(reported)},
            )
        return {"reported_patch": reported}

    return _report_node


async def report_node(state: PipelineState) -> dict[str, Any]:
    """Report node: only verified patches are reported."""
    if state.verified:
        return {"reported_patch": state.patch_diff}
    return {"reported_patch": ""}
