"""Lightweight static feature extraction for dynamic model routing."""

from __future__ import annotations

import ast
import re
from typing import Literal

from pydantic import BaseModel, Field

from patchpilot.providers.base import DataClass, LLMRequest

_FILE_HEADER_RE = re.compile(r"^---\s+(?:Test\s+)?File:\s+([^\s-]+)\s+---\s*$", re.MULTILINE)
_ERROR_TYPE_RE = re.compile(
    r"\b(AssertionError|AttributeError|TypeError|KeyError|IndexError|ValueError|"
    r"SyntaxError|NameError|RuntimeError|ZeroDivisionError|TimeoutError)\b"
)


class TaskFeatures(BaseModel):
    """Static task features extracted from an `LLMRequest` or repository files."""

    data_class: DataClass = "private"
    node_kind: Literal["analyze", "propose", "unknown"] = "unknown"
    repo_file_count: int = 0
    non_test_file_count: int = 0
    ast_node_count: int = 0
    max_ast_depth: int = 0
    function_count: int = 0
    estimated_prompt_tokens: int = 0
    failing_error_type: str = "unknown"
    predicted_taxonomy: str = "unknown"
    matched_weak_patterns: list[str] = Field(default_factory=list)
    is_retry: bool = False
    retry_attempt: int = 1
    low_confidence_signal: bool = False


def _compute_ast_stats(source: str) -> tuple[int, int, int]:
    """Return `(node_count, max_depth, function_count)` for Python source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return (0, 0, 0)

    node_count = 0
    func_count = 0

    def _walk_depth(node: ast.AST, depth: int) -> int:
        nonlocal node_count, func_count
        node_count += 1
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_count += 1
        max_d = depth
        for child in ast.iter_child_nodes(node):
            child_d = _walk_depth(child, depth + 1)
            if child_d > max_d:
                max_d = child_d
        return max_d

    max_depth = _walk_depth(tree, 1)
    return (node_count, max_depth, func_count)


def _extract_embedded_files(text: str) -> dict[str, str]:
    """Extract embedded `--- File: <path> ---` sections from a pipeline prompt."""
    matches = list(_FILE_HEADER_RE.finditer(text))
    if not matches:
        return {}

    extracted: dict[str, str] = {}
    for idx, match in enumerate(matches):
        path = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        snippet = text[start:end].strip()
        extracted[path] = snippet
    return extracted


def _is_test_path(path: str) -> bool:
    norm = path.replace("\\", "/").lower()
    name = norm.rsplit("/", 1)[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or "/tests/" in f"/{norm}"
    )


def _detect_taxonomy_and_weak_patterns(text: str, code_text: str) -> tuple[str, list[str]]:
    """Detect bug taxonomy class and known local-model weak-spot patterns."""
    combined_lower = f"{text}\n{code_text}".lower()
    code_lower = code_text.lower()
    weak_patterns: list[str] = []

    # 1. Check complex SQL injection signals (e.g. login_query, UNION SELECT, raw SQL f-strings)
    has_sql = any(
        kw in combined_lower
        for kw in (
            "select ",
            "sql_injection",
            "build_login_query",
            "build_audit_query",
            "union select",
            "where username =",
            "where user_id =",
        )
    )
    if has_sql:
        if "login_query" in combined_lower or "build_login_query" in combined_lower:
            weak_patterns.append("sql_injection:login_query")
        if "union select" in combined_lower or "pass_hash" in combined_lower:
            weak_patterns.append("sql_injection:complex_union_auth")
        if re.search(r"f[\"']select\s+.*\{", code_lower):
            weak_patterns.append("sql_injection:fstring_query")

    # 2. Check None-handling weak spots (specifically profile_display_name / explicit None values)
    has_none_bug = any(
        kw in combined_lower
        for kw in (
            "none_handling",
            "format_user_display_name",
            "profile_display_name",
            "'nonetype' object has no attribute",
            "'nonetype' object is not subscriptable",
            "'nonetype' object is not callable",
            '"full_name": none',
            "'full_name': none",
            '"tag": none',
            "'tag': none",
            '"author": none',
            "'author': none",
            '"timeout": none',
            "'timeout': none",
            '"x-scope": none',
            "'x-scope': none",
            "callback=none",
            "attributeserror when",
            "attributeerror when",
            "typeerror when",
        )
    )
    if has_none_bug:
        if (
            "format_user_display_name" in combined_lower
            or "profile_display_name" in combined_lower
            or ("full_name" in combined_lower and "username" in combined_lower)
        ):
            weak_patterns.append("none_handling:profile_display_name")
        if ".strip()" in code_lower and (
            '["full_name"]' in code_lower or "['full_name']" in code_lower
        ):
            weak_patterns.append("none_handling:unconditional_strip_on_nullable")
        if (
            "'nonetype' object has no attribute" in combined_lower
            or "'nonetype' object is not subscriptable" in combined_lower
            or "attributeerror when" in combined_lower
            or "typeerror when" in combined_lower
        ):
            weak_patterns.append("none_handling:nonetype_deref")

    # Predict primary taxonomy
    if has_sql and ("select " in code_lower or "sql_injection" in combined_lower):
        predicted_taxonomy = "sql_injection"
    elif has_none_bug:
        predicted_taxonomy = "none_handling"
    elif (
        "missing_await" in combined_lower
        or "coroutine" in combined_lower
        or "async def " in code_lower
    ):
        predicted_taxonomy = "missing_await"
    elif (
        "resource_leak" in combined_lower
        or ".close()" in combined_lower
        or "cleanup" in combined_lower
    ):
        predicted_taxonomy = "resource_leak"
    elif (
        "bad_default_arg" in combined_lower
        or "def " in code_lower
        and ("=[]" in code_lower.replace(" ", "") or "={}" in code_lower.replace(" ", ""))
    ):
        predicted_taxonomy = "bad_default_arg"
    elif (
        "off_by_one" in combined_lower or "paginate" in combined_lower or "offset" in combined_lower
    ):
        predicted_taxonomy = "off_by_one"
    elif "wrong_operator" in combined_lower:
        predicted_taxonomy = "wrong_operator"
    else:
        predicted_taxonomy = "unknown"

    return predicted_taxonomy, sorted(set(weak_patterns))


def extract_task_features(
    request: LLMRequest,
    *,
    files: dict[str, str] | None = None,
    failing_test_output: str = "",
) -> TaskFeatures:
    """Extract lightweight static features from an `LLMRequest` and optional repo context."""
    full_prompt = "\n".join(m.content for m in request.messages)
    prompt_lower = full_prompt.lower()

    # Determine pipeline node kind
    if "static analysis and fault localization engine" in prompt_lower:
        node_kind: Literal["analyze", "propose", "unknown"] = "analyze"
    elif (
        "autonomous automated code repair agent" in prompt_lower
        or "<<<<<<< search:" in prompt_lower
    ):
        node_kind = "propose"
    else:
        node_kind = "unknown"

    # Extract embedded files if not explicitly passed
    repo_files = dict(files) if files is not None else _extract_embedded_files(full_prompt)

    total_ast_nodes = 0
    max_depth = 0
    total_funcs = 0
    non_test_count = 0
    py_code_blobs: list[str] = []

    if repo_files:
        for path, content in repo_files.items():
            if not path.endswith(".py"):
                continue
            if not _is_test_path(path):
                non_test_count += 1
                py_code_blobs.append(content)
                nodes, depth, funcs = _compute_ast_stats(content)
                total_ast_nodes += nodes
                if depth > max_depth:
                    max_depth = depth
                total_funcs += funcs
    else:
        # Fallback: try parsing the prompt or code blocks inside the prompt directly
        nodes, depth, funcs = _compute_ast_stats(full_prompt)
        total_ast_nodes = nodes
        max_depth = depth
        total_funcs = funcs
        if nodes > 0:
            non_test_count = 1
            py_code_blobs.append(full_prompt)

    # Detect retry / cascade signals from prompt feedback
    is_retry = "previous attempt failed with test output / error:" in prompt_lower
    retry_attempt = 1
    if is_retry:
        attempt_matches = re.findall(r"attempt\s+(\d+)\s+(?:verification\s+)?failed", prompt_lower)
        if attempt_matches:
            try:
                retry_attempt = int(attempt_matches[-1]) + 1
            except ValueError:
                retry_attempt = 2
        else:
            retry_attempt = 2

    # Detect failing error type
    search_error_text = f"{failing_test_output}\n{full_prompt}"
    err_match = _ERROR_TYPE_RE.search(search_error_text)
    failing_error_type = err_match.group(1) if err_match else "unknown"

    # Detect taxonomy and weak spots
    code_joined = "\n".join(py_code_blobs) if py_code_blobs else full_prompt
    predicted_taxonomy, weak_patterns = _detect_taxonomy_and_weak_patterns(full_prompt, code_joined)

    # Low-confidence signal: retry after failed verification or normalization error
    low_confidence = bool(
        is_retry
        or "normalization error:" in prompt_lower
        or "failed before sandbox execution" in prompt_lower
        or len(weak_patterns) > 0
    )

    # Lightweight token estimation (~4 chars per token or word-based bound)
    word_count = len(full_prompt.split())
    char_est = max(1, len(full_prompt) // 4)
    estimated_tokens = max(word_count, char_est)

    return TaskFeatures(
        data_class=request.data_class,
        node_kind=node_kind,
        repo_file_count=len(repo_files) if repo_files else (1 if total_ast_nodes > 0 else 0),
        non_test_file_count=non_test_count,
        ast_node_count=total_ast_nodes,
        max_ast_depth=max_depth,
        function_count=total_funcs,
        estimated_prompt_tokens=estimated_tokens,
        failing_error_type=failing_error_type,
        predicted_taxonomy=predicted_taxonomy,
        matched_weak_patterns=weak_patterns,
        is_retry=is_retry,
        retry_attempt=retry_attempt,
        low_confidence_signal=low_confidence,
    )
