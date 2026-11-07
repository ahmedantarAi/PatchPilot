"""Unit tests for multi-file real-world pipeline refinements.

Covers:
1. Two-tier resilient SEARCH/REPLACE matcher (comments, whitespace, if/elif drift, ambiguity).
2. Stack-trace fault localization file extraction.
3. Concise pytest failure extraction for retry feedback.
"""

from __future__ import annotations

import pytest

from patchpilot.pipeline.nodes import (
    _select_priority_files,
    extract_concise_pytest_failure,
    extract_traceback_source_files,
)
from patchpilot.pipeline.normalization import (
    _tier2_fallback_replace,
    normalize_patch,
)

# ---------------------------------------------------------------------------
# 1. Two-Tier Resilient SEARCH/REPLACE Matcher
# ---------------------------------------------------------------------------


def test_tier1_exact_match_preserved() -> None:
    files = {"core.py": "def add(a, b):\n    return a - b\n"}
    raw = "<<<<<<< SEARCH:core.py\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
    diff = normalize_patch(raw, files)
    assert "-    return a - b" in diff
    assert "+    return a + b" in diff


def test_tier2_inline_comment_stripped_match() -> None:
    # Target in file has a trailing inline comment that model omitted in SEARCH block
    files = {
        "docopt.py": (
            "def parse():\n"
            '    pu = printable_usage.split()[1:]  # split and drop "usage:"\n'
            "    return pu\n"
        )
    }
    raw = (
        "<<<<<<< SEARCH:docopt.py\n"
        "    pu = printable_usage.split()[1:]\n"
        "=======\n"
        "    pu = [w for w in printable_usage.split()[1:] if not w.startswith('<')]\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(raw, files)
    assert "-    pu = printable_usage.split()[1:]  # split and drop" in diff
    assert "+    pu = [w for w in printable_usage.split()[1:] if not w.startswith('<')]" in diff


def test_tier2_if_elif_header_drift_multi_line() -> None:
    # Model emitted 'if' instead of 'elif' on a multi-line block (jsonpointer case)
    files = {
        "jsonpointer.py": (
            "def set_pointer(doc, pointer, value):\n"
            "    if isinstance(parent, list):\n"
            "        parent[part] = value\n"
            "    elif isinstance(parent, Sequence) and part == '-':\n"
            "        parent.append(value)\n"
            "    else:\n"
            "        parent[part] = value\n"
        )
    }
    raw = (
        "<<<<<<< SEARCH:jsonpointer.py\n"
        "    if isinstance(parent, Sequence) and part == '-':\n"
        "        parent.append(value)\n"
        "=======\n"
        "    elif isinstance(parent, Sequence) and part == '-':\n"
        "        parent.append(value)\n"
        "        parent.sort()\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(raw, files)
    assert "+        parent.sort()" in diff


def test_tier2_indentation_adjustment() -> None:
    # Model emitted search and replace blocks without indentation (0 spaces instead of 4)
    orig = "def run():\n    val = compute_initial()\n    return val\n"
    search = "val = compute_initial()\nreturn val\n"
    replace = "val = compute_better()\nreturn val\n"
    res = _tier2_fallback_replace(orig, search, replace)
    assert res is not None
    assert "    val = compute_better()\n    return val\n" in res


def test_tier2_ambiguous_match_rejected_fail_closed() -> None:
    # Multiple identical lines in file -> must fail closed to prevent ambiguous corruption
    files = {
        "multi.py": (
            "def f1():\n    return True  # comment 1\ndef f2():\n    return True  # comment 2\n"
        )
    }
    raw = "<<<<<<< SEARCH:multi.py\n    return True\n=======\n    return False\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="Search block not found"):
        normalize_patch(raw, files)


def test_tier2_whitespace_token_collapse() -> None:
    # Minor intra-line whitespace variation (e.g. extra spaces around =)
    orig = "def setup():\n    limit   =   100\n    return limit\n"
    search = "limit = 100\nreturn limit"
    replace = "limit = 200\nreturn limit"
    res = _tier2_fallback_replace(orig, search, replace)
    assert res is not None
    assert "    limit = 200\n" in res


# ---------------------------------------------------------------------------
# 2. Stack-Trace Fault Localization
# ---------------------------------------------------------------------------


def test_extract_traceback_source_files_pytest_style() -> None:
    tb = (
        "tests/test_core.py:45: in test_from_dict\n"
        "    res = from_dict(data_class=User, data={'name': 'alice'})\n"
        "dacite/core.py:68: in from_dict\n"
        "    return _build_value(type_=data_type, data=data, config=config)\n"
        "dacite/core.py:92: in _build_value\n"
        "    raise WrongTypeError()\n"
        "E   dacite.exceptions.WrongTypeError: ...\n"
    )
    files = {
        "tests/test_core.py": "def test_from_dict(): ...",
        "dacite/core.py": "def from_dict(): ...",
        "dacite/types.py": "def is_optional(): ...",
    }
    extracted = extract_traceback_source_files(tb, files, max_files=2)
    # Excludes test_core.py (is_test_file=True), excludes types.py (not in stack trace)
    assert extracted == ["dacite/core.py"]


def test_extract_traceback_source_files_python_tb_style() -> None:
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/workspace/tests/test_parser.py", line 12, in test_run\n'
        '    parse("foo")\n'
        '  File "/workspace/markdown_it/parser_core.py", line 34, in process\n'
        "    state = self.State(src, env)\n"
        "TypeError: State() missing 1 required positional argument\n"
    )
    files = {
        "tests/test_parser.py": "...",
        "markdown_it/parser_core.py": "...",
        "markdown_it/rules.py": "...",
    }
    extracted = extract_traceback_source_files(tb, files, max_files=3)
    assert extracted == ["markdown_it/parser_core.py"]


def test_select_priority_files_small_repo_keeps_all() -> None:
    # Small single or two-file repo (seeded_eval pattern) must keep all files
    files = {
        "mod.py": "def fn(): pass\n",
        "tests/test_mod.py": "def test_fn(): pass\n",
    }
    selected = _select_priority_files(files, failing_test_output="")
    assert selected == ["mod.py"]


def test_select_priority_files_multifile_prioritizes_traceback() -> None:
    files = {f"pkg/sub{i}.py": f"# Submodule {i}\ndef fn{i}(): pass\n" * 100 for i in range(10)}
    tb = "pkg/sub7.py:15: in fn7\n    raise ValueError('oops')"
    selected = _select_priority_files(files, failing_test_output=tb, max_files=2)
    assert "pkg/sub7.py" in selected
    assert len(selected) <= 2


# ---------------------------------------------------------------------------
# 3. Concise Pytest Failure Extraction for Retries
# ---------------------------------------------------------------------------


def test_extract_concise_pytest_failure_failures_block() -> None:
    stdout = (
        "============================= test session starts =============================\n"
        "collecting ... collected 10 items\n"
        "tests/test_a.py .........F\n"
        "=================================== FAILURES ===================================\n"
        "_________________________________ test_split __________________________________\n"
        ">       assert res == ['a', 'b']\n"
        "E       AssertionError: assert ['a'] == ['a', 'b']\n"
        "tests/test_a.py:22: AssertionError\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_a.py::test_split - AssertionError: assert ['a'] == ['a', 'b']\n"
        "========================= 1 failed, 9 passed in 0.23s ==========================\n"
    )
    res = extract_concise_pytest_failure(stdout, "")
    assert "=== FAILURES ===" in res
    assert "AssertionError: assert ['a'] == ['a', 'b']" in res
    assert "test session starts" not in res  # session noise trimmed


def test_extract_concise_pytest_failure_traceback_fallback() -> None:
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/app/repo/cli.py", line 5, in <module>\n'
        "    import missing_pkg\n"
        "ModuleNotFoundError: No module named 'missing_pkg'\n"
    )
    res = extract_concise_pytest_failure("", stderr)
    assert "Traceback (most recent call last):" in res
    assert "ModuleNotFoundError" in res
