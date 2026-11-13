"""Tests for 4-tier safe SEARCH/REPLACE normalization, fence stripping, and candidate feedback."""

from __future__ import annotations

import pytest

from patchpilot.pipeline.normalization import (
    _strip_markdown_code_fences,
    normalize_patch,
)

# ---------------------------------------------------------------------------
# Tier 1: Exact / CRLF with unique-match enforcement
# ---------------------------------------------------------------------------


def test_tier1_crlf_preserves_file_line_endings() -> None:
    files = {
        "win.py": "def a():\r\n    return 1\r\n\r\ndef b():\r\n    return 2\r\n",
    }
    raw = "<<<<<<< SEARCH:win.py\n    return 1\n=======\n    return 10\n>>>>>>> REPLACE"
    diff = normalize_patch(raw, files)
    assert "-    return 1\r\n" in diff
    assert "+    return 10\r\n" in diff
    # Untouched lines must not appear as spurious CRLF->LF changes
    assert "-def b():" not in diff


def test_tier1_rejects_ambiguous_multi_match() -> None:
    files = {
        "dup.py": (
            "def first(x: int) -> int:\n"
            "    return x + 1\n"
            "\n"
            "def second(x: int) -> int:\n"
            "    return x + 1\n"
        ),
    }
    raw = "<<<<<<< SEARCH:dup.py\n    return x + 1\n=======\n    return x + 2\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="Ambiguous SEARCH block"):
        normalize_patch(raw, files)


# ---------------------------------------------------------------------------
# Tier 2: Line-rstrip() match
# ---------------------------------------------------------------------------


def test_tier2_rstrip_ignores_trailing_whitespace_only() -> None:
    files = {
        "mod.py": "def fn():   \n    val = 1\t  \n    return val\n",
    }
    raw = (
        "<<<<<<< SEARCH:mod.py\n"
        "def fn():\n"
        "    val = 1\n"
        "    return val\n"
        "=======\n"
        "def fn():\n"
        "    val = 2\n"
        "    return val\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(raw, files)
    assert "+    val = 2" in diff


def test_tier2_rejects_ambiguous_rstrip_matches() -> None:
    files = {
        "mod.py": "def a():\n    x = 1   \n\ndef b():\n    x = 1\t\n",
    }
    raw = "<<<<<<< SEARCH:mod.py\n    x = 1\n=======\n    x = 2\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="Ambiguous SEARCH block"):
        normalize_patch(raw, files)


# ---------------------------------------------------------------------------
# Tier 3: Uniform indentation-delta & tab/space preservation
# ---------------------------------------------------------------------------


def test_tier3_uniform_indentation_delta_under_and_over_indented() -> None:
    files = {
        "nested.py": (
            "class Calc:\n"
            "    def compute(self, x: int) -> int:\n"
            "        if x < 0:\n"
            "            return 0\n"
            "        return x * 2\n"
        ),
    }
    # Search block dedented by 8 spaces compared to file
    raw_under = (
        "<<<<<<< SEARCH:nested.py\n"
        "if x < 0:\n"
        "    return 0\n"
        "return x * 2\n"
        "=======\n"
        "if x <= 0:\n"
        "    return -1\n"
        "return x * 2\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(raw_under, files)
    assert "-        if x < 0:" in diff
    assert "+        if x <= 0:" in diff
    assert "+            return -1" in diff

    # Search block over-indented by 4 spaces compared to file
    raw_over = (
        "<<<<<<< SEARCH:nested.py\n"
        "            if x < 0:\n"
        "                return 0\n"
        "            return x * 2\n"
        "=======\n"
        "            if x <= 0:\n"
        "                return -1\n"
        "            return x * 2\n"
        ">>>>>>> REPLACE"
    )
    diff_over = normalize_patch(raw_over, files)
    assert "+        if x <= 0:" in diff_over
    assert "+            return -1" in diff_over


def test_tier3_preserves_tabs_and_never_converts_tabs_to_spaces() -> None:
    files = {
        "tabbed.py": (
            "class Worker:\n"
            "\tdef run(self, n: int) -> int:\n"
            "\t\tif n <= 0:\n"
            "\t\t\treturn 0\n"
            "\t\treturn n + 1\n"
        ),
    }
    # Dedented search block using tabs for inner indent
    raw = (
        "<<<<<<< SEARCH:tabbed.py\n"
        "if n <= 0:\n"
        "\treturn 0\n"
        "return n + 1\n"
        "=======\n"
        "if n < 0:\n"
        "\treturn -1\n"
        "return n + 1\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(raw, files)
    assert "+\t\tif n < 0:\n" in diff
    assert "+\t\t\treturn -1\n" in diff
    assert " \t\treturn n + 1\n" in diff

    # Spaces inside nested indent must NOT match tab-indented nested line in Tier 3 (2 lines)
    space_mismatch = (
        "<<<<<<< SEARCH:tabbed.py\n"
        "if n <= 0:\n"
        "    return 0\n"
        "=======\n"
        "if n < 0:\n"
        "    return 0\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Search block not found"):
        normalize_patch(space_mismatch, files)


def test_tier3_requires_at_least_two_non_empty_lines_and_rejects_duplicates() -> None:
    files = {
        "single.py": "def f():\n        return 1\n",
    }
    # Only 1 non-empty line -> Tier 3 must NOT trigger
    one_line = "<<<<<<< SEARCH:single.py\nreturn 1\n=======\nreturn 2\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="Search block not found"):
        normalize_patch(one_line, files)

    # Duplicate windows matching at different indents -> rejected as ambiguous
    files_dup = {
        "dup_indent.py": (
            "def f():\n    x = 1\n    y = 2\n    if True:\n        x = 1\n        y = 2\n"
        ),
    }
    dup_sr = "<<<<<<< SEARCH:dup_indent.py\nx = 1\ny = 2\n=======\nx = 10\ny = 20\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="Ambiguous SEARCH block"):
        normalize_patch(dup_sr, files_dup)


# ---------------------------------------------------------------------------
# Tier 4: AST-anchored fuzzy fallback & invariants
# ---------------------------------------------------------------------------


def test_tier4_ast_anchored_fuzzy_match_succeeds() -> None:
    files = {
        "algo.py": (
            "def normalize_score(values: list[int], scale: int) -> list[int]:\n"
            "    if not values:\n"
            "        return []\n"
            "    total_sum = sum(values)\n"
            "    # compute scaled items\n"
            "    result = [v * scale for v in values]\n"
            "    return result\n"
        ),
    }
    # Slight fuzzy diff in middle comment/line, >= 3 non-empty lines, matching first/last anchors
    raw = (
        "<<<<<<< SEARCH:algo.py\n"
        "def normalize_score(values: list[int], scale: int) -> list[int]:\n"
        "    if not values:\n"
        "        return []\n"
        "    total_sum = sum(values)\n"
        "    result = [v * scale for v in values]\n"
        "    return result\n"
        "=======\n"
        "def normalize_score(values: list[int], scale: int) -> list[int]:\n"
        "    if not values:\n"
        "        return []\n"
        "    total_sum = sum(values) or 1\n"
        "    result = [v * scale // total_sum for v in values]\n"
        "    return result\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(raw, files)
    assert "+    result = [v * scale // total_sum for v in values]" in diff


def test_tier4_rejects_broken_syntax_with_ast_parse() -> None:
    files = {
        "broken.py": (
            "def compute(items: list[int]) -> int:\n"
            "    acc = 0\n"
            "    # sum values\n"
            "    for item in items:\n"
            "        acc += item\n"
            "    return acc\n"
        ),
    }
    raw = (
        "<<<<<<< SEARCH:broken.py\n"
        "def compute(items: list[int]) -> int:\n"
        "    acc = 0\n"
        "    for item in items:\n"
        "        acc += item\n"
        "    return acc\n"
        "=======\n"
        "def compute(items: list[int]) -> int:\n"
        "    acc = (\n"  # syntax error
        "    return acc\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="invalid Python syntax"):
        normalize_patch(raw, files)


def test_tier4_rejects_low_runner_up_margin() -> None:
    files = {
        "ambig.py": (
            "def branch_a(x: int) -> int:\n"
            "    if x > 0:\n"
            "        val = x + 1\n"
            "        extra = val * 2\n"
            "        return extra\n"
            "    return 0\n"
            "\n"
            "def branch_b(x: int) -> int:\n"
            "    if x > 0:\n"
            "        val = x + 2\n"
            "        extra = val * 2\n"
            "        return extra\n"
            "    return 0\n"
        ),
    }
    # Matches both `if x > 0: ... return extra` blocks with very close ratios (margin < 0.10)
    raw = (
        "<<<<<<< SEARCH:ambig.py\n"
        "    if x > 0:\n"
        "        val = x + 1\n"
        "        extra = val * 3\n"
        "        return extra\n"
        "=======\n"
        "    if x > 0:\n"
        "        val = x + 10\n"
        "        return val\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Ambiguous SEARCH block"):
        normalize_patch(raw, files)


# ---------------------------------------------------------------------------
# Invariants: if vs elif, string whitespace, # comments, reward-hacking guard
# ---------------------------------------------------------------------------


def test_invariant_never_equates_if_with_elif() -> None:
    files = {
        "cond.py": (
            "def classify(x: int) -> str:\n"
            "    if x > 10:\n"
            "        return 'high'\n"
            "    elif x > 0:\n"
            "        return 'low'\n"
            "    return 'zero'\n"
        ),
    }
    # SEARCH replaces `elif` with `if` on an inner line -> must NOT match even in Tier 4
    raw = (
        "<<<<<<< SEARCH:cond.py\n"
        "def classify(x: int) -> str:\n"
        "    if x > 10:\n"
        "        return 'high'\n"
        "    if x > 0:\n"
        "        return 'low'\n"
        "    return 'zero'\n"
        "=======\n"
        "def classify(x: int) -> str:\n"
        "    return 'fixed'\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Search block not found"):
        normalize_patch(raw, files)


def test_invariant_never_collapses_string_whitespace_or_strips_comments() -> None:
    files = {
        "strings.py": 'def msg():\n    return "alpha   beta"\n',
        "comments.py": "def calc(x):\n    y = x + 1  # keep comment\n    return y\n",
    }
    # Collapsing internal whitespace in string literal must NOT match
    str_sr = (
        "<<<<<<< SEARCH:strings.py\n"
        "def msg():\n"
        '    return "alpha beta"\n'
        "=======\n"
        "def msg():\n"
        '    return "fixed"\n'
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Search block not found"):
        normalize_patch(str_sr, files)

    # Stripping `# keep comment` on a 2-line block must NOT match Tiers 1-3 (<3 skips Tier 4)
    cmt_sr = (
        "<<<<<<< SEARCH:comments.py\n"
        "    y = x + 1\n"
        "    return y\n"
        "=======\n"
        "    y = x + 2\n"
        "    return y\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Search block not found"):
        normalize_patch(cmt_sr, files)


def test_invariant_reward_hacking_guard_still_blocks_forbidden_implicit_tier3() -> None:
    files = {
        "app.py": "def run():\n    return 1\n",
        "tests/test_app.py": "def test_run():\n    x = 1\n    assert x == 1\n",
    }
    # Implicit Tier-3 block matching inside tests/test_app.py must still trigger guard
    raw = "<<<<<<< SEARCH\nx = 1\nassert x == 1\n=======\nx = 2\nassert x == 2\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="Reward-hacking guard"):
        normalize_patch(raw, files)


# ---------------------------------------------------------------------------
# Multi-file target resolution & ambiguity
# ---------------------------------------------------------------------------


def test_multi_file_target_resolution_across_tiers_and_ambiguity() -> None:
    files = {
        "alpha.py": "def alpha_fn():\n    if True:\n        a = 1\n        b = 2\n",
        "beta.py": "def beta_fn():\n    if True:\n        a = 1\n        b = 2\n",
    }
    # Implicit block matching both alpha.py and beta.py must raise ambiguous error
    ambig_sr = (
        "<<<<<<< SEARCH\n"
        "        a = 1\n"
        "        b = 2\n"
        "=======\n"
        "        a = 9\n"
        "        b = 2\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="Ambiguous SEARCH block: matches multiple files"):
        normalize_patch(ambig_sr, files)

    # Implicit Tier-3 block uniquely matching gamma.py resolves to gamma.py
    files_unique = {
        **files,
        "gamma.py": (
            "def gamma_fn():\n    if True:\n        unique_var = 42\n        return unique_var\n"
        ),
    }
    uniq_sr = (
        "<<<<<<< SEARCH\n"
        "unique_var = 42\n"
        "return unique_var\n"
        "=======\n"
        "unique_var = 99\n"
        "return unique_var\n"
        ">>>>>>> REPLACE"
    )
    diff = normalize_patch(uniq_sr, files_unique)
    assert "--- a/gamma.py\n+++ b/gamma.py\n" in diff
    assert "+        unique_var = 99" in diff


# ---------------------------------------------------------------------------
# Prose-preamble fence stripping & candidate line feedback
# ---------------------------------------------------------------------------


def test_prose_preamble_fence_stripping() -> None:
    files = {"mod.py": "x = 1\ny = 2\n"}

    # Prose preamble + non-patch example fence + SEARCH/REPLACE fence + trailing prose
    sr_with_prose = (
        "I found the bug in `mod.py`. Avoid doing:\n"
        "```python\n"
        "x = 0\n"
        "```\n"
        "Here is the fix:\n\n"
        "```python\n"
        "<<<<<<< SEARCH:mod.py\n"
        "x = 1\n"
        "=======\n"
        "x = 10\n"
        ">>>>>>> REPLACE\n"
        "```\n"
        "This updates x cleanly.\n"
    )
    stripped_sr = _strip_markdown_code_fences(sr_with_prose)
    assert "I found the bug" not in stripped_sr
    assert "x = 0" not in stripped_sr
    assert "<<<<<<< SEARCH:mod.py" in stripped_sr
    diff_sr = normalize_patch(sr_with_prose, files)
    assert "+x = 10" in diff_sr

    # Prose preamble + unified diff fence
    udiff_with_prose = (
        "Sure, here is the unified diff:\n"
        "```diff\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-x = 1\n"
        "+x = 20\n"
        " y = 2\n"
        "```\n"
    )
    diff_u = normalize_patch(udiff_with_prose, files)
    assert "+x = 20" in diff_u


def test_candidate_line_feedback_on_failed_search() -> None:
    files = {
        "target.py": (
            "def first():\n"
            "    return 1\n"
            "\n"
            "def second(alpha: int, beta: int) -> int:\n"
            "    total = alpha + beta\n"
            "    return total * 2\n"
            "\n"
            "def third():\n"
            "    return 3\n"
        ),
    }
    bad_sr = (
        "<<<<<<< SEARCH:target.py\n"
        "def second(alpha: int, beta: int) -> int:\n"
        "    total = alpha - beta + 999\n"
        "    return total * 999\n"
        "=======\n"
        "def second(alpha: int, beta: int) -> int:\n"
        "    return alpha + beta\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError) as exc_info:
        normalize_patch(bad_sr, files)

    msg = str(exc_info.value)
    assert "Search block not found in target.py" in msg
    assert "Closest candidate lines" in msg
    assert "4: def second" in msg
    assert "5:     total = alpha + beta" in msg
