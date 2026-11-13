"""Formatting of findings and verified patches into GitHub PR review comments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patchpilot.pipeline.models import Finding
from patchpilot.review.github import PostedReviewComment

_HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


@dataclass(frozen=True)
class DiffSuggestion:
    """One contiguous change group of a unified diff, mapped to head-file (`RIGHT`) lines.

    The verified patch is generated against the PR head content, so its `-`/context line
    numbers are exactly the line numbers of the PR's new file (GitHub `side="RIGHT"`).
    `start_line..end_line` is the inclusive range that `replacement_code` replaces.
    """

    file: str
    start_line: int
    end_line: int
    replacement_code: str
    raw_hunk: str


@dataclass
class _Hunk:
    file: str
    old_start: int
    lines: list[str] = field(default_factory=list)


def _parse_hunks(patch_diff: str) -> list[_Hunk]:
    """Parse every hunk of every file in a unified diff (tolerant of inexact hunk counts)."""
    hunks: list[_Hunk] = []
    lines = patch_diff.splitlines()
    target_file = ""
    current: _Hunk | None = None
    old_left = new_left = 0

    for idx, line in enumerate(lines):
        if line.startswith("diff --git "):
            current = None
            target_file = ""
            continue
        if (
            line.startswith("--- ")
            and idx + 2 < len(lines)
            and lines[idx + 1].startswith("+++ ")
            and lines[idx + 2].startswith("@@")
        ):
            current = None
            continue
        if (
            line.startswith("+++ ")
            and idx > 0
            and lines[idx - 1].startswith("--- ")
            and idx + 1 < len(lines)
            and lines[idx + 1].startswith("@@")
        ):
            raw_path = line[4:].split("\t", 1)[0].strip()
            if raw_path.startswith("b/"):
                raw_path = raw_path[2:]
            if raw_path != "/dev/null":
                target_file = raw_path
            current = None
            continue
        match = _HUNK_HEADER_RE.match(line)
        if match:
            old_left = int(match.group(2)) if match.group(2) is not None else 1
            new_left = int(match.group(4)) if match.group(4) is not None else 1
            current = _Hunk(file=target_file, old_start=int(match.group(1)), lines=[line])
            hunks.append(current)
            continue
        if current is None:
            continue
        current.lines.append(line)
        if line.startswith("-"):
            old_left -= 1
        elif line.startswith("+"):
            new_left -= 1
        elif line.startswith("\\"):
            pass
        else:
            old_left -= 1
            new_left -= 1
    return [h for h in hunks if h.file]


def _suggestions_for_hunk(hunk: _Hunk) -> list[DiffSuggestion]:
    raw_hunk = "\n".join(hunk.lines)
    out: list[DiffSuggestion] = []
    old_no = hunk.old_start
    prev_ctx: tuple[int, str] | None = None
    removed: list[int] = []
    added: list[str] = []

    def flush(next_ctx: tuple[int, str] | None) -> None:
        nonlocal removed, added
        if removed:
            start, end, repl = removed[0], removed[-1], list(added)
        elif added and prev_ctx is not None:
            start = end = prev_ctx[0]
            repl = [prev_ctx[1], *added]
        elif added and next_ctx is not None:
            start = end = next_ctx[0]
            repl = [*added, next_ctx[1]]
        else:
            removed, added = [], []
            return
        out.append(DiffSuggestion(hunk.file, start, end, "\n".join(repl), raw_hunk))
        removed, added = [], []

    for hline in hunk.lines[1:]:
        if hline.startswith("-"):
            removed.append(old_no)
            old_no += 1
        elif hline.startswith("+"):
            added.append(hline[1:])
        elif hline.startswith("\\"):
            continue
        else:
            ctx = (old_no, hline[1:] if hline.startswith(" ") else hline)
            if removed or added:
                flush(ctx)
            prev_ctx = ctx
            old_no += 1
    if removed or added:
        flush(None)
    return out


def extract_diff_suggestions(patch_diff: str) -> list[DiffSuggestion]:
    """Return one suggestion per contiguous change group across ALL hunks and files."""
    if not patch_diff.strip():
        return []
    suggestions: list[DiffSuggestion] = []
    for hunk in _parse_hunks(patch_diff):
        suggestions.extend(_suggestions_for_hunk(hunk))
    return suggestions


def extract_diff_suggestion(patch_diff: str) -> tuple[str, int, str, str] | None:
    """Return `(target_file, changed_line, replacement_code, raw_hunk)` for the first change."""
    suggestions = extract_diff_suggestions(patch_diff)
    if not suggestions:
        return None
    first = suggestions[0]
    return (first.file, first.start_line, first.replacement_code, first.raw_hunk)


def format_verified_patch_comment(
    *,
    file: str,
    line: int,
    finding: Finding | None,
    replacement_code: str,
    raw_hunk: str,
    start_line: int | None = None,
) -> PostedReviewComment:
    """Format a verified patch as a GitHub PR suggestion block (` ```suggestion `)."""
    cat = finding.category if finding is not None else "verified_fix"
    explanation = (
        finding.explanation
        if finding is not None
        else "Automated fix verified in sandbox pytest suite."
    )
    span = f"{start_line}-{line}" if start_line is not None and start_line < line else f"{line}"
    suggestion_lines = ["```suggestion", *([replacement_code] if replacement_code else []), "```"]
    body_lines = [
        f"**PatchPilot Verified Fix** (`{cat}` at `{file}:{span}`)",
        "",
        explanation,
        "",
        *suggestion_lines,
        "",
        "<details><summary>Verified Unified Diff Hunk</summary>",
        "",
        "```diff",
        raw_hunk,
        "```",
        "</details>",
    ]
    end_line = max(1, line)
    begin = max(1, start_line) if start_line is not None else None
    return PostedReviewComment(
        path=file,
        line=end_line,
        start_line=begin if begin is not None and begin < end_line else None,
        side="RIGHT",
        body="\n".join(body_lines),
        verified_suggestion=True,
    )


def format_unverified_finding_comment(finding: Finding) -> PostedReviewComment:
    """Format an unverified finding as a descriptive risk comment without a code suggestion."""
    body_lines = [
        f"**PatchPilot Finding** (`{finding.category}` at `{finding.file}:{finding.line}`)",
        "",
        f"Risk / Analysis: {finding.explanation}",
        "",
        "_Note: No sandbox-verified code suggestion is attached for this finding._",
    ]
    return PostedReviewComment(
        path=finding.file,
        line=max(1, finding.line),
        body="\n".join(body_lines),
        verified_suggestion=False,
    )


def format_review_comments(
    findings: list[Finding],
    reported_patch: str,
) -> list[PostedReviewComment]:
    """Build the list of PR review comments from findings and any verified patch."""
    comments: list[PostedReviewComment] = []
    suggestions = extract_diff_suggestions(reported_patch)
    used: set[int] = set()

    assigned: dict[int, int] = {}
    # Pass 1: findings whose line falls inside a suggestion's range
    for si, sug in enumerate(suggestions):
        for i, f in enumerate(findings):
            if i not in used and f.file == sug.file and sug.start_line <= f.line <= sug.end_line:
                assigned[si] = i
                used.add(i)
                break
    # Pass 2: fall back to the primary (non-lint) finding on the same file, then any
    for si, sug in enumerate(suggestions):
        if si in assigned:
            continue
        candidates = [i for i, f in enumerate(findings) if i not in used and f.file == sug.file]
        preferred = [i for i in candidates if findings[i].category not in ("F401", "E501")]
        pool = preferred or candidates
        if pool:
            chosen = pool[0]
            assigned[si] = chosen
            used.add(chosen)

    for si, sug in enumerate(suggestions):
        idx = assigned.get(si)
        comments.append(
            format_verified_patch_comment(
                file=sug.file,
                line=sug.end_line,
                start_line=sug.start_line,
                finding=findings[idx] if idx is not None else None,
                replacement_code=sug.replacement_code,
                raw_hunk=sug.raw_hunk,
            )
        )

    for idx, f in enumerate(findings):
        if idx in used:
            continue
        comments.append(format_unverified_finding_comment(f))

    return comments
