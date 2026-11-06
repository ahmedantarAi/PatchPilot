"""Formatting of findings and verified patches into GitHub PR review comments."""

from __future__ import annotations

import re

from patchpilot.pipeline.models import Finding
from patchpilot.review.github import PostedReviewComment

_HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def extract_diff_suggestion(patch_diff: str) -> tuple[str, int, str, str] | None:
    """Extract `(target_file, changed_line, replacement_code, raw_hunk)` from a unified diff."""
    if not patch_diff.strip():
        return None

    target_file = ""
    lines = patch_diff.splitlines()
    hunk_lines: list[str] = []
    in_hunk = False
    old_start = 1

    for line in lines:
        if line.startswith("+++ "):
            raw_path = line[4:].strip()
            if raw_path.startswith("b/"):
                raw_path = raw_path[2:]
            target_file = raw_path
            continue
        if line.startswith("--- "):
            continue
        match = _HUNK_HEADER_RE.match(line)
        if match:
            if in_hunk and hunk_lines:
                break
            in_hunk = True
            old_start = int(match.group(1))
            hunk_lines.append(line)
            continue
        if in_hunk:
            if line.startswith("diff --git "):
                break
            hunk_lines.append(line)

    if not target_file or not hunk_lines:
        return None

    # Find the specific modified line range (+ / - lines inside the hunk)
    cur_line = old_start
    first_changed_line: int | None = None
    added_lines: list[str] = []

    for hline in hunk_lines[1:]:
        if hline.startswith("-"):
            if first_changed_line is None:
                first_changed_line = cur_line
            cur_line += 1
        elif hline.startswith("+"):
            if first_changed_line is None:
                first_changed_line = cur_line
            added_lines.append(hline[1:])
        elif hline.startswith(" "):
            if first_changed_line is not None and added_lines:
                break
            cur_line += 1

    effective_line = first_changed_line if first_changed_line is not None else old_start
    replacement_code = "\n".join(added_lines)
    raw_hunk = "\n".join(hunk_lines)
    return (target_file, effective_line, replacement_code, raw_hunk)


def format_verified_patch_comment(
    *,
    file: str,
    line: int,
    finding: Finding | None,
    replacement_code: str,
    raw_hunk: str,
) -> PostedReviewComment:
    """Format a verified patch as a GitHub PR suggestion block (` ```suggestion `)."""
    cat = finding.category if finding is not None else "verified_fix"
    explanation = (
        finding.explanation
        if finding is not None
        else "Automated fix verified in sandbox pytest suite."
    )
    body_lines = [
        f"**PatchPilot Verified Fix** (`{cat}` at `{file}:{line}`)",
        "",
        explanation,
        "",
        "```suggestion",
        replacement_code,
        "```",
        "",
        "<details><summary>Verified Unified Diff Hunk</summary>",
        "",
        "```diff",
        raw_hunk,
        "```",
        "</details>",
    ]
    return PostedReviewComment(
        path=file,
        line=max(1, line),
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
    extracted = extract_diff_suggestion(reported_patch) if reported_patch.strip() else None

    matched_finding_idx: int | None = None
    if extracted is not None:
        patch_file, patch_line, replacement_code, raw_hunk = extracted
        # Pair the verified patch with the primary fixable/LLM finding on that file if present
        for idx, f in enumerate(findings):
            if f.file == patch_file and f.category not in ("F401", "E501"):
                matched_finding_idx = idx
                break
        if matched_finding_idx is None:
            for idx, f in enumerate(findings):
                if f.file == patch_file:
                    matched_finding_idx = idx
                    break

        paired_finding = findings[matched_finding_idx] if matched_finding_idx is not None else None
        comments.append(
            format_verified_patch_comment(
                file=patch_file,
                line=patch_line,
                finding=paired_finding,
                replacement_code=replacement_code,
                raw_hunk=raw_hunk,
            )
        )

    for idx, f in enumerate(findings):
        if idx == matched_finding_idx:
            continue
        comments.append(format_unverified_finding_comment(f))

    return comments
