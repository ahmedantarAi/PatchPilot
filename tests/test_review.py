"""Phase 7 unit tests for GitHub PR integration, comment formatting, and `patchpilot review` CLI."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

from patchpilot.providers.mock import MockProvider
from patchpilot.review import (
    GitHubClient,
    GitHubError,
    extract_diff_suggestion,
    parse_repo_slug,
    run_review,
)
from patchpilot.sandbox.runner import SandboxResult
from patchpilot.sandbox.tar_and_diff import JUnitReport


async def _fast_verify(files: dict[str, str], patch: str) -> SandboxResult:
    return SandboxResult(
        exit_code=0,
        patch_applied=bool(patch.strip()),
        junit=JUnitReport(passed=1, failures=0, errors=0),
    )


def test_local_repo_dir_dry_run_matches_phase7_contract(tmp_path: Path) -> None:
    """Local `--repo-dir tests/fixtures/pr_demo --dry-run` matches Phase 7 summary contract."""
    report_path = tmp_path / "review_report.json"
    res = asyncio.run(
        run_review(
            repo_dir=Path("tests/fixtures/pr_demo"),
            dry_run=True,
            provider_name="mock",
            verify_fn=_fast_verify,
            report_out=report_path,
        )
    )

    assert res.summary_line() == "findings=2 verified_patches=1 comments_posted=0 (dry-run)"
    assert res.findings_count == 2
    assert res.verified_patches_count == 1
    assert res.comments_posted_count == 0
    assert len(res.comments) == 2

    # One comment is a verified suggestion with a ```suggestion block; the other is unverified
    verified_comments = [c for c in res.comments if c.verified_suggestion]
    unverified_comments = [c for c in res.comments if not c.verified_suggestion]
    assert len(verified_comments) == 1
    assert len(unverified_comments) == 1
    assert "```suggestion" in verified_comments[0].body
    assert "if quantity >= 10:" in verified_comments[0].body
    assert "```suggestion" not in unverified_comments[0].body
    assert report_path.is_file()


def test_github_pr_diff_fetch_and_live_comment_posting() -> None:
    """Mocked GitHub REST API responses verify PR fetch, decoding, and review comment posting."""
    requests_seen: list[httpx.Request] = []
    posted_payloads: list[dict[str, object]] = []

    buggy_py = (
        '"""Discount module."""\n\n'
        "import os\n\n\n"
        "def calc(q: int) -> int:\n"
        "    if q > 10:\n"
        "        return 5\n"
        "    return 0\n"
    )
    b64_content = base64.b64encode(buggy_py.encode("utf-8")).decode("ascii")
    pr_diff = (
        "--- a/discount.py\n"
        "+++ b/discount.py\n"
        "@@ -6,3 +6,3 @@\n"
        " def calc(q: int) -> int:\n"
        "-    if q >= 10:\n"
        "+    if q > 10:\n"
        "         return 5\n"
    )
    fix_diff = (
        "--- a/discount.py\n"
        "+++ b/discount.py\n"
        "@@ -6,4 +6,4 @@\n"
        " def calc(q: int) -> int:\n"
        "-    if q > 10:\n"
        "+    if q >= 10:\n"
        "         return 5\n"
        "     return 0\n"
    )

    def github_handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        accept = request.headers.get("Accept", "")
        path = request.url.path

        if request.method == "GET" and path == "/repos/acme/widgets/pulls/42":
            if "diff" in accept:
                return httpx.Response(200, text=pr_diff)
            return httpx.Response(
                200,
                json={
                    "title": "Update discount threshold",
                    "head": {"sha": "abc1234deadbeef"},
                    "base": {"sha": "000111222", "repo": {"private": False}},
                },
            )
        if request.method == "GET" and path == "/repos/acme/widgets/pulls/42/files":
            return httpx.Response(
                200,
                json=[{"filename": "discount.py", "status": "modified"}],
            )
        if request.method == "GET" and path == "/repos/acme/widgets/contents/discount.py":
            return httpx.Response(
                200,
                json={"content": b64_content, "encoding": "base64"},
            )
        if request.method == "POST" and path == "/repos/acme/widgets/pulls/42/reviews":
            body = json.loads(request.content.decode("utf-8"))
            posted_payloads.append(body)
            return httpx.Response(201, json={"id": 999})

        return httpx.Response(404, json={"message": "not found"})

    gh_client = GitHubClient(
        token="ghp_test_token_do_not_expose_98765",
        transport=httpx.MockTransport(github_handler),
    )

    res = asyncio.run(
        run_review(
            repo="https://github.com/acme/widgets",
            pr_number=42,
            dry_run=False,
            provider_name="mock",
            provider_override=MockProvider(canned=fix_diff),
            github_client=gh_client,
            verify_fn=_fast_verify,
        )
    )

    assert res.findings_count == 2
    assert res.verified_patches_count == 1
    assert res.comments_posted_count == 2
    assert res.summary_line() == "findings=2 verified_patches=1 comments_posted=2"
    assert len(posted_payloads) == 1
    assert posted_payloads[0]["commit_id"] == "abc1234deadbeef"
    comments_sent = posted_payloads[0]["comments"]
    assert isinstance(comments_sent, list)
    assert len(comments_sent) == 2
    assert any("```suggestion" in str(c.get("body", "")) for c in comments_sent)


def test_github_token_security_masking_and_missing_token_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GITHUB_TOKEN` is never exposed in repr/errors and missing token fails fast."""
    secret_tok = "ghp_super_secret_token_value_12345"
    client_with_tok = GitHubClient(
        token=secret_tok,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(401, text=f"Unauthorized token {secret_tok}")
        ),
    )

    assert secret_tok not in repr(client_with_tok)
    assert secret_tok not in str(client_with_tok)

    with pytest.raises(GitHubError) as exc_info:
        asyncio.run(client_with_tok.fetch_pull_request("owner/repo", 1))
    assert secret_tok not in str(exc_info.value)
    assert "***REDACTED***" in str(exc_info.value)

    # Missing token fails fast before any network or LLM call when dry_run=False
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    empty_client = GitHubClient(token="")
    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        asyncio.run(
            run_review(
                repo="owner/repo",
                pr_number=7,
                dry_run=False,
                provider_name="mock",
                github_client=empty_client,
            )
        )


def test_extract_diff_suggestion_and_repo_slug_parsing() -> None:
    """Verify unified diff hunk extraction and GitHub repository slug normalization."""
    assert parse_repo_slug("owner/repo") == "owner/repo"
    assert parse_repo_slug("https://github.com/owner/repo.git/") == "owner/repo"

    with pytest.raises(GitHubError):
        parse_repo_slug("invalid-slug-without-slash")

    sample_diff = (
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -10,4 +10,4 @@\n"
        " def check(x: int) -> bool:\n"
        "-    return x > 5\n"
        "+    return x >= 5\n"
        " \n"
    )
    extracted = extract_diff_suggestion(sample_diff)
    assert extracted is not None
    file_path, line_no, replacement, raw_hunk = extracted
    assert file_path == "pkg/module.py"
    assert line_no == 11
    assert replacement == "    return x >= 5"
    assert "@@ -10,4 +10,4 @@" in raw_hunk
