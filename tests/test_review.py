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


# ---------------------------------------------------------------------------
# Regression tests: multi-hunk RIGHT mapping, pagination, path encoding, privacy
# ---------------------------------------------------------------------------

from patchpilot.pipeline.models import Finding  # noqa: E402
from patchpilot.review import extract_diff_suggestions, format_review_comments  # noqa: E402
from patchpilot.review.github import PostedReviewComment, quote_repo_path  # noqa: E402

_MULTI_HUNK_PATCH = (
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -2,3 +2,3 @@\n"
    " a = 1\n"
    "-b = 2\n"
    "+b = 22\n"
    " c = 3\n"
    "@@ -20,4 +20,5 @@\n"
    " def f():\n"
    "-    return 1\n"
    "-    # old\n"
    "+    x = 1\n"
    "+    return x\n"
    " \n"
    "@@ -40,2 +41,3 @@\n"
    " tail = 1\n"
    "+tail2 = 2\n"
    " end = 0\n"
)


def test_multi_hunk_patch_maps_every_hunk_to_right_side_ranges() -> None:
    sugs = extract_diff_suggestions(_MULTI_HUNK_PATCH)
    assert [(s.file, s.start_line, s.end_line) for s in sugs] == [
        ("pkg/mod.py", 3, 3),
        ("pkg/mod.py", 21, 22),
        ("pkg/mod.py", 40, 40),  # pure insertion anchors on preceding context line
    ]
    assert sugs[0].replacement_code == "b = 22"
    assert sugs[1].replacement_code == "    x = 1\n    return x"
    assert sugs[2].replacement_code == "tail = 1\ntail2 = 2"
    # backward-compatible single-suggestion API returns the first change
    assert extract_diff_suggestion(_MULTI_HUNK_PATCH) == (
        "pkg/mod.py",
        3,
        "b = 22",
        sugs[0].raw_hunk,
    )


def test_multi_file_patch_and_comment_formatting_uses_start_line_right_side() -> None:
    patch = _MULTI_HUNK_PATCH + (
        "--- a/other.py\n+++ b/other.py\n@@ -1,2 +1,1 @@\n-gone = 1\n keep = 2\n"
    )
    findings = [
        Finding(file="pkg/mod.py", line=21, category="logic", explanation="range finding"),
    ]
    comments = format_review_comments(findings, patch)
    verified = [c for c in comments if c.verified_suggestion]
    assert len(verified) == 4
    ranged = [c for c in verified if c.start_line is not None]
    assert len(ranged) == 1
    assert (ranged[0].start_line, ranged[0].line, ranged[0].side) == (21, 22, "RIGHT")
    assert "range finding" in ranged[0].body
    deletion = [c for c in verified if c.path == "other.py"][0]
    assert deletion.line == 1
    assert "```suggestion\n```" in deletion.body
    assert all(c.side == "RIGHT" for c in comments)


def test_post_review_comments_sends_start_line_and_right_side() -> None:
    posted: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={})

    client = GitHubClient(token="ghp_x_token_12345", transport=httpx.MockTransport(handler))
    comments = [
        PostedReviewComment(path="a.py", line=22, start_line=21, body="multi"),
        PostedReviewComment(path="a.py", line=5, body="single"),
    ]
    assert asyncio.run(client.post_review_comments("o/r", 1, comments)) == 2
    sent = posted[0]["comments"]
    assert isinstance(sent, list)
    assert sent[0]["start_line"] == 21 and sent[0]["start_side"] == "RIGHT"
    assert sent[0]["line"] == 22 and sent[0]["side"] == "RIGHT"
    assert "start_line" not in sent[1]


def test_pr_files_pagination_follows_link_next_and_fetches_extra_file() -> None:
    seen: list[str] = []
    total = 45

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        path = request.url.path
        if path == "/repos/o/r/pulls/3":
            if "diff" in request.headers.get("Accept", ""):
                return httpx.Response(200, text="")
            return httpx.Response(
                200,
                json={"head": {"sha": "h1"}, "base": {"sha": "b1", "repo": {"private": True}}},
            )
        if path == "/repos/o/r/pulls/3/files":
            page = int(request.url.params.get("page", "1"))
            assert request.url.params.get("per_page") == "100" or page > 1
            names = [f"f{i}.py" for i in range(total)]
            chunk = names[(page - 1) * 30 : page * 30]
            headers = {}
            if page * 30 < total:
                nxt = f"https://api.github.com/repos/o/r/pulls/3/files?per_page=100&page={page + 1}"
                headers["Link"] = f'<{nxt}>; rel="next"'
            return httpx.Response(
                200,
                json=[{"filename": n, "status": "added", "content": "x = 1\n"} for n in chunk],
                headers=headers,
            )
        if path == "/repos/o/r/contents/lib/unchanged.py":
            assert request.url.params.get("ref") == "h1"
            return httpx.Response(
                200,
                json={"content": base64.b64encode(b"u = 1\n").decode(), "encoding": "base64"},
            )
        return httpx.Response(404, json={})

    client = GitHubClient(token="ghp_tok_pag_123456", transport=httpx.MockTransport(handler))
    ctx = asyncio.run(client.fetch_pull_request("o/r", 3, extra_paths=["lib/unchanged.py"]))
    assert len([k for k in ctx.files if k.startswith("f")]) == total
    assert ctx.files["lib/unchanged.py"] == "u = 1\n"
    assert sum("/pulls/3/files" in u for u in seen) == 2


def test_pagination_does_not_follow_foreign_origin_link() -> None:
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.path.endswith("/files"):
            return httpx.Response(
                200,
                json=[],
                headers={"Link": '<https://evil.example/steal>; rel="next"'},
            )
        if request.url.path == "/repos/o/r/pulls/3":
            return httpx.Response(200, json={"head": {"sha": "h"}, "base": {"repo": {}}})
        return httpx.Response(200, text="")

    client = GitHubClient(token="ghp_tok_origin_1234", transport=httpx.MockTransport(handler))
    asyncio.run(client.fetch_pull_request("o/r", 3))
    assert "evil.example" not in seen_hosts


def test_file_paths_are_url_encoded_per_segment() -> None:
    assert quote_repo_path("dir one/a#b/c%d?.py") == "dir%20one/a%23b/c%25d%3F.py"
    raw_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raw = request.url.raw_path.decode("ascii")
        if "/contents/" in raw:
            raw_paths.append(raw)
            return httpx.Response(
                200, json={"content": base64.b64encode(b"ok\n").decode(), "encoding": "base64"}
            )
        if raw.startswith("/repos/o/r/pulls/9/files"):
            return httpx.Response(
                200, json=[{"filename": "my dir/we#ird %name?.py", "status": "modified"}]
            )
        if raw == "/repos/o/r/pulls/9":
            return httpx.Response(200, json={"head": {"sha": "s1"}, "base": {"repo": {}}})
        return httpx.Response(200, text="")

    secret = "ghp_secret_for_encoding_test"
    client = GitHubClient(token=secret, transport=httpx.MockTransport(handler))
    ctx = asyncio.run(client.fetch_pull_request("o/r", 9))
    assert ctx.files["my dir/we#ird %name?.py"] == "ok\n"
    assert raw_paths == ["/repos/o/r/contents/my%20dir/we%23ird%20%25name%3F.py?ref=s1"]
    assert secret not in repr(client) and secret not in str(client)


@pytest.mark.parametrize("dirname", ["fixtures", "evals", "my_evals_fixtures"])
def test_local_repo_dir_defaults_to_private_even_for_fixtures_and_evals(
    tmp_path: Path, dirname: str
) -> None:
    from patchpilot.providers.base import LLMRequest

    repo = tmp_path / dirname / "case"
    repo.mkdir(parents=True)
    (repo / "m.py").write_text("import os\n\ndef f():\n    return 1\n", encoding="utf-8")

    class Spy(MockProvider):
        classes: list[str] = []

        async def generate(self, request: LLMRequest):  # type: ignore[no-untyped-def]
            Spy.classes.append(request.data_class)
            return await super().generate(request)

    Spy.classes = []
    asyncio.run(
        run_review(
            repo_dir=repo,
            dry_run=True,
            provider_name="mock",
            provider_override=Spy(canned="x"),
            verify_fn=_fast_verify,
        )
    )
    assert Spy.classes, "expected at least one LLM call"
    assert set(Spy.classes) == {"private"}

    Spy.classes = []
    asyncio.run(
        run_review(
            repo_dir=repo,
            dry_run=True,
            provider_name="mock",
            provider_override=Spy(canned="x"),
            verify_fn=_fast_verify,
            assume_public=True,
        )
    )
    assert set(Spy.classes) == {"public"}
