"""GitHub REST API client using plain `httpx` (no third-party GitHub SDKs)."""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, Field

GITHUB_API_BASE_URL = "https://api.github.com"
_REPO_SLUG_RE = re.compile(
    r"^(?:https?://github\.com/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_GH_TOKEN_RE = re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,})")


def read_github_token_from_env() -> str:
    """Read `GITHUB_TOKEN` strictly from the environment."""
    return (os.environ.get("GITHUB_TOKEN") or "").strip()


def _redact_token(text: str, token: str = "") -> str:
    out = text
    for candidate in (token, read_github_token_from_env()):
        if candidate:
            out = out.replace(candidate, "***REDACTED***")
            encoded = quote(candidate, safe="")
            if encoded and encoded != candidate:
                out = out.replace(encoded, "***REDACTED***")
    out = _GH_TOKEN_RE.sub("***REDACTED***", out)
    return out


class GitHubError(RuntimeError):
    """Raised when GitHub API requests fail or required credentials are missing."""

    def __init__(self, message: str = "") -> None:
        super().__init__(_redact_token(str(message)))


class PullRequestContext(BaseModel):
    """Fetched metadata, unified diff, and file contents for a GitHub Pull Request."""

    repo: str
    pr_number: int
    title: str = ""
    head_sha: str = ""
    base_sha: str = ""
    is_private: bool = True
    diff: str = ""
    files: dict[str, str] = Field(default_factory=dict)


class PostedReviewComment(BaseModel):
    """Formatted review comment ready to be posted (or dry-run recorded) on a PR."""

    path: str
    line: int
    body: str
    start_line: int | None = None
    side: str = "RIGHT"
    verified_suggestion: bool = False


def parse_repo_slug(repo_arg: str) -> str:
    """Normalize `'owner/repo'` or `'https://github.com/owner/repo'` into `'owner/repo'`."""
    stripped = repo_arg.strip()
    match = _REPO_SLUG_RE.match(stripped)
    if not match:
        raise GitHubError(
            f"github: invalid repository identifier {repo_arg!r}; expected 'owner/repo'"
        )
    return match.group(1)


def quote_repo_path(path: str) -> str:
    """Percent-encode each path segment while keeping `/` separators intact."""
    return "/".join(quote(seg, safe="") for seg in path.split("/"))


MAX_LIST_PAGES = 30


class GitHubClient:
    """Plain `httpx` client for fetching PR diffs/files and posting review comments."""

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = GITHUB_API_BASE_URL,
        timeout_s: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = (token if token is not None else read_github_token_from_env()).strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._transport = transport

    @property
    def has_token(self) -> bool:
        return bool(self._token and "CHANGEME" not in self._token)

    def require_token_for_live_write(self) -> None:
        """Fail fast with a clean error message if `GITHUB_TOKEN` is missing for non-dry-run."""
        if not self.has_token:
            raise GitHubError(
                "github: GITHUB_TOKEN environment variable is required when running "
                "without --dry-run"
            )

    def __repr__(self) -> str:
        masked = "***REDACTED***" if self._token else "<unset>"
        return f"GitHubClient(base_url={self.base_url!r}, token={masked!r})"

    def __str__(self) -> str:
        return self.__repr__()

    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "PatchPilot/0.1",
        }
        if self.has_token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _get(
        self, path: str, *, accept: str = "application/vnd.github+json"
    ) -> httpx.Response:
        url = path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                resp = await client.get(url, headers=self._headers(accept=accept))
        except httpx.HTTPError as exc:
            msg = _redact_token(str(exc), self._token)
            raise GitHubError(
                f"github: HTTP request failed ({type(exc).__name__}): {msg}"
            ) from None

        if resp.status_code != 200:
            body_preview = _redact_token(resp.text[:200], self._token)
            raise GitHubError(
                f"github: GET {path} returned HTTP {resp.status_code}: {body_preview}"
            )
        return resp

    def _is_same_origin(self, url: str) -> bool:
        if url.startswith("/") and not url.startswith("//"):
            return True
        a, b = urlsplit(url), urlsplit(self.base_url)
        return (a.scheme.lower(), a.netloc.lower()) == (b.scheme.lower(), b.netloc.lower())

    def _normalize_page_url(self, url: str) -> str:
        return url if url.startswith(("http://", "https://")) else f"{self.base_url}{url}"

    async def _get_json_pages(self, path: str) -> list[Any]:
        """GET a list endpoint with `per_page=100`, following `Link: rel="next"` headers."""
        sep = "&" if "?" in path else "?"
        next_url: str | None = path if "per_page=" in path else f"{path}{sep}per_page=100"
        items: list[Any] = []
        visited: set[str] = set()
        for _ in range(MAX_LIST_PAGES):
            if next_url is None:
                break
            norm_url = self._normalize_page_url(next_url)
            if norm_url in visited:
                break
            visited.add(norm_url)
            resp = await self._get(next_url)
            try:
                page = resp.json()
            except ValueError:
                break
            if not isinstance(page, list):
                break
            items.extend(page)
            candidate = resp.links.get("next", {}).get("url")
            # Never forward the token to a foreign origin.
            next_url = candidate if candidate and self._is_same_origin(candidate) else None
        return items

    async def fetch_file(self, repo: str, path: str, ref: str = "") -> str | None:
        """Fetch one repository file (changed or unchanged) at `ref`; `None` if unavailable."""
        slug = parse_repo_slug(repo)
        ref_query = f"?ref={quote(ref, safe='')}" if ref else ""
        try:
            resp = await self._get(f"/repos/{slug}/contents/{quote_repo_path(path)}{ref_query}")
            data = resp.json()
        except (GitHubError, ValueError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("content"), str):
            return None
        raw = str(data["content"])
        if str(data.get("encoding", "base64")).lower() == "base64":
            return base64.b64decode(raw).decode("utf-8", errors="replace")
        return raw

    async def fetch_pull_request(
        self,
        repo: str,
        pr_number: int,
        *,
        extra_paths: Iterable[str] = (),
    ) -> PullRequestContext:
        """Fetch PR metadata, unified diff, and changed Python file contents via GitHub REST API."""
        slug = parse_repo_slug(repo)
        if pr_number < 1:
            raise GitHubError(f"github: pr_number must be >= 1, got {pr_number}")

        # 1. Fetch PR metadata JSON
        meta_resp = await self._get(f"/repos/{slug}/pulls/{pr_number}")
        try:
            meta = meta_resp.json()
        except ValueError:
            raise GitHubError("github: invalid JSON in pull request metadata response") from None
        if not isinstance(meta, dict):
            raise GitHubError("github: expected JSON object for pull request metadata")

        title = str(meta.get("title", ""))
        head_obj = meta.get("head") if isinstance(meta.get("head"), dict) else {}
        base_obj = meta.get("base") if isinstance(meta.get("base"), dict) else {}
        head_sha = str(head_obj.get("sha", "")) if isinstance(head_obj, dict) else ""
        base_sha = str(base_obj.get("sha", "")) if isinstance(base_obj, dict) else ""

        base_repo = base_obj.get("repo") if isinstance(base_obj, dict) else None
        is_private = bool(base_repo.get("private", True)) if isinstance(base_repo, dict) else True

        # 2. Fetch PR unified diff
        diff_resp = await self._get(
            f"/repos/{slug}/pulls/{pr_number}",
            accept="application/vnd.github.v3.diff",
        )
        pr_diff = diff_resp.text

        # 3. Fetch changed files list and their contents at head_sha
        files_list = await self._get_json_pages(f"/repos/{slug}/pulls/{pr_number}/files")

        repo_files: dict[str, str] = {}
        for item in files_list:
            if not isinstance(item, dict):
                continue
            filename = str(item.get("filename", "")).strip()
            status = str(item.get("status", "modified")).lower()
            if not filename or status == "removed":
                continue
            if isinstance(item.get("content"), str):
                repo_files[filename] = str(item["content"])
                continue
            fetched = await self.fetch_file(slug, filename, head_sha)
            if fetched is not None:
                repo_files[filename] = fetched

        # Unchanged files needed for cross-file context / SEARCH-REPLACE normalization
        for extra in extra_paths:
            extra = extra.strip()
            if extra and extra not in repo_files:
                fetched = await self.fetch_file(slug, extra, head_sha)
                if fetched is not None:
                    repo_files[extra] = fetched

        return PullRequestContext(
            repo=slug,
            pr_number=pr_number,
            title=title,
            head_sha=head_sha,
            base_sha=base_sha,
            is_private=is_private,
            diff=pr_diff,
            files=repo_files,
        )

    async def post_review_comments(
        self,
        repo: str,
        pr_number: int,
        comments: list[PostedReviewComment],
        *,
        commit_id: str = "",
    ) -> int:
        """Post formatted review comments via `/repos/{owner}/{repo}/pulls/{pr}/reviews`."""
        self.require_token_for_live_write()
        if not comments:
            return 0

        slug = parse_repo_slug(repo)
        url = f"{self.base_url}/repos/{slug}/pulls/{pr_number}/reviews"

        review_comments: list[dict[str, Any]] = []
        for c in comments:
            entry: dict[str, Any] = {
                "path": c.path,
                "line": max(1, c.line),
                "side": "RIGHT",
                "body": c.body,
            }
            if c.start_line is not None and 1 <= c.start_line < c.line:
                entry["start_line"] = c.start_line
                entry["start_side"] = "RIGHT"
            review_comments.append(entry)

        payload: dict[str, Any] = {
            "event": "COMMENT",
            "body": f"PatchPilot automated review: {len(comments)} finding(s).",
            "comments": review_comments,
        }
        if commit_id:
            payload["commit_id"] = commit_id

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            msg = _redact_token(str(exc), self._token)
            raise GitHubError(
                f"github: failed to post review comments ({type(exc).__name__}): {msg}"
            ) from None

        if resp.status_code not in (200, 201):
            body_preview = _redact_token(resp.text[:200], self._token)
            raise GitHubError(
                f"github: POST review returned HTTP {resp.status_code}: {body_preview}"
            )

        return len(comments)
