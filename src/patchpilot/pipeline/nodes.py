"""LangGraph pipeline nodes: analyze, propose, verify, report."""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from patchpilot.pipeline.models import Finding, PipelineState
from patchpilot.pipeline.normalization import normalize_patch
from patchpilot.providers.base import LLMProvider, LLMRequest, Message, ProviderError
from patchpilot.retrieval import Embedder, QdrantStore
from patchpilot.sandbox.runner import SandboxResult, SandboxRunner

logger = logging.getLogger("patchpilot.pipeline")

VerifyCallable = Callable[[dict[str, str], str], Awaitable[SandboxResult]]


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
    py_files = {p: c for p, c in files.items() if p.endswith(".py")}
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
    target_file: str = "",
    retrieved_context: list[str] | None = None,
) -> list[Finding]:
    """Perform LLM diff and fault inspection producing typed Finding objects."""
    prompt_lines = [
        "You are PatchPilot static analysis and diff inspection engine.",
        "Inspect the provided repository files and existing static findings to identify bugs.",
        "Emits findings as lines in format: FINDING: <file>:<line>: <category>: <explanation>",
    ]
    if target_file:
        prompt_lines.append(f"Primary target file: {target_file}")
    if findings:
        prompt_lines.append("Initial linter findings:")
        for f in findings:
            prompt_lines.append(f"- {f.file}:{f.line} [{f.category}] {f.explanation}")
    if retrieved_context:
        prompt_lines.append("Retrieved context:")
        for ctx in retrieved_context:
            prompt_lines.append(ctx)

    prompt_lines.append("Files:")
    for path, content in sorted(files.items()):
        prompt_lines.append(f"--- File: {path} ---")
        prompt_lines.append(content)

    req = LLMRequest(
        messages=[Message(role="user", content="\n".join(prompt_lines))],
        temperature=0.0,
        data_class=data_class,
    )
    llm_findings: list[Finding] = []
    try:
        resp = await provider.generate(req)
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
            target = target_file or next(iter(files.keys()), "unknown")
            llm_findings.append(
                Finding(
                    file=target,
                    line=1,
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
) -> Callable[[PipelineState], Awaitable[dict[str, Any]]]:
    """Create the analyze node pure function over PipelineState."""

    async def analyze_node(state: PipelineState) -> dict[str, Any]:
        # 1. ast.parse
        ast_findings = run_ast_parse(state.files)

        # 2. ruff check --output-format json --isolated
        ruff_findings = run_ruff_check(state.files)

        all_findings = ast_findings + ruff_findings

        # 3. Retrieval from Qdrant if available
        retrieved_context: list[str] = []
        if qdrant is not None and embedder is not None:
            try:
                healthy = await qdrant.is_healthy()
                if healthy:
                    query_text = f"{state.target_file}\n" + "\n".join(
                        f"{f.file}:{f.category}" for f in all_findings
                    )
                    query_vecs = await embedder.embed([query_text])
                    if query_vecs:
                        hits = await qdrant.search("repo_code", query_vecs[0], limit=3)
                        for hit in hits:
                            payload = hit.get("payload")
                            if isinstance(payload, dict) and "text" in payload:
                                retrieved_context.append(str(payload["text"]))
            except Exception as exc:
                logger.info("Qdrant retrieval degraded gracefully during analyze: %s", exc)

        # 4. LLM diff analysis
        llm_findings = await run_llm_diff_analysis(
            files=state.files,
            provider=provider,
            data_class=state.data_class,
            findings=all_findings,
            target_file=state.target_file,
            retrieved_context=retrieved_context,
        )
        combined_findings = all_findings + llm_findings

        return {
            "findings": combined_findings,
            "retrieved_context": retrieved_context,
        }

    return analyze_node


def create_propose_node(
    provider: LLMProvider,
) -> Callable[[PipelineState], Awaitable[dict[str, Any]]]:
    """Create the propose node pure function over PipelineState."""

    async def propose_node(state: PipelineState) -> dict[str, Any]:
        prompt_parts = [
            "You are PatchPilot, an autonomous automated code repair agent.",
            "Fix the bug in the repository so that all tests pass.",
            "Provide your fix as SEARCH/REPLACE blocks formatted as:",
            "<<<<<<< SEARCH",
            "<exact original lines to replace>",
            "=======",
            "<replacement lines>",
            ">>>>>>> REPLACE",
            "(A raw unified diff is also accepted.)",
        ]
        if state.target_file:
            prompt_parts.append(f"Target file: {state.target_file}")
        if state.findings:
            prompt_parts.append("Identified findings:")
            for f in state.findings:
                prompt_parts.append(f"- {f.file}:{f.line} [{f.category}] {f.explanation}")
        if state.retrieved_context:
            prompt_parts.append("Retrieved repository context:")
            for ctx in state.retrieved_context:
                prompt_parts.append(ctx)
        if state.feedback:
            prompt_parts.append("PREVIOUS ATTEMPT FAILED WITH TEST OUTPUT / ERROR:")
            prompt_parts.append(state.feedback)
            prompt_parts.append("Please analyze the test failure and adjust your fix accordingly.")

        prompt_parts.append("Repository files:")
        for path, content in sorted(state.files.items()):
            prompt_parts.append(f"--- File: {path} ---")
            prompt_parts.append(content)

        req = LLMRequest(
            messages=[Message(role="user", content="\n".join(prompt_parts))],
            temperature=0.0,
            data_class=state.data_class,
        )

        try:
            resp = await provider.generate(req)
            raw_patch = resp.text
        except ProviderError as exc:
            logger.warning("LLM generation failed in propose: %s", exc)
            return {
                "raw_patch": "",
                "patch_diff": "",
                "error": f"LLM error: {exc}",
            }

        try:
            canonical_diff = normalize_patch(
                raw_patch,
                state.files,
                default_file=state.target_file,
            )
            return {
                "raw_patch": raw_patch,
                "patch_diff": canonical_diff,
                "error": None,
            }
        except Exception as exc:
            logger.info("Patch normalization failed: %s", exc)
            return {
                "raw_patch": raw_patch,
                "patch_diff": "",
                "error": f"Normalization error: {exc}",
            }

    return propose_node


def create_verify_node(
    sandbox_runner: SandboxRunner | None = None,
    verify_fn: VerifyCallable | None = None,
) -> Callable[[PipelineState], Awaitable[dict[str, Any]]]:
    """Create the verify node pure function over PipelineState."""

    async def verify_node(state: PipelineState) -> dict[str, Any]:
        attempt = state.attempt + 1

        if not state.patch_diff.strip():
            return {
                "attempt": attempt,
                "patch_applies": False,
                "target_test_passes": False,
                "regressions": 0,
                "verified": False,
                "pass_at_1": False if attempt == 1 else state.pass_at_1,
                "pass_at_k": False if attempt >= state.max_attempts else state.pass_at_k,
                "feedback": state.error or "No patch proposed or patch could not be normalized.",
            }

        # Run verification in sandbox
        if verify_fn is not None:
            result = await verify_fn(state.files, state.patch_diff)
        else:
            runner = sandbox_runner or SandboxRunner()
            result = await runner.run(
                files=state.files,
                patch_diff=state.patch_diff,
                pytest_args=["-q"],
            )

        patch_applies = bool(result.patch_applied)
        tests_passed = (
            result.exit_code == 0
            and result.junit.failures == 0
            and result.junit.errors == 0
            and result.junit.passed > 0
        )
        regressions = result.junit.failures + result.junit.errors if not tests_passed else 0
        verified = bool(patch_applies and tests_passed and regressions == 0)

        pass_at_1 = state.pass_at_1
        if attempt == 1:
            pass_at_1 = verified

        pass_at_k = state.pass_at_k or verified
        if attempt >= state.max_attempts and not verified:
            pass_at_k = False

        feedback = ""
        if not verified:
            feedback = (
                f"Attempt {attempt} verification failed:\n"
                f"Patch applied: {patch_applies}\n"
                f"Exit code: {result.exit_code}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}\n"
                f"Error: {result.error}"
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
        }

    return verify_node


def should_retry(state: PipelineState) -> Literal["propose", "report"]:
    """Conditional edge from verify: loop back to propose if attempt < max_attempts."""
    if state.verified:
        return "report"
    if state.attempt < state.max_attempts:
        return "propose"
    return "report"


async def report_node(state: PipelineState) -> dict[str, Any]:
    """Report node: only verified patches are reported."""
    if state.verified:
        return {"reported_patch": state.patch_diff}
    return {"reported_patch": ""}
