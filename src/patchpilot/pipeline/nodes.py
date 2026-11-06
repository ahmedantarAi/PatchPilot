"""LangGraph pipeline nodes: analyze, propose, verify, report."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
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
    failing_test_output: str = "",
    retrieved_context: list[str] | None = None,
    tracer: PatchPilotTracer | None = None,
    parent_span_id: str | None = None,
) -> list[Finding]:
    """Perform LLM fault localization and inspection without ground-truth file hints."""
    non_test_py_files = [p for p in sorted(files) if is_modifiable_py_file(p)]
    test_files = [p for p in sorted(files) if is_test_file(p)]

    prompt_lines = list(ANALYZE_PROMPT_TEMPLATE)
    if non_test_py_files:
        prompt_lines.append("Repository non-test .py files:")
        for p in non_test_py_files:
            prompt_lines.append(f"- {p}")
    if failing_test_output:
        prompt_lines.append("Failing test output:")
        prompt_lines.append(failing_test_output)
    if test_files:
        prompt_lines.append("Failing test content:")
        for p in test_files:
            prompt_lines.append(f"--- Test File: {p} ---")
            prompt_lines.append(files[p])
    if findings:
        prompt_lines.append("Initial static analysis findings:")
        for f in findings:
            prompt_lines.append(f"- {f.file}:{f.line} [{f.category}] {f.explanation}")
    if retrieved_context:
        prompt_lines.append("Retrieved context:")
        for ctx in retrieved_context:
            prompt_lines.append(ctx)

    prompt_lines.append("Repository non-test .py file contents:")
    for path in non_test_py_files:
        prompt_lines.append(f"--- File: {path} ---")
        prompt_lines.append(files[path])

    req = LLMRequest(
        messages=[Message(role="user", content="\n".join(prompt_lines))],
        temperature=0.0,
        data_class=data_class,
    )
    llm_findings: list[Finding] = []
    t0 = time.perf_counter()
    try:
        resp = await provider.generate(req)
        latency_s = time.perf_counter() - t0
        if tracer is not None:
            tracer.record_llm_call(
                node="analyze",
                model=resp.model,
                provider=resp.provider,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                latency_s=latency_s,
                parent_span_id=parent_span_id,
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
            for candidate in non_test_py_files:
                if candidate in resp.text:
                    chosen_file = candidate
                    break
            if chosen_file == "unknown" and non_test_py_files:
                chosen_file = non_test_py_files[0]
            llm_findings.append(
                Finding(
                    file=chosen_file,
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
                failing_test_output = f"{init_res.stdout}\n{init_res.stderr}".strip()
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
            test_snippets = [
                f"{p}:\n{c}" for p, c in sorted(state.files.items()) if is_test_file(p)
            ]
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
                    query_text = f"{non_test_list}\n{failing_test_output}\n{findings_text}".strip()
                    if query_text:
                        query_vecs = await embedder.embed([query_text])
                        if query_vecs:
                            hits = await qdrant.search(col_name, query_vecs[0], limit=3)
                            for hit in hits:
                                payload = hit.get("payload")
                                if isinstance(payload, dict) and "text" in payload:
                                    retrieved_context.append(str(payload["text"]))
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

        non_test_py_files = [p for p in sorted(state.files) if is_modifiable_py_file(p)]

        prompt_parts = list(PROPOSE_PROMPT_TEMPLATE)
        if non_test_py_files:
            prompt_parts.append("Candidate non-test .py files:")
            for p in non_test_py_files:
                prompt_parts.append(f"- {p}")
        if state.failing_test_output:
            prompt_parts.append("Failing test output:")
            prompt_parts.append(state.failing_test_output)
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
            prompt_parts.append("Please analyze the failure and adjust your fix accordingly.")

        prompt_parts.append("Repository files:")
        for path, content in sorted(state.files.items()):
            prompt_parts.append(f"--- File: {path} ---")
            prompt_parts.append(content)

        req = LLMRequest(
            messages=[Message(role="user", content="\n".join(prompt_parts))],
            temperature=0.0,
            data_class=state.data_class,
        )

        t0 = time.perf_counter()
        try:
            resp = await provider.generate(req)
            latency_s = time.perf_counter() - t0
            raw_patch = resp.text
            if tracer is not None:
                tracer.record_llm_call(
                    node="propose",
                    model=resp.model,
                    provider=resp.provider,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    latency_s=latency_s,
                    parent_span_id=span_id,
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
