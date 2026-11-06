"""Dataset exporter for Phase 8 trace distillation and QLoRA fine-tuning (seeded_train only)."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from patchpilot.evals.generator import generate_seeded_eval, generate_seeded_train, load_pool_cases
from patchpilot.evals.models import CaseDefinition, FrozenManifest
from patchpilot.evals.taxonomy import (
    EVAL_TEMPLATES_PER_CLASS,
    compute_case_content_hash,
    compute_source_hash,
)
from patchpilot.pipeline.nodes import PROPOSE_PROMPT_TEMPLATE
from patchpilot.pipeline.normalization import is_modifiable_py_file, is_test_file
from patchpilot.sandbox.tar_and_diff import apply_unified_diff

DISTILLATION_SYSTEM_PROMPT = "\n".join(PROPOSE_PROMPT_TEMPLATE)


class DistillationLeakageError(RuntimeError):
    """Raised when an eval-only case, template, or hash is detected during training export."""


class ChatMessage(BaseModel):
    """Single ChatML message (`system`, `user`, or `assistant`)."""

    role: Literal["system", "user", "assistant"]
    content: str


class DistillationSample(BaseModel):
    """Single JSONL training record for QLoRA fine-tuning on `seeded_train`."""

    case_id: str
    pool: str
    taxonomy: str
    template: str
    seed: int
    content_hash: str
    source: Literal["seeded_train_reference", "verified_trace"] = "seeded_train_reference"
    messages: list[ChatMessage]


class ExportSummary(BaseModel):
    """Summary of a completed `export-train` dataset export."""

    output_path: str
    sample_count: int
    sha256: str
    eval_leakage_checked: int
    overlap_count: int = 0


def diff_to_search_replace(files: dict[str, str], patch_diff: str) -> str:
    """Convert a unified diff against `files` into a canonical `SEARCH/REPLACE` block."""
    non_test_py = [p for p in sorted(files) if is_modifiable_py_file(p)]
    if not non_test_py:
        raise ValueError("No modifiable non-test .py file found in case files")

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for rel_p, content in files.items():
            dest = work_dir / rel_p
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        apply_unified_diff(patch_diff, work_dir)

        updated_files: dict[str, str] = {}
        for path in non_test_py:
            patched_file = work_dir / path
            if patched_file.is_file():
                updated_files[path] = patched_file.read_text(encoding="utf-8")

    blocks: list[str] = []

    for path in non_test_py:
        orig_text = files[path]
        new_text = updated_files.get(path, orig_text)
        if orig_text == new_text:
            continue
        blocks.append(
            f"<<<<<<< SEARCH:{path}\n"
            f"{orig_text.rstrip()}\n"
            "=======\n"
            f"{new_text.rstrip()}\n"
            ">>>>>>> REPLACE"
        )

    if not blocks:
        raise ValueError("Unified diff produced no modifications on non-test .py files")

    return "\n\n".join(blocks)


def build_user_prompt_for_case(case: CaseDefinition) -> str:
    """Construct the user task prompt (candidate files, failing test, and repo contents)."""
    non_test_py = [p for p in sorted(case.files) if is_modifiable_py_file(p)]
    test_files = [p for p in sorted(case.files) if is_test_file(p)]

    lines: list[str] = []
    if case.metadata.description:
        lines.append(f"Task description: {case.metadata.description}")
    if non_test_py:
        lines.append("Candidate non-test .py files:")
        for p in non_test_py:
            lines.append(f"- {p}")
    if test_files:
        lines.append("Failing test suite:")
        for p in test_files:
            lines.append(f"--- Test File: {p} ---")
            lines.append(case.files[p].rstrip())

    lines.append("Repository files:")
    for path in non_test_py:
        lines.append(f"--- File: {path} ---")
        lines.append(case.files[path].rstrip())

    return "\n".join(lines)


def _collect_forbidden_eval_fingerprints(
    *,
    datasets_dir: Path,
    frozen_path: Path,
    eval_cases: list[CaseDefinition] | None = None,
) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    """Return `(eval_templates, content_hashes, source_hashes, diff_hashes, signatures)`."""
    eval_templates = {tpl for tpls in EVAL_TEMPLATES_PER_CLASS.values() for tpl in tpls}
    content_hashes: set[str] = set()
    source_hashes: set[str] = set()
    diff_hashes: set[str] = set()
    signatures: set[str] = set()

    if frozen_path.is_file():
        manifest = FrozenManifest.model_validate_json(frozen_path.read_text(encoding="utf-8"))
        for pool_manifest in manifest.pools.values():
            content_hashes.update(pool_manifest.cases.values())

    loaded_eval = (
        eval_cases if eval_cases is not None else load_pool_cases(datasets_dir, "seeded_eval")
    )
    if not loaded_eval:
        loaded_eval = generate_seeded_eval(base_seed=1337, count=120)

    for ec in loaded_eval:
        content_hashes.add(ec.metadata.content_hash)
        content_hashes.add(compute_case_content_hash(ec.files, ec.patch_diff))
        diff_hashes.add(hashlib.sha256(ec.patch_diff.strip().encode("utf-8")).hexdigest())
        signatures.add(ec.metadata.mutation_signature)
        for path, src in ec.files.items():
            if is_modifiable_py_file(path):
                source_hashes.add(compute_source_hash(src))

    return (eval_templates, content_hashes, source_hashes, diff_hashes, signatures)


def validate_case_zero_leakage(
    case: CaseDefinition,
    *,
    eval_templates: set[str],
    eval_content_hashes: set[str],
    eval_source_hashes: set[str],
    eval_diff_hashes: set[str],
    eval_signatures: set[str],
) -> None:
    """Validate that `case` is strictly from `seeded_train` and shares zero hashes with eval."""
    if case.metadata.pool != "seeded_train":
        raise DistillationLeakageError(
            f"Refusing to export case {case.metadata.case_id!r} from pool "
            f"{case.metadata.pool!r}: only 'seeded_train' is allowed"
        )

    if case.metadata.template in eval_templates:
        raise DistillationLeakageError(
            f"Refusing to export case {case.metadata.case_id!r}: template "
            f"{case.metadata.template!r} is reserved for seeded_eval"
        )

    recomputed_hash = compute_case_content_hash(case.files, case.patch_diff)
    if case.metadata.content_hash in eval_content_hashes or recomputed_hash in eval_content_hashes:
        raise DistillationLeakageError(
            f"Refusing to export case {case.metadata.case_id!r}: content_hash matches "
            "a frozen seeded_eval / safety case"
        )

    diff_hash = hashlib.sha256(case.patch_diff.strip().encode("utf-8")).hexdigest()
    if diff_hash in eval_diff_hashes:
        raise DistillationLeakageError(
            f"Refusing to export case {case.metadata.case_id!r}: patch_diff hash matches "
            "a seeded_eval case"
        )

    if case.metadata.mutation_signature in eval_signatures:
        raise DistillationLeakageError(
            f"Refusing to export case {case.metadata.case_id!r}: mutation_signature matches "
            "a seeded_eval case"
        )

    for path, src in case.files.items():
        if is_modifiable_py_file(path):
            src_hash = compute_source_hash(src)
            if src_hash in eval_source_hashes:
                raise DistillationLeakageError(
                    f"Refusing to export case {case.metadata.case_id!r}: normalized source hash "
                    f"of {path!r} matches a seeded_eval case"
                )


def format_case_as_distillation_sample(case: CaseDefinition) -> DistillationSample:
    """Convert a validated `seeded_train` `CaseDefinition` into a `DistillationSample`."""
    user_prompt = build_user_prompt_for_case(case)
    assistant_patch = diff_to_search_replace(case.files, case.patch_diff)
    recomputed_hash = compute_case_content_hash(case.files, case.patch_diff)

    return DistillationSample(
        case_id=case.metadata.case_id,
        pool=case.metadata.pool,
        taxonomy=case.metadata.taxonomy,
        template=case.metadata.template,
        seed=case.metadata.seed,
        content_hash=recomputed_hash,
        source="seeded_train_reference",
        messages=[
            ChatMessage(role="system", content=DISTILLATION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
            ChatMessage(role="assistant", content=assistant_patch),
        ],
    )


def export_distillation_dataset(
    output_path: Path,
    *,
    datasets_dir: Path = Path("evals/datasets"),
    frozen_path: Path = Path("evals/FROZEN.json"),
    cases_override: list[CaseDefinition] | None = None,
    eval_cases_override: list[CaseDefinition] | None = None,
) -> ExportSummary:
    """Export `seeded_train` cases to JSONL in ChatML format after strict zero-leakage checks."""
    (
        eval_templates,
        eval_content_hashes,
        eval_source_hashes,
        eval_diff_hashes,
        eval_signatures,
    ) = _collect_forbidden_eval_fingerprints(
        datasets_dir=datasets_dir,
        frozen_path=frozen_path,
        eval_cases=eval_cases_override,
    )

    if cases_override is not None:
        train_cases = list(cases_override)
    else:
        train_cases = load_pool_cases(datasets_dir, "seeded_train")
        if not train_cases:
            train_cases = generate_seeded_train(base_seed=1337, count=360)

    if not train_cases:
        raise ValueError("No seeded_train cases found to export")

    ordered_cases = sorted(train_cases, key=lambda c: c.metadata.case_id)
    jsonl_lines: list[str] = []

    for case in ordered_cases:
        validate_case_zero_leakage(
            case,
            eval_templates=eval_templates,
            eval_content_hashes=eval_content_hashes,
            eval_source_hashes=eval_source_hashes,
            eval_diff_hashes=eval_diff_hashes,
            eval_signatures=eval_signatures,
        )
        sample = format_case_as_distillation_sample(case)
        jsonl_lines.append(sample.model_dump_json())

    payload_text = "\n".join(jsonl_lines) + "\n"
    file_sha256 = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload_text, encoding="utf-8")

    return ExportSummary(
        output_path=output_path.as_posix(),
        sample_count=len(jsonl_lines),
        sha256=file_sha256,
        eval_leakage_checked=len(eval_content_hashes),
        overlap_count=0,
    )
