"""Dataset generator and materializer for seeded evaluation, training, and safety pools."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from patchpilot.evals.models import BugTaxonomy, CaseDefinition, CaseMetadata, SplitManifest
from patchpilot.evals.taxonomy import (
    EVAL_TEMPLATES_PER_CLASS,
    TRAIN_TEMPLATES_PER_CLASS,
    build_seeded_case,
    compute_case_content_hash,
    compute_source_hash,
)
from patchpilot.pipeline.normalization import is_modifiable_py_file
from patchpilot.sandbox.fixtures import HOSTILE_FIXTURES


def _select_buggy_source_for_hash(files: Mapping[str, str]) -> str:
    """Select non-test Python source content deterministically for source-hash deduplication."""
    non_test_py_paths = sorted(path for path in files if is_modifiable_py_file(path))
    if not non_test_py_paths:
        return ""
    if len(non_test_py_paths) == 1:
        return files[non_test_py_paths[0]]
    return "\n".join(f"# {path}\n{files[path]}" for path in non_test_py_paths)


def _round_robin_targets(
    total_cases: int,
    template_map: dict[BugTaxonomy, list[str]],
) -> list[tuple[BugTaxonomy, str]]:
    """Generate balanced sequence of (taxonomy, template) pairs for target size."""
    taxonomies = list(BugTaxonomy)
    pairs: list[tuple[BugTaxonomy, str]] = []
    tax_idx = 0
    tpl_indices = {tax: 0 for tax in taxonomies}

    while len(pairs) < total_cases:
        tax = taxonomies[tax_idx % len(taxonomies)]
        tpl_list = template_map.get(tax, [])
        if not tpl_list:
            tax_idx += 1
            continue
        t_idx = tpl_indices[tax] % len(tpl_list)
        tpl = tpl_list[t_idx]
        tpl_indices[tax] += 1
        pairs.append((tax, tpl))
        tax_idx += 1

    return pairs


def generate_seeded_eval(base_seed: int = 1337, count: int = 120) -> list[CaseDefinition]:
    """Deterministically generate the seeded_eval pool using reserved eval templates."""
    pairs = _round_robin_targets(count, EVAL_TEMPLATES_PER_CLASS)
    cases: list[CaseDefinition] = []

    for idx, (tax, tpl) in enumerate(pairs, start=1):
        case_id = f"{idx:04d}"
        # Seed derivation: base_seed + multiplier * idx
        seed = base_seed + (idx * 37) + 11
        case = build_seeded_case(
            case_id=case_id,
            pool="seeded_eval",
            taxonomy=tax,
            template=tpl,
            seed=seed,
        )
        cases.append(case)

    return cases


def generate_seeded_train(
    base_seed: int = 1337,
    count: int = 360,
    eval_cases: list[CaseDefinition] | None = None,
) -> list[CaseDefinition]:
    """Deterministically generate the train dataset pool (seeded_train) with strict zero-overlap."""
    if eval_cases is None:
        eval_cases = generate_seeded_eval(base_seed=base_seed, count=120)

    # Collect rejection sets from eval cases
    eval_content_hashes = {c.metadata.content_hash for c in eval_cases}
    eval_source_hashes = {
        compute_source_hash(_select_buggy_source_for_hash(c.files)) for c in eval_cases
    }
    eval_diff_hashes = {
        hashlib.sha256(c.patch_diff.strip().encode("utf-8")).hexdigest() for c in eval_cases
    }
    eval_signatures = {c.metadata.mutation_signature for c in eval_cases}

    # Template rejection set: eval templates are strictly prohibited in train
    reserved_eval_templates = {tpl for tpls in EVAL_TEMPLATES_PER_CLASS.values() for tpl in tpls}

    pairs = _round_robin_targets(count, TRAIN_TEMPLATES_PER_CLASS)
    cases: list[CaseDefinition] = []
    seen_train_content_hashes: set[str] = set()

    # Disjoint seed space for training: base_seed + 100_000
    train_seed_offset = base_seed + 100_000
    candidate_step = 0

    for idx, (tax, tpl) in enumerate(pairs, start=1):
        if tpl in reserved_eval_templates:
            raise ValueError(
                f"Template {tpl} is reserved for seeded_eval and cannot be used in seeded_train"
            )

        case_id = f"{idx:04d}"
        while True:
            candidate_step += 1
            seed = train_seed_offset + (candidate_step * 53) + 7
            candidate = build_seeded_case(
                case_id=case_id,
                pool="seeded_train",
                taxonomy=tax,
                template=tpl,
                seed=seed,
            )

            c_hash = candidate.metadata.content_hash
            c_diff_hash = hashlib.sha256(candidate.patch_diff.strip().encode("utf-8")).hexdigest()
            c_src_hash = compute_source_hash(_select_buggy_source_for_hash(candidate.files))
            c_sig = candidate.metadata.mutation_signature

            # Overlap checks against eval and prior train cases
            if c_hash in eval_content_hashes or c_hash in seen_train_content_hashes:
                continue
            if c_diff_hash in eval_diff_hashes:
                continue
            if c_src_hash in eval_source_hashes:
                continue
            if c_sig in eval_signatures:
                continue

            # Accepted candidate
            cases.append(candidate)
            seen_train_content_hashes.add(c_hash)
            break

    return cases


def generate_safety_cases() -> list[CaseDefinition]:
    """Expose the 8 hostile containment fixtures as evaluation cases."""
    cases: list[CaseDefinition] = []

    for fixture in HOSTILE_FIXTURES:
        content_hash = compute_case_content_hash(fixture.files, fixture.patch_diff)
        meta = CaseMetadata(
            case_id=fixture.name,
            pool="safety",
            taxonomy="containment",
            template="hostile_sandbox_fixture",
            mutation_signature=f"safety::{fixture.name}",
            seed=0,
            content_hash=content_hash,
            patch_diff=fixture.patch_diff,
            pytest_args=["-q"],
            license="MIT",
            license_source="self-generated",
            description=fixture.description,
        )
        case = CaseDefinition(
            metadata=meta,
            files=dict(fixture.files),
            symlinks=dict(fixture.symlinks),
            patch_diff=fixture.patch_diff,
        )
        cases.append(case)

    return cases


def materialize_case(case: CaseDefinition, target_dir: Path) -> Path:
    """Write a case definition to disk under target_dir/<pool>/<case_id>/."""
    case_dir = target_dir / case.metadata.pool / case.metadata.case_id
    if case_dir.exists():
        shutil.rmtree(case_dir)
    repo_dir = case_dir / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, content in sorted(case.files.items()):
        fpath = repo_dir / rel_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")

    if case.symlinks:
        symlinks_json = repo_dir / "__symlinks__.json"
        symlinks_json.write_text(json.dumps(case.symlinks, indent=2), encoding="utf-8")

    patch_file = case_dir / "patch.diff"
    patch_file.write_text(case.patch_diff, encoding="utf-8")

    meta_file = case_dir / "meta.json"
    meta_file.write_text(case.metadata.model_dump_json(indent=2), encoding="utf-8")

    return case_dir


def materialize_pool(cases: list[CaseDefinition], target_dir: Path) -> None:
    """Materialize a full list of cases to the filesystem."""
    for case in cases:
        materialize_case(case, target_dir)


def load_pool_cases(datasets_dir: Path, pool: str) -> list[CaseDefinition]:
    """Read all materialized cases for a pool from disk."""
    pool_dir = datasets_dir / pool
    if not pool_dir.exists():
        return []

    cases: list[CaseDefinition] = []
    case_dirs = sorted([d for d in pool_dir.iterdir() if d.is_dir()], key=lambda d: d.name)

    for cdir in case_dirs:
        meta_file = cdir / "meta.json"
        if not meta_file.exists():
            continue
        meta = CaseMetadata.model_validate_json(meta_file.read_text(encoding="utf-8"))

        repo_dir = cdir / "repo"
        files: dict[str, str] = {}
        symlinks: dict[str, str] = {}

        if repo_dir.exists():
            for fpath in repo_dir.rglob("*"):
                if fpath.is_file():
                    rel_posix = fpath.relative_to(repo_dir).as_posix()
                    if rel_posix == "__symlinks__.json":
                        symlinks_raw = json.loads(fpath.read_text(encoding="utf-8"))
                        if isinstance(symlinks_raw, dict):
                            symlinks = {str(k): str(v) for k, v in symlinks_raw.items()}
                    else:
                        files[rel_posix] = fpath.read_text(encoding="utf-8")

        patch_file = cdir / "patch.diff"
        patch_diff = (
            patch_file.read_text(encoding="utf-8") if patch_file.exists() else meta.patch_diff
        )

        cases.append(
            CaseDefinition(
                metadata=meta,
                files=files,
                symlinks=symlinks,
                patch_diff=patch_diff,
            )
        )

    return cases


def build_and_materialize_all(
    *,
    base_seed: int = 1337,
    datasets_dir: Path,
    eval_count: int = 120,
    train_count: int = 360,
) -> SplitManifest:
    """Build and materialize seeded_eval, seeded_train, and safety datasets."""
    eval_cases = generate_seeded_eval(base_seed=base_seed, count=eval_count)
    train_cases = generate_seeded_train(
        base_seed=base_seed,
        count=train_count,
        eval_cases=eval_cases,
    )
    safety_cases = generate_safety_cases()

    materialize_pool(eval_cases, datasets_dir)
    materialize_pool(train_cases, datasets_dir)
    materialize_pool(safety_cases, datasets_dir)

    eval_hashes = [c.metadata.content_hash for c in eval_cases]
    train_hashes = [c.metadata.content_hash for c in train_cases]
    safety_hashes = [c.metadata.content_hash for c in safety_cases]

    overlap = len(set(eval_hashes).intersection(set(train_hashes)))
    eval_templates = {c.metadata.template for c in eval_cases}
    train_templates = {c.metadata.template for c in train_cases}
    template_overlap = len(eval_templates.intersection(train_templates))

    split_manifest = SplitManifest(
        generator_version="1.0.0",
        base_seed=base_seed,
        seeded_eval_count=len(eval_cases),
        seeded_train_count=len(train_cases),
        safety_count=len(safety_cases),
        seeded_eval_hashes=eval_hashes,
        seeded_train_hashes=train_hashes,
        safety_hashes=safety_hashes,
        overlap_count=overlap,
        eval_template_count=len(eval_templates),
        train_template_count=len(train_templates),
        template_overlap_count=template_overlap,
    )

    manifest_path = datasets_dir / "split_manifest.json"
    manifest_path.write_text(split_manifest.model_dump_json(indent=2), encoding="utf-8")

    return split_manifest
