"""Tests for Phase 3a: generator determinism, split separation, freeze, and sandbox verify."""

from __future__ import annotations

from pathlib import Path

import pytest

from patchpilot.evals.freeze import freeze_eval_split, verify_frozen_eval_split
from patchpilot.evals.generator import (
    build_and_materialize_all,
    generate_safety_cases,
    generate_seeded_eval,
    generate_seeded_train,
)
from patchpilot.evals.taxonomy import (
    BASE_TEMPLATES,
    EVAL_TEMPLATES_PER_CLASS,
    TRAIN_TEMPLATES_PER_CLASS,
    get_eval_templates_per_class,
    get_template_distribution,
    get_train_templates_per_class,
)
from patchpilot.evals.verifier import run_verify


def test_generator_determinism() -> None:
    """Same seed must produce bit-for-bit identical cases, signatures, and hashes."""
    run_a = generate_seeded_eval(base_seed=1337, count=40)
    run_b = generate_seeded_eval(base_seed=1337, count=40)

    assert len(run_a) == len(run_b) == 40
    for case_a, case_b in zip(run_a, run_b, strict=True):
        assert case_a.metadata.case_id == case_b.metadata.case_id
        assert case_a.metadata.content_hash == case_b.metadata.content_hash
        assert case_a.metadata.mutation_signature == case_b.metadata.mutation_signature
        assert case_a.patch_diff == case_b.patch_diff
        assert case_a.files == case_b.files


def test_taxonomy_and_template_coverage() -> None:
    """At least 8 distinct base templates per taxonomy class (56 total across 7 classes)."""
    assert len(BASE_TEMPLATES) >= 56

    distribution = get_template_distribution()
    assert len(distribution) == 7  # 7 SPEC §7 classes

    for tax_name, tpls in distribution.items():
        assert len(tpls) >= 8, f"Taxonomy {tax_name} has fewer than 8 templates: {len(tpls)}"
        assert len(set(tpls)) == len(tpls), f"Duplicate templates registered for {tax_name}"


def test_template_split_disjoint_and_counts() -> None:
    """reserve about 25% of templates per class for seeded_eval ONLY; train must never use them."""
    eval_dist = get_eval_templates_per_class()
    train_dist = get_train_templates_per_class()

    assert len(eval_dist) == 7
    assert len(train_dist) == 7
    assert len(EVAL_TEMPLATES_PER_CLASS) == 7
    assert len(TRAIN_TEMPLATES_PER_CLASS) == 7

    total_eval_tpls = 0
    total_train_tpls = 0

    all_eval_set: set[str] = set()
    all_train_set: set[str] = set()

    for tax in eval_dist:
        eval_tpls = set(eval_dist[tax])
        train_tpls = set(train_dist[tax])

        # Exactly 2 eval (25%) and 6 train (75%)
        assert len(eval_tpls) == 2, f"Expected 2 eval templates for {tax}, got {len(eval_tpls)}"
        assert len(train_tpls) == 6, f"Expected 6 train templates for {tax}, got {len(train_tpls)}"

        # Strict disjointness per class
        overlap = eval_tpls.intersection(train_tpls)
        assert len(overlap) == 0, f"Template overlap in taxonomy {tax}: {overlap}"

        total_eval_tpls += len(eval_tpls)
        total_train_tpls += len(train_tpls)
        all_eval_set.update(eval_tpls)
        all_train_set.update(train_tpls)

    assert total_eval_tpls == 14
    assert total_train_tpls == 42
    assert len(all_eval_set.intersection(all_train_set)) == 0

    # Test actual pool generation enforces disjointness
    eval_cases = generate_seeded_eval(base_seed=1337, count=120)
    train_cases = generate_seeded_train(base_seed=1337, count=360, eval_cases=eval_cases)

    used_eval_templates = {c.metadata.template for c in eval_cases}
    used_train_templates = {c.metadata.template for c in train_cases}

    assert used_eval_templates.issubset(all_eval_set)
    assert used_train_templates.issubset(all_train_set)

    pool_template_overlap = used_eval_templates.intersection(used_train_templates)
    assert len(pool_template_overlap) == 0, (
        f"Templates appeared in both eval and train pools: {pool_template_overlap}"
    )


def test_train_eval_split_zero_overlap() -> None:
    """seeded_eval and seeded_train must have zero overlap by hash and signature."""
    eval_cases = generate_seeded_eval(base_seed=1337, count=120)
    train_cases = generate_seeded_train(base_seed=1337, count=360, eval_cases=eval_cases)

    eval_hashes = {c.metadata.content_hash for c in eval_cases}
    train_hashes = {c.metadata.content_hash for c in train_cases}
    overlap_hashes = eval_hashes.intersection(train_hashes)
    assert len(overlap_hashes) == 0, f"Found {len(overlap_hashes)} overlapping content hashes"

    eval_signatures = {c.metadata.mutation_signature for c in eval_cases}
    train_signatures = {c.metadata.mutation_signature for c in train_cases}
    overlap_signatures = eval_signatures.intersection(train_signatures)
    assert len(overlap_signatures) == 0, f"Found {len(overlap_signatures)} overlapping signatures"


def test_safety_cases_materialization() -> None:
    """The 8 hostile fixtures are cleanly exposed as safety cases."""
    safety_cases = generate_safety_cases()
    assert len(safety_cases) == 8
    fixture_names = {c.metadata.case_id for c in safety_cases}
    assert "network_egress" in fixture_names
    assert "symlink_escape" in fixture_names
    assert "memory_balloon" in fixture_names


def test_frozen_check_fails_when_eval_case_is_edited(tmp_path: Path) -> None:
    """Editing any eval case file or patch after freezing must cause verify-frozen to fail."""
    datasets_dir = tmp_path / "datasets"
    build_and_materialize_all(
        base_seed=1337,
        datasets_dir=datasets_dir,
        eval_count=10,
        train_count=10,
    )

    frozen_file = tmp_path / "FROZEN.json"
    manifest = freeze_eval_split(datasets_dir, frozen_file, base_seed=1337)
    assert manifest.overall_sha256

    # 1. First verification must succeed
    valid, msg = verify_frozen_eval_split(frozen_file, datasets_dir)
    assert valid is True
    assert "OK" in msg

    # 2. Mutate one eval case source file
    eval_case_py = next((datasets_dir / "seeded_eval" / "0001" / "repo").glob("*.py"))
    original_code = eval_case_py.read_text(encoding="utf-8")
    eval_case_py.write_text(original_code + "\n# TAMPERED_CODE_COMMENT\n", encoding="utf-8")

    # 3. Verification must now fail
    valid_tampered, err_msg = verify_frozen_eval_split(frozen_file, datasets_dir)
    assert valid_tampered is False
    assert "Content hash mismatch" in err_msg or "SHA-256 mismatch" in err_msg


@pytest.mark.docker
def test_docker_sandbox_verify_sample_and_cache(tmp_path: Path) -> None:
    """Verify a 10-case sample in Docker: pre-fix must fail, post-fix must pass, caching works."""
    datasets_dir = tmp_path / "datasets"
    build_and_materialize_all(
        base_seed=1337,
        datasets_dir=datasets_dir,
        eval_count=10,
        train_count=10,
    )

    cache_file = tmp_path / "test_cache.json"
    report_file = tmp_path / "test_report.json"

    # Cold run: 10 containers executed sequentially
    report_cold = run_verify(
        datasets_dir=datasets_dir,
        pools=["seeded_eval"],
        limit=10,
        cache_path=cache_file,
        report_path=report_file,
    )

    assert report_cold.total == 10
    assert report_cold.verified == 10
    assert report_cold.failed == 0
    assert report_cold.cached == 0
    assert report_cold.train_eval_overlap == 0

    for c in report_cold.cases:
        assert c.pre_fix_failed is True
        assert c.patch_applied is True
        assert c.post_fix_passed is True
        assert c.verified is True

    # Warm run: identical sample hits the cache immediately (< 1s)
    report_warm = run_verify(
        datasets_dir=datasets_dir,
        pools=["seeded_eval"],
        limit=10,
        cache_path=cache_file,
        report_path=report_file,
    )

    assert report_warm.total == 10
    assert report_warm.verified == 10
    assert report_warm.failed == 0
    assert report_warm.cached == 10
    assert report_warm.duration_s < 2.0
