"""Tests for Phase 3a: generator determinism, split separation, freeze, and sandbox verify."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from patchpilot.config import LangfuseSettings, Settings
from patchpilot.evals.__main__ import main as evals_main
from patchpilot.evals.freeze import freeze_eval_split, verify_frozen_eval_split
from patchpilot.evals.generator import (
    _select_buggy_source_for_hash,
    build_and_materialize_all,
    generate_safety_cases,
    generate_seeded_eval,
    generate_seeded_train,
)
from patchpilot.evals.runner import (
    EvalSummaryReport,
    _build_effective_run_id,
    _compute_config_hash,
    _resolve_default_model_name,
    run_eval_pool,
)
from patchpilot.evals.taxonomy import (
    BASE_TEMPLATES,
    EVAL_TEMPLATES_PER_CLASS,
    TRAIN_TEMPLATES_PER_CLASS,
    compute_source_hash,
    get_eval_templates_per_class,
    get_template_distribution,
    get_train_templates_per_class,
)
from patchpilot.evals.verifier import run_verify
from patchpilot.sandbox.runner import JUnitReport, SandboxResult


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


def test_evals_run_cli_openrouter_and_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`evals run --provider openrouter` parses cleanly and resolves the OpenRouter model."""
    captured: dict[str, Any] = {}

    async def fake_run_eval_pool(**kwargs: Any) -> tuple[list[Any], EvalSummaryReport]:
        captured.update(kwargs)
        summary = EvalSummaryReport(
            run_id="seeded_train_openrouter_s1337_deadbeef_limit1",
            pool=str(kwargs["pool"]),
            provider=str(kwargs["provider_name"]),
            model_name="google/gemma-4-31b-it:free",
            seed=int(kwargs["seed"]),
            n_runs=int(kwargs["runs"]),
            prompt_version_hash="0" * 64,
            total_observations=1,
            pass_at_1_count=1,
            pass_at_1_rate=1.0,
            pass_at_1_ci_95=(0.206549, 1.0),
            pass_at_3_count=1,
            pass_at_3_rate=1.0,
            pass_at_3_ci_95=(0.206549, 1.0),
        )
        return [], summary

    monkeypatch.setattr("patchpilot.evals.__main__.run_eval_pool", fake_run_eval_pool)

    rc = evals_main(
        [
            "run",
            "--pool",
            "seeded_train",
            "--provider",
            "openrouter",
            "--limit",
            "1",
            "--results-dir",
            str(tmp_path / "results"),
        ]
    )
    assert rc == 0
    assert captured["pool"] == "seeded_train"
    assert captured["provider_name"] == "openrouter"
    assert captured["limit"] == 1

    cfg_openrouter = Settings(
        _env_file=None,
        openrouter_model="google/gemma-4-31b-it:free",
        frontier_provider="openrouter",
    )
    assert _resolve_default_model_name("openrouter", cfg_openrouter) == "google/gemma-4-31b-it:free"
    assert (
        _resolve_default_model_name("router", cfg_openrouter)
        == "router(qwen2.5-coder:7b+google/gemma-4-31b-it:free)"
    )

    cfg_empty_openrouter = Settings(_env_file=None, openrouter_model="")
    assert _resolve_default_model_name("openrouter", cfg_empty_openrouter) == "openrouter"

    cfg_gemini_frontier = Settings(
        _env_file=None,
        frontier_provider="gemini",
        gemini_model="gemini-3.6-flash",
    )
    assert (
        _resolve_default_model_name("router", cfg_gemini_frontier)
        == "router(qwen2.5-coder:7b+gemini-3.6-flash)"
    )


def test_evals_limit_run_uses_distinct_run_id_and_does_not_overwrite_full_report(
    tmp_path: Path,
) -> None:
    """Partial `--limit` runs must get distinct config_hash/run_id/report paths from full runs."""
    full_cfg_hash = _compute_config_hash(
        pool="seeded_train",
        provider="mock",
        model_name="mock-1",
        seed=1337,
        prompt_version_hash="a" * 64,
        limit=None,
    )
    limit_cfg_hash = _compute_config_hash(
        pool="seeded_train",
        provider="mock",
        model_name="mock-1",
        seed=1337,
        prompt_version_hash="a" * 64,
        limit=1,
    )
    assert full_cfg_hash != limit_cfg_hash

    full_run_id = _build_effective_run_id(
        pool="seeded_train",
        provider_name="mock",
        seed=1337,
        config_hash=full_cfg_hash,
        limit=None,
    )
    limit_run_id = _build_effective_run_id(
        pool="seeded_train",
        provider_name="mock",
        seed=1337,
        config_hash=limit_cfg_hash,
        limit=1,
    )
    assert full_run_id != limit_run_id
    assert not full_run_id.endswith("_limit1")
    assert limit_run_id.endswith("_limit1")
    assert (
        _build_effective_run_id(
            pool="seeded_train",
            provider_name="mock",
            seed=1337,
            config_hash=limit_cfg_hash,
            limit=1,
            run_id="custom_smoke",
        )
        == "custom_smoke_limit1"
    )
    assert (
        _build_effective_run_id(
            pool="seeded_train",
            provider_name="mock",
            seed=1337,
            config_hash=limit_cfg_hash,
            limit=1,
            run_id="custom_smoke_limit1",
        )
        == "custom_smoke_limit1"
    )

    datasets_dir = tmp_path / "datasets"
    results_dir = tmp_path / "results"
    build_and_materialize_all(
        base_seed=1337,
        datasets_dir=datasets_dir,
        eval_count=2,
        train_count=2,
    )

    async def fast_verify(fs: dict[str, str], patch: str) -> SandboxResult:
        return SandboxResult(
            exit_code=0,
            patch_applied=True,
            junit=JUnitReport(passed=1, failures=0, errors=0),
        )

    async def _run() -> tuple[Any, Any]:
        full_reports, _ = await run_eval_pool(
            pool="seeded_train",
            provider_name="mock",
            runs=1,
            seed=1337,
            limit=None,
            datasets_dir=datasets_dir,
            results_dir=results_dir,
            verify_fn=fast_verify,
            langfuse_settings=LangfuseSettings(_env_file=None),
        )
        limit_reports, _ = await run_eval_pool(
            pool="seeded_train",
            provider_name="mock",
            runs=1,
            seed=1337,
            limit=1,
            datasets_dir=datasets_dir,
            results_dir=results_dir,
            verify_fn=fast_verify,
            langfuse_settings=LangfuseSettings(_env_file=None),
        )
        return full_reports[0], limit_reports[0]

    full_report, limit_report = asyncio.run(_run())

    full_path = Path(full_report.output_file)
    limit_path = Path(limit_report.output_file)
    assert full_path != limit_path
    assert full_path.is_file()
    assert limit_path.is_file()
    assert "_limit1" not in full_path.name
    assert "_limit1" in limit_path.name

    saved_full = json.loads(full_path.read_text(encoding="utf-8"))
    saved_limit = json.loads(limit_path.read_text(encoding="utf-8"))
    assert saved_full["total_cases"] == 2
    assert saved_limit["total_cases"] == 1


def test_select_buggy_source_for_hash_excludes_tests() -> None:
    """Source hashing must select non-test .py module(s), never test_*.py or test helpers."""
    test_src = "def test_bug():\n    assert add(1, 2) == 3\n"
    mod_src = "def add(a: int, b: int) -> int:\n    return a - b\n"

    # Even when test_foo.py appears first in insertion order, non-test source is hashed
    files_test_first = {
        "test_foo.py": test_src,
        "foo_test.py": test_src,
        "tests/helpers.py": "HELPER = 1\n",
        "conftest.py": "import pytest\n",
        "README.md": "# docs\n",
        "foo.py": mod_src,
    }
    selected = _select_buggy_source_for_hash(files_test_first)
    assert selected == mod_src
    assert compute_source_hash(selected) == compute_source_hash(mod_src)
    assert compute_source_hash(selected) != compute_source_hash(test_src)

    # Multiple non-test .py modules are combined deterministically in sorted path order
    multi_files = {
        "test_pkg.py": test_src,
        "pkg/zeta.py": "Z = 2\n",
        "pkg/alpha.py": "A = 1\n",
    }
    multi_selected = _select_buggy_source_for_hash(multi_files)
    assert multi_selected == "# pkg/alpha.py\nA = 1\n\n# pkg/zeta.py\nZ = 2\n"
    assert "test_pkg.py" not in multi_selected

    # Test-only files yield empty string
    assert _select_buggy_source_for_hash({"test_only.py": test_src}) == ""

    # Generated seeded_eval case also selects its non-test source rather than test_*.py
    sample_case = generate_seeded_eval(base_seed=1337, count=1)[0]
    non_test_keys = [
        k for k in sample_case.files if not k.startswith("test_") and k.endswith(".py")
    ]
    assert len(non_test_keys) == 1
    assert _select_buggy_source_for_hash(sample_case.files) == sample_case.files[non_test_keys[0]]
