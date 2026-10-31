"""PatchPilot evaluation dataset generation, verification, and freezing module (Phase 3a)."""

from __future__ import annotations

from patchpilot.evals.freeze import freeze_eval_split, verify_frozen_eval_split
from patchpilot.evals.generator import (
    build_and_materialize_all,
    generate_safety_cases,
    generate_seeded_eval,
    generate_seeded_train,
    load_pool_cases,
    materialize_pool,
)
from patchpilot.evals.interfaces import (
    DatasetBuilder,
    FindingsEvalProvider,
    RealworldEvalProvider,
)
from patchpilot.evals.models import (
    BugTaxonomy,
    CaseDefinition,
    CaseMetadata,
    FrozenManifest,
    FrozenPoolManifest,
    SplitManifest,
    VerifyCaseResult,
    VerifySummaryReport,
)
from patchpilot.evals.taxonomy import (
    BASE_TEMPLATES,
    TAXONOMY_TEMPLATE_GENERATORS,
    build_seeded_case,
    compute_case_content_hash,
    compute_source_hash,
    compute_unified_diff,
    get_template_distribution,
)
from patchpilot.evals.verifier import (
    VerifyCache,
    compute_verify_cache_key,
    run_verify,
    verify_single_case,
)

__all__ = [
    "BASE_TEMPLATES",
    "BugTaxonomy",
    "CaseDefinition",
    "CaseMetadata",
    "DatasetBuilder",
    "FindingsEvalProvider",
    "FrozenManifest",
    "FrozenPoolManifest",
    "RealworldEvalProvider",
    "SplitManifest",
    "TAXONOMY_TEMPLATE_GENERATORS",
    "VerifyCache",
    "VerifyCaseResult",
    "VerifySummaryReport",
    "build_and_materialize_all",
    "build_seeded_case",
    "compute_case_content_hash",
    "compute_source_hash",
    "compute_unified_diff",
    "compute_verify_cache_key",
    "freeze_eval_split",
    "generate_safety_cases",
    "generate_seeded_eval",
    "generate_seeded_train",
    "get_template_distribution",
    "load_pool_cases",
    "materialize_pool",
    "run_verify",
    "verify_frozen_eval_split",
    "verify_single_case",
]
