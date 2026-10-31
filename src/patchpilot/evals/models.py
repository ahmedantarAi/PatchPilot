"""Data models and schemas for Phase 3 evaluation datasets, splits, and verifier."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class BugTaxonomy(StrEnum):
    OFF_BY_ONE = "off_by_one"
    WRONG_OPERATOR = "wrong_operator"
    NONE_HANDLING = "none_handling"
    MISSING_AWAIT = "missing_await"
    RESOURCE_LEAK = "resource_leak"
    BAD_DEFAULT_ARG = "bad_default_arg"
    SQL_INJECTION = "sql_injection"


class CaseMetadata(BaseModel):
    case_id: str
    pool: str
    taxonomy: str
    template: str
    mutation_signature: str
    seed: int
    content_hash: str
    patch_diff: str
    pytest_args: list[str] = Field(default_factory=lambda: ["-q"])
    license: str = "MIT"
    license_source: str = "self-generated"
    description: str = ""


class CaseDefinition(BaseModel):
    metadata: CaseMetadata
    files: dict[str, str]
    symlinks: dict[str, str] = Field(default_factory=dict)
    patch_diff: str


class VerifyCaseResult(BaseModel):
    case_id: str
    pool: str
    content_hash: str
    verified: bool
    pre_fix_failed: bool
    patch_applied: bool
    post_fix_passed: bool
    duration_s: float
    cached: bool = False
    error: str | None = None
    container_name: str = ""
    image_digest: str = ""


class VerifySummaryReport(BaseModel):
    total: int
    verified: int
    failed: int
    cached: int
    duration_s: float
    train_eval_overlap: int = 0
    cases: list[VerifyCaseResult] = Field(default_factory=list)


class FrozenPoolManifest(BaseModel):
    count: int
    cases: dict[str, str] = Field(default_factory=dict)  # case_id -> content_hash


class FrozenManifest(BaseModel):
    generator_version: str = "1.0.0"
    base_seed: int
    pools: dict[str, FrozenPoolManifest] = Field(default_factory=dict)
    overall_sha256: str


class SplitManifest(BaseModel):
    generator_version: str = "1.0.0"
    base_seed: int
    seeded_eval_count: int
    seeded_train_count: int
    safety_count: int
    seeded_eval_hashes: list[str]
    seeded_train_hashes: list[str]
    safety_hashes: list[str]
    overlap_count: int = 0
    eval_template_count: int = 14
    train_template_count: int = 42
    template_overlap_count: int = 0
