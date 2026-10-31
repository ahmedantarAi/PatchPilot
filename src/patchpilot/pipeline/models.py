"""Typed state and data models for the PatchPilot LangGraph pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from patchpilot.providers.base import DataClass


class Finding(BaseModel):
    """Typed finding emitted by static analysis and LLM diff inspection (SPEC §3)."""

    file: str
    line: int
    category: str
    explanation: str
    fixable: bool = True


class PipelineState(BaseModel):
    """Typed Pydantic state passed through the LangGraph pipeline nodes."""

    case_id: str
    files: dict[str, str] = Field(default_factory=dict)
    target_file: str = ""
    findings: list[Finding] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)
    raw_patch: str = ""
    patch_diff: str = ""
    feedback: str = ""
    attempt: int = 0
    max_attempts: int = 3
    patch_applies: bool = False
    target_test_passes: bool = False
    regressions: int = 0
    verified: bool = False
    pass_at_1: bool = False
    pass_at_k: bool = False
    error: str | None = None
    provider: str = "mock"
    data_class: DataClass = "private"
    reported_patch: str = ""


class PipelineRunSummary(BaseModel):
    """Single-line JSON output format matching PLAN.md Phase 4 acceptance contract."""

    case: str
    provider: str
    patch_applies: bool
    target_test_passes: bool
    regressions: int
    verified: bool
