"""LangGraph pipeline for PatchPilot (Phase 4)."""

from __future__ import annotations

from patchpilot.pipeline.graph import build_pipeline_graph, run_pipeline
from patchpilot.pipeline.models import Finding, PipelineRunSummary, PipelineState
from patchpilot.pipeline.nodes import (
    VerifyCallable,
    create_analyze_node,
    create_propose_node,
    create_verify_node,
    report_node,
    should_retry,
)
from patchpilot.pipeline.normalization import (
    normalize_patch,
    normalize_search_replace,
    normalize_unified_diff,
)

__all__ = [
    "Finding",
    "PipelineState",
    "PipelineRunSummary",
    "VerifyCallable",
    "build_pipeline_graph",
    "create_analyze_node",
    "create_propose_node",
    "create_verify_node",
    "normalize_patch",
    "normalize_search_replace",
    "normalize_unified_diff",
    "report_node",
    "run_pipeline",
    "should_retry",
]
