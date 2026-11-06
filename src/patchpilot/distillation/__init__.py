"""Phase 8 Trace Distillation & QLoRA Fine-Tuning Pipeline (`src/patchpilot/distillation/`)."""

from __future__ import annotations

from patchpilot.distillation.compare import (
    DistillationComparisonReport,
    ModelComparisonMetrics,
    compare_eval_runs,
    format_comparison_markdown,
    load_run_reports,
)
from patchpilot.distillation.exporter import (
    DISTILLATION_SYSTEM_PROMPT,
    ChatMessage,
    DistillationLeakageError,
    DistillationSample,
    ExportSummary,
    build_user_prompt_for_case,
    diff_to_search_replace,
    export_distillation_dataset,
    format_case_as_distillation_sample,
    validate_case_zero_leakage,
)

__all__ = [
    "ChatMessage",
    "DISTILLATION_SYSTEM_PROMPT",
    "DistillationComparisonReport",
    "DistillationLeakageError",
    "DistillationSample",
    "ExportSummary",
    "ModelComparisonMetrics",
    "build_user_prompt_for_case",
    "compare_eval_runs",
    "diff_to_search_replace",
    "export_distillation_dataset",
    "format_case_as_distillation_sample",
    "format_comparison_markdown",
    "load_run_reports",
    "validate_case_zero_leakage",
]
