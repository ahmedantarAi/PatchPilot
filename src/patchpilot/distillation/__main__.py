"""CLI entrypoint for Phase 8 trace distillation and QLoRA comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from patchpilot.distillation.compare import (
    compare_eval_runs,
    format_comparison_markdown,
    load_run_reports,
)
from patchpilot.distillation.exporter import (
    DistillationLeakageError,
    export_distillation_dataset,
)
from patchpilot.evals.runner import EvalIntegrityError

DEFAULT_DATASETS_DIR = Path("evals/datasets")
DEFAULT_FROZEN_FILE = Path("evals/FROZEN.json")
DEFAULT_RESULTS_DIR = Path("evals/results")
DEFAULT_OUTPUT_JSONL = Path("data/distillation_train.jsonl")


def cmd_export_train(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    try:
        summary = export_distillation_dataset(
            output_path=output_path,
            datasets_dir=Path(args.datasets_dir),
            frozen_path=Path(args.frozen),
        )
    except (DistillationLeakageError, ValueError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        f"exported {summary.sample_count} seeded_train samples to {summary.output_path} "
        f"sha256={summary.sha256} (train/eval overlap=0)"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    try:
        base_reports = load_run_reports(args.baseline, results_dir=results_dir)
        cand_reports = load_run_reports(args.candidate, results_dir=results_dir)
        report = compare_eval_runs(
            base_reports,
            cand_reports,
            datasets_dir=Path(args.datasets_dir),
            frozen_path=Path(args.frozen),
            require_min_runs=int(args.min_runs),
        )
    except (EvalIntegrityError, FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    md = format_comparison_markdown(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patchpilot.distillation",
        description="PatchPilot Phase 8 distillation exporter and QLoRA comparison CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_export = subparsers.add_parser(
        "export-train",
        help="Export seeded_train dataset to ChatML JSONL for Colab QLoRA fine-tuning",
    )
    p_export.add_argument(
        "--output",
        "--out",
        dest="output",
        type=str,
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Output JSONL path (default: data/distillation_train.jsonl)",
    )
    p_export.add_argument(
        "--datasets-dir",
        type=str,
        default=str(DEFAULT_DATASETS_DIR),
        help="Datasets root directory",
    )
    p_export.add_argument(
        "--frozen",
        type=str,
        default=str(DEFAULT_FROZEN_FILE),
        help="Path to FROZEN.json",
    )

    p_compare = subparsers.add_parser(
        "compare",
        help="Compare baseline vs distilled candidate evaluation runs on seeded_eval",
    )
    p_compare.add_argument("--baseline", required=True, help="Baseline run ID prefix or JSON path")
    p_compare.add_argument(
        "--candidate", required=True, help="Candidate run ID prefix or JSON path"
    )
    p_compare.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing evaluation run JSON reports",
    )
    p_compare.add_argument(
        "--datasets-dir",
        type=str,
        default=str(DEFAULT_DATASETS_DIR),
        help="Datasets root directory",
    )
    p_compare.add_argument(
        "--frozen",
        type=str,
        default=str(DEFAULT_FROZEN_FILE),
        help="Path to FROZEN.json",
    )
    p_compare.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum number of runs required per configuration (default: 3)",
    )
    p_compare.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write markdown comparison report",
    )

    args = parser.parse_args(argv)
    if args.command == "export-train":
        return cmd_export_train(args)
    if args.command == "compare":
        return cmd_compare(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
