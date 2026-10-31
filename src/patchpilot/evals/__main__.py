"""CLI entrypoint for Phase 3 evaluation datasets: build, verify, freeze, and verify-frozen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from patchpilot.evals.freeze import freeze_eval_split, verify_frozen_eval_split
from patchpilot.evals.generator import build_and_materialize_all
from patchpilot.evals.taxonomy import BASE_TEMPLATES, get_template_distribution
from patchpilot.evals.verifier import run_verify

DEFAULT_DATASETS_DIR = Path("evals/datasets")
DEFAULT_FROZEN_FILE = Path("evals/FROZEN.json")
DEFAULT_REPORT_FILE = Path("evals/results/verify_report.json")


def cmd_build(args: argparse.Namespace) -> int:
    datasets_dir = Path(args.datasets_dir)
    seed = int(args.seed)

    distribution = get_template_distribution()
    print("Bug taxonomy template distribution:")
    for tax_name, tpls in sorted(distribution.items()):
        print(f"  {tax_name}: {len(tpls)} unique templates ({', '.join(tpls)})")
    print(f"Total unique base templates: {len(BASE_TEMPLATES)}")

    manifest = build_and_materialize_all(
        base_seed=seed,
        datasets_dir=datasets_dir,
        eval_count=args.eval_count,
        train_count=args.train_count,
    )

    print(
        f"built seeded_eval={manifest.seeded_eval_count} "
        f"seeded_train={manifest.seeded_train_count} "
        f"safety={manifest.safety_count}"
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    datasets_dir = Path(args.datasets_dir)
    report_file = Path(args.report_out)
    limit = args.limit or args.sample

    report = run_verify(
        datasets_dir=datasets_dir,
        limit=limit,
        report_path=report_file,
    )

    if report.train_eval_overlap > 0:
        print(
            f"FAIL: train/eval overlap detected ({report.train_eval_overlap} overlapping cases)",
            file=sys.stderr,
        )
        return 1

    if report.failed > 0:
        print(
            f"FAIL: {report.failed}/{report.total} cases failed verification",
            file=sys.stderr,
        )
        for c in report.cases:
            if not c.verified:
                print(
                    f"  - {c.pool}/{c.case_id}: pre_fail={c.pre_fix_failed} "
                    f"patch_applied={c.patch_applied} post_pass={c.post_fix_passed} err={c.error}",
                    file=sys.stderr,
                )
        return 1

    print(
        f"OK: every seeded case fails before fix and passes after fix; "
        f"train/eval overlap = {report.train_eval_overlap}"
    )
    print(
        f"Verified {report.verified}/{report.total} cases "
        f"({report.cached} cached) in {report.duration_s:.2f}s"
    )
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    datasets_dir = Path(args.datasets_dir)
    frozen_file = Path(args.out)
    seed = int(args.seed)

    manifest = freeze_eval_split(
        datasets_dir=datasets_dir,
        output_path=frozen_file,
        base_seed=seed,
    )
    print(f"wrote {frozen_file.as_posix()} sha256={manifest.overall_sha256}")
    return 0


def cmd_verify_frozen(args: argparse.Namespace) -> int:
    datasets_dir = Path(args.datasets_dir)
    frozen_file = Path(args.frozen)

    valid, msg = verify_frozen_eval_split(
        frozen_path=frozen_file,
        datasets_dir=datasets_dir,
    )
    if valid:
        print(msg)
        return 0
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patchpilot.evals",
        description="PatchPilot Phase 3 evaluation harness CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = subparsers.add_parser("build", help="Build deterministic seeded datasets")
    p_build.add_argument("--seed", type=int, default=1337, help="Base PRNG seed (default: 1337)")
    p_build.add_argument("--eval-count", type=int, default=120, help="Number of eval cases")
    p_build.add_argument("--train-count", type=int, default=360, help="Number of train cases")
    p_build.add_argument(
        "--datasets-dir",
        type=str,
        default=str(DEFAULT_DATASETS_DIR),
        help="Datasets output directory",
    )

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify seeded cases in sandbox")
    p_verify.add_argument(
        "--datasets-dir",
        type=str,
        default=str(DEFAULT_DATASETS_DIR),
        help="Datasets root directory",
    )
    p_verify.add_argument("--limit", type=int, default=None, help="Limit number of cases verified")
    p_verify.add_argument(
        "--sample", type=int, default=None, help="Sample N cases for verification"
    )
    p_verify.add_argument(
        "--report-out",
        type=str,
        default=str(DEFAULT_REPORT_FILE),
        help="Path to write JSON verify report",
    )

    # freeze
    p_freeze = subparsers.add_parser("freeze", help="Freeze evaluation split to FROZEN.json")
    p_freeze.add_argument(
        "--datasets-dir",
        type=str,
        default=str(DEFAULT_DATASETS_DIR),
        help="Datasets root directory",
    )
    p_freeze.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_FROZEN_FILE),
        help="Output FROZEN.json path",
    )
    p_freeze.add_argument("--seed", type=int, default=1337, help="Base seed to record")

    # verify-frozen
    p_verify_frozen = subparsers.add_parser(
        "verify-frozen", help="Verify eval dataset against frozen manifest"
    )
    p_verify_frozen.add_argument(
        "--datasets-dir",
        type=str,
        default=str(DEFAULT_DATASETS_DIR),
        help="Datasets root directory",
    )
    p_verify_frozen.add_argument(
        "--frozen",
        type=str,
        default=str(DEFAULT_FROZEN_FILE),
        help="Path to FROZEN.json",
    )

    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "freeze":
        return cmd_freeze(args)
    elif args.command == "verify-frozen":
        return cmd_verify_frozen(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
