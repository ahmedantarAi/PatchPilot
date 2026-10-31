"""CLI entrypoint for PatchPilot: python -m patchpilot run ..."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from patchpilot.config import Settings
from patchpilot.pipeline.graph import run_pipeline
from patchpilot.pipeline.models import PipelineRunSummary, PipelineState
from patchpilot.providers import get_provider
from patchpilot.providers.base import LLMProvider
from patchpilot.providers.mock import MockProvider
from patchpilot.retrieval import HashEmbedder, OllamaEmbedder, QdrantStore
from patchpilot.sandbox.runner import SandboxRunner

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("patchpilot.cli")


def _resolve_case_dir(case_arg: str) -> Path:
    """Resolve a case argument (e.g. 'seeded_eval/0001' or '0001') to its directory."""
    raw = Path(case_arg)
    if raw.is_dir():
        return raw
    candidate1 = Path("evals/datasets") / case_arg
    if candidate1.is_dir():
        return candidate1
    candidate2 = Path("evals/datasets/seeded_eval") / case_arg
    if candidate2.is_dir():
        return candidate2
    raise FileNotFoundError(f"Case directory not found for {case_arg!r}")


def _load_case_files(case_dir: Path) -> tuple[dict[str, str], str, str]:
    """Load files, target file name, and known patch from a case directory."""
    repo_dir = case_dir / "repo"
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"repo directory not found in {case_dir}")

    files: dict[str, str] = {}
    for p in sorted(repo_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(repo_dir)).replace("\\", "/")
            files[rel] = p.read_text(encoding="utf-8")

    known_patch = ""
    patch_path = case_dir / "patch.diff"
    if patch_path.is_file():
        known_patch = patch_path.read_text(encoding="utf-8")

    target_file = ""
    meta_path = case_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and "patch_diff" in meta:
                diff_str = str(meta["patch_diff"])
                for line in diff_str.splitlines():
                    if line.startswith("--- a/"):
                        target_file = line[6:].strip()
                        break
        except (ValueError, OSError):
            pass

    if not target_file:
        py_files = [f for f in files if f.endswith(".py") and not Path(f).name.startswith("test_")]
        if py_files:
            target_file = py_files[0]

    return files, target_file, known_patch


async def _run_command(case_arg: str, provider_name: str) -> int:
    settings = Settings()
    case_dir = _resolve_case_dir(case_arg)
    files, target_file, known_patch = _load_case_files(case_dir)

    # Provider setup
    provider: LLMProvider
    if provider_name == "mock":
        provider = MockProvider(canned=known_patch)
    else:
        provider = get_provider(provider_name, settings)  # type: ignore[arg-type]

    # Embedder setup
    embedder = (
        OllamaEmbedder(base_url=settings.ollama_base_url, model=settings.ollama_model)
        if provider_name == "ollama"
        else HashEmbedder(dim=64)
    )

    # Qdrant setup
    qdrant = QdrantStore(base_url="http://127.0.0.1:6333", timeout_s=1.0)
    try:
        if await qdrant.is_healthy():
            await qdrant.ensure_collection("repo_code", dim=64)
            points = []
            non_test_files = [
                (p, c)
                for p, c in files.items()
                if p.endswith(".py") and not Path(p).name.startswith("test_")
            ]
            if non_test_files:
                texts = [f"{p}:\n{c}" for p, c in non_test_files]
                vecs = await embedder.embed(texts)
                for i, ((p, c), v) in enumerate(zip(non_test_files, vecs, strict=True)):
                    points.append({"id": i + 1, "vector": v, "payload": {"file": p, "text": c}})
                await qdrant.upsert("repo_code", points)
    except Exception as exc:
        logger.info("Qdrant indexing failed or degraded: %s", exc)

    initial_state = PipelineState(
        case_id=case_arg,
        files=files,
        target_file=target_file,
        data_class="public",  # Seeded eval cases are public per SPEC §9 and Req 5
        provider=provider_name,
    )

    final_state = await run_pipeline(
        initial_state,
        provider=provider,
        sandbox_runner=SandboxRunner(),
        qdrant=qdrant,
        embedder=embedder,
    )

    summary = PipelineRunSummary(
        case=case_arg,
        provider=provider_name,
        patch_applies=final_state.patch_applies,
        target_test_passes=final_state.target_test_passes,
        regressions=final_state.regressions,
        verified=final_state.verified,
    )
    print(summary.model_dump_json())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="patchpilot", description="PatchPilot CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser("run", help="Run the LangGraph repair pipeline on a case")
    run_parser.add_argument("--case", required=True, help="Case identifier (e.g. seeded_eval/0001)")
    run_parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "ollama", "gemini"],
        help="LLM provider name (default: mock)",
    )

    args = parser.parse_args()

    if args.subcommand == "run":
        exit_code = asyncio.run(_run_command(args.case, args.provider))
        sys.exit(exit_code)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
