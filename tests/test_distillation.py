"""Phase 8 unit tests for distillation export, zero-leakage guard, notebook, and comparison."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from patchpilot.distillation import (
    DistillationLeakageError,
    DistillationSample,
    compare_eval_runs,
    export_distillation_dataset,
    format_comparison_markdown,
)
from patchpilot.evals.generator import generate_seeded_eval, load_pool_cases
from patchpilot.evals.runner import (
    EvalCaseRunResult,
    EvalIntegrityError,
    EvalRunReport,
    wilson_ci_95,
)
from patchpilot.evals.taxonomy import EVAL_TEMPLATES_PER_CLASS
from patchpilot.pipeline.nodes import compute_prompt_version_hash
from patchpilot.pipeline.normalization import normalize_patch


def test_export_train_determinism_and_zero_leakage(tmp_path: Path) -> None:
    """Exporting `seeded_train` produces 360 deterministic samples with zero eval overlap."""
    out1 = tmp_path / "train_1.jsonl"
    out2 = tmp_path / "train_2.jsonl"

    summary1 = export_distillation_dataset(out1)
    summary2 = export_distillation_dataset(out2)

    assert summary1.sample_count == 360
    assert summary2.sample_count == 360
    assert summary1.sha256 == summary2.sha256
    assert summary1.overlap_count == 0
    assert out1.read_bytes() == out2.read_bytes()

    eval_templates = {tpl for tpls in EVAL_TEMPLATES_PER_CLASS.values() for tpl in tpls}
    lines = [line for line in out1.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 360

    for line in lines:
        sample = DistillationSample.model_validate_json(line)
        assert sample.pool == "seeded_train"
        assert sample.template not in eval_templates


def test_export_train_rejects_seeded_eval_case_or_hash_leakage(tmp_path: Path) -> None:
    """Attempting to export any `seeded_eval` case or hash raises `DistillationLeakageError`."""
    eval_cases = load_pool_cases(Path("evals/datasets"), "seeded_eval")
    if not eval_cases:
        eval_cases = generate_seeded_eval(base_seed=1337, count=5)

    train_cases = load_pool_cases(Path("evals/datasets"), "seeded_train")[:3]
    assert len(train_cases) == 3

    # 1. Direct inclusion of a seeded_eval case
    contaminated_direct = [*train_cases, eval_cases[0]]
    with pytest.raises(DistillationLeakageError, match="seeded_train"):
        export_distillation_dataset(
            tmp_path / "leak_direct.jsonl",
            cases_override=contaminated_direct,
        )

    # 2. Disguised eval case with pool spoofed to 'seeded_train' (caught by template & hash guard)
    spoofed_case = eval_cases[0].model_copy(
        update={"metadata": eval_cases[0].metadata.model_copy(update={"pool": "seeded_train"})}
    )
    with pytest.raises(DistillationLeakageError, match="seeded_eval"):
        export_distillation_dataset(
            tmp_path / "leak_spoofed.jsonl",
            cases_override=[*train_cases, spoofed_case],
        )


def test_exported_jsonl_chatml_and_search_replace_format_roundtrip(tmp_path: Path) -> None:
    """Every exported JSONL sample has valid messages and normalizes to exact patch_diff."""
    train_cases = load_pool_cases(Path("evals/datasets"), "seeded_train")[:10]
    out_file = tmp_path / "sample_10.jsonl"

    summary = export_distillation_dataset(out_file, cases_override=train_cases)
    assert summary.sample_count == 10

    lines = [line for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case, line in zip(train_cases, lines, strict=True):
        sample = DistillationSample.model_validate_json(line)
        assert [m.role for m in sample.messages] == ["system", "user", "assistant"]

        sys_msg, user_msg, asst_msg = sample.messages
        assert "<<<<<<< SEARCH:" in sys_msg.content
        assert "Candidate non-test .py files:" in user_msg.content
        assert "Failing test suite:" in user_msg.content
        assert asst_msg.content.startswith("<<<<<<< SEARCH:")
        assert "\n=======\n" in asst_msg.content
        assert asst_msg.content.endswith(">>>>>>> REPLACE")

        # Round-trip check: normalizing the assistant SEARCH/REPLACE reproduces canonical patch_diff
        reconstructed_diff = normalize_patch(asst_msg.content, case.files)
        assert reconstructed_diff.strip() == case.patch_diff.strip()


def test_colab_notebook_syntax_and_qlora_structure() -> None:
    """Validate `notebooks/train_qlora_colab.ipynb` JSON structure, AST syntax, and QLoRA config."""
    nb_path = Path("notebooks/train_qlora_colab.ipynb")
    assert nb_path.is_file(), "Colab QLoRA notebook is missing"

    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb_data.get("nbformat") == 4
    assert nb_data.get("metadata", {}).get("colab", {}).get("gpuType") == "T4"

    cells = nb_data.get("cells", [])
    assert isinstance(cells, list) and len(cells) >= 5

    code_cells = [c for c in cells if isinstance(c, dict) and c.get("cell_type") == "code"]
    assert len(code_cells) >= 4

    combined_source = ""
    for idx, cell in enumerate(code_cells):
        src_lines = cell.get("source", [])
        src_text = "".join(src_lines) if isinstance(src_lines, list) else str(src_lines)
        combined_source += f"\n# --- Cell {idx} ---\n{src_text}"
        # Every code cell must parse cleanly as valid Python 3.11 syntax
        ast.parse(src_text, filename=f"train_qlora_colab_cell_{idx}.py")

    # Verify required QLoRA hyperparameters and export steps are present
    assert "load_in_4bit=True" in combined_source
    assert "LORA_RANK = 16" in combined_source
    assert "LORA_ALPHA = 32" in combined_source
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert proj in combined_source
    assert "save_pretrained_gguf" in combined_source
    assert "Modelfile" in combined_source


def _make_fake_run_report(
    run_id: str,
    run_index: int,
    model_name: str,
    pass1: int,
    pass3: int,
    total: int = 120,
    time_s: float = 48.0,
    prompt_hash: str | None = None,
) -> EvalRunReport:
    ph = prompt_hash or compute_prompt_version_hash()
    cases = [
        EvalCaseRunResult(
            case_id=f"{i:04d}",
            pool="seeded_eval",
            taxonomy="off_by_one",
            template="pagination_offset",
            run_index=run_index,
            seed=1337,
            provider="ollama",
            model_name=model_name,
            prompt_version_hash=ph,
            attempts=1 if i <= pass1 else (2 if i <= pass3 else 3),
            verified=i <= pass3,
            pass_at_1=i <= pass1,
            pass_at_3=i <= pass3,
            prompt_tokens=200,
            completion_tokens=60,
            total_tokens=260,
            total_time_s=round(time_s / total, 4),
        )
        for i in range(1, total + 1)
    ]
    return EvalRunReport(
        run_id=run_id,
        run_index=run_index,
        timestamp="2026-11-07T00:00:00Z",
        pool="seeded_eval",
        provider="ollama",
        model_name=model_name,
        seed=1337,
        prompt_version_hash=ph,
        total_cases=total,
        pass_at_1_count=pass1,
        pass_at_1_rate=round(pass1 / total, 6),
        pass_at_1_ci_95=wilson_ci_95(pass1, total),
        pass_at_3_count=pass3,
        pass_at_3_rate=round(pass3 / total, 6),
        pass_at_3_ci_95=wilson_ci_95(pass3, total),
        total_prompt_tokens=total * 200,
        total_completion_tokens=total * 60,
        total_tokens=total * 260,
        total_time_s=time_s,
        cases=cases,
    )


def test_distillation_compare_report_and_min_runs_guard() -> None:
    """`compare_eval_runs` enforces `n_runs >= 3`, verifies `FROZEN.json`, and formats deltas."""
    base_runs = [
        _make_fake_run_report("baseline_7b", idx, "qwen2.5-coder:7b", pass1=105, pass3=108)
        for idx in (1, 2, 3)
    ]
    cand_runs = [
        _make_fake_run_report(
            "distilled_7b_qlora",
            idx,
            "patchpilot-qwen2.5-coder:7b-qlora",
            pass1=112,
            pass3=115,
            time_s=42.0,
        )
        for idx in (1, 2, 3)
    ]

    # Fewer than 3 runs must fail closed
    with pytest.raises(EvalIntegrityError, match="n_runs >= 3"):
        compare_eval_runs(base_runs[:2], cand_runs)

    report = compare_eval_runs(base_runs, cand_runs)
    assert report.train_eval_overlap == 0
    assert report.baseline.n_runs == 3
    assert report.candidate.n_runs == 3
    assert report.pass_at_1_delta_pp > 0.0
    assert report.pass_at_3_delta_pp > 0.0
    assert report.latency_delta_s < 0.0

    md = format_comparison_markdown(report)
    assert "frozen split: OK (sha256=" in md
    assert "train/eval overlap: 0" in md
    assert "patchpilot-qwen2.5-coder:7b-qlora" in md
    assert "+5.83 pp" in md
