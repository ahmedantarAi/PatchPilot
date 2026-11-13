"""Assembly of the LangGraph state machine: analyze -> propose <-> verify -> report."""

from __future__ import annotations

import logging
from typing import Any, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from patchpilot.pipeline.models import PipelineState
from patchpilot.pipeline.nodes import (
    VerifyCallable,
    compute_prompt_version_hash,
    create_analyze_node,
    create_propose_node,
    create_report_node,
    create_verify_node,
    should_retry,
)
from patchpilot.providers.base import LLMProvider
from patchpilot.retrieval import Embedder, QdrantStore, case_collection_name
from patchpilot.sandbox.runner import SandboxRunner
from patchpilot.tracing import PatchPilotTracer

logger = logging.getLogger("patchpilot.pipeline")


def build_pipeline_graph(
    provider: LLMProvider,
    sandbox_runner: SandboxRunner | None = None,
    verify_fn: VerifyCallable | None = None,
    qdrant: QdrantStore | None = None,
    embedder: Embedder | None = None,
    tracer: PatchPilotTracer | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build and compile the LangGraph state machine for PatchPilot."""
    builder: Any = StateGraph(PipelineState)

    analyze_node = create_analyze_node(
        provider=provider,
        qdrant=qdrant,
        embedder=embedder,
        sandbox_runner=sandbox_runner,
        tracer=tracer,
    )
    propose_node = create_propose_node(provider=provider, tracer=tracer)
    verify_node = create_verify_node(
        sandbox_runner=sandbox_runner,
        verify_fn=verify_fn,
        tracer=tracer,
    )
    rep_node = create_report_node(tracer=tracer)

    builder.add_node("analyze", analyze_node)
    builder.add_node("propose", propose_node)
    builder.add_node("verify", verify_node)
    builder.add_node("report", rep_node)

    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "propose")
    builder.add_edge("propose", "verify")
    builder.add_conditional_edges(
        "verify",
        should_retry,
        {
            "propose": "propose",
            "report": "report",
        },
    )
    builder.add_edge("report", END)

    return cast(CompiledStateGraph[Any, Any, Any, Any], builder.compile())


async def run_pipeline(
    initial_state: PipelineState,
    provider: LLMProvider,
    sandbox_runner: SandboxRunner | None = None,
    verify_fn: VerifyCallable | None = None,
    qdrant: QdrantStore | None = None,
    embedder: Embedder | None = None,
    tracer: PatchPilotTracer | None = None,
) -> PipelineState:
    """Execute the pipeline state machine to completion and return final PipelineState."""
    active_tracer = tracer or PatchPilotTracer()
    prompt_hash = compute_prompt_version_hash()
    active_tracer.start_trace(
        case_id=initial_state.case_id,
        metadata={
            "provider": initial_state.provider,
            "data_class": initial_state.data_class,
            "prompt_version_hash": prompt_hash,
        },
    )

    col_name = case_collection_name(initial_state.files)
    final_state: PipelineState | None = None
    try:
        if qdrant is not None and embedder is not None:
            try:
                await qdrant.index_case_files(initial_state.files, embedder)
            except Exception as exc:
                logger.info("Qdrant per-case indexing failed or degraded: %s", exc)

        graph = build_pipeline_graph(
            provider=provider,
            sandbox_runner=sandbox_runner,
            verify_fn=verify_fn,
            qdrant=qdrant,
            embedder=embedder,
            tracer=active_tracer,
        )
        result_dict = await graph.ainvoke(initial_state)
        validated = PipelineState.model_validate(result_dict)
        total_prompt = sum(c.prompt_tokens for c in active_tracer.llm_calls)
        total_comp = sum(c.completion_tokens for c in active_tracer.llm_calls)
        total_lat = round(sum(c.latency_s for c in active_tracer.llm_calls), 6)
        model_name = (
            active_tracer.llm_calls[-1].model
            if active_tracer.llm_calls
            else getattr(provider, "default_model", provider.name)
        )
        final_state = validated.model_copy(
            update={
                "trace_id": active_tracer.trace_id,
                "prompt_version_hash": prompt_hash,
                "model_name": str(model_name),
                "llm_calls": list(active_tracer.llm_calls),
                "sandbox_runs": list(active_tracer.sandbox_runs),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_comp,
                "total_tokens": total_prompt + total_comp,
                "total_latency_s": total_lat,
            }
        )
        return final_state
    finally:
        if qdrant is not None and embedder is not None:
            try:
                await qdrant.delete_collection(col_name)
            except Exception as exc:
                logger.info("Qdrant per-case collection cleanup degraded: %s", exc)
        try:
            await active_tracer.finish_trace(
                case_id=initial_state.case_id,
                verified=final_state.verified if final_state else False,
                pass_at_1=final_state.pass_at_1 if final_state else False,
                pass_at_k=final_state.pass_at_k if final_state else False,
                attempts=final_state.attempt if final_state else 0,
                metadata={"prompt_version_hash": prompt_hash},
            )
        except Exception as exc:
            logger.debug("Langfuse trace finalization degraded: %s", exc)
