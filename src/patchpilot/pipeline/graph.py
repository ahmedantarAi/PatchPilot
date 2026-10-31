"""Assembly of the LangGraph state machine: analyze -> propose <-> verify -> report."""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from patchpilot.pipeline.models import PipelineState
from patchpilot.pipeline.nodes import (
    VerifyCallable,
    create_analyze_node,
    create_propose_node,
    create_verify_node,
    report_node,
    should_retry,
)
from patchpilot.providers.base import LLMProvider
from patchpilot.retrieval import Embedder, QdrantStore
from patchpilot.sandbox.runner import SandboxRunner


def build_pipeline_graph(
    provider: LLMProvider,
    sandbox_runner: SandboxRunner | None = None,
    verify_fn: VerifyCallable | None = None,
    qdrant: QdrantStore | None = None,
    embedder: Embedder | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build and compile the LangGraph state machine for PatchPilot."""
    builder: Any = StateGraph(PipelineState)

    analyze_node = create_analyze_node(provider=provider, qdrant=qdrant, embedder=embedder)
    propose_node = create_propose_node(provider=provider)
    verify_node = create_verify_node(sandbox_runner=sandbox_runner, verify_fn=verify_fn)

    builder.add_node("analyze", analyze_node)
    builder.add_node("propose", propose_node)
    builder.add_node("verify", verify_node)
    builder.add_node("report", report_node)

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
) -> PipelineState:
    """Execute the pipeline state machine to completion and return final PipelineState."""
    graph = build_pipeline_graph(
        provider=provider,
        sandbox_runner=sandbox_runner,
        verify_fn=verify_fn,
        qdrant=qdrant,
        embedder=embedder,
    )
    result_dict = await graph.ainvoke(initial_state)
    return PipelineState.model_validate(result_dict)
