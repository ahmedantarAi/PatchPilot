"""Thin Langfuse v4 OTLP/REST tracing wrapper with graceful no-op degradation."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel

from patchpilot.config import LangfuseSettings

logger = logging.getLogger("patchpilot.tracing")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _otel_attr(key: str, val: str | int | float | bool) -> dict[str, Any]:
    if isinstance(val, bool):
        return {"key": key, "value": {"boolValue": val}}
    if isinstance(val, int):
        return {"key": key, "value": {"intValue": str(val)}}
    if isinstance(val, float):
        return {"key": key, "value": {"doubleValue": val}}
    return {"key": key, "value": {"stringValue": str(val)}}


class LLMCallMetric(BaseModel):
    """Recorded metrics for a single LLM provider call."""

    node: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0


class SandboxRunMetric(BaseModel):
    """Recorded metrics for a single sandbox container execution."""

    node: str
    exit_code: int
    duration_s: float
    patch_applied: bool
    passed: int = 0
    failures: int = 0
    errors: int = 0
    container_name: str = ""
    image_digest: str = ""


class NodeSpanMetric(BaseModel):
    """Recorded metrics for a single LangGraph pipeline node span."""

    span_id: str
    node: str
    start_time: str
    end_time: str
    duration_s: float


def _is_langfuse_container_running() -> bool:
    """Return True if a local langfuse-web container is currently running."""
    try:
        proc = subprocess.run(
            ["docker", "ps", "--filter", "name=langfuse-web", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        return bool(proc.returncode == 0 and proc.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


class PatchPilotTracer:
    """Thin wrapper over Langfuse v4 OTLP & public REST API; degrades to no-op when down."""

    def __init__(
        self,
        settings: LangfuseSettings | None = None,
        *,
        enabled: bool = True,
        timeout_s: float = 3.0,
        transport: httpx.AsyncBaseTransport | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.settings = settings or LangfuseSettings()
        self._enabled = enabled
        self.timeout_s = timeout_s
        self._transport = transport
        self.trace_id: str = trace_id or uuid.uuid4().hex
        self.root_span_id: str = uuid.uuid4().hex[:16]
        self.trace_start_ns: int = time.time_ns()
        self.trace_started_at: str = _iso_now()
        self.trace_name: str = "patchpilot-case-run"
        self.node_spans: list[NodeSpanMetric] = []
        self.llm_calls: list[LLMCallMetric] = []
        self.sandbox_runs: list[SandboxRunMetric] = []
        self._otel_spans: list[dict[str, Any]] = []
        self._PendingNodeSpans: dict[str, tuple[int, list[dict[str, Any]]]] = {}

    def __repr__(self) -> str:
        return f"PatchPilotTracer(trace_id={self.trace_id!r}, configured={self.is_configured()})"

    def is_configured(self) -> bool:
        return bool(self._enabled and self.settings.is_configured)

    def _auth(self) -> tuple[str, str]:
        return (self.settings.effective_public_key, self.settings.effective_secret_key)

    def _base_url(self) -> str:
        return self.settings.host.rstrip("/")

    async def check_health(self, *, wait_s: float = 0.0) -> bool:
        """Check if Langfuse /api/public/health returns HTTP 200."""
        if not self.is_configured():
            return False

        url = f"{self._base_url()}/api/public/health"
        deadline = time.monotonic() + max(0.0, wait_s)

        while True:
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_s, transport=self._transport
                ) as client:
                    res = await client.get(url)
                    if res.status_code == 200:
                        return True
            except (httpx.HTTPError, OSError):
                pass

            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(1.0)

    def start_trace(
        self,
        *,
        case_id: str,
        name: str = "patchpilot-case-run",
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record the start of a case trace."""
        self.trace_name = name
        self.trace_start_ns = time.time_ns()
        self.trace_started_at = _iso_now()
        return self.trace_id

    def start_node_span(
        self,
        node_name: str,
        input_data: dict[str, Any] | None = None,
    ) -> tuple[str, float, str]:
        """Start a span for a LangGraph pipeline node (`analyze`, `propose`, `verify`, `report`)."""
        span_id = uuid.uuid4().hex[:16]
        perf_start = time.perf_counter()
        iso_start = _iso_now()
        attrs = [
            _otel_attr("langfuse.observation.type", "span"),
            _otel_attr("patchpilot.node", node_name),
        ]
        if input_data:
            for k, v in input_data.items():
                if isinstance(v, str | int | float | bool):
                    attrs.append(_otel_attr(f"patchpilot.input.{k}", v))
        self._PendingNodeSpans[span_id] = (time.time_ns(), attrs)
        return span_id, perf_start, iso_start

    def end_node_span(
        self,
        span_id: str,
        node_name: str,
        perf_start: float,
        iso_start: str,
        output_data: dict[str, Any] | None = None,
    ) -> None:
        """Complete a span for a LangGraph pipeline node."""
        duration_s = round(max(0.0, time.perf_counter() - perf_start), 6)
        iso_end = _iso_now()
        self.node_spans.append(
            NodeSpanMetric(
                span_id=span_id,
                node=node_name,
                start_time=iso_start,
                end_time=iso_end,
                duration_s=duration_s,
            )
        )
        start_ns, attrs = self._PendingNodeSpans.pop(
            span_id,
            (time.time_ns() - int(duration_s * 1e9), []),
        )
        end_ns = max(start_ns + 1, time.time_ns())
        attrs.append(_otel_attr("patchpilot.duration_s", duration_s))
        if output_data:
            for k, v in output_data.items():
                if isinstance(v, str | int | float | bool):
                    attrs.append(_otel_attr(f"patchpilot.output.{k}", v))

        self._otel_spans.append(
            {
                "traceId": self.trace_id,
                "spanId": span_id,
                "parentSpanId": self.root_span_id,
                "name": f"node:{node_name}",
                "kind": 1,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(end_ns),
                "attributes": attrs,
            }
        )

    def record_llm_call(
        self,
        *,
        node: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float,
        parent_span_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> LLMCallMetric:
        """Record an LLM generation call with model, token counts, and latency."""
        total_tok = prompt_tokens + completion_tokens
        metric = LLMCallMetric(
            node=node,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tok,
            latency_s=round(max(0.0, latency_s), 6),
        )
        self.llm_calls.append(metric)

        end_ns = time.time_ns()
        start_ns = max(1, end_ns - int(metric.latency_s * 1e9))
        parent_id = (
            parent_span_id[:16]
            if parent_span_id and len(parent_span_id) >= 16
            else self.root_span_id
        )
        self._otel_spans.append(
            {
                "traceId": self.trace_id,
                "spanId": uuid.uuid4().hex[:16],
                "parentSpanId": parent_id,
                "name": f"llm:{node}:{provider}",
                "kind": 3,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(end_ns),
                "attributes": [
                    _otel_attr("langfuse.observation.type", "generation"),
                    _otel_attr("gen_ai.system", provider),
                    _otel_attr("gen_ai.request.model", model),
                    _otel_attr("gen_ai.response.model", model),
                    _otel_attr("gen_ai.usage.input_tokens", prompt_tokens),
                    _otel_attr("gen_ai.usage.output_tokens", completion_tokens),
                    _otel_attr("gen_ai.usage.total_tokens", total_tok),
                    _otel_attr("patchpilot.node", node),
                    _otel_attr("patchpilot.latency_s", metric.latency_s),
                ],
            }
        )
        return metric

    def record_sandbox_run(
        self,
        *,
        node: str,
        exit_code: int,
        duration_s: float,
        patch_applied: bool,
        passed: int = 0,
        failures: int = 0,
        errors: int = 0,
        container_name: str = "",
        image_digest: str = "",
        parent_span_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> SandboxRunMetric:
        """Record a sandbox container execution span."""
        metric = SandboxRunMetric(
            node=node,
            exit_code=exit_code,
            duration_s=round(max(0.0, duration_s), 6),
            patch_applied=patch_applied,
            passed=passed,
            failures=failures,
            errors=errors,
            container_name=container_name,
            image_digest=image_digest,
        )
        self.sandbox_runs.append(metric)

        end_ns = time.time_ns()
        start_ns = max(1, end_ns - int(metric.duration_s * 1e9))
        parent_id = (
            parent_span_id[:16]
            if parent_span_id and len(parent_span_id) >= 16
            else self.root_span_id
        )
        self._otel_spans.append(
            {
                "traceId": self.trace_id,
                "spanId": uuid.uuid4().hex[:16],
                "parentSpanId": parent_id,
                "name": f"sandbox:{node}",
                "kind": 1,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(end_ns),
                "attributes": [
                    _otel_attr("langfuse.observation.type", "span"),
                    _otel_attr("patchpilot.node", node),
                    _otel_attr("patchpilot.sandbox.exit_code", exit_code),
                    _otel_attr("patchpilot.sandbox.patch_applied", patch_applied),
                    _otel_attr("patchpilot.sandbox.passed", passed),
                    _otel_attr("patchpilot.sandbox.failures", failures),
                    _otel_attr("patchpilot.sandbox.errors", errors),
                    _otel_attr("patchpilot.sandbox.duration_s", metric.duration_s),
                ],
            }
        )
        return metric

    async def flush_events(self) -> bool:
        """Flush buffered OTLP spans to Langfuse `/api/public/otel/v1/traces`."""
        if not self.is_configured() or not self._otel_spans:
            return False

        url = f"{self._base_url()}/api/public/otel/v1/traces"
        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            _otel_attr("service.name", "patchpilot"),
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "patchpilot.tracer", "version": "0.1.0"},
                            "spans": list(self._otel_spans),
                        }
                    ],
                }
            ]
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                res = await client.post(
                    url,
                    json=payload,
                    auth=self._auth(),
                )
                if res.status_code in (200, 201, 202, 207):
                    self._otel_spans.clear()
                    return True
                return False
        except (httpx.HTTPError, OSError) as exc:
            logger.debug("Langfuse OTLP flush skipped (unreachable): %s", type(exc).__name__)
            return False

    async def finish_trace(
        self,
        *,
        case_id: str,
        verified: bool,
        pass_at_1: bool,
        pass_at_k: bool,
        attempts: int,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Finalize the root trace span and flush all spans to Langfuse via OTLP."""
        end_ns = max(self.trace_start_ns + 1, time.time_ns())
        root_attrs = [
            _otel_attr("langfuse.trace.name", self.trace_name),
            _otel_attr("langfuse.observation.type", "span"),
            _otel_attr("patchpilot.case_id", case_id),
            _otel_attr("patchpilot.verified", verified),
            _otel_attr("patchpilot.pass_at_1", pass_at_1),
            _otel_attr("patchpilot.pass_at_3", pass_at_k),
            _otel_attr("patchpilot.attempts", attempts),
            _otel_attr("patchpilot.llm_call_count", len(self.llm_calls)),
            _otel_attr("patchpilot.sandbox_run_count", len(self.sandbox_runs)),
        ]
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, str | int | float | bool):
                    root_attrs.append(_otel_attr(f"patchpilot.meta.{k}", v))

        self._otel_spans.insert(
            0,
            {
                "traceId": self.trace_id,
                "spanId": self.root_span_id,
                "name": self.trace_name,
                "kind": 1,
                "startTimeUnixNano": str(self.trace_start_ns),
                "endTimeUnixNano": str(end_ns),
                "attributes": root_attrs,
            },
        )
        return await self.flush_events()

    async def verify_trace_visible(
        self,
        trace_id: str | None = None,
        *,
        timeout_s: float = 25.0,
        poll_interval_s: float = 0.5,
    ) -> bool:
        """Poll Langfuse v4 observations/traces API until `trace_id` is indexed in ClickHouse."""
        if not self.is_configured():
            return False

        target_id = trace_id or self.trace_id
        obs_url = f"{self._base_url()}/api/public/v2/observations"
        trace_url = f"{self._base_url()}/api/public/traces/{target_id}"
        deadline = time.monotonic() + max(0.1, timeout_s)

        while True:
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_s, transport=self._transport
                ) as client:
                    res_obs = await client.get(
                        obs_url,
                        params={"traceId": target_id},
                        auth=self._auth(),
                    )
                    if res_obs.status_code == 200:
                        data = res_obs.json()
                        items = data.get("data") if isinstance(data, dict) else None
                        if isinstance(items, list) and any(
                            isinstance(it, dict) and it.get("traceId") == target_id for it in items
                        ):
                            return True

                    res_tr = await client.get(trace_url, auth=self._auth())
                    if res_tr.status_code == 200:
                        tdata = res_tr.json()
                        if isinstance(tdata, dict) and tdata.get("id") == target_id:
                            return True
            except (httpx.HTTPError, ValueError, OSError):
                pass

            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll_interval_s)

    async def ensure_dataset(self, dataset_name: str, description: str = "") -> bool:
        """Ensure a Langfuse dataset exists for the evaluation pool."""
        if not self.is_configured():
            return False

        payload = {"name": dataset_name, "description": description}
        for endpoint in ("/api/public/v2/datasets", "/api/public/datasets"):
            url = f"{self._base_url()}{endpoint}"
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_s, transport=self._transport
                ) as client:
                    res = await client.post(url, json=payload, auth=self._auth())
                    if res.status_code in (200, 201, 409):
                        return True
            except (httpx.HTTPError, OSError):
                return False
        return False

    async def upsert_dataset_item(
        self,
        *,
        dataset_name: str,
        item_id: str,
        input_data: dict[str, Any],
        expected_output: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Create or update a dataset item in Langfuse."""
        if not self.is_configured():
            return False

        url = f"{self._base_url()}/api/public/dataset-items"
        payload: dict[str, Any] = {
            "datasetName": dataset_name,
            "id": item_id,
            "input": input_data,
            "expectedOutput": expected_output,
            "metadata": metadata or {},
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                res = await client.post(url, json=payload, auth=self._auth())
                return res.status_code in (200, 201)
        except (httpx.HTTPError, OSError):
            return False

    async def create_dataset_run_item(
        self,
        *,
        run_name: str,
        run_description: str,
        item_id: str,
        trace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Link a case trace or observation to a Langfuse dataset run."""
        if not self.is_configured():
            return False

        url = f"{self._base_url()}/api/public/dataset-run-items"
        payload: dict[str, Any] = {
            "runName": run_name,
            "runDescription": run_description,
            "datasetItemId": item_id,
            "traceId": trace_id,
            "metadata": metadata or {},
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                res = await client.post(url, json=payload, auth=self._auth())
                return res.status_code in (200, 201)
        except (httpx.HTTPError, OSError):
            return False


async def run_trace_smoke(
    settings: LangfuseSettings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[bool, str]:
    """Execute the Phase 5 `trace-smoke` check against Langfuse or return no-op when down."""
    cfg = settings or LangfuseSettings()
    tracer = PatchPilotTracer(settings=cfg, transport=transport)
    if not tracer.is_configured():
        return False, "tracing disabled: no-op"

    healthy = await tracer.check_health(wait_s=0.0)
    if not healthy and transport is None and _is_langfuse_container_running():
        healthy = await tracer.check_health(wait_s=60.0)

    if not healthy:
        return False, "tracing disabled: no-op"

    tracer.start_trace(
        case_id="trace-smoke",
        name="patchpilot-trace-smoke",
        input_data={"smoke": True},
        metadata={"phase": 5},
    )
    span_id, perf_start, iso_start = tracer.start_node_span("analyze", {"smoke": True})
    tracer.record_llm_call(
        node="analyze",
        model="mock-smoke",
        provider="mock",
        prompt_tokens=12,
        completion_tokens=8,
        latency_s=0.01,
        parent_span_id=span_id,
    )
    tracer.end_node_span(span_id, "analyze", perf_start, iso_start, {"findings": 1})
    tracer.record_sandbox_run(
        node="verify",
        exit_code=0,
        duration_s=0.05,
        patch_applied=True,
        passed=1,
    )
    ingested = await tracer.finish_trace(
        case_id="trace-smoke",
        verified=True,
        pass_at_1=True,
        pass_at_k=True,
        attempts=1,
    )
    if not ingested:
        return False, "tracing disabled: no-op"

    visible = await tracer.verify_trace_visible(
        tracer.trace_id,
        timeout_s=30.0 if transport is None else 1.0,
        poll_interval_s=0.5 if transport is None else 0.05,
    )
    return visible, f"trace_id={tracer.trace_id} visible={str(visible).lower()}"
