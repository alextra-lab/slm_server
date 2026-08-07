"""OTLP span export for slm_server request telemetry (FRE-1071, ADR-0129 D5/D8).

Every completed request becomes one span exported over OTLP/HTTP to the
OpenTelemetry Collector. The Collector is the single egress point for traces
(ADR-0129 D5); nothing here writes to a search index, and no index name is
formatted client-side any more.

Two properties are load-bearing:

* **The caller's trace context is continued, not replaced.** A W3C ``traceparent``
  on the inbound request makes this span a child of the caller's span, inside the
  caller's trace. That is the only thing that puts SLM work inside the calling
  turn, and it cannot be faked from the calling repository.
* **Export is fail-soft.** ``emit_request_span`` swallows everything, and the
  network hop happens on the batch processor's own thread, so a Collector that
  is down, slow or absent can never reach the request path.

Spans are created after the fact, with an explicit start time derived from the
measured elapsed time, because telemetry is assembled once the response is
complete. They are leaf spans — nothing nests inside them — so no live current
span is needed while the backend call is in flight.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import structlog
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import AttributeValue

log = structlog.get_logger()

EmitPath = Literal["chat", "responses", "rerank", "streaming"]

#: The four sites that complete a request and export a span. Each sets its own
#: value on ``slm.emit_path``; without a distinct value per path, a path that
#: stopped emitting is indistinguishable from one that is merely idle.
EMIT_PATHS: Final[tuple[EmitPath, ...]] = ("chat", "responses", "rerank", "streaming")

_DEFAULT_OTLP_ENDPOINT: Final = "http://localhost:4318"
_OTLP_PROTOCOL: Final = "http/protobuf"
_INSTRUMENTATION_SCOPE: Final = "slm_server.telemetry"

# gen_ai.operation.name per emit path. Semantic conventions name "chat"; they
# define no rerank operation, so that value is this project's own (ADR-0129 D2:
# semconv where one exists, a namespaced project key where none does).
_OPERATION_BY_PATH: Final[dict[str, str]] = {
    "chat": "chat",
    "responses": "chat",
    "streaming": "chat",
    "rerank": "rerank",
}

# (span attribute, telemetry-doc field). A field whose value is None is dropped
# rather than sent: None is not a legal attribute value, and a zero would be a
# measurement this request never made.
_ATTRIBUTE_SOURCES: Final[tuple[tuple[str, str], ...]] = (
    ("gen_ai.request.model", "model_id"),
    ("gen_ai.system", "backend"),
    ("gen_ai.usage.input_tokens", "prompt_tokens"),
    ("gen_ai.usage.output_tokens", "completion_tokens"),
    ("http.response.status_code", "status"),
    ("slm.backend_port", "port"),
    ("slm.session_id", "session_id"),
    # The caller's pre-traceparent identity. Kept as an attribute so the join to
    # the calling turn stays recoverable until the caller injects a traceparent
    # (owned by FRE-1067). Deliberately not turned into a synthetic parent span
    # context: that would fabricate a parent that exists in no trace store.
    ("slm.client.trace_id", "trace_id"),
    ("slm.prefill_ms", "prefill_ms"),
    ("slm.decode_ms", "decode_ms"),
    ("slm.prompt_n", "prompt_n"),
    ("slm.predicted_n", "predicted_n"),
    ("slm.cache_reuse", "cache_reuse"),
    ("slm.ttfb_ms", "ttfb_ms"),
    ("slm.heartbeat_count", "heartbeat_count"),
    ("slm.client_disconnected", "client_disconnected"),
)


@dataclass(frozen=True)
class TelemetrySettings:
    """Resolved export configuration.

    Attributes:
        enabled: Whether spans are exported at all.
        otlp_endpoint: Base OTLP/HTTP endpoint of the Collector, no trailing slash.
        service_name: Value published as the ``service.name`` resource attribute.
    """

    enabled: bool
    otlp_endpoint: str
    service_name: str

    @property
    def traces_endpoint(self) -> str:
        """The signal-specific URL the span exporter actually posts to."""
        return f"{self.otlp_endpoint}/v1/traces"


def load_settings() -> TelemetrySettings:
    """Resolve export settings from the environment.

    Export is on by default. The old shipper was opt-in, and staying opt-in would
    mean telemetry that has to be remembered to switch on — the failure mode
    ADR-0129 exists to end. ``SLM_OTEL_ENABLED=false`` turns it off.

    Returns:
        The resolved settings.
    """
    endpoint = os.getenv("SLM_OTLP_ENDPOINT") or _DEFAULT_OTLP_ENDPOINT
    return TelemetrySettings(
        enabled=os.getenv("SLM_OTEL_ENABLED", "true").lower() != "false",
        otlp_endpoint=endpoint.rstrip("/"),
        service_name=os.getenv("SLM_OTEL_SERVICE_NAME", "slm-server"),
    )


_settings: TelemetrySettings = load_settings()
_provider: TracerProvider | None = None
_propagator: Final = TraceContextTextMapPropagator()


def _build_exporter() -> OTLPSpanExporter:
    """Construct the span exporter from the resolved settings.

    Factored out so the effective-configuration artifact and the exporter cannot
    drift: both read the same resolved endpoint, and a test asserts they agree.

    Returns:
        An exporter pointed at the Collector's traces endpoint.
    """
    return OTLPSpanExporter(endpoint=_settings.traces_endpoint)


def init_tracing() -> None:
    """Install the tracer provider. Idempotent; safe to call when disabled.

    Called once from the router lifespan rather than lazily on first use, so that
    processor construction and its exporter thread never land inline on a served
    request.

    Settings are re-resolved here rather than trusted from import time, so startup
    is the authority on configuration and the module is not import-order sensitive.
    """
    global _provider, _settings

    if _provider is not None:
        return
    _settings = load_settings()
    if not _settings.enabled:
        log.info("otlp_export_disabled", reason="SLM_OTEL_ENABLED=false")
        return

    try:
        provider = TracerProvider(
            resource=Resource.create({"service.name": _settings.service_name})
        )
        provider.add_span_processor(BatchSpanProcessor(_build_exporter()))
        _provider = provider
        log.info(
            "otlp_export_initialised",
            endpoint=_settings.traces_endpoint,
            protocol=_OTLP_PROTOCOL,
            service_name=_settings.service_name,
        )
    except Exception as exc:  # noqa: BLE001 - telemetry never breaks startup
        log.warning("otlp_init_failed", error=str(exc), endpoint=_settings.traces_endpoint)


def shutdown_tracing() -> None:
    """Flush and tear down the tracer provider. Idempotent."""
    global _provider

    provider, _provider = _provider, None
    if provider is None:
        return
    try:
        provider.shutdown()
    except Exception as exc:  # noqa: BLE001 - telemetry never breaks shutdown
        log.warning("otlp_shutdown_failed", error=str(exc))


def effective_config() -> dict[str, object]:
    """Report the export configuration this process actually resolved.

    Published over HTTP because the acceptance verifier runs on a host that
    cannot otherwise inspect this one. Derived from live state rather than
    hand-written, so it cannot claim an endpoint the exporter does not use.

    Returns:
        A JSON-serialisable mapping naming the OTLP endpoint and export state.
    """
    return {
        "service_name": _settings.service_name,
        "otlp_endpoint": _settings.otlp_endpoint,
        "otlp_traces_endpoint": _settings.traces_endpoint,
        "otlp_protocol": _OTLP_PROTOCOL,
        "telemetry_enabled": _settings.enabled,
        "tracer_provider_initialised": _provider is not None,
        "emit_paths": list(EMIT_PATHS),
        "elasticsearch_export": False,
    }


def _span_attributes(doc: Mapping[str, object], emit_path: EmitPath) -> dict[str, AttributeValue]:
    """Project a telemetry doc onto span attributes, dropping absent values."""
    attributes: dict[str, AttributeValue] = {
        "slm.emit_path": emit_path,
        "gen_ai.operation.name": _OPERATION_BY_PATH[emit_path],
    }
    for attribute, field in _ATTRIBUTE_SOURCES:
        value = doc.get(field)
        if isinstance(value, bool | int | float | str):
            attributes[attribute] = value
    return attributes


def emit_request_span(
    doc: Mapping[str, object],
    *,
    emit_path: EmitPath,
    headers: Mapping[str, str],
) -> None:
    """Export one span describing a completed request.

    Fail-soft by contract: this never raises and never blocks on the network. The
    body is wrapped whole, and the batch processor's ``on_end`` only enqueues.

    The span is backdated from the doc's ``total_ms``, which is rounded to 0.1 ms,
    so the reconstructed duration carries up to 0.05 ms of error — immaterial
    against the tolerances any consumer applies to a model call.

    Args:
        doc: The request telemetry doc, as built by the router.
        emit_path: Which of the four emit sites produced this request.
        headers: Inbound request headers, used to continue the caller's trace.
    """
    try:
        provider = _provider
        if provider is None:
            return

        end_time = time.time_ns()
        total_ms = doc.get("total_ms")
        start_time = end_time
        if isinstance(total_ms, int | float) and not isinstance(total_ms, bool):
            start_time = end_time - int(total_ms * 1_000_000)

        model_id = doc.get("model_id")
        operation = _OPERATION_BY_PATH[emit_path]
        name = f"{operation} {model_id}" if isinstance(model_id, str) else operation

        span = provider.get_tracer(_INSTRUMENTATION_SCOPE).start_span(
            name,
            context=_propagator.extract(carrier=dict(headers)),
            kind=SpanKind.SERVER,
            start_time=start_time,
            attributes=_span_attributes(doc, emit_path),
        )
        status = doc.get("status")
        if isinstance(status, int) and status >= 400:
            span.set_status(Status(StatusCode.ERROR, f"backend returned {status}"))
        span.end(end_time)
    except Exception as exc:  # noqa: BLE001 - a telemetry fault must never reach the request
        log.warning("otlp_emit_failed", error=str(exc), emit_path=emit_path)
