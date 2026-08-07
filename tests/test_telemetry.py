"""Tests for slm_server.telemetry — OTLP span export (FRE-1071, ADR-0129 D5/D8).

The Elasticsearch shipper these tests used to cover is gone; AC-3 asserts its
absence rather than its behaviour.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import slm_server.telemetry as telemetry_module
from slm_server.telemetry import (
    EMIT_PATHS,
    effective_config,
    emit_request_span,
    load_settings,
)

# A known W3C traceparent: version-traceid-spanid-flags. `01` = sampled.
_TRACE_ID_HEX = "4bf92f3577b34da6a3ce929d0e0e4736"
_PARENT_SPAN_HEX = "00f067aa0ba902b7"
_TRACEPARENT = f"00-{_TRACE_ID_HEX}-{_PARENT_SPAN_HEX}-01"
_UNSAMPLED_TRACEPARENT = f"00-{_TRACE_ID_HEX}-{_PARENT_SPAN_HEX}-00"

_DOC: dict[str, object] = {
    "trace_id": "d4cd4a06f3f14b6ea1f4d0c1a3b25f90",
    "span_id": None,
    "session_id": "sess-1",
    "model_id": "test/model",
    "backend": "mlx",
    "port": 8501,
    "prompt_tokens": 11,
    "completion_tokens": 22,
    "prefill_ms": 5.0,
    "decode_ms": 6.0,
    "prompt_n": 11,
    "predicted_n": 22,
    "cache_reuse": 3,
    "total_ms": 250.0,
    "ttfb_ms": 40.0,
    "heartbeat_count": 0,
    "client_disconnected": False,
    "status": 200,
    "ts": "2026-08-07T00:00:00+00:00",
}


@pytest.fixture
def exported() -> Iterator[InMemorySpanExporter]:
    """Install an in-memory tracer provider for the duration of one test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = telemetry_module._provider
    telemetry_module._provider = provider
    try:
        yield exporter
    finally:
        telemetry_module._provider = previous


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected exactly one span, got {len(spans)}"
    return spans[0]


# ── AC-1: an incoming trace context is continued, not replaced ────────────────────────────────────


def test_incoming_traceparent_is_continued(exported: InMemorySpanExporter) -> None:
    """The exported span joins the caller's trace, under the caller's span.

    Trace-id equality alone is gameable — a span that merely copied the id would
    pass it — so the parent linkage and its remoteness are asserted too.
    """
    emit_request_span(
        _DOC,
        emit_path="chat",
        headers={"traceparent": _TRACEPARENT, "tracestate": "vendor=abc"},
    )

    span = _only_span(exported)
    assert format(span.context.trace_id, "032x") == _TRACE_ID_HEX
    assert span.parent is not None
    assert format(span.parent.span_id, "016x") == _PARENT_SPAN_HEX
    assert span.parent.is_remote is True
    assert span.context.trace_state.get("vendor") == "abc"


def test_absent_traceparent_starts_a_new_trace(exported: InMemorySpanExporter) -> None:
    """With no incoming context the span is a valid root, not a broken child."""
    emit_request_span(_DOC, emit_path="chat", headers={})

    span = _only_span(exported)
    assert span.parent is None
    assert span.context.trace_id != 0
    assert format(span.context.trace_id, "032x") != _TRACE_ID_HEX


def test_unsampled_parent_exports_nothing(exported: InMemorySpanExporter) -> None:
    """A caller that sampled the trace out is respected, not overridden.

    Pinned deliberately: the parent-based sampler doing this is correct OTel
    behaviour, and a future reader should not "fix" it into always-on.
    """
    emit_request_span(_DOC, emit_path="chat", headers={"traceparent": _UNSAMPLED_TRACEPARENT})

    assert exported.get_finished_spans() == ()


# ── AC-2: every emit path is distinguishable ──────────────────────────────────────────────────────


@pytest.mark.parametrize("emit_path", EMIT_PATHS)
def test_each_emit_path_sets_its_attribute(exported: InMemorySpanExporter, emit_path: str) -> None:
    emit_request_span(_DOC, emit_path=emit_path, headers={})  # type: ignore[arg-type]

    span = _only_span(exported)
    assert span.attributes is not None
    assert span.attributes["slm.emit_path"] == emit_path


def test_emit_path_values_are_distinct(exported: InMemorySpanExporter) -> None:
    """Four paths, four values — a shared value makes a dead path look idle."""
    for emit_path in EMIT_PATHS:
        emit_request_span(_DOC, emit_path=emit_path, headers={})  # type: ignore[arg-type]

    values = [s.attributes["slm.emit_path"] for s in exported.get_finished_spans()]  # type: ignore[index]
    assert len(values) == len(EMIT_PATHS)
    assert len(set(values)) == len(EMIT_PATHS)


# ── AC-3: the Elasticsearch writer is gone ────────────────────────────────────────────────────────


def test_source_has_no_elasticsearch_writer() -> None:
    """No ES URL formatting survives anywhere in src/ — including an undated one.

    Removing only the date suffix would leave a writer pointed at an undated
    index, which is why this looks for the writer's whole vocabulary rather than
    for `strftime`.
    """
    src = Path(__file__).resolve().parent.parent / "src"
    forbidden = ("SLM_ES_", "slm-requests", "/_doc", "ship_request_complete")

    offenders = [
        f"{path.relative_to(src)}:{lineno}: {token}"
        for path in src.rglob("*.py")
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        for token in forbidden
        if token in line
    ]
    assert offenders == [], f"Elasticsearch write path survives: {offenders}"


def test_module_exposes_no_shipper() -> None:
    assert not hasattr(telemetry_module, "ship_request_complete")


# ── AC-5: the effective-configuration artifact ────────────────────────────────────────────────────


def test_effective_config_reports_resolved_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The artifact is derived from resolved runtime state, not hand-written.

    Asserted by moving the setting and watching the artifact move with it — a
    hard-coded literal would not.
    """
    monkeypatch.setenv("SLM_OTLP_ENDPOINT", "http://otel-collector:4318")
    monkeypatch.setattr(telemetry_module, "_settings", load_settings())

    config = effective_config()
    assert config["otlp_endpoint"] == "http://otel-collector:4318"
    assert config["otlp_traces_endpoint"] == "http://otel-collector:4318/v1/traces"
    assert config["otlp_protocol"] == "http/protobuf"
    assert config["elasticsearch_export"] is False
    assert config["emit_paths"] == list(EMIT_PATHS)


def test_default_endpoint_is_the_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stock instance names the Collector, with no environment to help it."""
    monkeypatch.delenv("SLM_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr(telemetry_module, "_settings", load_settings())

    assert effective_config()["otlp_endpoint"] == "http://localhost:4318"


def test_reported_endpoint_is_the_exporter_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the artifact reports is what the exporter is actually built with.

    Without this the artifact could name the Collector while the exporter shipped
    somewhere else entirely, and AC-5 would pass on a lie. `_endpoint` is an SDK
    private; reaching for it in a test is the price of checking the real object.
    """
    monkeypatch.setenv("SLM_OTLP_ENDPOINT", "http://otel-collector:4318")
    monkeypatch.setattr(telemetry_module, "_settings", load_settings())

    exporter = telemetry_module._build_exporter()
    assert exporter._endpoint == effective_config()["otlp_traces_endpoint"]


# ── AC-6: export stays fail-soft ──────────────────────────────────────────────────────────────────


def test_emit_never_raises_when_tracing_is_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure anywhere inside the emit is swallowed, never propagated."""

    class _ExplodingProvider:
        def get_tracer(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("tracer is broken")

    monkeypatch.setattr(telemetry_module, "_provider", _ExplodingProvider())

    emit_request_span(_DOC, emit_path="chat", headers={})  # must not raise


def test_emit_is_a_noop_without_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before init_tracing (or with telemetry disabled) emitting does nothing."""
    monkeypatch.setattr(telemetry_module, "_provider", None)

    emit_request_span(_DOC, emit_path="chat", headers={})  # must not raise


def test_emit_survives_an_unreachable_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """The strong form: a real OTLP exporter to a closed port, exported inline.

    SimpleSpanProcessor makes the network hop happen on this thread, so an export
    that was not fail-soft would surface right here. Production uses
    BatchSpanProcessor, which is strictly safer.
    """
    monkeypatch.setenv("SLM_OTLP_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setattr(telemetry_module, "_settings", load_settings())

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(telemetry_module._build_exporter()))
    monkeypatch.setattr(telemetry_module, "_provider", provider)

    emit_request_span(_DOC, emit_path="chat", headers={})  # must not raise


# ── AC-7: token usage under semantic conventions ──────────────────────────────────────────────────


def test_span_carries_semconv_token_usage(exported: InMemorySpanExporter) -> None:
    emit_request_span(_DOC, emit_path="chat", headers={})

    attributes = _only_span(exported).attributes
    assert attributes is not None
    assert attributes["gen_ai.usage.input_tokens"] == _DOC["prompt_tokens"]
    assert attributes["gen_ai.usage.output_tokens"] == _DOC["completion_tokens"]


def test_absent_usage_omits_the_token_attributes(exported: InMemorySpanExporter) -> None:
    """None is not a legal attribute value — the keys are dropped, not zeroed."""
    doc = {**_DOC, "prompt_tokens": None, "completion_tokens": None}

    emit_request_span(doc, emit_path="rerank", headers={})

    attributes = _only_span(exported).attributes
    assert attributes is not None
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes


# ── Span shape ────────────────────────────────────────────────────────────────────────────────────


def test_span_duration_reflects_the_measured_elapsed_time(exported: InMemorySpanExporter) -> None:
    """A backdated span is a measurement, not a zero-width marker."""
    emit_request_span(_DOC, emit_path="chat", headers={})

    span = _only_span(exported)
    assert span.end_time is not None and span.start_time is not None
    elapsed_ms = (span.end_time - span.start_time) / 1_000_000
    assert elapsed_ms == pytest.approx(250.0, abs=0.1)


def test_span_carries_model_and_backend_identity(exported: InMemorySpanExporter) -> None:
    emit_request_span(_DOC, emit_path="chat", headers={})

    span = _only_span(exported)
    attributes = span.attributes
    assert attributes is not None
    assert span.name == "chat test/model"
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.request.model"] == "test/model"
    assert attributes["gen_ai.system"] == "mlx"
    assert attributes["slm.backend_port"] == 8501
    assert attributes["slm.session_id"] == "sess-1"
    assert attributes["http.response.status_code"] == 200


def test_rerank_uses_its_own_operation_name(exported: InMemorySpanExporter) -> None:
    emit_request_span(_DOC, emit_path="rerank", headers={})

    span = _only_span(exported)
    assert span.attributes is not None
    assert span.attributes["gen_ai.operation.name"] == "rerank"
    assert span.name == "rerank test/model"


def test_legacy_client_trace_id_rides_as_an_attribute(exported: InMemorySpanExporter) -> None:
    """Keeps the pre-traceparent join recoverable while FRE-1067 is outstanding."""
    emit_request_span(_DOC, emit_path="chat", headers={})

    attributes = _only_span(exported).attributes
    assert attributes is not None
    assert attributes["slm.client.trace_id"] == _DOC["trace_id"]


def test_error_status_marks_the_span_as_error(exported: InMemorySpanExporter) -> None:
    emit_request_span({**_DOC, "status": 503}, emit_path="chat", headers={})

    span = _only_span(exported)
    assert span.status.is_ok is False


# ── Startup wiring ────────────────────────────────────────────────────────────────────────────────


def test_init_tracing_installs_a_real_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The startup path is exercised, not only substituted.

    Every other test here injects its own provider and conftest disables export,
    so nothing else would notice init_tracing() failing: its own except clause
    swallows the error and leaves _provider None: production exports nothing while
    the suite stays green. This is the one test that runs the real wiring.
    """
    monkeypatch.setenv("SLM_OTEL_ENABLED", "true")
    monkeypatch.setenv("SLM_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setattr(telemetry_module, "_provider", None)

    try:
        telemetry_module.init_tracing()

        provider = telemetry_module._provider
        assert isinstance(provider, TracerProvider)
        assert provider.resource.attributes["service.name"] == "slm-server"
        assert effective_config()["tracer_provider_initialised"] is True
        # A provider with no processor attached silently drops every span, which
        # is indistinguishable from working until you look in the trace store.
        assert provider._active_span_processor._span_processors  # noqa: SLF001
    finally:
        telemetry_module.shutdown_tracing()

    assert telemetry_module._provider is None
    assert effective_config()["tracer_provider_initialised"] is False


def test_init_tracing_is_a_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """SLM_OTEL_ENABLED=false must leave no provider and start no exporter thread."""
    monkeypatch.setenv("SLM_OTEL_ENABLED", "false")
    monkeypatch.setattr(telemetry_module, "_provider", None)

    telemetry_module.init_tracing()

    assert telemetry_module._provider is None


def test_effective_config_hides_endpoint_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """A credential embedded in the endpoint is not republished to the verifier."""
    monkeypatch.setenv("SLM_OTLP_ENDPOINT", "https://user:hunter2@collector.internal:4318")
    monkeypatch.setattr(telemetry_module, "_settings", load_settings())

    config = effective_config()

    assert config["otlp_endpoint"] == "https://collector.internal:4318"
    assert config["otlp_traces_endpoint"] == "https://collector.internal:4318/v1/traces"
    assert "hunter2" not in repr(config)
