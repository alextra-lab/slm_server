# FRE-1071 — slm_server exports OTLP to the Collector; the Elasticsearch writer is removed

Ticket: FRE-1071 (Approved) · Backing ADR: ADR-0129 D5/D8 (personal_agent repo)
Repo: `slm_server` (separate repository, separate release cycle)

## Objective

`slm_server` stops writing `slm-requests-YYYY.MM.DD` documents to Elasticsearch and instead
exports OTLP spans to the OpenTelemetry Collector, continuing the caller's trace context so its
spans land inside the calling turn's trace.

## Current state (verified 2026-08-07)

- `src/slm_server/telemetry.py` — 54 lines, one function `ship_request_complete(doc)`, POSTs to
  `{_ES_URL}/{_ES_INDEX_PREFIX}-{YYYY.MM.DD}/_doc`. Fail-soft (`except Exception` → `log.warning`).
- Four emit sites in `router.py`, all of the shape
  `log.info("request_complete", **tel_doc)` + `asyncio.create_task(ship_request_complete(tel_doc))`:

  | line | path | function |
  |---|---|---|
  | 658 | streaming | `_stream_backend_response._emit_telemetry` |
  | 857 | chat | `chat_completions` (non-streaming) |
  | 1134 | rerank | `rerank` |
  | 1355 | responses | `responses._emit_responses_telemetry` |

  (The ticket cites `528, 696, 971, 1175` — stale; the file shifted when FRE-241/FRE-980 landed.)
- All four build their doc through the single helper `_build_request_telemetry(...)`.
- Identity today comes from `x-trace-id` / `x-span-id` / `x-session-id` headers, used only for the
  structlog line and the doc. There is no W3C `traceparent` handling anywhere.
- `/v1/embeddings` emits no `request_complete` and is out of scope (four paths, not five).

## Decisions taken in this plan

1. **OTLP/HTTP (`http/protobuf`), not gRPC.** Avoids pulling `grpcio` (a wheel-heavy dep on Apple
   Silicon) into a repo that deliberately keeps its core dependency list lean.
2. **Default endpoint `http://localhost:4318`**, overridable with `SLM_OTLP_ENDPOINT`. slm_server
   runs natively on the Mac and the Collector (FRE-1070) is a local `docker-compose` service, so
   loopback is the right default. **Assumption flagged:** FRE-1070 has not merged, so the Collector's
   port is taken from the OTLP/HTTP default rather than read from a committed compose file.
3. **Enabled by default** (`SLM_OTEL_ENABLED=false` disables), unlike the old opt-in-by-`SLM_ES_URL`
   shipper. AC-5 requires a default local instance to report the Collector as its endpoint. Codex
   review argued for opt-in until FRE-1070 merges; declined, because telemetry that must be
   remembered to switch on is exactly the failure mode ADR-0129 exists to end. Cost is recorded in
   Risks and is one env var to reverse.
4. **Export is enqueued on the request path; the network hop is not.** `BatchSpanProcessor.on_end`
   appends to a bounded queue and returns — no I/O. So the four call sites drop
   `asyncio.create_task(...)` and call a plain function.
5. **The tracer provider is built at lifespan startup, not lazily on first request.** Codex flagged
   that lazy init would put processor construction and thread start inline on one unlucky request.
   Startup init removes that entirely; `emit_request_span` is a no-op if the provider is absent.
6. **Spans are emitted after the fact with explicit start/end times.** Telemetry is built once the
   response is complete, so the span is created with `start_time = end_time - total_ms`, using
   `time.time_ns()`. This keeps the diff at the four call sites to one line each rather than
   restructuring every handler around a context manager. These are **leaf summary spans** — nothing
   nests inside them, so the absence of a live current span during the backend call costs nothing.
   `total_ms` in the doc is rounded to 0.1 ms, so the reconstructed duration carries ±0.05 ms of
   error; documented in the docstring, immaterial against the 10% tolerance ADR-0129 AC-4 applies.
7. **The default parent-based sampler is kept.** An incoming `traceparent` with flags `00` therefore
   exports nothing — correct OTel semantics (the caller's sampling decision is respected), and
   asserted by a test so a later reader does not "fix" it.
8. **The structlog `request_complete` line stays.** ADR-0129 D5 is explicit that logs keep their
   existing path; only the ES *writer* goes.
9. **`x-trace-id` is not used as a trace-context fallback**, but it is carried onto the span as
   `slm.client.trace_id`. AC-1 asks for `traceparent`; synthesising a parent `SpanContext` from a
   UUID would mean fabricating a span id for a span that does not exist. Carrying the legacy id as an
   attribute keeps the join recoverable without inventing trace structure. **This has a consequence
   for the chain, flagged to master rather than solved here** — see below.

## Chain gap: owned by FRE-1067, not by this ticket (owner ruling, 2026-08-07)

`personal_agent` sends `X-Trace-Id` and not a W3C `traceparent` on its outbound call to the SLM
server (`src/personal_agent/llm_client/client.py:429`). Until it does, every span this ticket exports
starts a **new** trace, so ADR-0129 AC-6's "every such span's `trace_id` equals the calling turn's
ledger `trace_id`" will not hold yet.

**The owner ruled this is FRE-1067's (B3), folded in there as AC-14** — B3 already owns the
model-call span and is the natural instrumentation site. No ticket is filed from here.

Two premises in the codex review were stale and are corrected: `personal_agent` **does** carry
`opentelemetry-sdk>=1.44.0` and `opentelemetry-exporter-otlp>=1.44.0` (FRE-1064, deployed 08:09
today, SDK bootstrapped at startup), and since FRE-1065 merged this morning `X-Trace-Id` already
carries a **32-hex OTel trace id**, not a UUID. That makes `slm.client.trace_id` a directly joinable
value in the interim rather than a foreign-format breadcrumb.

Synthesising a parent `SpanContext` from `X-Trace-Id` was considered and **rejected by the owner**:
it fabricates a parent span id for a span that does not exist in Tempo — manufacturing identity
rather than reading it, the exact divergence FRE-1065's same-trace guard was added to prevent. A
trace whose root is a phantom is worse than two honest traces.

Nothing here blocks this ticket: AC-1 is stated in terms of an incoming `traceparent`, so it is
provable today, and none of the seven criteria depends on the caller's side.

## Steps

### 1 — Dependencies — DONE
`pyproject.toml`: added `opentelemetry-api`, `opentelemetry-sdk`,
`opentelemetry-exporter-otlp-proto-http` (all resolved at 1.44.0). Verified: `uv sync` resolves,
`import opentelemetry.sdk` succeeds, `protobuf` stays at 7.35.0 (not downgraded by
`opentelemetry-proto`).

### 2 — Failing tests first (TDD)
New `tests/test_telemetry.py` (replacing the ES tests wholesale) using the SDK's
`InMemorySpanExporter` + `SimpleSpanProcessor` through a fixture that swaps the module's tracer
provider, and new cases in the router test files. One test per acceptance criterion:

- AC-1 `test_incoming_traceparent_is_continued` — POST with
  `traceparent: 00-<32 hex>-<16 hex>-01`. Codex flagged trace-id equality alone as gameable, so this
  asserts **four** things: exported span's `trace_id` == the supplied trace id, its
  `parent.span_id` == the supplied parent span id, `parent.is_remote` is True, and `tracestate`
  survives. A span that merely copied a trace id fails on the parent assertions.
- AC-1b `test_absent_traceparent_starts_a_new_trace` — no header → valid non-zero trace id, no parent.
- AC-1c `test_unsampled_parent_exports_nothing` — `traceparent` flags `00` → zero spans exported
  (decision 7, pinned so it is not later "fixed").
- AC-2 four tests, one per path, asserting `slm.emit_path` == `chat` / `responses` / `rerank` /
  `streaming`, plus `test_emit_path_values_are_distinct` over the four exercised paths. **All three**
  `_stream_backend_response` callers are covered — chat-stream (`router.py:816`), responses-stream
  (`:1281`) and the responses→chat fallback stream (`:1321`); Codex caught that the first draft named
  only two, which would have left the fallback path's carrier unwired.
- AC-3 `test_no_elasticsearch_request_on_any_path` — capture every outbound httpx URL across all
  four paths, assert none is an ES write; plus `test_source_has_no_elasticsearch_writer`, a grep of
  `src/` for `SLM_ES_`, `slm-requests`, `/_doc`.
- AC-5 `test_effective_config_endpoint_names_the_collector` — `GET /v1/telemetry/effective-config`
  returns 200 JSON whose `otlp_endpoint` equals the resolved Collector endpoint, and which is
  derived from module state (assert it tracks a monkeypatched setting, not a literal). Codex flagged
  that a reported endpoint could diverge from the one the exporter is actually built with, so
  `test_reported_endpoint_is_the_exporter_endpoint` constructs the real `OTLPSpanExporter` through
  the same code path and asserts its `_endpoint` equals the reported `otlp_traces_endpoint`
  (a deliberate, commented reach into an SDK private, test-only).
- AC-6 `test_request_succeeds_when_span_emit_raises` (call-site fail-soft) and
  `test_request_succeeds_when_collector_unreachable` (real `OTLPSpanExporter` at a closed port
  behind a `SimpleSpanProcessor`, so the export runs inline — the strong form).
- AC-7 `test_span_carries_semconv_token_usage` — `gen_ai.usage.input_tokens` /
  `gen_ai.usage.output_tokens` equal the `usage` the backend response reported.

AC-4 ("no existing `slm-requests-*` document is modified") needs no test: it is a negative
implementation requirement, satisfied by the diff containing no migration, reindex or backfill.

The 23 existing test references to `ship_request_complete` across `test_router_chat.py`,
`test_router_rerank.py` and `test_telemetry.py` are re-pointed at `emit_request_span`, preserving
their doc-schema assertions and picking up `emit_path` coverage for free.

Verify: `uv run pytest tests/test_telemetry.py` fails for the right reasons before implementation.

### 3 — Rewrite `src/slm_server/telemetry.py`
Public surface:

```python
EMIT_PATHS: Final = ("chat", "responses", "rerank", "streaming")
EmitPath = Literal["chat", "responses", "rerank", "streaming"]

@dataclass(frozen=True)
class TelemetrySettings:
    enabled: bool
    otlp_endpoint: str
    service_name: str

def load_settings() -> TelemetrySettings: ...          # env, mirrors watchdog.load_settings
def effective_config() -> dict[str, object]: ...        # AC-5, from resolved state
def emit_request_span(doc: Mapping[str, object], *, emit_path: EmitPath,
                      headers: Mapping[str, str]) -> None: ...   # AC-1/2/6/7
```

- `emit_request_span` extracts the parent context with `TraceContextTextMapPropagator().extract(headers)`,
  starts a span named `f"{operation} {model_id}"` with explicit `start_time`/`end_time`, sets the
  attributes below, sets span status `ERROR` for `status >= 400`, and ends it. **The entire body is
  the first statement inside one `try/except Exception`** → `log.warning("otlp_emit_failed", ...)`,
  preserving today's fail-soft contract. Its parameters are a mapping, a literal and
  `dict(request.headers)` — none can raise at the call site, so the four call sites are *not*
  additionally wrapped (Codex suggested they should be; declined as four copies of a guard that has
  nothing left to catch).
- `init_tracing()` / `shutdown_tracing()` are called from the router lifespan. `init_tracing` is
  idempotent, sets `Resource({"service.name": settings.service_name})`, and installs
  `BatchSpanProcessor(OTLPSpanExporter(...))`. Importing the module opens no socket and starts no
  thread, so tests can install their own provider.

Attributes:

| key | source |
|---|---|
| `gen_ai.operation.name` | `chat` for chat/responses/streaming, `rerank` for rerank |
| `gen_ai.system` | `doc["backend"]` |
| `gen_ai.request.model` | `doc["model_id"]` |
| `gen_ai.usage.input_tokens` | `doc["prompt_tokens"]` (AC-7) |
| `gen_ai.usage.output_tokens` | `doc["completion_tokens"]` (AC-7) |
| `slm.emit_path` | the `emit_path` argument (AC-2) |
| `slm.backend_port`, `slm.prefill_ms`, `slm.decode_ms`, `slm.prompt_n`, `slm.predicted_n`, `slm.cache_reuse`, `slm.ttfb_ms`, `slm.heartbeat_count`, `slm.client_disconnected`, `slm.session_id` | the doc |
| `slm.client.trace_id` | the doc's legacy `x-trace-id`, when present (decision 9) |
| `http.response.status_code` | `doc["status"]` |

`None` values are dropped rather than sent (OTel rejects `None` attribute values).

### 4 — Wire the four call sites in `router.py`
Replace `asyncio.create_task(ship_request_complete(tel_doc))` with
`emit_request_span(tel_doc, emit_path="…", headers=…)` at 658 / 857 / 1134 / 1355.

`_stream_backend_response` has no `Request`, so it gains a
`carrier: Mapping[str, str] | None = None` parameter, passed as `dict(request.headers)` from **all
three** call sites: chat-stream (`:816`), responses-stream (`:1281`) and the responses→chat fallback
stream (`:1321`). All three report `emit_path="streaming"` — that is the ticket's four-path model,
where "streaming" is an emit site, not an endpoint.

Update the import at `router.py:18`, and add `init_tracing()` / `shutdown_tracing()` to the lifespan.

**Fold-in:** the comment at `router.py:1257-1261` claims "Telemetry stays off here: this endpoint has
never emitted `request_complete`", but the call two lines below it uses `_stream_backend_response`'s
default `emit_telemetry=True`, so telemetry *is* emitted. The comment is stale and sits inside the
lines this change edits; correcting it is folded in rather than ticketed.

### 5 — Effective-config endpoint
`GET /v1/telemetry/effective-config` → `JSONResponse(effective_config())`:

```json
{
  "service_name": "slm-server",
  "otlp_endpoint": "http://localhost:4318",
  "otlp_traces_endpoint": "http://localhost:4318/v1/traces",
  "otlp_protocol": "http/protobuf",
  "telemetry_enabled": true,
  "tracer_provider_initialized": false,
  "emit_paths": ["chat", "responses", "rerank", "streaming"],
  "elasticsearch_export": false
}
```

### 6 — Docs
`README.md` — a Telemetry section: the OTLP endpoint, the env vars, the effective-config endpoint,
and the note that ES shipping is gone. `CLAUDE.md` — update the `telemetry.py` module description.

### 7 — Quality gates
`uv run pytest` · `uv run mypy src/` · `uv run ruff check src/ tests/` · `uv run ruff format src/ tests/`
· code-review skill at `high` (src logic + a new network egress path) · security-review (new outbound
egress + a new unauthenticated endpoint).

## Risks

- **A new unauthenticated endpoint.** `/v1/telemetry/effective-config` exposes the OTLP endpoint and
  service name. The router already exposes `/v1/models` and `/v1/backends/health` unauthenticated and
  binds loopback/LAN, so this adds no new class of exposure — but the artifact must not echo
  credentials. It reports no secrets, and there are none to report on this path (the Collector hop is
  unauthenticated loopback). To be confirmed by security-review.
- **Export noise when the Collector is down.** Enabled-by-default means a failing export every batch
  interval until FRE-1070 lands. Fail-soft and rate-limited by the batch processor; noted in the
  runbook rather than engineered around.
- **Stale `.env`.** The gitignored `.env` still carries `SLM_ES_URL` and CF-Access credentials. The
  code ignores them after this change; cleaning it is a runbook item for master, not a diff item.
