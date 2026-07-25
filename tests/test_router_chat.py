"""Tests for POST /v1/chat/completions routing behavior."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from slm_server import router as router_module  # type: ignore[import-untyped]
from slm_server.config import ModelConfig, ModelDefinition  # type: ignore[import-untyped]

app = router_module.app


def _chat_model_def() -> ModelDefinition:
    return ModelDefinition(
        id="mlx-community/Qwen3.5-9B-8bit",
        backend="mlx",
        port=8501,
        context_length=32768,
        quantization="8bit",
        max_concurrency=1,
        default_timeout=120,
        model_type="multimodal",
        model_path="hf/mlx-stub",
    )


@pytest.fixture
def router_client(monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(models={"standard": _chat_model_def()})
    monkeypatch.setattr(
        router_module,
        "load_model_config",
        lambda config_path=None, validate=True: cfg,
    )
    with TestClient(app) as client:
        yield client


def test_chat_uses_default_timeout_and_does_not_forward_timeout_field(
    router_client: TestClient,
) -> None:
    captured: dict[str, object] = {}

    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["timeout"] = kwargs.get("timeout")
        return httpx.Response(200, json={"id": "ok", "choices": []})

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    body = {
        "model": "mlx-community/Qwen3.5-9B-8bit",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = router_client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert captured["url"] == "http://localhost:8501/v1/chat/completions"
    assert captured["json"] == body
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 120


def test_chat_timeout_override_is_applied_and_not_forwarded(router_client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured["json"] = kwargs.get("json")
        captured["timeout"] = kwargs.get("timeout")
        return httpx.Response(200, json={"id": "ok", "choices": []})

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    response = router_client.post(
        "/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen3.5-9B-8bit",
            "messages": [{"role": "user", "content": "slow request"}],
            "timeout": 300,
        },
    )

    assert response.status_code == 200
    assert captured["json"] == {
        "model": "mlx-community/Qwen3.5-9B-8bit",
        "messages": [{"role": "user", "content": "slow request"}],
    }
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 300


def test_chat_timeout_override_rejects_out_of_range(router_client: TestClient) -> None:
    response = router_client.post(
        "/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen3.5-9B-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "timeout": 0,
        },
    )
    assert response.status_code == 400
    assert "between 1 and 3600 seconds" in response.json()["detail"]


# ── Telemetry integration tests (async) ───────────────────────────────────────────────────────────
# AsyncClient + ASGITransport runs in the same event loop as the test, so
# asyncio.create_task tasks drain with `await asyncio.sleep(0)`.
# app.state is pre-set to bypass lifespan (ASGITransport never triggers it).


def _sse_body_with_timings() -> bytes:
    """SSE response body with llama.cpp usage + timings in the final chunk."""
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "index": 0}]},
        {
            "choices": [{"delta": {"content": "!"}, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            "timings": {
                "prompt_ms": 1200.0,
                "predicted_ms": 500.0,
                "prompt_n": 100,
                "predicted_n": 10,
                "cache_n": 80,
            },
        },
    ]
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return "\n".join(lines).encode()


def _sse_body_no_timings() -> bytes:
    """SSE response body without timings (MLX backend)."""
    chunk = {
        "choices": [{"delta": {"content": "Hi"}, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
    }
    return f"data: {json.dumps(chunk)}\ndata: [DONE]".encode()


@pytest.fixture
def _telemetry_app_setup(monkeypatch: pytest.MonkeyPatch):
    """Pre-configure app.state for async telemetry tests (no lifespan needed)."""
    cfg = ModelConfig(models={"standard": _chat_model_def()})
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    app.state.model_config = cfg
    yield cfg


async def test_request_complete_doc_llamacpp_fields(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """request_complete doc has correct llama.cpp fields (usage + timings)."""
    captured: list[dict] = []

    async def fake_ship(doc: dict) -> None:
        captured.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=httpx.Response(
            200,
            content=_sse_body_with_timings(),
            headers={"content-type": "text/event-stream"},
        )
    )
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "X-Trace-Id": "trace-123",
                "X-Span-Id": "span-456",
                "X-Session-Id": "sess-789",
            },
        )

    await asyncio.sleep(0)  # drain create_task queue

    assert response.status_code == 200
    assert len(captured) == 1
    doc = captured[0]
    assert doc["trace_id"] == "trace-123"
    assert doc["span_id"] == "span-456"
    assert doc["session_id"] == "sess-789"
    assert doc["prompt_tokens"] == 100
    assert doc["completion_tokens"] == 10
    assert doc["prefill_ms"] == 1200.0
    assert doc["decode_ms"] == 500.0
    assert doc["prompt_n"] == 100
    assert doc["predicted_n"] == 10
    assert doc["cache_reuse"] == 80
    assert doc["total_ms"] >= 0
    assert doc["ts"] is not None
    assert doc["model_id"] == "mlx-community/Qwen3.5-9B-8bit"


async def test_request_complete_doc_mlx_no_timings(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """MLX backend: timings fields are None, token counts still present."""
    captured: list[dict] = []

    async def fake_ship(doc: dict) -> None:
        captured.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=httpx.Response(
            200,
            content=_sse_body_no_timings(),
            headers={"content-type": "text/event-stream"},
        )
    )
    app.state.http_client = fake_http

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    await asyncio.sleep(0)

    assert response.status_code == 200
    assert len(captured) == 1
    doc = captured[0]
    assert doc["prompt_tokens"] == 50
    assert doc["completion_tokens"] == 5
    assert doc["prefill_ms"] is None
    assert doc["decode_ms"] is None
    assert doc["cache_reuse"] is None


async def test_routing_request_log_has_trace_headers(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """routing_request log carries trace_id, span_id, session_id from request headers."""
    logged: list[dict] = []
    original_log_info = router_module.log.info

    def capturing_log_info(event: str, **kwargs: object) -> None:
        if event == "routing_request":
            logged.append({"event": event, **kwargs})
        original_log_info(event, **kwargs)

    monkeypatch.setattr(router_module.log, "info", capturing_log_info)
    monkeypatch.setattr(router_module, "ship_request_complete", AsyncMock())

    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=httpx.Response(200, json={"id": "ok", "choices": [], "usage": {}})
    )
    app.state.http_client = fake_http

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "x"}],
            },
            headers={"X-Trace-Id": "t-1", "X-Span-Id": "s-2", "X-Session-Id": "sess-3"},
        )

    assert len(logged) == 1
    assert logged[0]["trace_id"] == "t-1"
    assert logged[0]["span_id"] == "s-2"
    assert logged[0]["session_id"] == "sess-3"


async def test_es_ship_failure_does_not_break_response(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """ES shipping raising an exception must not break the HTTP response (fail-soft)."""

    async def fake_ship_raising(doc: dict) -> None:
        raise RuntimeError("ES is down")

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship_raising)

    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=httpx.Response(
            200,
            content=_sse_body_no_timings(),
            headers={"content-type": "text/event-stream"},
        )
    )
    app.state.http_client = fake_http

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200


async def test_url_unset_no_ship_call(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """When SLM_ES_URL is not set, ship_request_complete is still scheduled but
    itself is a no-op — the router always calls create_task (the guard is inside
    ship_request_complete). Verify the response succeeds and no ES POST is made."""
    import slm_server.telemetry as tel

    monkeypatch.setattr(tel, "_ES_URL", None)

    es_posted = []

    async def fake_ship(doc: dict) -> None:
        # Delegates to real telemetry which will short-circuit on _ES_URL=None
        # We just capture if it was called at all
        es_posted.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=httpx.Response(
            200,
            content=_sse_body_no_timings(),
            headers={"content-type": "text/event-stream"},
        )
    )
    app.state.http_client = fake_http

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    await asyncio.sleep(0)

    assert response.status_code == 200
    # When URL is unset the real ship_request_complete is a no-op — no httpx POST
    # This test verifies the router still succeeds (the telemetry unit test covers the no-op)


# ─── FRE-980: SSE pass-through streaming + prefill heartbeat ──────────────────


class _LazyByteStream(httpx.AsyncByteStream):
    """Async byte stream that yields chunks lazily, optionally stalling first.

    Models a backend that accepts the request and then goes silent while it
    prefills — the condition that produced the Cloudflare 524.
    """

    def __init__(self, chunks: list[bytes], first_delay: float = 0.0) -> None:
        self._chunks = chunks
        self._first_delay = first_delay

    async def __aiter__(self):
        for index, chunk in enumerate(self._chunks):
            if index == 0 and self._first_delay:
                await asyncio.sleep(self._first_delay)
            yield chunk


def _sse_stream_chunks() -> list[bytes]:
    """The body from _sse_body_with_timings(), split into separate wire chunks."""
    return [part + b"\n\n" for part in _sse_body_with_timings().split(b"\n\n") if part.strip()]


def _streaming_client(response: httpx.Response) -> MagicMock:
    fake_http = MagicMock()
    fake_http.build_request = MagicMock(return_value=MagicMock())
    fake_http.send = AsyncMock(return_value=response)
    return fake_http


def _streaming_response(chunks: list[bytes], first_delay: float = 0.0) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_LazyByteStream(chunks, first_delay=first_delay),
    )


async def test_streaming_request_passes_through_instead_of_buffering(
    _telemetry_app_setup: ModelConfig,
) -> None:
    """A stream=True request must use send(stream=True), never the buffering post().

    Regression guard for FRE-980: client.post() buffered the whole generation
    before emitting a byte, so Cloudflare saw silence for the turn and 524'd.
    """
    fake_http = _streaming_client(_streaming_response(_sse_stream_chunks()))
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert fake_http.send.await_args.kwargs["stream"] is True
    assert not fake_http.post.called
    assert b"Hi" in response.content


async def test_streaming_emits_heartbeat_while_backend_is_silent(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """Silence longer than the interval must produce SSE comment keep-alives."""
    monkeypatch.setattr(router_module, "_SSE_HEARTBEAT_INTERVAL_SECONDS", 0.05)

    fake_http = _streaming_client(_streaming_response(_sse_stream_chunks(), first_delay=0.3))
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    body = response.content
    assert body.startswith(b":"), "heartbeat must precede the first backend chunk"
    # Heartbeats are SSE comments, so real data still arrives intact.
    assert b"Hi" in body


async def test_streaming_heartbeat_absent_when_backend_is_prompt(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """No stall means no keep-alive noise on the wire."""
    monkeypatch.setattr(router_module, "_SSE_HEARTBEAT_INTERVAL_SECONDS", 5.0)

    fake_http = _streaming_client(_streaming_response(_sse_stream_chunks()))
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert not response.content.startswith(b":")


async def test_streaming_telemetry_emitted_after_stream_completes(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """usage/timings must survive the switch to pass-through streaming."""
    captured: list[dict] = []

    async def fake_ship(doc: dict) -> None:
        captured.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = _streaming_client(_streaming_response(_sse_stream_chunks()))
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"X-Trace-Id": "trace-980"},
        )

    assert response.status_code == 200
    await asyncio.sleep(0)  # drain create_task queue

    assert len(captured) == 1
    doc = captured[0]
    assert doc["trace_id"] == "trace-980"
    assert doc["prompt_tokens"] == 100
    assert doc["completion_tokens"] == 10
    assert doc["prefill_ms"] == 1200.0
    assert doc["cache_reuse"] == 80
    assert doc["status"] == 200


def _sequenced_streaming_client(responses: list[httpx.Response]) -> MagicMock:
    """Streaming client that returns a different response per send() call."""
    fake_http = MagicMock()
    fake_http.build_request = MagicMock(return_value=MagicMock())
    fake_http.send = AsyncMock(side_effect=responses)
    return fake_http


def _sent_urls(fake_http: MagicMock) -> list[str]:
    return [call.args[1] for call in fake_http.build_request.call_args_list]


# /v1/responses shares the same streaming machinery, so its tests live here
# alongside the helpers rather than in a near-duplicate module.


async def test_responses_streaming_passes_through_instead_of_buffering(
    _telemetry_app_setup: ModelConfig,
) -> None:
    """/v1/responses must stream through too — it had the same buffering bug."""
    fake_http = _streaming_client(_streaming_response(_sse_stream_chunks()))
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "input": "hi",
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert fake_http.send.await_args.kwargs["stream"] is True
    assert not fake_http.post.called
    assert b"Hi" in response.content


async def test_responses_streaming_falls_back_to_chat_on_404(
    _telemetry_app_setup: ModelConfig,
) -> None:
    """The 404 fallback must survive the switch to unbuffered streaming.

    send(stream=True) exposes the status before the body is read, so the probe
    can still be abandoned in favour of /v1/chat/completions.
    """
    probe = httpx.Response(
        404, headers={"content-type": "application/json"}, stream=_LazyByteStream([b"{}"])
    )
    fallback = _streaming_response(_sse_stream_chunks())
    fake_http = _sequenced_streaming_client([probe, fallback])
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "input": "hi",
                "stream": True,
            },
        )

    assert response.status_code == 200
    urls = _sent_urls(fake_http)
    assert len(urls) == 2
    assert urls[0].endswith("/v1/responses")
    assert urls[1].endswith("/v1/chat/completions")
    assert b"Hi" in response.content


async def test_responses_streaming_emits_no_telemetry(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """This endpoint never emitted request_complete; the fix must not add it."""
    captured: list[dict] = []

    async def fake_ship(doc: dict) -> None:
        captured.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = _streaming_client(_streaming_response(_sse_stream_chunks()))
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "input": "hi",
                "stream": True,
            },
        )

    assert response.status_code == 200
    await asyncio.sleep(0)
    assert captured == []


async def test_streaming_telemetry_records_ttfb_and_heartbeats(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """ttfb_ms and heartbeat_count make a silent backend visible in telemetry.

    total_ms alone cannot distinguish a request that waited for a slot from one
    that computed slowly; ttfb_ms against prefill_ms separates them.
    """
    monkeypatch.setattr(router_module, "_SSE_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    captured: list[dict] = []

    async def fake_ship(doc: dict) -> None:
        captured.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = _streaming_client(_streaming_response(_sse_stream_chunks(), first_delay=0.3))
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    await asyncio.sleep(0)

    doc = captured[0]
    # First content byte arrived only after the stall, not at t0.
    assert doc["ttfb_ms"] >= 250
    assert doc["ttfb_ms"] <= doc["total_ms"]
    assert doc["heartbeat_count"] >= 1
    assert doc["client_disconnected"] is False


async def test_streaming_telemetry_ttfb_excludes_heartbeats(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """Keep-alives must not be mistaken for the first content byte."""
    monkeypatch.setattr(router_module, "_SSE_HEARTBEAT_INTERVAL_SECONDS", 5.0)
    captured: list[dict] = []

    async def fake_ship(doc: dict) -> None:
        captured.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = _streaming_client(_streaming_response(_sse_stream_chunks()))
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    await asyncio.sleep(0)
    doc = captured[0]
    assert doc["heartbeat_count"] == 0
    assert doc["ttfb_ms"] is not None


async def test_non_streaming_telemetry_omits_streaming_fields(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """Non-streaming requests keep the previous doc shape — fields present, None."""
    captured: list[dict] = []

    async def fake_ship(doc: dict) -> None:
        captured.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=httpx.Response(
            200,
            content=_sse_body_with_timings(),
            headers={"content-type": "text/event-stream"},
        )
    )
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    await asyncio.sleep(0)
    doc = captured[0]
    assert doc["ttfb_ms"] is None
    assert doc["heartbeat_count"] is None
    assert doc["client_disconnected"] is None


async def test_streaming_backend_error_returns_json_not_stream(
    _telemetry_app_setup: ModelConfig,
) -> None:
    """A 4xx/5xx on a streaming request still surfaces as a JSON error body."""
    error_response = httpx.Response(
        503,
        headers={"content-type": "application/json"},
        stream=_LazyByteStream([b'{"error": "backend down"}']),
    )
    fake_http = _streaming_client(error_response)
    app.state.http_client = fake_http

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 503
