"""Tests for POST /v1/rerank routing."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import structlog
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from slm_server import router as router_module
from slm_server.config import ModelConfig, ModelDefinition

app = router_module.app

# Exact request_complete schema, shared across chat and rerank (see _build_request_telemetry).
_TELEMETRY_KEYS = {
    "trace_id", "span_id", "session_id", "model_id", "backend", "port",
    "prompt_tokens", "completion_tokens", "prefill_ms", "decode_ms",
    "prompt_n", "predicted_n", "cache_reuse", "total_ms", "status", "ts",
    # Streaming-only signals — present but None for rerank, preserving the
    # invariant that every endpoint ships an identical key set.
    "ttfb_ms", "heartbeat_count", "client_disconnected",
}


def _rerank_model_def() -> ModelDefinition:
    return ModelDefinition(
        id="test/rerank-model",
        backend="llamacpp",
        port=9998,
        context_length=8192,
        quantization="GGUF",
        max_concurrency=1,
        default_timeout=60,
        model_type="rerank",
        model_path="hf/rerank-stub",
    )


@pytest.fixture
def router_client(monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(models={"rr": _rerank_model_def()})
    monkeypatch.setattr(
        router_module,
        "load_model_config",
        lambda config_path=None, validate=True: cfg,
    )
    with TestClient(app) as client:
        yield client


def test_rerank_missing_model_field(router_client: TestClient) -> None:
    r = router_client.post("/v1/rerank", json={"query": "q"})
    assert r.status_code == 400


def test_rerank_unknown_model(router_client: TestClient) -> None:
    r = router_client.post(
        "/v1/rerank",
        json={"model": "unknown/model", "query": "q", "documents": []},
    )
    assert r.status_code == 404


def test_rerank_forwards_to_backend(router_client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return httpx.Response(
            200,
            json={"results": [], "model": "test/rerank-model"},
        )

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    body = {"model": "test/rerank-model", "query": "hello", "documents": ["a", "b"]}
    r = router_client.post("/v1/rerank", json=body)
    assert r.status_code == 200
    assert r.json()["model"] == "test/rerank-model"
    assert captured["url"] == "http://localhost:9998/v1/rerank"
    assert captured["json"] == body


def test_routing_rerank_request_stamps_identity_headers(router_client: TestClient) -> None:
    """FRE-701: routing_rerank_request carries trace/session/span from the request headers."""

    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "model": "test/rerank-model"})

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    with structlog.testing.capture_logs() as logs:
        r = router_client.post(
            "/v1/rerank",
            json={"model": "test/rerank-model", "query": "q", "documents": ["a"]},
            headers={"X-Trace-Id": "t1", "X-Span-Id": "s1", "X-Session-Id": "sess1"},
        )

    assert r.status_code == 200
    events = [e for e in logs if e["event"] == "routing_rerank_request"]
    assert len(events) == 1
    e = events[0]
    assert e["trace_id"] == "t1"
    assert e["span_id"] == "s1"
    assert e["session_id"] == "sess1"


def test_routing_rerank_request_without_headers_logs_absent(router_client: TestClient) -> None:
    """FRE-701: a request without identity headers still logs (values None), no error."""

    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "model": "test/rerank-model"})

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    with structlog.testing.capture_logs() as logs:
        r = router_client.post(
            "/v1/rerank",
            json={"model": "test/rerank-model", "query": "q", "documents": ["a"]},
        )

    assert r.status_code == 200
    events = [e for e in logs if e["event"] == "routing_rerank_request"]
    assert len(events) == 1
    e = events[0]
    assert e["trace_id"] is None
    assert e["span_id"] is None
    assert e["session_id"] is None


@pytest.fixture
def _telemetry_app_setup(monkeypatch: pytest.MonkeyPatch):
    """Pre-configure app.state for async telemetry tests (ASGITransport skips lifespan)."""
    cfg = ModelConfig(models={"rr": _rerank_model_def()})
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    app.state.model_config = cfg
    yield cfg


async def _post_rerank(headers: dict[str, str] | None = None) -> httpx.Response:
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        return await client.post(
            "/v1/rerank",
            json={"model": "test/rerank-model", "query": "q", "documents": ["a", "b"]},
            headers=headers or {},
        )


async def test_rerank_ships_request_complete_consistent_schema(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """Rerank ships a request_complete doc with EXACTLY the chat schema: generation-only
    fields present-but-None, trace context from headers, usage carried through."""
    captured: list[dict] = []

    async def fake_ship(doc: dict) -> None:
        captured.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "results": [],
                "model": "test/rerank-model",
                "usage": {"prompt_tokens": 55, "total_tokens": 55},
            },
        )
    )
    app.state.http_client = fake_http

    response = await _post_rerank(
        headers={"X-Trace-Id": "t1", "X-Span-Id": "s1", "X-Session-Id": "sess1"}
    )
    await asyncio.sleep(0)  # drain create_task queue

    assert response.status_code == 200
    assert len(captured) == 1
    doc = captured[0]

    # Exact schema match with chat — extra or missing keys must fail.
    assert doc.keys() == _TELEMETRY_KEYS
    assert doc["trace_id"] == "t1"
    assert doc["span_id"] == "s1"
    assert doc["session_id"] == "sess1"
    assert doc["model_id"] == "test/rerank-model"
    assert doc["prompt_tokens"] == 55
    assert doc["status"] == 200
    assert doc["total_ms"] >= 0
    assert doc["ts"] is not None
    # generation-only fields are not applicable to a reranker → present but None
    for f in ("completion_tokens", "prefill_ms", "decode_ms", "prompt_n", "predicted_n", "cache_reuse"):
        assert doc[f] is None


async def test_rerank_ships_telemetry_on_backend_error_status(
    monkeypatch: pytest.MonkeyPatch, _telemetry_app_setup: ModelConfig
) -> None:
    """Telemetry is emitted even when the backend returns a 4xx/5xx, with status recorded."""
    captured: list[dict] = []

    async def fake_ship(doc: dict) -> None:
        captured.append(doc)

    monkeypatch.setattr(router_module, "ship_request_complete", fake_ship)

    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=httpx.Response(500, json={"error": "backend boom"})
    )
    app.state.http_client = fake_http

    response = await _post_rerank()
    await asyncio.sleep(0)  # drain create_task queue

    assert response.status_code == 500
    assert len(captured) == 1
    doc = captured[0]
    assert doc.keys() == _TELEMETRY_KEYS
    assert doc["status"] == 500
    assert doc["prompt_tokens"] is None  # no usage in error body


def test_rerank_backend_unreachable(router_client: TestClient) -> None:
    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=None)

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    r = router_client.post(
        "/v1/rerank",
        json={"model": "test/rerank-model", "query": "q", "documents": []},
    )
    assert r.status_code == 503


def test_rerank_backend_timeout(router_client: TestClient) -> None:
    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    r = router_client.post(
        "/v1/rerank",
        json={"model": "test/rerank-model", "query": "q", "documents": []},
    )
    assert r.status_code == 504
