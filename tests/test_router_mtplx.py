"""Router behaviour for backend: mtplx entries (spec §2)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from slm_server import router as router_module  # type: ignore[import-untyped]
from slm_server.config import ModelConfig, ModelDefinition  # type: ignore[import-untyped]

app = router_module.app
prepare = router_module._prepare_chat_template_kwargs


def _mtplx_def(**overrides: object) -> ModelDefinition:
    fields: dict[str, object] = {
        "id": "mtplx-test",
        "backend": "mtplx",
        "port": 8600,
        "quantization": "4bit",
        "default_timeout": 600,
        "model_path": "/path/to/pack",
        "mtp_depth": 3,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    fields.update(overrides)
    return ModelDefinition(**fields)


def _llama_def(**overrides: object) -> ModelDefinition:
    fields: dict[str, object] = {
        "id": "llama-test",
        "backend": "llamacpp",
        "port": 8502,
        "quantization": "Q4",
        "default_timeout": 600,
        "model_path": "/path/to/model.gguf",
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
    }
    fields.update(overrides)
    return ModelDefinition(**fields)


# ── helper ────────────────────────────────────────────────────────────────


def test_llamacpp_injects_config_kwargs_only_when_request_has_none() -> None:
    model_def = _llama_def()
    assert prepare({"model": "x"}, model_def)["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    body = {"model": "x", "chat_template_kwargs": {"enable_thinking": False}}
    assert prepare(body, model_def) == body


def test_mtplx_merges_config_and_request_kwargs_per_key() -> None:
    body = {"model": "x", "chat_template_kwargs": {"enable_thinking": False, "foo": 1}}
    out = prepare(body, _mtplx_def())
    assert out["chat_template_kwargs"] == {"enable_thinking": False, "foo": 1}


def test_mtplx_uses_config_kwargs_when_request_has_none() -> None:
    out = prepare({"model": "x"}, _mtplx_def())
    assert out["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in out


def test_mtplx_lifts_request_reasoning_effort_to_top_level() -> None:
    body = {"model": "x", "chat_template_kwargs": {"reasoning_effort": "low"}}
    out = prepare(body, _mtplx_def())
    assert out["reasoning_effort"] == "low"
    assert out["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_mtplx_top_level_reasoning_effort_wins() -> None:
    body = {
        "model": "x",
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    assert prepare(body, _mtplx_def())["reasoning_effort"] == "xhigh"


def test_mtplx_does_not_lift_from_config_kwargs() -> None:
    model_def = _mtplx_def(chat_template_kwargs={"reasoning_effort": "low"})
    assert "reasoning_effort" not in prepare({"model": "x"}, model_def)


def test_prepare_does_not_mutate_the_request_body() -> None:
    body = {"model": "x", "chat_template_kwargs": {"reasoning_effort": "low"}}
    snapshot = {"model": "x", "chat_template_kwargs": {"reasoning_effort": "low"}}
    prepare(body, _mtplx_def())
    assert body == snapshot


# ── through the router ─────────────────────────────────────────────────────


@pytest.fixture
def mtplx_client(monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(models={"mtplx": _mtplx_def()})
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    with TestClient(app) as client:
        yield client


def test_chat_forwards_translated_body_and_headers(mtplx_client: TestClient) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["headers"] = {k.lower(): v for k, v in (kwargs.get("headers") or {}).items()}
        return httpx.Response(200, json={"id": "ok", "choices": []})

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    response = mtplx_client.post(
        "/v1/chat/completions",
        json={
            "model": "mtplx-test",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
        headers={
            "X-Session-Id": "sess-1",
            "X-Trace-Id": "trace-1",
            "X-Span-Id": "span-1",
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        },
    )

    assert response.status_code == 200
    assert captured["url"] == "http://localhost:8600/v1/chat/completions"
    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["json"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }
    for header in ("x-session-id", "x-trace-id", "x-span-id", "traceparent"):
        assert header in captured["headers"]


def test_responses_fallback_translates_the_chat_body(mtplx_client: TestClient) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/v1/responses"):
            return httpx.Response(404, json={"error": "not found"})
        captured["json"] = kwargs.get("json")
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "hi"}}]},
        )

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    response = mtplx_client.post(
        "/v1/responses",
        json={
            "model": "mtplx-test",
            "input": "hi",
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
    )

    assert response.status_code == 200
    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["json"]["chat_template_kwargs"]["enable_thinking"] is True


class _FakeStreamingClient:
    """Stands in for the shared httpx.AsyncClient on the streaming path.

    `bodies` maps a URL suffix to (status, SSE bytes). `sent` records each forwarded body.
    """

    def __init__(self, bodies: dict[str, tuple[int, bytes]]) -> None:
        self.bodies = bodies
        self.sent: list[tuple[str, Any]] = []

    def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
        self.sent.append((url, kwargs.get("json")))
        return httpx.Request(method, url)

    async def send(self, request: httpx.Request, stream: bool = False) -> httpx.Response:
        for suffix, (status, body) in self.bodies.items():
            if str(request.url).endswith(suffix):
                return httpx.Response(
                    status,
                    headers={"content-type": "text/event-stream"},
                    content=body,
                    request=request,
                )
        raise AssertionError(f"unexpected backend URL: {request.url}")


async def test_streaming_responses_fallback_translates_the_chat_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ModelConfig(models={"mtplx": _mtplx_def()})
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    app.state.model_config = cfg
    fake = _FakeStreamingClient(
        {
            "/v1/responses": (404, b""),
            "/v1/chat/completions": (200, b"data: [DONE]\n\n"),
        }
    )
    app.state.http_client = fake

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "mtplx-test",
                "input": "hi",
                "stream": True,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
        )

    assert response.status_code == 200
    url, body = fake.sent[-1]
    assert url == "http://localhost:8600/v1/chat/completions"
    assert body["reasoning_effort"] == "low"
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
