"""Tests for POST /v1/chat/completions routing behavior."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

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
