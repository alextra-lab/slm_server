"""Tests for POST /v1/rerank routing."""

import httpx
import pytest
from fastapi.testclient import TestClient

from slm_server import router as router_module
from slm_server.config import ModelConfig, ModelDefinition

app = router_module.app


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
