"""Tests for POST /v1/embeddings routing."""

import httpx
import pytest
from fastapi.testclient import TestClient

from slm_server import router as router_module
from slm_server.config import ModelConfig, ModelDefinition

app = router_module.app


def _embedding_model_def() -> ModelDefinition:
    return ModelDefinition(
        id="test/embed-model",
        backend="llamacpp",
        port=9999,
        context_length=2048,
        quantization="GGUF",
        max_concurrency=1,
        default_timeout=30,
        model_type="embeddings",
        model_path="hf/embeddings-stub",  # HF-style id avoids local path checks in tests
    )


@pytest.fixture
def router_client(monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(models={"emb": _embedding_model_def()})
    monkeypatch.setattr(
        router_module,
        "load_model_config",
        lambda config_path=None, validate=True: cfg,
    )
    with TestClient(app) as client:
        yield client


def test_embeddings_missing_model_field(router_client: TestClient) -> None:
    r = router_client.post("/v1/embeddings", json={"input": "hello"})
    assert r.status_code == 400


def test_embeddings_unknown_model(router_client: TestClient) -> None:
    r = router_client.post(
        "/v1/embeddings",
        json={"model": "unknown/model", "input": "hello"},
    )
    assert r.status_code == 404


def test_embeddings_forwards_to_backend(router_client: TestClient) -> None:
    captured: dict[str, object] = {}

    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2]}],
                "model": "test/embed-model",
            },
        )

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    r = router_client.post(
        "/v1/embeddings",
        json={"model": "test/embed-model", "input": "hello"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["embedding"] == [0.1, 0.2]
    assert captured["url"] == "http://localhost:9999/v1/embeddings"
    assert captured["json"] == {"model": "test/embed-model", "input": "hello"}


def test_embeddings_backend_unreachable(router_client: TestClient) -> None:
    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=None)

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    r = router_client.post(
        "/v1/embeddings",
        json={"model": "test/embed-model", "input": "hello"},
    )
    assert r.status_code == 503
    assert "unreachable" in r.json()["error"]["message"].lower()


def test_embeddings_backend_timeout(router_client: TestClient) -> None:
    async def fake_post(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    r = router_client.post(
        "/v1/embeddings",
        json={"model": "test/embed-model", "input": "hello"},
    )
    assert r.status_code == 504
