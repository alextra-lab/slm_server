"""Tests for model selection behavior in router."""

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from slm_server import router as router_module
from slm_server.config import ModelConfig, ModelDefinition
from slm_server.router import _get_model_definition, app


def _model(
    model_id: str,
    port: int,
    *,
    enabled: bool = True,
) -> ModelDefinition:
    return ModelDefinition(
        id=model_id,
        backend="llamacpp",
        port=port,
        context_length=8192,
        quantization="Q4_K_M",
        max_concurrency=1,
        default_timeout=60,
        model_path="/tmp/model.gguf",
        enabled=enabled,
    )


def test_get_model_definition_ignores_disabled_duplicate() -> None:
    cfg = ModelConfig(
        models={
            "standard": _model("unsloth/qwen3.6-35-A3B", 8501, enabled=False),
            "reasoning": _model("unsloth/qwen3.6-35-A3B", 8502, enabled=True),
        }
    )

    selected = _get_model_definition("unsloth/qwen3.6-35-A3B", cfg)

    assert selected.port == 8502
    assert selected.enabled is True


def test_get_model_definition_rejects_multiple_enabled_duplicates() -> None:
    cfg = ModelConfig(
        models={
            "reasoning_a": _model("unsloth/qwen3.6-35-A3B", 8501, enabled=True),
            "reasoning_b": _model("unsloth/qwen3.6-35-A3B", 8502, enabled=True),
        }
    )

    with pytest.raises(HTTPException) as exc:
        _get_model_definition("unsloth/qwen3.6-35-A3B", cfg)

    assert exc.value.status_code == 409


def test_get_model_definition_returns_disabled_message_when_only_disabled() -> None:
    cfg = ModelConfig(
        models={
            "reasoning": _model("unsloth/qwen3.6-35-A3B", 8502, enabled=False),
        }
    )

    with pytest.raises(HTTPException) as exc:
        _get_model_definition("unsloth/qwen3.6-35-A3B", cfg)

    assert exc.value.status_code == 503


def _client_with(monkeypatch: pytest.MonkeyPatch, cfg: ModelConfig) -> TestClient:
    """The router lifespan reloads config from disk, so patch the loader rather
    than assigning app.state.model_config, which gets overwritten on startup."""
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    return TestClient(app)


def test_list_models_omits_disabled_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """/v1/models is a discovery endpoint: a client must not be offered a model
    that has no backend. Routing already refuses disabled entries, so listing
    them only invites a request that is guaranteed to fail."""
    cfg = ModelConfig(
        models={
            "live": _model("vendor/live", 8502),
            "off": _model("vendor/off", 8504, enabled=False),
        }
    )
    with _client_with(monkeypatch, cfg) as client:
        body = client.get("/v1/models").json()

    ids = [m["id"] for m in body["data"]]
    assert ids == ["vendor/live"]
    assert "vendor/off" not in ids


def test_list_models_returns_empty_list_when_all_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ModelConfig(models={"off": _model("vendor/off", 8504, enabled=False)})
    with _client_with(monkeypatch, cfg) as client:
        body = client.get("/v1/models").json()

    assert body == {"object": "list", "data": []}


def test_list_models_hides_disabled_entry_sharing_a_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two entries may share a port with only one enabled, as flash-next and the
    Qwen3.6 primary both claim 8502. Listing both makes the choice look arbitrary."""
    cfg = ModelConfig(
        models={
            "live": _model("vendor/live", 8502),
            "shadow": _model("vendor/shadow", 8502, enabled=False),
        }
    )
    with _client_with(monkeypatch, cfg) as client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]

    assert ids == ["vendor/live"]
