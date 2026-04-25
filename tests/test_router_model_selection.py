"""Tests for model selection behavior in router."""

import pytest
from fastapi import HTTPException

from slm_server.config import ModelConfig, ModelDefinition
from slm_server.router import _get_model_definition


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
