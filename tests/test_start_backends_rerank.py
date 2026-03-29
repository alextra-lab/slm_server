"""Tests for llama.cpp rerank launch flags and config validation."""

from pathlib import Path

import pytest

from slm_server.config import ModelConfig, ModelDefinition, validate_model_config
from slm_server.start_backends import build_llama_native_command, build_llamacpp_command


def test_build_llama_native_command_rerank_flags(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    cmd = build_llama_native_command(
        gguf,
        8500,
        2048,
        "q8",
        1,
        None,
        "test/rerank",
        "/usr/bin/true",
        model_type="rerank",
    )
    assert "--embedding" in cmd
    assert "--pooling" in cmd
    assert cmd[cmd.index("--pooling") + 1] == "rank"
    assert "--reranking" in cmd
    assert "--chat-template-kwargs" not in cmd


def test_build_llamacpp_command_rerank_raises(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    with pytest.raises(ValueError, match="native llama-server"):
        build_llamacpp_command(
            gguf,
            8501,
            2048,
            "q8",
            1,
            model_type="rerank",
        )


def test_validate_model_config_rerank_mlx_warns() -> None:
    cfg = ModelConfig(
        models={
            "r": ModelDefinition(
                id="x/y",
                backend="mlx",
                port=8601,
                quantization="8bit",
                default_timeout=10,
                model_type="rerank",
                model_path="hf/demo",
            )
        }
    )
    issues = validate_model_config(cfg)
    assert any("rerank" in i and "llamacpp" in i for i in issues)


def test_validate_model_config_rerank_lm_only_warnings() -> None:
    cfg = ModelConfig(
        models={
            "r": ModelDefinition(
                id="x/y",
                backend="llamacpp",
                port=8602,
                quantization="GGUF",
                default_timeout=10,
                model_type="rerank",
                model_path="hf/demo",
                tool_call_parser="qwen3",
            )
        }
    )
    issues = validate_model_config(cfg)
    assert any("tool_call_parser" in i for i in issues)
