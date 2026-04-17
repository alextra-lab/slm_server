"""Tests for llama.cpp launch flags when model_type is embeddings."""

from pathlib import Path

from slm_server.config import ModelConfig, ModelDefinition, validate_model_config
from slm_server.start_backends import build_llama_native_command, build_llamacpp_command


def test_build_llama_native_command_includes_embedding_flag(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    cmd = build_llama_native_command(
        gguf,
        8500,
        2048,
        "q8",
        1,
        None,
        "test/model",
        "/usr/bin/true",
        model_type="embeddings",
    )
    assert "--embedding" in cmd
    assert "--chat-template-kwargs" not in cmd


def test_build_llama_native_command_lm_still_uses_chat_template_kwargs(
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    cmd = build_llama_native_command(
        gguf,
        8500,
        2048,
        "q8",
        1,
        {"enable_thinking": False},
        "test/model",
        "/usr/bin/true",
        model_type="lm",
    )
    assert "--embedding" not in cmd
    assert "--chat-template-kwargs" in cmd


def test_build_llamacpp_command_embedding_mode(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    cmd = build_llamacpp_command(
        gguf,
        8501,
        2048,
        "q8",
        1,
        model_type="embeddings",
    )
    i = cmd.index("--embedding")
    assert cmd[i + 1] == "true"


def test_validate_model_config_embeddings_warns_on_lm_only_options() -> None:
    cfg = ModelConfig(
        models={
            "e": ModelDefinition(
                id="x/y",
                backend="llamacpp",
                port=8600,
                quantization="GGUF",
                default_timeout=10,
                model_type="embeddings",
                model_path="hf/demo",
                enable_auto_tool_choice=True,
                tool_call_parser="qwen3",
                reasoning_parser="qwen3",
                chat_template_kwargs={"enable_thinking": True},
            )
        }
    )
    issues = validate_model_config(cfg)
    joined = " ".join(issues)
    assert "enable_auto_tool_choice" in joined
    assert "tool_call_parser" in joined
    assert "reasoning_parser" in joined
    assert "chat_template_kwargs" in joined
