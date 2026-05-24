"""Tests for llama.cpp launch flags when model_type is multimodal (vision)."""

from pathlib import Path

import pytest

from slm_server.config import ModelConfig, ModelDefinition, validate_model_config
from slm_server.start_backends import build_llama_native_command, build_llamacpp_command


def test_build_llama_native_command_includes_mmproj_for_multimodal(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"")
    cmd = build_llama_native_command(
        gguf,
        8502,
        131072,
        "UD-Q4_K_XL",
        1,
        None,
        "test/model",
        "/usr/bin/true",
        model_type="multimodal",
        mmproj_path=mmproj,
    )
    assert "--mmproj" in cmd
    assert cmd[cmd.index("--mmproj") + 1] == str(mmproj)


def test_build_llama_native_command_no_mmproj_for_lm(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"")
    cmd = build_llama_native_command(
        gguf,
        8502,
        16384,
        "UD-Q4_K_XL",
        1,
        None,
        "test/model",
        "/usr/bin/true",
        model_type="lm",
        mmproj_path=mmproj,
    )
    assert "--mmproj" not in cmd


def test_build_llama_native_command_nonexistent_mmproj_raises(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    with pytest.raises(ValueError, match="mmproj_path does not exist"):
        build_llama_native_command(
            gguf,
            8502,
            131072,
            "UD-Q4_K_XL",
            1,
            None,
            "test/model",
            "/usr/bin/true",
            model_type="multimodal",
            mmproj_path=tmp_path / "missing.gguf",
        )


def test_build_llamacpp_command_multimodal_raises(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    with pytest.raises(ValueError, match="native llama-server"):
        build_llamacpp_command(
            gguf,
            8502,
            131072,
            "UD-Q4_K_XL",
            1,
            model_type="multimodal",
        )


def test_validate_model_config_warns_multimodal_llamacpp_missing_mmproj() -> None:
    cfg = ModelConfig(
        models={
            "vision": ModelDefinition(
                id="test/vl-model",
                backend="llamacpp",
                port=8600,
                quantization="Q4_K_XL",
                default_timeout=300,
                model_type="multimodal",
                model_path="hf/demo",
            )
        }
    )
    issues = validate_model_config(cfg)
    assert any("mmproj_path" in i for i in issues)


def test_validate_model_config_warns_nonexistent_mmproj(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    cfg = ModelConfig(
        models={
            "vision": ModelDefinition(
                id="test/vl-model",
                backend="llamacpp",
                port=8600,
                quantization="Q4_K_XL",
                default_timeout=300,
                model_type="multimodal",
                model_path=str(gguf),
                mmproj_path=str(tmp_path / "missing.gguf"),
            )
        }
    )
    issues = validate_model_config(cfg)
    assert any("mmproj_path does not exist" in i for i in issues)


def test_validate_model_config_no_issues_with_valid_multimodal(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"")
    cfg = ModelConfig(
        models={
            "vision": ModelDefinition(
                id="test/vl-model",
                backend="llamacpp",
                port=8600,
                quantization="Q4_K_XL",
                default_timeout=300,
                model_type="multimodal",
                model_path=str(gguf),
                mmproj_path=str(mmproj),
            )
        }
    )
    issues = validate_model_config(cfg)
    assert not any("mmproj" in i for i in issues)


def test_validate_model_config_warns_mmproj_on_non_multimodal(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"")
    cfg = ModelConfig(
        models={
            "lm": ModelDefinition(
                id="test/lm-model",
                backend="llamacpp",
                port=8600,
                quantization="Q4_K_XL",
                default_timeout=300,
                model_type="lm",
                model_path=str(gguf),
                mmproj_path=str(mmproj),
            )
        }
    )
    issues = validate_model_config(cfg)
    assert any("mmproj_path is set but model_type is lm" in i for i in issues)
