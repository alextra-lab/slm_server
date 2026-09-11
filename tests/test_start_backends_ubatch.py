"""Tests for the physical micro-batch size (native llama-server --ubatch-size).

A warm agent round appends about 10k tokens to a 60k cached context. At the
llama.cpp default of 512 that prefill runs in ~20 micro-batches, each reading the
whole stored KV. Measured on 2026-09-11 with Qwen3.8-Flash-Next: 2048 ran the
warm prefill about 13% faster than 512, and 4096 added nothing more.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_server.config import ModelDefinition
from slm_server.start_backends import build_llama_native_command


def _native(gguf: Path, **kwargs) -> list[str]:
    return build_llama_native_command(
        gguf, 8502, 131072, "UD-IQ4_XS", 1, None, "test/model", "/usr/bin/true", **kwargs
    )


def test_ubatch_size_emits_flag(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    cmd = _native(gguf, ubatch_size=2048)
    assert cmd[cmd.index("--ubatch-size") + 1] == "2048"


def test_no_ubatch_flag_when_unset(tmp_path: Path) -> None:
    """Unset keeps llama.cpp's own default rather than pinning one here."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    assert "--ubatch-size" not in _native(gguf)


def _definition(tmp_path: Path, **kwargs) -> ModelDefinition:
    return ModelDefinition(
        id="test/flash",
        backend="llamacpp",
        port=8502,
        quantization="UD-IQ4_XS",
        default_timeout=600,
        model_path=str(tmp_path / "model.gguf"),
        **kwargs,
    )


def test_model_definition_accepts_ubatch_size(tmp_path: Path) -> None:
    assert _definition(tmp_path, ubatch_size=2048).ubatch_size == 2048


def test_model_definition_ubatch_size_defaults_none(tmp_path: Path) -> None:
    assert _definition(tmp_path).ubatch_size is None


def test_model_definition_rejects_zero_ubatch_size(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _definition(tmp_path, ubatch_size=0)
