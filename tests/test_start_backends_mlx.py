"""Tests for mlx-openai-server launch flags built by build_mlx_command."""

from __future__ import annotations

import pytest

from slm_server.start_backends import build_mlx_command, validate_served_model_name


@pytest.fixture(autouse=True)
def _fake_mlx_binary(monkeypatch, tmp_path):
    """Pretend mlx-openai-server is installed and supports the expected flags.

    Avoids requiring the optional `mlx` extra to be installed in CI.
    """
    fake_bin = tmp_path / "mlx-openai-server"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(
        "slm_server.start_backends.find_command_in_venv", lambda _name: str(fake_bin)
    )
    monkeypatch.setattr(
        "slm_server.start_backends.get_mlx_launch_supported_flags",
        lambda _cmd: {"max-concurrency", "served-model-name"},
    )


def test_build_mlx_command_passes_served_model_name() -> None:
    cmd = build_mlx_command(
        model_path="/tmp/models/org/Some-Model",
        port=8501,
        context_length=4096,
        max_concurrency=1,
        model_type="lm",
        served_model_name="org/some-model",
    )
    assert "--served-model-name" in cmd
    idx = cmd.index("--served-model-name")
    assert cmd[idx + 1] == "org/some-model"


def test_build_mlx_command_omits_flag_without_id() -> None:
    cmd = build_mlx_command(
        model_path="/tmp/models/org/Some-Model",
        port=8501,
        context_length=4096,
        max_concurrency=1,
        model_type="lm",
    )
    assert "--served-model-name" not in cmd


def test_build_mlx_command_skips_flag_when_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(
        "slm_server.start_backends.get_mlx_launch_supported_flags",
        lambda _cmd: {"queue-size"},
    )
    cmd = build_mlx_command(
        model_path="/tmp/models/org/Some-Model",
        port=8501,
        context_length=4096,
        max_concurrency=1,
        model_type="lm",
        served_model_name="org/some-model",
    )
    assert "--served-model-name" not in cmd


def test_validate_served_model_name_rejects_shell_metachars() -> None:
    with pytest.raises(ValueError):
        validate_served_model_name("foo;rm -rf /")
    with pytest.raises(ValueError):
        validate_served_model_name("foo bar")
    with pytest.raises(ValueError):
        validate_served_model_name("")


def test_validate_served_model_name_allows_hf_style_ids() -> None:
    assert validate_served_model_name("mlx-community/Qwen3.5-9B-8bit") == (
        "mlx-community/Qwen3.5-9B-8bit"
    )
    assert validate_served_model_name(None) is None
