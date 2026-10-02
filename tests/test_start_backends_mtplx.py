"""build_mtplx_command, find_mtplx_binary and the mtplx launcher branch (spec §1, §3)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from slm_server import start_backends as sb
from slm_server.config import ModelConfig, ModelDefinition


def _pack(tmp_path: Path, depth_max: int = 3) -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "config.json").write_text("{}")
    (pack / "mtplx_runtime.json").write_text(json.dumps({"mtp_depth_max": depth_max}))
    return pack


def _mtplx(pack: Path | str, **overrides: object) -> ModelDefinition:
    fields: dict[str, object] = {
        "id": "mtplx-qwen38-27b",
        "backend": "mtplx",
        "port": 8600,
        "quantization": "4bit",
        "default_timeout": 600,
        "model_path": str(pack),
        "mtp_depth": 3,
    }
    fields.update(overrides)
    return ModelDefinition(**fields)


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_minimal_command(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    cmd = sb.build_mtplx_command(_mtplx(pack), "/opt/mtplx")
    assert cmd[:2] == ["/opt/mtplx", "serve"]
    assert _flag(cmd, "--model") == str(pack)
    assert _flag(cmd, "--model-id") == "mtplx-qwen38-27b"
    assert _flag(cmd, "--host") == "127.0.0.1"
    assert _flag(cmd, "--port") == "8600"
    assert "--no-auth" in cmd
    assert _flag(cmd, "--depth") == "3"
    assert _flag(cmd, "--preserve-thinking") == "off"


def test_optional_flags(tmp_path: Path) -> None:
    model_def = _mtplx(
        _pack(tmp_path),
        context_length=131072,
        reasoning_effort="medium",
        preserve_thinking="scoped",
        reasoning_parser="qwen3",
        mtplx_profile="turbo",
        mtplx_batching_preset="agent",
        mtplx_fan_mode="default",
        temp=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=0.0,
    )
    cmd = sb.build_mtplx_command(model_def, "/opt/mtplx")
    assert _flag(cmd, "--context-window") == "131072"
    assert _flag(cmd, "--reasoning-effort") == "medium"
    assert _flag(cmd, "--preserve-thinking") == "scoped"
    assert _flag(cmd, "--reasoning-parser") == "qwen3"
    assert _flag(cmd, "--profile") == "turbo"
    assert _flag(cmd, "--batching-preset") == "agent"
    assert _flag(cmd, "--fan-mode") == "default"
    assert _flag(cmd, "--default-temperature") == "0.6"
    assert _flag(cmd, "--default-top-p") == "0.95"
    assert _flag(cmd, "--default-top-k") == "20"
    assert _flag(cmd, "--default-presence-penalty") == "0.0"


def test_unset_optional_flags_are_absent(tmp_path: Path) -> None:
    cmd = sb.build_mtplx_command(_mtplx(_pack(tmp_path)), "/opt/mtplx")
    for flag in (
        "--context-window",
        "--reasoning-effort",
        "--reasoning-parser",
        "--profile",
        "--batching-preset",
        "--fan-mode",
        "--default-temperature",
    ):
        assert flag not in cmd


def test_host_field_never_changes_bind_address(tmp_path: Path) -> None:
    cmd = sb.build_mtplx_command(_mtplx(_pack(tmp_path), host="0.0.0.0"), "/opt/mtplx")
    assert _flag(cmd, "--host") == "127.0.0.1"


def test_missing_depth_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mtp_depth"):
        sb.build_mtplx_command(_mtplx(_pack(tmp_path), mtp_depth=None), "/opt/mtplx")


def test_depth_above_pack_limit_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mtp_depth_max"):
        sb.build_mtplx_command(_mtplx(_pack(tmp_path, depth_max=3), mtp_depth=5), "/opt/mtplx")


def test_huggingface_id_raises() -> None:
    with pytest.raises(ValueError, match="local directory"):
        sb.build_mtplx_command(_mtplx("Youssofal/Some-Pack"), "/opt/mtplx")


def test_unknown_reasoning_parser_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reasoning_parser"):
        sb.build_mtplx_command(_mtplx(_pack(tmp_path), reasoning_parser="glm4_moe"), "/opt/mtplx")


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_binary_lookup_prefers_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = _executable(tmp_path / "mtplx-override")
    monkeypatch.setenv("SLM_MTPLX_BIN", str(override))
    monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/local/bin/mtplx")
    assert sb.find_mtplx_binary() == str(override)


def test_binary_lookup_uses_path_next(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_MTPLX_BIN", raising=False)
    monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/local/bin/mtplx")
    assert sb.find_mtplx_binary() == "/usr/local/bin/mtplx"


def test_binary_lookup_falls_back_to_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_MTPLX_BIN", raising=False)
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    home = tmp_path / "home"
    (home / ".mtplx" / "bin").mkdir(parents=True)
    fallback = _executable(home / ".mtplx" / "bin" / "mtplx")
    monkeypatch.setattr(sb.Path, "home", classmethod(lambda cls: home))
    assert sb.find_mtplx_binary() == str(fallback)


def test_binary_lookup_returns_none_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLM_MTPLX_BIN", raising=False)
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    monkeypatch.setattr(sb.Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))
    assert sb.find_mtplx_binary() is None


class _FakeProcess:
    pid = 4242
    returncode = None

    def poll(self) -> None:
        return None


def test_start_model_server_launches_mtplx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack = _pack(tmp_path)
    model_def = _mtplx(pack)
    launched: dict[str, object] = {}

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakeProcess:
        launched["cmd"] = cmd
        launched["stderr_name"] = getattr(kwargs.get("stderr"), "name", None)
        return _FakeProcess()

    monkeypatch.setattr(sb, "find_mtplx_binary", lambda: "/opt/mtplx")
    monkeypatch.setattr(sb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sb.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sb, "LOG_DIR", tmp_path / "logs")

    process = sb.start_model_server(model_def, ModelConfig(models={"m": model_def}))

    assert process is not None
    cmd = launched["cmd"]
    assert isinstance(cmd, list) and cmd[:2] == ["/opt/mtplx", "serve"]
    assert str(launched["stderr_name"]).endswith(os.sep + "mtplx-mtplx-qwen38-27b-8600.log")


def test_start_model_server_without_binary_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_def = _mtplx(_pack(tmp_path))
    monkeypatch.setattr(sb, "find_mtplx_binary", lambda: None)
    monkeypatch.setattr(sb, "LOG_DIR", tmp_path / "logs")
    assert sb.start_model_server(model_def, ModelConfig(models={"m": model_def})) is None
