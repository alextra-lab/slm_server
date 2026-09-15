"""Validation rules for backend: mtplx entries (spec §1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_server import config as config_module
from slm_server.config import (
    ModelConfig,
    ModelDefinition,
    mtplx_config_errors,
    mtplx_config_warnings,
    read_mtp_depth_max,
    validate_model_config,
)


def _pack(tmp_path: Path, runtime: dict | None = None) -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "config.json").write_text("{}")
    (pack / "mtplx_runtime.json").write_text(json.dumps(runtime or {"mtp_depth_max": 3}))
    return pack


def _mtplx(pack: Path | str, **overrides: object) -> ModelDefinition:
    fields: dict[str, object] = {
        "id": "mtplx-test",
        "backend": "mtplx",
        "port": 8600,
        "quantization": "4bit",
        "default_timeout": 600,
        "model_path": str(pack),
        "mtp_depth": 3,
    }
    fields.update(overrides)
    return ModelDefinition(**fields)


def _issues(model_def: ModelDefinition) -> list[str]:
    return validate_model_config(ModelConfig(models={"m": model_def}))


def test_valid_mtplx_entry_has_no_issues(tmp_path: Path) -> None:
    assert _issues(_mtplx(_pack(tmp_path))) == []


def test_missing_depth_is_an_error(tmp_path: Path) -> None:
    errors = mtplx_config_errors(_mtplx(_pack(tmp_path), mtp_depth=None))
    assert any("mtp_depth" in e for e in errors)


def test_depth_above_pack_limit_is_an_error(tmp_path: Path) -> None:
    errors = mtplx_config_errors(_mtplx(_pack(tmp_path, {"mtp_depth_max": 3}), mtp_depth=4))
    assert any("mtp_depth_max" in e for e in errors)


def test_depth_limit_defaults_to_ceiling_without_the_key(tmp_path: Path) -> None:
    pack = _pack(tmp_path, {"other": 1})
    assert read_mtp_depth_max(pack) == 6
    assert mtplx_config_errors(_mtplx(pack, mtp_depth=6)) == []


def test_depth_above_ceiling_rejected_by_model(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _mtplx(_pack(tmp_path), mtp_depth=7)


def test_pack_without_runtime_file_is_an_error(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "config.json").write_text("{}")
    errors = mtplx_config_errors(_mtplx(pack))
    assert any("mtplx_runtime.json" in e for e in errors)


def test_huggingface_id_is_an_error() -> None:
    errors = mtplx_config_errors(_mtplx("Youssofal/Some-Pack"))
    assert any("local directory" in e for e in errors)


@pytest.mark.parametrize("key", ["reasoning_effort", "preserve_thinking"])
def test_effort_keys_in_config_kwargs_are_an_error(tmp_path: Path, key: str) -> None:
    model_def = _mtplx(_pack(tmp_path), chat_template_kwargs={key: "medium"})
    assert any(key in e for e in mtplx_config_errors(model_def))


def test_unknown_enum_value_rejected_by_model(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _mtplx(_pack(tmp_path), reasoning_effort="extreme")


def test_llamacpp_only_field_warns(tmp_path: Path) -> None:
    warnings = mtplx_config_warnings(_mtplx(_pack(tmp_path), min_p=0.0))
    assert any("min_p" in w for w in warnings)


def test_concurrency_above_one_with_serial_preset_warns(tmp_path: Path) -> None:
    warnings = mtplx_config_warnings(_mtplx(_pack(tmp_path), max_concurrency=3))
    assert any("max_concurrency" in w for w in warnings)


def test_concurrency_with_agent_preset_does_not_warn(tmp_path: Path) -> None:
    model_def = _mtplx(_pack(tmp_path), max_concurrency=3, mtplx_batching_preset="agent")
    assert not any("max_concurrency" in w for w in mtplx_config_warnings(model_def))


def test_unusual_host_warns(tmp_path: Path) -> None:
    warnings = mtplx_config_warnings(_mtplx(_pack(tmp_path), host="192.168.1.5"))
    assert any("host" in w for w in warnings)


def test_fan_mode_without_daemon_socket_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "THERMALFORGE_SOCKET", tmp_path / "absent.sock")
    warnings = mtplx_config_warnings(_mtplx(_pack(tmp_path), mtplx_fan_mode="smart"))
    assert any("thermalforge" in w for w in warnings)


def test_mtplx_only_field_on_llamacpp_warns(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    model_def = ModelDefinition(
        id="llama-test",
        backend="llamacpp",
        port=8502,
        quantization="Q4",
        default_timeout=600,
        model_path=str(gguf),
        mtp_depth=3,
    )
    assert any("mtp_depth" in issue for issue in _issues(model_def))


def test_validate_prefixes_mtplx_issues_with_role(tmp_path: Path) -> None:
    issues = _issues(_mtplx(_pack(tmp_path), mtp_depth=None))
    assert any(issue.startswith("m: ") and "mtp_depth" in issue for issue in issues)
