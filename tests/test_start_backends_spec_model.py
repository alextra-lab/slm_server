"""Tests for the sidecar draft model (-md) and the llama-server binary override.

Qwen3.6 carries its MTP head inside the main GGUF, so --spec-type alone was enough.
Qwen3.8-Flash-Next ships the head as a separate file, which llama-server only picks
up when it is named with -md. The binary override exists because Homebrew's build
cannot load that model at all.
"""

from pathlib import Path

import pytest

from slm_server.config import ModelDefinition
from slm_server.start_backends import build_llama_native_command, find_native_llama_server


def _native(gguf: Path, **kwargs) -> list[str]:
    return build_llama_native_command(
        gguf, 8502, 131072, "UD-IQ4_XS", 1, None, "test/model", "/usr/bin/true", **kwargs
    )


def test_spec_model_path_emits_md_flag(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    draft = tmp_path / "mtp-shared.gguf"
    draft.write_bytes(b"")
    cmd = _native(gguf, spec_model_path=draft, spec_type="draft-mtp", spec_draft_n_max=1)
    assert "-md" in cmd
    assert cmd[cmd.index("-md") + 1] == str(draft)


def test_md_precedes_spec_type(tmp_path: Path) -> None:
    """llama-server needs the draft model named before the spec flags."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    draft = tmp_path / "mtp-shared.gguf"
    draft.write_bytes(b"")
    cmd = _native(gguf, spec_model_path=draft, spec_type="draft-mtp", spec_draft_n_max=1)
    assert cmd.index("-md") < cmd.index("--spec-type")


def test_no_md_flag_when_unset(tmp_path: Path) -> None:
    """Qwen3.6 keeps its head in the main GGUF and must not gain a -md flag."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    cmd = _native(gguf, spec_type="draft-mtp", spec_draft_n_max=1)
    assert "-md" not in cmd
    assert "--spec-type" in cmd


def test_missing_spec_model_path_raises(tmp_path: Path) -> None:
    """A typo in the sidecar path must fail loudly, not start a server without MTP."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="spec_model_path"):
        _native(gguf, spec_model_path=tmp_path / "absent.gguf", spec_type="draft-mtp")


def test_model_definition_accepts_spec_model_path(tmp_path: Path) -> None:
    draft = tmp_path / "mtp-shared.gguf"
    draft.write_bytes(b"")
    md = ModelDefinition(
        id="test/flash",
        backend="llamacpp",
        port=8502,
        quantization="UD-IQ4_XS",
        default_timeout=600,
        model_path=str(tmp_path / "model.gguf"),
        spec_model_path=str(draft),
    )
    assert md.spec_model_path == str(draft)


def test_model_definition_spec_model_path_defaults_none(tmp_path: Path) -> None:
    md = ModelDefinition(
        id="test/model",
        backend="llamacpp",
        port=8502,
        quantization="UD-Q4_K_XL",
        default_timeout=600,
        model_path=str(tmp_path / "model.gguf"),
    )
    assert md.spec_model_path is None


def test_binary_override_wins_over_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_LLAMA_SERVER_BIN", "/usr/bin/true")
    assert find_native_llama_server() == "/usr/bin/true"


def test_binary_override_falls_back_when_not_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale override must not strand the launcher; PATH still applies."""
    dud = tmp_path / "not-a-binary"
    dud.write_text("")
    monkeypatch.setenv("SLM_LLAMA_SERVER_BIN", str(dud))
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/true")
    assert find_native_llama_server() == "/usr/bin/true"


def test_binary_override_absent_uses_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/llama-server")
    assert find_native_llama_server() == "/opt/homebrew/bin/llama-server"
