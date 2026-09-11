"""Backend stderr must go to a file, never to a pipe nobody reads.

llama.cpp logs every request at INFO level. An unread pipe fills at ~64 KB on
macOS, its logger then queues 512 more lines, and every thread that logs blocks:
the server freezes mid-request and ignores SIGTERM. Reproduced on 2026-09-11 with
a 0.6B model, which froze at request 153 and recovered the moment the pipe was
drained. It is the cause of the watchdog's "no first byte within 300s" stalls.
"""

import subprocess
from pathlib import Path

import pytest

from slm_server import start_backends
from slm_server.config import ModelConfig, ModelDefinition


class _Running:
    pid = 4242
    returncode = None

    def poll(self) -> None:
        return None


@pytest.fixture
def launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    md = ModelDefinition(
        id="test/flash",
        backend="llamacpp",
        port=8502,
        quantization="UD-IQ4_XS",
        default_timeout=600,
        model_path=str(gguf),
    )
    cfg = ModelConfig(models={"reasoning": md})
    monkeypatch.setenv("SLM_LLAMA_SERVER_BIN", "/usr/bin/true")
    monkeypatch.setattr(start_backends, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(start_backends.time, "sleep", lambda _s: None)
    seen: dict = {}

    def fake_popen(cmd, **kwargs):
        seen.update(kwargs)
        return _Running()

    monkeypatch.setattr(start_backends.subprocess, "Popen", fake_popen)

    def run():
        assert start_backends.start_model_server(md, cfg) is not None
        return seen

    return run, tmp_path / "logs" / "llama-test_flash-8502.log"


def test_stderr_goes_to_a_log_file_not_a_pipe(launch) -> None:
    run, log_path = launch
    kwargs = run()
    assert kwargs["stderr"] is not subprocess.PIPE
    assert Path(kwargs["stderr"].name) == log_path


def test_previous_log_is_kept_as_prev(launch) -> None:
    """A restart after a stall must not erase what the stalled process logged."""
    run, log_path = launch
    log_path.parent.mkdir(parents=True)
    log_path.write_text("previous lifetime\n")
    run()
    assert (log_path.parent / (log_path.name + ".prev")).read_text() == "previous lifetime\n"
    assert log_path.exists()
