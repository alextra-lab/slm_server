"""Tests for backend launcher process lifecycle behavior."""

import signal

from slm_server import start_backends
from slm_server.config import ModelConfig, ModelDefinition


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 12345
        self.terminated = False
        self.killed = False
        self.returncode = None

    def wait(self, timeout: float | None = None) -> None:
        if timeout is None:
            raise RuntimeError("SIGTERM handler was not installed")
        self.returncode = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _model() -> ModelDefinition:
    return ModelDefinition(
        id="test/model",
        backend="llamacpp",
        port=8600,
        quantization="Q4_K_M",
        default_timeout=60,
        model_path="/tmp/model.gguf",
    )


def test_main_cleans_up_started_servers_on_sigterm(monkeypatch) -> None:
    handlers = {}
    process = FakeProcess()
    cfg = ModelConfig(models={"reasoning": _model()})

    monkeypatch.setattr(start_backends, "load_model_config", lambda: cfg)
    monkeypatch.setattr(start_backends, "start_model_server", lambda _model_def, _cfg: process)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler))

    def wait_until_signal(timeout: float | None = None) -> None:
        if timeout is None:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return
        process.returncode = 0

    process.wait = wait_until_signal

    start_backends.main()

    assert process.terminated is True
    assert process.killed is False
