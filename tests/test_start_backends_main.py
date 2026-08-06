"""Tests for backend launcher process lifecycle behavior."""

import signal
from pathlib import Path

import pytest

from slm_server import start_backends
from slm_server import watchdog as wd
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


def test_main_exits_non_zero_when_every_backend_is_abandoned(monkeypatch, tmp_path: Path) -> None:
    """Nothing serving must not be reported as success.

    This exit code previously was 0, chosen so a LaunchAgent's KeepAlive would
    not relaunch into churn. That apparatus is gone. The consumer is now a
    human's shell, and exit 0 there says "started" while no model is running —
    the same report-success-with-nothing-behind-it defect the watchdog exists
    to remove, aimed at a person about to assume the server is up.
    """
    cfg = ModelConfig(models={"reasoning": _model()})
    monkeypatch.setattr(start_backends, "load_model_config", lambda: cfg)
    monkeypatch.setattr(start_backends, "start_model_server", lambda _m, _c: None)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(
        start_backends,
        "load_watchdog_settings",
        lambda: wd.WatchdogSettings(
            sweep_interval_seconds=0.0,
            restart_cooldown_seconds=0.0,
            max_restarts=2,
            request_dir=tmp_path / "requests",
            log_path=tmp_path / "watchdog.jsonl",
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        start_backends.main()

    assert excinfo.value.code == 1, "a launcher exiting with nothing serving is a failure"


def test_main_cleans_up_started_servers_on_sigterm(monkeypatch, tmp_path: Path) -> None:
    handlers = {}
    process = FakeProcess()
    cfg = ModelConfig(models={"reasoning": _model()})

    monkeypatch.setattr(start_backends, "load_model_config", lambda: cfg)
    monkeypatch.setattr(start_backends, "start_model_server", lambda _model_def, _cfg: process)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler))
    # This test is about shutdown, not timing: keep the supervisor from waiting
    # on the (absent) model path or sleeping between sweeps, and keep its log
    # out of the repo.
    monkeypatch.setattr(
        start_backends,
        "load_watchdog_settings",
        lambda: wd.WatchdogSettings(
            sweep_interval_seconds=0.0,
            restart_cooldown_seconds=0.0,
            request_dir=tmp_path / "requests",
            log_path=tmp_path / "watchdog.jsonl",
        ),
    )

    # The supervisor polls for liveness instead of blocking in wait(), so the
    # signal arrives during a poll. What the test checks is unchanged: SIGTERM
    # must terminate the children it started, without escalating to SIGKILL.
    def poll_then_signal() -> int | None:
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return None

    process.poll = poll_then_signal

    start_backends.main()

    assert process.terminated is True
    assert process.killed is False
