"""Tests for the backend watchdog (FRE-241).

The cases that matter are the ones a naive supervisor gets wrong: a model that
is alive but not serving, and a broken configuration that would otherwise be
restarted forever.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from slm_server import watchdog as wd  # type: ignore[import-untyped]


@dataclass
class _StubModelDef:
    """Minimal stand-in for ModelDefinition."""

    id: str = "unsloth/qwen3.6-35-A3B"
    port: int = 8502
    model_path: str = ""


class FakePopen:
    """A process that can be made to die, or to ignore SIGTERM like a wedge.

    A SIGSTOPped process does not act on SIGTERM until it is continued, so
    `terminate_hangs` models the case the watchdog exists to survive.
    """

    def __init__(self, pid: int = 1000, alive: bool = True, terminate_hangs: bool = False) -> None:
        self.pid = pid
        self._alive = alive
        self._terminate_hangs = terminate_hangs
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        if not self._terminate_hangs:
            self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        if self._alive:
            raise subprocess.TimeoutExpired("llama-server", timeout or 0)
        return 0


@pytest.fixture
def settings(tmp_path: Path) -> wd.WatchdogSettings:
    """Settings isolated to tmp_path, with restart bounds small enough to hit."""
    return wd.WatchdogSettings(
        failure_threshold=2,
        stall_seconds=300.0,
        sweep_interval_seconds=0.0,
        max_restarts=3,
        restart_window_seconds=600.0,
        restart_cooldown_seconds=0.0,
        mount_wait_seconds=0.0,
        request_dir=tmp_path / "requests",
        log_path=tmp_path / "watchdog.jsonl",
    )


@pytest.fixture(autouse=True)
def no_lsof(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop stray-reaping from shelling out to the real lsof during tests."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(wd.subprocess, "run", fake_run)


def _events(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Failure counting
# --------------------------------------------------------------------------


def test_single_ordinary_failure_does_not_trip() -> None:
    tracker = wd.BackendHealthTracker(failure_threshold=2)
    assert tracker.record_failure(8502, "server_error") is False


def test_two_consecutive_failures_trip() -> None:
    tracker = wd.BackendHealthTracker(failure_threshold=2)
    tracker.record_failure(8502, "unreachable")
    assert tracker.record_failure(8502, "unreachable") is True


def test_success_between_failures_resets_the_streak() -> None:
    tracker = wd.BackendHealthTracker(failure_threshold=2)
    tracker.record_failure(8502, "timeout")
    tracker.record_success(8502)
    assert tracker.record_failure(8502, "timeout") is False


def test_failures_are_counted_per_backend() -> None:
    tracker = wd.BackendHealthTracker(failure_threshold=2)
    tracker.record_failure(8502, "timeout")
    assert tracker.record_failure(8503, "timeout") is False


def test_a_stall_trips_on_its_own() -> None:
    """300s of silence is outside the whole observed distribution, so one is enough."""
    tracker = wd.BackendHealthTracker(failure_threshold=2)
    assert tracker.record_failure(8502, "stall") is True


# --------------------------------------------------------------------------
# Stall detection
# --------------------------------------------------------------------------


def test_request_silent_past_threshold_is_stalled() -> None:
    tracker = wd.BackendHealthTracker(stall_seconds=300.0)
    tracker.begin_request(8502, now=0.0)
    assert tracker.sweep_stalled(now=299.0) == []
    assert tracker.sweep_stalled(now=300.0) == [8502]


def test_a_stalled_request_is_reported_only_once() -> None:
    """Otherwise every sweep re-trips while the restart is still under way."""
    tracker = wd.BackendHealthTracker(stall_seconds=300.0)
    tracker.begin_request(8502, now=0.0)
    assert tracker.sweep_stalled(now=400.0) == [8502]
    assert tracker.sweep_stalled(now=500.0) == []


def test_request_that_produced_a_first_byte_never_stalls() -> None:
    tracker = wd.BackendHealthTracker(stall_seconds=300.0)
    token = tracker.begin_request(8502, now=0.0)
    tracker.first_byte(token)
    assert tracker.sweep_stalled(now=100_000.0) == []


def test_finished_request_is_no_longer_in_flight() -> None:
    tracker = wd.BackendHealthTracker(stall_seconds=300.0)
    token = tracker.begin_request(8502, now=0.0)
    tracker.end_request(token)
    assert tracker.sweep_stalled(now=999.0) == []


# --------------------------------------------------------------------------
# The router-to-launcher request channel
# --------------------------------------------------------------------------


def test_restart_request_round_trips(settings: wd.WatchdogSettings) -> None:
    request = wd.RestartRequest(8502, "m", "stall", "no first byte", "2026-08-05T00:00:00Z")
    assert wd.write_restart_request(settings.request_dir, request) is True
    assert wd.read_restart_requests(settings.request_dir) == [request]


def test_repeat_trips_for_one_backend_collapse_to_one_request(
    settings: wd.WatchdogSettings,
) -> None:
    """A wedged backend trips repeatedly; that must not queue several restarts."""
    for _ in range(3):
        wd.write_restart_request(
            settings.request_dir, wd.RestartRequest(8502, "m", "stall", "", "t")
        )
    assert len(wd.read_restart_requests(settings.request_dir)) == 1


def test_clearing_a_request_removes_it(settings: wd.WatchdogSettings) -> None:
    wd.write_restart_request(settings.request_dir, wd.RestartRequest(8502, "m", "stall", "", "t"))
    wd.clear_restart_request(settings.request_dir, 8502)
    assert wd.read_restart_requests(settings.request_dir) == []


def test_malformed_request_is_discarded_not_retried_forever(
    settings: wd.WatchdogSettings,
) -> None:
    settings.request_dir.mkdir(parents=True)
    bad = settings.request_dir / "restart-8502.json"
    bad.write_text("{not json")
    assert wd.read_restart_requests(settings.request_dir) == []
    assert not bad.exists()


def test_reading_an_absent_directory_is_not_an_error(settings: wd.WatchdogSettings) -> None:
    assert wd.read_restart_requests(settings.request_dir) == []


# --------------------------------------------------------------------------
# Restarting — AC-1 dead, AC-2 wedged, AC-3 bounded
# --------------------------------------------------------------------------


def _supervisor(
    settings: wd.WatchdogSettings, started: list[FakePopen]
) -> tuple[wd.BackendSupervisor, list[FakePopen]]:
    """Build a supervisor whose start_fn hands out fresh, distinguishable pids."""
    made: list[FakePopen] = []

    def start_fn(_model_def: object) -> FakePopen:
        process = FakePopen(pid=2000 + len(made))
        made.append(process)
        return process

    supervisor = wd.BackendSupervisor(settings, start_fn=start_fn, sleep_fn=lambda _s: None)
    return supervisor, made


def test_exited_backend_is_restarted(settings: wd.WatchdogSettings) -> None:
    """AC-1: the process died and something brings it back."""
    dead = FakePopen(pid=1000, alive=False)
    supervisor, made = _supervisor(settings, [dead])
    supervisor.register(8502, "reasoning", _StubModelDef(), dead)

    assert supervisor.check_exited() == [8502]
    assert len(made) == 1
    assert supervisor.current_processes() == [("unsloth/qwen3.6-35-A3B", made[0])]


def test_live_backend_is_left_alone(settings: wd.WatchdogSettings) -> None:
    alive = FakePopen(pid=1000, alive=True)
    supervisor, made = _supervisor(settings, [alive])
    supervisor.register(8502, "reasoning", _StubModelDef(), alive)

    assert supervisor.check_exited() == []
    assert made == []


def test_wedged_backend_is_replaced_even_though_its_process_lives(
    settings: wd.WatchdogSettings,
) -> None:
    """AC-2: the case this ticket exists for.

    The process is alive and ignoring SIGTERM, so nothing watching for exit
    would ever fire. The watchdog must escalate to SIGKILL and replace it.
    """
    wedged = FakePopen(pid=1000, alive=True, terminate_hangs=True)
    supervisor, made = _supervisor(settings, [wedged])
    supervisor.register(8502, "reasoning", _StubModelDef(), wedged)

    assert supervisor.check_exited() == []  # a liveness check sees nothing wrong

    wd.write_restart_request(
        settings.request_dir, wd.RestartRequest(8502, "m", "stall", "no first byte", "t")
    )
    assert supervisor.check_requests() == [8502]

    assert wedged.terminated is True
    assert wedged.killed is True, "SIGTERM was ignored; the watchdog must escalate"
    assert len(made) == 1
    assert made[0].pid != wedged.pid, "the original process must be replaced, not waited on"
    assert wd.read_restart_requests(settings.request_dir) == []


def test_restarts_are_bounded_then_the_backend_is_abandoned(
    settings: wd.WatchdogSettings,
) -> None:
    """AC-3: a configuration that cannot start must not churn forever."""
    supervisor, made = _supervisor(settings, [])
    supervisor.register(8502, "reasoning", _StubModelDef(), FakePopen(alive=False))

    for _ in range(settings.max_restarts):
        assert supervisor.restart(8502, "process_exited") is True

    assert supervisor.restart(8502, "process_exited") is False
    assert supervisor.live_count == 0
    assert len(made) == settings.max_restarts

    abandoned = [e for e in _events(settings.log_path) if e["event"] == "backend_abandoned"]
    assert len(abandoned) == 1
    assert abandoned[0]["port"] == 8502
    assert "restarts within" in abandoned[0]["reason"], "the reason must be readable"


def test_an_abandoned_backend_is_not_restarted_again(settings: wd.WatchdogSettings) -> None:
    supervisor, made = _supervisor(settings, [])
    supervisor.register(8502, "reasoning", _StubModelDef(), FakePopen(alive=False))
    for _ in range(settings.max_restarts + 1):
        supervisor.restart(8502, "process_exited")

    before = len(made)
    assert supervisor.restart(8502, "process_exited") is False
    assert supervisor.check_exited() == []
    assert len(made) == before


def test_supervision_loop_stops_once_everything_is_abandoned(
    settings: wd.WatchdogSettings,
) -> None:
    """The loop must terminate rather than spin on a stack that cannot run."""
    supervisor, _made = _supervisor(settings, [])
    supervisor.register(8502, "reasoning", _StubModelDef(), FakePopen(alive=False))
    for _ in range(settings.max_restarts + 1):
        supervisor.restart(8502, "process_exited")

    supervisor.run()  # returns only because live_count reached 0

    events = {e["event"] for e in _events(settings.log_path)}
    assert {"supervisor_started", "supervisor_stopped"} <= events


def test_restart_waits_for_an_unmounted_model_path(
    settings: wd.WatchdogSettings, tmp_path: Path
) -> None:
    """Models live on an external volume that may not be mounted yet at boot."""
    missing = _StubModelDef(model_path=str(tmp_path / "not-mounted" / "model.gguf"))
    supervisor, made = _supervisor(settings, [])
    supervisor.register(8502, "reasoning", missing, FakePopen(alive=False))

    assert supervisor.restart(8502, "process_exited") is True
    events = {e["event"] for e in _events(settings.log_path)}
    assert "waiting_for_model_path" in events
    assert len(made) == 1, "it still launches after the wait rather than giving up silently"


def test_restart_of_an_unknown_port_is_ignored(settings: wd.WatchdogSettings) -> None:
    supervisor, made = _supervisor(settings, [])
    wd.write_restart_request(settings.request_dir, wd.RestartRequest(9999, "m", "stall", "", "t"))
    assert supervisor.check_requests() == [9999]
    assert made == []


# --------------------------------------------------------------------------
# Stray reaping — which processes may be killed
# --------------------------------------------------------------------------


def test_stray_reaping_only_considers_listeners(
    settings: wd.WatchdogSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: plain `lsof -ti tcp:PORT` also matches *clients* of that port.

    The router holds pooled keep-alive connections to the backend it just failed
    against, so on the running stack `lsof -ti tcp:8502` returned both the
    backend (35559) and the router (35572). Reaping that set would SIGKILL the
    router on the very first restart. Only listeners may be considered.
    """
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        seen.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr(wd.subprocess, "run", fake_run)

    supervisor, _made = _supervisor(settings, [])
    supervisor._reap_port_strays(8502)

    assert seen, "lsof was never invoked"
    assert "-sTCP:LISTEN" in seen[0], f"selector would match clients too: {seen[0]}"


def test_stray_reaping_never_kills_the_supervisor_itself(
    settings: wd.WatchdogSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if lsof somehow names us, we must not kill ourselves or our parent."""
    killed: list[int] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=f"{os.getpid()}\n{os.getppid()}\n", stderr=""
        )

    monkeypatch.setattr(wd.subprocess, "run", fake_run)
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append(pid))

    supervisor, _made = _supervisor(settings, [])
    supervisor._reap_port_strays(8502)

    assert killed == []
    refused = [e for e in _events(settings.log_path) if e["event"] == "stray_kill_refused"]
    assert len(refused) == 2


def test_a_genuine_stray_listener_is_killed(
    settings: wd.WatchdogSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not defeat the purpose: real strays still get reaped."""
    killed: list[int] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="98765\n", stderr="")

    monkeypatch.setattr(wd.subprocess, "run", fake_run)
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append(pid))

    supervisor, _made = _supervisor(settings, [])
    supervisor._reap_port_strays(8502)

    assert killed == [98765]


# --------------------------------------------------------------------------
# The restart log
# --------------------------------------------------------------------------


def test_every_restart_records_the_pid_it_replaced(settings: wd.WatchdogSettings) -> None:
    """The log is the first durable measurement of how often a model wedges."""
    old = FakePopen(pid=1000, alive=False)
    supervisor, made = _supervisor(settings, [old])
    supervisor.register(8502, "reasoning", _StubModelDef(), old)
    supervisor.restart(8502, "stall", "no first byte within 300s")

    succeeded = [e for e in _events(settings.log_path) if e["event"] == "restart_succeeded"]
    assert len(succeeded) == 1
    assert succeeded[0]["old_pid"] == 1000
    assert succeeded[0]["new_pid"] == made[0].pid
    assert succeeded[0]["reason"] == "stall"


def test_router_watchdog_publishes_a_request_when_it_trips(
    settings: wd.WatchdogSettings,
) -> None:
    watchdog = wd.RouterWatchdog(settings, model_ids={8502: "unsloth/qwen3.6-35-A3B"})

    watchdog.record_failure(8502, "timeout", "backend did not answer")
    assert wd.read_restart_requests(settings.request_dir) == []

    watchdog.record_failure(8502, "timeout", "backend did not answer")
    pending = wd.read_restart_requests(settings.request_dir)
    assert len(pending) == 1
    assert pending[0].port == 8502
    assert pending[0].model_id == "unsloth/qwen3.6-35-A3B"


def test_a_disabled_watchdog_records_and_restarts_nothing(
    settings: wd.WatchdogSettings,
) -> None:
    disabled = wd.WatchdogSettings(
        enabled=False, request_dir=settings.request_dir, log_path=settings.log_path
    )
    watchdog = wd.RouterWatchdog(disabled)
    for _ in range(5):
        watchdog.record_failure(8502, "unreachable")
    assert wd.read_restart_requests(settings.request_dir) == []
    assert not settings.log_path.exists()
