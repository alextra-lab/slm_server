"""Detect backends that have stopped serving and restart them (FRE-241).

The problem this solves is not a dead process — that is the easy half. It is a
*stuck* model: the `llama-server` process is alive, still holding its listening
socket, and no longer able to produce a token. A supervisor that watches for
process exit never fires on it, and neither health path in the router can see
it: `/health` returns a hardcoded string for the router itself, and
`/v1/backends/health` only proves a socket answered, not that a model can decode.

So responsiveness is judged from the router's own errors on real traffic:

* **unreachable** — `httpx.ConnectError`; the backend is gone. Fails fast.
* **timeout** — `httpx.TimeoutException`; the request outlived its budget.
* **server_error** — the backend answered 5xx. Client 4xx is the caller's fault
  and is deliberately not counted.
* **stall** — a request has been in flight past `stall_seconds` without a single
  byte coming back. This is the signal that catches a wedge quickly; see below.

Two thresholds, because the two signals carry different weight:

`failure_threshold` consecutive failures trip a restart, but a stall trips one on
its own. That asymmetry is measured rather than assumed. Across the 798
`request_complete` documents shipped to Elasticsearch, router-observed time to
first byte peaked at 30.4s (p50 1.4s, p95 29.6s) and backend-reported prefill
peaked at 243.2s. A backend silent for 300s is therefore outside the whole
observed distribution, which makes one such observation decisive. Ordinary
errors are noisier and wait for a second consecutive occurrence.

Detection lives in the router because that is where real traffic is, but the
router does not own the backend processes — the `slm_server backends` launcher
does. The two are separate processes, so a trip is published as a small JSON
file (written atomically) that `BackendSupervisor` picks up on its next sweep.
A file rather than a control socket: nothing new listens, and the request is
still on disk afterwards for anyone reconstructing what happened.

Every decision is appended to a JSONL log. Because router error paths never
emitted telemetry, failures have never been recorded anywhere durable — that log
is the first measurement of how often a model actually wedges.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Literal

import structlog

log = structlog.get_logger(__name__)

FailureKind = Literal["unreachable", "timeout", "server_error", "stall"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back on unset or unparseable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("watchdog_bad_env_value", name=name, value=raw, using=default)
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back on unset or unparseable."""
    return int(_env_float(name, float(default)))


@dataclass(frozen=True)
class WatchdogSettings:
    """Tunables for detection and restart.

    Attributes:
        enabled: Master switch. When False the watchdog records nothing and
            restarts nothing.
        failure_threshold: Consecutive ordinary failures before a restart trips.
        stall_seconds: Seconds an in-flight request may go without a first byte
            before it counts as a stall.
        sweep_interval_seconds: How often the router checks for stalls and the
            supervisor checks for requests and dead children.
        max_restarts: Restart attempts allowed inside `restart_window_seconds`
            before a backend is abandoned.
        restart_window_seconds: Rolling window over which `max_restarts` applies.
        restart_cooldown_seconds: Pause between consecutive restart attempts.
        mount_wait_seconds: How long to wait for a model path to appear before
            launching anyway. Covers the external volume not being mounted yet
            after a reboot.
        request_dir: Directory holding pending restart requests.
        log_path: JSONL file recording every watchdog decision.
    """

    enabled: bool = True
    failure_threshold: int = 2
    stall_seconds: float = 300.0
    sweep_interval_seconds: float = 10.0
    max_restarts: int = 5
    restart_window_seconds: float = 600.0
    restart_cooldown_seconds: float = 10.0
    mount_wait_seconds: float = 120.0
    request_dir: Path = _REPO_ROOT / "logs" / "watchdog" / "requests"
    log_path: Path = _REPO_ROOT / "logs" / "watchdog.jsonl"


def load_settings() -> WatchdogSettings:
    """Build settings from `SLM_WATCHDOG_*` environment variables.

    `SLM_WATCHDOG_REQUEST_DIR` and `SLM_WATCHDOG_LOG_PATH` exist so a test run
    can be pointed somewhere harmless. Without them the suite would publish
    restart requests into the directory the live launcher is watching, and
    running tests would restart the owner's running models.

    Returns:
        Settings with environment overrides applied over the defaults.
    """
    defaults = WatchdogSettings()
    request_dir = os.getenv("SLM_WATCHDOG_REQUEST_DIR")
    log_path = os.getenv("SLM_WATCHDOG_LOG_PATH")
    return WatchdogSettings(
        enabled=os.getenv("SLM_WATCHDOG_ENABLED", "true").lower() != "false",
        failure_threshold=_env_int("SLM_WATCHDOG_FAILURE_THRESHOLD", 2),
        stall_seconds=_env_float("SLM_WATCHDOG_STALL_SECONDS", 300.0),
        sweep_interval_seconds=_env_float("SLM_WATCHDOG_SWEEP_SECONDS", 10.0),
        max_restarts=_env_int("SLM_WATCHDOG_MAX_RESTARTS", 5),
        restart_window_seconds=_env_float("SLM_WATCHDOG_RESTART_WINDOW_SECONDS", 600.0),
        restart_cooldown_seconds=_env_float("SLM_WATCHDOG_RESTART_COOLDOWN_SECONDS", 10.0),
        mount_wait_seconds=_env_float("SLM_WATCHDOG_MOUNT_WAIT_SECONDS", 120.0),
        request_dir=Path(request_dir) if request_dir else defaults.request_dir,
        log_path=Path(log_path) if log_path else defaults.log_path,
    )


# --------------------------------------------------------------------------
# Event log
# --------------------------------------------------------------------------


def append_event(log_path: Path, event: str, **fields: object) -> None:
    """Append one JSON line to the watchdog log.

    Fail-soft: a watchdog that crashes the thing it supervises is worse than one
    that loses a log line.

    Args:
        log_path: Destination JSONL file. Parent directories are created.
        event: Event name, e.g. "restart_requested".
        **fields: Additional JSON-serialisable fields.
    """
    doc: dict[str, object] = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as handle:
            handle.write(json.dumps(doc, default=str) + "\n")
    except OSError as exc:
        log.warning("watchdog_log_write_failed", error=str(exc), path=str(log_path))


# --------------------------------------------------------------------------
# Restart requests — the router-to-launcher handoff
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RestartRequest:
    """A request to restart one backend.

    Attributes:
        port: Backend port to restart. Identifies the backend across processes.
        model_id: Model id, for the log.
        reason: The failure kind that tripped the restart.
        detail: Human-readable explanation.
        requested_at: ISO-8601 timestamp.
    """

    port: int
    model_id: str
    reason: str
    detail: str
    requested_at: str

    def to_json(self) -> str:
        """Serialise to a JSON object string."""
        return json.dumps(
            {
                "port": self.port,
                "model_id": self.model_id,
                "reason": self.reason,
                "detail": self.detail,
                "requested_at": self.requested_at,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> RestartRequest:
        """Parse from a JSON object string.

        Args:
            raw: JSON text previously produced by `to_json`.

        Returns:
            The parsed request.

        Raises:
            ValueError: If the payload is not a JSON object or lacks `port`.
        """
        data = json.loads(raw)
        if not isinstance(data, dict) or "port" not in data:
            raise ValueError("restart request payload missing 'port'")
        return cls(
            port=int(data["port"]),
            model_id=str(data.get("model_id", "")),
            reason=str(data.get("reason", "unknown")),
            detail=str(data.get("detail", "")),
            requested_at=str(data.get("requested_at", "")),
        )


def _request_path(request_dir: Path, port: int) -> Path:
    """Path of the pending-request file for a port."""
    return request_dir / f"restart-{port}.json"


def write_restart_request(request_dir: Path, request: RestartRequest) -> bool:
    """Publish a restart request atomically.

    Written to a temporary file and moved into place with `os.replace`, so the
    supervisor never reads a half-written payload. One file per port: repeated
    trips for the same backend collapse instead of queueing restarts.

    Args:
        request_dir: Directory for pending requests.
        request: The request to publish.

    Returns:
        True if the request was written.
    """
    target = _request_path(request_dir, request.port)
    tmp = target.with_suffix(".tmp")
    try:
        request_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(request.to_json())
        os.replace(tmp, target)
        return True
    except OSError as exc:
        log.warning("watchdog_request_write_failed", error=str(exc), port=request.port)
        return False


def read_restart_requests(request_dir: Path) -> list[RestartRequest]:
    """Read every pending restart request.

    Unreadable or malformed files are logged and removed rather than left to be
    retried forever.

    Args:
        request_dir: Directory for pending requests.

    Returns:
        The pending requests, in arbitrary order.
    """
    if not request_dir.is_dir():
        return []
    requests: list[RestartRequest] = []
    for path in sorted(request_dir.glob("restart-*.json")):
        try:
            requests.append(RestartRequest.from_json(path.read_text()))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            log.warning("watchdog_request_unreadable", path=str(path), error=str(exc))
            path.unlink(missing_ok=True)
    return requests


def clear_restart_request(request_dir: Path, port: int) -> None:
    """Delete the pending request for a port, if present."""
    _request_path(request_dir, port).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Detection — runs inside the router
# --------------------------------------------------------------------------


@dataclass
class _InFlight:
    """A request currently outstanding against a backend."""

    port: int
    started: float
    first_byte: bool = False
    stalled: bool = False


@dataclass
class BackendHealthTracker:
    """Counts real-traffic outcomes per backend and decides when to restart.

    Pure bookkeeping: it never restarts anything itself, it returns the ports
    that have tripped and lets the caller act. That keeps the decision logic
    testable without processes, sockets or clocks.

    Attributes:
        failure_threshold: Consecutive ordinary failures that trip a restart.
        stall_seconds: Silence before an in-flight request counts as stalled.
    """

    failure_threshold: int = 2
    stall_seconds: float = 300.0
    _consecutive: dict[int, int] = field(default_factory=dict)
    _in_flight: dict[int, _InFlight] = field(default_factory=dict)
    _tokens: Iterator[int] = field(default_factory=lambda: count(1))

    def record_success(self, port: int) -> None:
        """Reset the failure streak for a backend that just served a request."""
        self._consecutive.pop(port, None)

    def record_failure(self, port: int, kind: FailureKind) -> bool:
        """Record a failed request and report whether a restart should trip.

        A stall trips on its own; every other kind waits for
        `failure_threshold` consecutive occurrences.

        Args:
            port: Backend port that failed.
            kind: What went wrong.

        Returns:
            True if this failure trips a restart.
        """
        if kind == "stall":
            self._consecutive[port] = 0
            return True
        streak = self._consecutive.get(port, 0) + 1
        self._consecutive[port] = streak
        return streak >= self.failure_threshold

    def consecutive_failures(self, port: int) -> int:
        """Current consecutive-failure streak for a backend."""
        return self._consecutive.get(port, 0)

    def begin_request(self, port: int, *, now: float | None = None) -> int:
        """Register a request as in flight.

        Args:
            port: Backend the request was sent to.
            now: Monotonic timestamp; defaults to the current time.

        Returns:
            A token identifying this request, for `first_byte`/`end_request`.
        """
        token = next(self._tokens)
        self._in_flight[token] = _InFlight(
            port=port, started=time.monotonic() if now is None else now
        )
        return token

    def first_byte(self, token: int) -> None:
        """Note that a request has produced its first byte, clearing stall risk."""
        entry = self._in_flight.get(token)
        if entry is not None:
            entry.first_byte = True

    def end_request(self, token: int) -> None:
        """Deregister a request that has finished, succeeded or not."""
        self._in_flight.pop(token, None)

    def sweep_stalled(self, *, now: float | None = None) -> list[int]:
        """Find in-flight requests that have gone silent for too long.

        Each stalled request is reported once, so a single wedged request does
        not re-trip on every sweep while the restart is under way.

        Args:
            now: Monotonic timestamp; defaults to the current time.

        Returns:
            Ports newly considered stalled.
        """
        moment = time.monotonic() if now is None else now
        stalled: list[int] = []
        for entry in self._in_flight.values():
            if entry.first_byte or entry.stalled:
                continue
            if moment - entry.started >= self.stall_seconds:
                entry.stalled = True
                stalled.append(entry.port)
        return stalled


class RouterWatchdog:
    """Wires the tracker to the restart-request channel.

    Args:
        settings: Watchdog configuration.
        model_ids: Port-to-model-id map, used only to label log entries.
    """

    def __init__(self, settings: WatchdogSettings, model_ids: dict[int, str] | None = None) -> None:
        self.settings = settings
        self.tracker = BackendHealthTracker(
            failure_threshold=settings.failure_threshold,
            stall_seconds=settings.stall_seconds,
        )
        self._model_ids = model_ids or {}

    def record_success(self, port: int) -> None:
        """Record a served request."""
        if self.settings.enabled:
            self.tracker.record_success(port)

    def record_failure(self, port: int, kind: FailureKind, detail: str = "") -> None:
        """Record a failed request and request a restart if it trips.

        Args:
            port: Backend port that failed.
            kind: What went wrong.
            detail: Extra context for the log and the request file.
        """
        if not self.settings.enabled:
            return
        tripped = self.tracker.record_failure(port, kind)
        append_event(
            self.settings.log_path,
            "backend_failure",
            port=port,
            model_id=self._model_ids.get(port),
            kind=kind,
            detail=detail,
            consecutive=self.tracker.consecutive_failures(port),
            tripped=tripped,
        )
        if tripped:
            self._request_restart(port, kind, detail)

    def _request_restart(self, port: int, kind: FailureKind, detail: str) -> None:
        """Publish a restart request for a backend."""
        request = RestartRequest(
            port=port,
            model_id=self._model_ids.get(port, ""),
            reason=kind,
            detail=detail,
            requested_at=datetime.now(UTC).isoformat(),
        )
        if write_restart_request(self.settings.request_dir, request):
            append_event(
                self.settings.log_path,
                "restart_requested",
                port=port,
                model_id=request.model_id,
                reason=kind,
                detail=detail,
            )
            log.warning("watchdog_restart_requested", port=port, reason=kind, detail=detail)

    def sweep(self) -> None:
        """Check for stalled in-flight requests and trip restarts for them."""
        if not self.settings.enabled:
            return
        for port in self.tracker.sweep_stalled():
            self.record_failure(
                port,
                "stall",
                f"no first byte within {self.settings.stall_seconds:.0f}s",
            )


# --------------------------------------------------------------------------
# Restart — runs inside the backends launcher
# --------------------------------------------------------------------------


@dataclass
class _Supervised:
    """Launcher-side state for one backend."""

    role: str
    model_def: object
    process: subprocess.Popen | None
    attempts: list[float] = field(default_factory=list)
    abandoned: bool = False


class BackendSupervisor:
    """Restarts backends that have died or been reported unusable.

    Replaces the launcher's original `for _, p in processes: p.wait()`, which
    restarted nothing under any circumstance.

    Restarts are bounded: `max_restarts` inside `restart_window_seconds`, after
    which the backend is abandoned with the reason written to the log. Without
    that bound, a model path that no longer exists would relaunch forever.

    Args:
        settings: Watchdog configuration.
        start_fn: Callable that launches a backend and returns its `Popen`, or
            None on failure. Injected so this class never imports the launcher
            and stays testable with fakes.
        sleep_fn: Sleep function, injected so tests do not wait in real time.
    """

    def __init__(
        self,
        settings: WatchdogSettings,
        start_fn: Callable[[object], subprocess.Popen | None],
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._start_fn = start_fn
        self._sleep = sleep_fn
        self._backends: dict[int, _Supervised] = {}

    def register(self, port: int, role: str, model_def: object, process: subprocess.Popen) -> None:
        """Track an already-started backend.

        Args:
            port: Backend port.
            role: Config role name, for logging.
            model_def: Model definition, passed back to `start_fn` on restart.
            process: The running process.
        """
        self._backends[port] = _Supervised(role=role, model_def=model_def, process=process)

    @property
    def live_count(self) -> int:
        """Number of backends still being supervised and not abandoned."""
        return sum(1 for entry in self._backends.values() if not entry.abandoned)

    def current_processes(self) -> list[tuple[str, subprocess.Popen]]:
        """Every process currently supervised.

        Shutdown must reap what is running *now*, not the processes started at
        boot — a restarted backend is a different pid and the original handle is
        long dead.

        Returns:
            (model id, process) pairs for backends with a live process.
        """
        return [
            (str(getattr(entry.model_def, "id", entry.role)), entry.process)
            for entry in self._backends.values()
            if entry.process is not None
        ]

    def _within_budget(self, entry: _Supervised, now: float) -> bool:
        """Whether another restart is allowed inside the rolling window."""
        cutoff = now - self.settings.restart_window_seconds
        entry.attempts = [stamp for stamp in entry.attempts if stamp >= cutoff]
        return len(entry.attempts) < self.settings.max_restarts

    def _abandon(self, port: int, entry: _Supervised, reason: str) -> None:
        """Stop restarting a backend and record why."""
        entry.abandoned = True
        append_event(
            self.settings.log_path,
            "backend_abandoned",
            port=port,
            role=entry.role,
            model_id=getattr(entry.model_def, "id", None),
            reason=reason,
            attempts=len(entry.attempts),
            window_seconds=self.settings.restart_window_seconds,
        )
        log.error(
            "watchdog_backend_abandoned",
            port=port,
            role=entry.role,
            reason=reason,
            attempts=len(entry.attempts),
        )

    def wait_for_model_path(self, model_def: object) -> None:
        """Block until the model file exists, bounded by `mount_wait_seconds`.

        Models live on an external volume. After a reboot the launcher can start
        before the volume mounts, and a launch attempted then fails for a reason
        that resolves itself seconds later.
        """
        raw_path = getattr(model_def, "model_path", None)
        if not raw_path:
            return
        path = Path(str(raw_path))
        if path.exists():
            return
        deadline = time.monotonic() + self.settings.mount_wait_seconds
        log.info("watchdog_waiting_for_model_path", path=str(path))
        append_event(self.settings.log_path, "waiting_for_model_path", path=str(path))
        while time.monotonic() < deadline:
            self._sleep(2.0)
            if path.exists():
                append_event(self.settings.log_path, "model_path_appeared", path=str(path))
                return
        append_event(self.settings.log_path, "model_path_still_missing", path=str(path))

    def _kill(self, port: int, process: subprocess.Popen | None) -> int | None:
        """Stop a backend process, escalating to SIGKILL, and reap port strays.

        A wedged process may be unable to act on SIGTERM at all — a SIGSTOPped
        process queues it until continued — so escalation is not optional. Any
        process still holding the port afterwards is killed too, because a stray
        holding the socket makes the relaunch fail to bind.

        Args:
            port: Backend port, used to find strays.
            process: The known process, if any.

        Returns:
            The pid that was stopped, or None.
        """
        pid: int | None = None
        if process is not None and process.poll() is None:
            pid = process.pid
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("watchdog_sigterm_ignored_killing", port=port, pid=pid)
                process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.error("watchdog_process_unkillable", port=port, pid=pid)
        elif process is not None:
            pid = process.pid
        self._reap_port_strays(port)
        return pid

    def _reap_port_strays(self, port: int) -> None:
        """SIGKILL anything still listening on a backend port."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("watchdog_lsof_failed", port=port, error=str(exc))
            return
        for line in result.stdout.split():
            try:
                stray = int(line)
            except ValueError:
                continue
            log.warning("watchdog_killing_port_stray", port=port, pid=stray)
            append_event(self.settings.log_path, "port_stray_killed", port=port, pid=stray)
            try:
                os.kill(stray, 9)
            except (OSError, ProcessLookupError) as exc:
                log.warning("watchdog_stray_kill_failed", port=port, pid=stray, error=str(exc))

    def restart(self, port: int, reason: str, detail: str = "") -> bool:
        """Replace one backend process.

        Args:
            port: Backend to restart.
            reason: Why, for the log.
            detail: Extra context for the log.

        Returns:
            True if a replacement process is running.
        """
        entry = self._backends.get(port)
        if entry is None or entry.abandoned:
            return False

        now = time.monotonic()
        if not self._within_budget(entry, now):
            self._abandon(
                port,
                entry,
                f"{len(entry.attempts)} restarts within "
                f"{self.settings.restart_window_seconds:.0f}s; last reason: {reason}",
            )
            return False

        entry.attempts.append(now)
        old_pid = self._kill(port, entry.process)
        entry.process = None
        append_event(
            self.settings.log_path,
            "restart_started",
            port=port,
            role=entry.role,
            model_id=getattr(entry.model_def, "id", None),
            reason=reason,
            detail=detail,
            old_pid=old_pid,
            attempt=len(entry.attempts),
        )

        self._sleep(self.settings.restart_cooldown_seconds)
        self.wait_for_model_path(entry.model_def)
        process = self._start_fn(entry.model_def)
        entry.process = process

        append_event(
            self.settings.log_path,
            "restart_succeeded" if process else "restart_failed",
            port=port,
            role=entry.role,
            model_id=getattr(entry.model_def, "id", None),
            reason=reason,
            old_pid=old_pid,
            new_pid=process.pid if process else None,
            attempt=len(entry.attempts),
        )
        if process:
            log.info("watchdog_restarted", port=port, old_pid=old_pid, new_pid=process.pid)
        else:
            log.error("watchdog_restart_failed", port=port, reason=reason)
        return process is not None

    def check_exited(self) -> list[int]:
        """Restart backends whose process has exited.

        This is the easy failure mode, and it is handled here as well as by the
        request channel so that a dead backend recovers even with no traffic at
        all to notice it.

        Returns:
            Ports that were found exited.
        """
        exited: list[int] = []
        for port, entry in self._backends.items():
            if entry.abandoned or entry.process is None:
                continue
            code = entry.process.poll()
            if code is None:
                continue
            exited.append(port)
            append_event(
                self.settings.log_path,
                "backend_exited",
                port=port,
                role=entry.role,
                model_id=getattr(entry.model_def, "id", None),
                pid=entry.process.pid,
                return_code=code,
            )
            log.warning("watchdog_backend_exited", port=port, return_code=code)
            self.restart(port, "process_exited", f"exit code {code}")
        return exited

    def check_requests(self) -> list[int]:
        """Service restart requests published by the router.

        Returns:
            Ports that had a pending request.
        """
        serviced: list[int] = []
        for request in read_restart_requests(self.settings.request_dir):
            clear_restart_request(self.settings.request_dir, request.port)
            serviced.append(request.port)
            if request.port not in self._backends:
                log.warning("watchdog_request_unknown_port", port=request.port)
                continue
            log.warning("watchdog_servicing_request", port=request.port, reason=request.reason)
            self.restart(request.port, request.reason, request.detail)
        return serviced

    def poll_once(self) -> None:
        """Run one supervision sweep: pending requests, then exited processes."""
        self.check_requests()
        self.check_exited()

    def run(self, should_continue: Callable[[], bool] = lambda: True) -> None:
        """Supervise until every backend is abandoned or `should_continue` is False.

        Args:
            should_continue: Loop predicate, injected for tests.
        """
        append_event(
            self.settings.log_path,
            "supervisor_started",
            ports=sorted(self._backends),
            failure_threshold=self.settings.failure_threshold,
            stall_seconds=self.settings.stall_seconds,
            max_restarts=self.settings.max_restarts,
        )
        while should_continue() and self.live_count > 0:
            self.poll_once()
            self._sleep(self.settings.sweep_interval_seconds)
        append_event(self.settings.log_path, "supervisor_stopped", live=self.live_count)
