"""Tests for watchdog detection inside the router (FRE-241).

These cover the seam the design depends on: the router deciding, from real
traffic, whether a backend is serving — and publishing a restart request when
it decides not.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from slm_server import router as router_module  # type: ignore[import-untyped]
from slm_server import watchdog as wd  # type: ignore[import-untyped]
from slm_server.config import ModelConfig, ModelDefinition  # type: ignore[import-untyped]

app = router_module.app


def _model_def() -> ModelDefinition:
    return ModelDefinition(
        id="unsloth/qwen3.6-35-A3B",
        backend="llamacpp",
        port=8502,
        context_length=32768,
        quantization="Q6_K_XL",
        max_concurrency=1,
        default_timeout=120,
        model_type="multimodal",
        model_path="hf/stub",
    )


@pytest.fixture
def watchdog_settings(tmp_path: Path) -> wd.WatchdogSettings:
    # A long sweep interval: these tests drive detection directly rather than
    # waiting on the background sweeper, which must not spin during the test.
    return wd.WatchdogSettings(
        failure_threshold=2,
        stall_seconds=300.0,
        sweep_interval_seconds=3600.0,
        request_dir=tmp_path / "requests",
        log_path=tmp_path / "watchdog.jsonl",
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, watchdog_settings: wd.WatchdogSettings):
    cfg = ModelConfig(models={"reasoning": _model_def()})
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    monkeypatch.setattr(router_module, "load_watchdog_settings", lambda: watchdog_settings)
    with TestClient(app) as test_client:
        yield test_client


def _chat(client: TestClient) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": "unsloth/qwen3.6-35-A3B", "messages": [{"role": "user", "content": "hi"}]},
    )


def test_unreachable_backend_trips_a_restart_after_the_threshold(
    client: TestClient, watchdog_settings: wd.WatchdogSettings
) -> None:
    async def refuse(url: str, **_kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    app.state.http_client.post = refuse  # type: ignore[method-assign]

    assert _chat(client).status_code == 503
    assert wd.read_restart_requests(watchdog_settings.request_dir) == []

    assert _chat(client).status_code == 503
    pending = wd.read_restart_requests(watchdog_settings.request_dir)
    assert len(pending) == 1
    assert pending[0].port == 8502
    assert pending[0].reason == "unreachable"


def test_timeouts_trip_a_restart(
    client: TestClient, watchdog_settings: wd.WatchdogSettings
) -> None:
    async def time_out(url: str, **_kwargs: object) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    app.state.http_client.post = time_out  # type: ignore[method-assign]

    assert _chat(client).status_code == 504
    assert _chat(client).status_code == 504

    pending = wd.read_restart_requests(watchdog_settings.request_dir)
    assert len(pending) == 1
    assert pending[0].reason == "timeout"


def test_a_served_request_between_failures_prevents_a_restart(
    client: TestClient, watchdog_settings: wd.WatchdogSettings
) -> None:
    """One transient error either side of a good turn is not a wedge."""
    outcomes = iter(["fail", "ok", "fail"])

    async def flaky(url: str, **_kwargs: object) -> httpx.Response:
        if next(outcomes) == "fail":
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"id": "ok", "choices": []})

    app.state.http_client.post = flaky  # type: ignore[method-assign]

    _chat(client)
    _chat(client)
    _chat(client)

    assert wd.read_restart_requests(watchdog_settings.request_dir) == []


def test_client_errors_are_never_blamed_on_the_backend(
    client: TestClient, watchdog_settings: wd.WatchdogSettings
) -> None:
    """A 400 says the caller sent nonsense, not that the model is unhealthy."""

    async def bad_request(url: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    app.state.http_client.post = bad_request  # type: ignore[method-assign]

    for _ in range(5):
        assert _chat(client).status_code == 400

    assert wd.read_restart_requests(watchdog_settings.request_dir) == []


def test_a_backend_degrading_into_429s_trips_a_restart(
    client: TestClient, watchdog_settings: wd.WatchdogSettings
) -> None:
    """429 describes the backend's condition, not the caller's request.

    Under the previous `status < 500` split this was scored as health, which
    also reset the failure streak — so this backend could have answered 429
    indefinitely and never tripped.
    """

    async def saturated(url: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(429, json={"error": "too many requests"})

    app.state.http_client.post = saturated  # type: ignore[method-assign]

    _chat(client)
    _chat(client)

    pending = wd.read_restart_requests(watchdog_settings.request_dir)
    assert len(pending) == 1
    assert pending[0].reason == "saturated"


def test_a_malformed_request_never_blames_the_backend(
    client: TestClient, watchdog_settings: wd.WatchdogSettings
) -> None:
    """422 is the caller's fault and must stay health, even in a long run."""

    async def unprocessable(url: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(422, json={"error": "unprocessable"})

    app.state.http_client.post = unprocessable  # type: ignore[method-assign]

    for _ in range(6):
        _chat(client)

    assert wd.read_restart_requests(watchdog_settings.request_dir) == []


def test_backend_5xx_counts_as_a_failure(
    client: TestClient, watchdog_settings: wd.WatchdogSettings
) -> None:
    async def broken(url: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    app.state.http_client.post = broken  # type: ignore[method-assign]

    _chat(client)
    _chat(client)

    pending = wd.read_restart_requests(watchdog_settings.request_dir)
    assert len(pending) == 1
    assert pending[0].reason == "server_error"


def test_requests_that_never_reached_a_backend_are_not_counted(
    client: TestClient, watchdog_settings: wd.WatchdogSettings
) -> None:
    """An unknown model never resolves a port, so no backend can be blamed."""
    for _ in range(5):
        client.post("/v1/chat/completions", json={"model": "nope", "messages": []})

    assert wd.read_restart_requests(watchdog_settings.request_dir) == []


def test_a_stream_that_breaks_after_its_first_byte_is_counted_as_a_failure(
    client: TestClient, watchdog_settings: wd.WatchdogSettings
) -> None:
    """A backend that emits one chunk then dies was previously scored a success.

    The middleware records the 200 that opened the stream before the body is
    consumed, and the first byte disables stall detection for that request, so
    nothing else would ever notice.
    """
    watchdog = app.state.watchdog
    handle = router_module._InFlightHandle(watchdog, 8502, watchdog.tracker.begin_request(8502))

    handle.failed("stream aborted after 12 bytes: peer closed")
    handle.failed("stream aborted after 4 bytes: peer closed")

    pending = wd.read_restart_requests(watchdog_settings.request_dir)
    assert len(pending) == 1
    assert pending[0].reason == "server_error"


def test_non_streaming_requests_are_not_stall_tracked(client: TestClient) -> None:
    """A buffered generation has no first byte until it finishes — observed
    total_ms reaches 890.9s — so stall tracking there would restart a healthy
    backend mid-turn."""
    tracked: list[int] = []

    async def slow_but_fine(url: str, **_kwargs: object) -> httpx.Response:
        tracked.append(len(app.state.watchdog.tracker._in_flight))
        return httpx.Response(200, json={"id": "ok", "choices": []})

    app.state.http_client.post = slow_but_fine  # type: ignore[method-assign]

    assert _chat(client).status_code == 200
    assert tracked == [0], "the non-streaming path must not register an in-flight request"
