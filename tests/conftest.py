"""Shared test fixtures.

The watchdog redirect below is a safety interlock, not a convenience. The
launcher watches a directory for restart requests, and any test that drives the
router into a backend failure publishes one. Pointed at the default location,
running the suite on this machine would restart the models actually serving
traffic. Every test therefore gets its own throwaway directory.

The OTLP redirect is the same kind of interlock. Any test that builds a
TestClient runs the router lifespan, which installs a live span exporter; left
at its default the suite would start an exporter thread per test, all pointed at
a Collector that is not there. Tests that want to observe spans install their own
in-memory provider (see test_telemetry.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_watchdog_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep watchdog restart requests and logs out of the live repo."""
    monkeypatch.setenv("SLM_WATCHDOG_REQUEST_DIR", str(tmp_path / "watchdog-requests"))
    monkeypatch.setenv("SLM_WATCHDOG_LOG_PATH", str(tmp_path / "watchdog.jsonl"))


@pytest.fixture(autouse=True)
def disable_otlp_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the router lifespan installing a real span exporter during tests."""
    monkeypatch.setenv("SLM_OTEL_ENABLED", "false")
