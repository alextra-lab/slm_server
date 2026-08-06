"""Shared test fixtures.

The watchdog redirect below is a safety interlock, not a convenience. The
launcher watches a directory for restart requests, and any test that drives the
router into a backend failure publishes one. Pointed at the default location,
running the suite on this machine would restart the models actually serving
traffic. Every test therefore gets its own throwaway directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_watchdog_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep watchdog restart requests and logs out of the live repo."""
    monkeypatch.setenv("SLM_WATCHDOG_REQUEST_DIR", str(tmp_path / "watchdog-requests"))
    monkeypatch.setenv("SLM_WATCHDOG_LOG_PATH", str(tmp_path / "watchdog.jsonl"))
