"""Tests for slm_server.telemetry — Elasticsearch request_complete shipper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import slm_server.telemetry as telemetry_module
from slm_server.telemetry import ship_request_complete  # type: ignore[import-untyped]

_SAMPLE_DOC = {"trace_id": "trace-abc", "span_id": "span-def", "ts": "2026-05-29T00:00:00Z"}


async def test_ship_skips_when_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No POST is attempted when SLM_ES_URL is not set (opt-in shipping)."""
    monkeypatch.setattr(telemetry_module, "_ES_URL", None)
    with patch("httpx.AsyncClient") as mock_class:
        await ship_request_complete(_SAMPLE_DOC)
        mock_class.assert_not_called()


async def test_ship_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """No POST when SLM_ES_ENABLED is False even if URL is set."""
    monkeypatch.setattr(telemetry_module, "_ES_URL", "http://localhost:9200")
    monkeypatch.setattr(telemetry_module, "_ES_ENABLED", False)
    with patch("httpx.AsyncClient") as mock_class:
        await ship_request_complete(_SAMPLE_DOC)
        mock_class.assert_not_called()


async def test_ship_posts_to_correct_url_with_cf_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST hits the dated slm-requests-* index with CF-Access headers attached."""
    monkeypatch.setattr(telemetry_module, "_ES_URL", "https://es.frenchforet.com")
    monkeypatch.setattr(telemetry_module, "_ES_ENABLED", True)
    monkeypatch.setattr(telemetry_module, "_ES_INDEX_PREFIX", "slm-requests")
    monkeypatch.setattr(telemetry_module, "_CF_CLIENT_ID", "cf-id")
    monkeypatch.setattr(telemetry_module, "_CF_CLIENT_SECRET", "cf-secret")
    monkeypatch.setattr(telemetry_module, "_ES_API_KEY", None)

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await ship_request_complete(_SAMPLE_DOC)

    mock_client.post.assert_called_once()
    url: str = mock_client.post.call_args[0][0]
    headers: dict = mock_client.post.call_args[1]["headers"]
    assert "es.frenchforet.com" in url
    assert "slm-requests-" in url
    assert url.endswith("/_doc")
    assert headers["CF-Access-Client-Id"] == "cf-id"
    assert headers["CF-Access-Client-Secret"] == "cf-secret"
    assert "Authorization" not in headers


async def test_ship_adds_api_key_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authorization: ApiKey header is added when SLM_ES_API_KEY is set."""
    monkeypatch.setattr(telemetry_module, "_ES_URL", "http://localhost:9200")
    monkeypatch.setattr(telemetry_module, "_ES_ENABLED", True)
    monkeypatch.setattr(telemetry_module, "_ES_INDEX_PREFIX", "slm-requests")
    monkeypatch.setattr(telemetry_module, "_CF_CLIENT_ID", None)
    monkeypatch.setattr(telemetry_module, "_CF_CLIENT_SECRET", None)
    monkeypatch.setattr(telemetry_module, "_ES_API_KEY", "my-api-key")

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await ship_request_complete(_SAMPLE_DOC)

    headers: dict = mock_client.post.call_args[1]["headers"]
    assert headers["Authorization"] == "ApiKey my-api-key"


async def test_ship_fail_soft_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """ship_request_complete never raises — swallows connection errors silently."""
    monkeypatch.setattr(telemetry_module, "_ES_URL", "http://localhost:9200")
    monkeypatch.setattr(telemetry_module, "_ES_ENABLED", True)
    monkeypatch.setattr(telemetry_module, "_CF_CLIENT_ID", None)
    monkeypatch.setattr(telemetry_module, "_CF_CLIENT_SECRET", None)
    monkeypatch.setattr(telemetry_module, "_ES_API_KEY", None)

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectError("refused")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await ship_request_complete(_SAMPLE_DOC)  # must not raise
