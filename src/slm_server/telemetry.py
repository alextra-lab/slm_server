"""Elasticsearch telemetry shipper for slm_server request_complete events.

Shipping is opt-in: set SLM_ES_URL to enable. When unset, this module is a
no-op and never attempts any network connection.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx
import structlog

log = structlog.get_logger()

# Module-level config — read once at import time; tests monkeypatch these directly.
_ES_URL: str | None = os.getenv("SLM_ES_URL")
_ES_ENABLED: bool = os.getenv("SLM_ES_ENABLED", "true").lower() != "false"
_ES_INDEX_PREFIX: str = os.getenv("SLM_ES_INDEX_PREFIX", "slm-requests")
_CF_CLIENT_ID: str | None = os.getenv("SLM_CF_ACCESS_CLIENT_ID")
_CF_CLIENT_SECRET: str | None = os.getenv("SLM_CF_ACCESS_CLIENT_SECRET")
_ES_API_KEY: str | None = os.getenv("SLM_ES_API_KEY")

if not _ES_URL:
    log.info("slm_telemetry_disabled", reason="SLM_ES_URL not set")


async def ship_request_complete(doc: dict) -> None:
    """POST a request_complete doc to the dated ES index.

    Fail-soft: swallows all exceptions so the request path is never affected.
    """
    if not _ES_URL or not _ES_ENABLED:
        return

    date_suffix = datetime.now(UTC).strftime("%Y.%m.%d")
    url = f"{_ES_URL.rstrip('/')}/{_ES_INDEX_PREFIX}-{date_suffix}/_doc"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _CF_CLIENT_ID and _CF_CLIENT_SECRET:
        headers["CF-Access-Client-Id"] = _CF_CLIENT_ID
        headers["CF-Access-Client-Secret"] = _CF_CLIENT_SECRET
    if _ES_API_KEY:
        headers["Authorization"] = f"ApiKey {_ES_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=doc, headers=headers, timeout=5.0)
            if resp.status_code >= 300:
                log.warning("es_ship_failed", status=resp.status_code, url=url)
    except Exception as exc:  # noqa: BLE001
        log.warning("es_ship_failed", error=str(exc), url=url)
