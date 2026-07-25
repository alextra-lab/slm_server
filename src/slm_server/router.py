"""FastAPI routing service that routes requests to backend model servers based on model ID."""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from structlog import get_logger

from slm_server.config import ModelConfig, ModelDefinition, load_model_config
from slm_server.telemetry import ship_request_complete

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - create shared resources on startup, cleanup on shutdown."""
    # Create shared HTTP client with connection pooling
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    app.state.http_client = httpx.AsyncClient(timeout=timeout, limits=limits)

    # Load config
    try:
        app.state.model_config = load_model_config()
        log.info("model_config_loaded", model_count=len(app.state.model_config.models))
    except Exception as e:
        log.error("failed_to_load_config", error=str(e))
        await app.state.http_client.aclose()
        raise

    yield

    # Cleanup: close HTTP client
    await app.state.http_client.aclose()
    log.info("application_shutdown")


app = FastAPI(title="SLM Server Router", version="0.2.0", lifespan=lifespan)


MIN_BACKEND_TIMEOUT_SECONDS = 1.0
MAX_BACKEND_TIMEOUT_SECONDS = 3600.0


def _resolve_backend_timeout_seconds(body: dict[str, Any], model_def: ModelDefinition) -> float:
    """Resolve per-request backend timeout, with model default fallback.

    Clients may pass `timeout` in the request body to override model defaults.
    This field is consumed by the router and not forwarded to model backends.
    """
    timeout_value = body.get("timeout")
    if timeout_value is None:
        return float(model_def.default_timeout)

    if isinstance(timeout_value, bool):
        raise HTTPException(
            status_code=400,
            detail="Invalid 'timeout' field. Must be a positive number of seconds.",
        )

    if isinstance(timeout_value, int | float):
        timeout_seconds = float(timeout_value)
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid 'timeout' field. Must be a positive number of seconds.",
        )

    if (
        timeout_seconds < MIN_BACKEND_TIMEOUT_SECONDS
        or timeout_seconds > MAX_BACKEND_TIMEOUT_SECONDS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid 'timeout' field. Must be between "
                f"{int(MIN_BACKEND_TIMEOUT_SECONDS)} and {int(MAX_BACKEND_TIMEOUT_SECONDS)} seconds."
            ),
        )

    return timeout_seconds


def _get_model_definition(model_id: str, config: ModelConfig) -> ModelDefinition:
    """Get model definition by ID.

    Args:
        model_id: Model identifier from request.
        config: Model configuration.

    Returns:
        ModelDefinition instance.

    Raises:
        HTTPException: If model not found in config.
    """
    enabled_matches: list[ModelDefinition] = []
    disabled_match_found = False

    # Route only to enabled model entries. This avoids selecting a disabled
    # duplicate entry when multiple config roles share the same model id.
    for model_def in config.models.values():
        if model_def.id != model_id:
            continue
        if model_def.enabled:
            enabled_matches.append(model_def)
        else:
            disabled_match_found = True

    if len(enabled_matches) == 1:
        return enabled_matches[0]

    if len(enabled_matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Model '{model_id}' is configured multiple times as enabled. "
                "Ensure model IDs are unique among enabled entries."
            ),
        )

    if disabled_match_found:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' is configured but currently disabled.",
        )

    available_models = [m.id for m in config.models.values() if m.enabled]
    raise HTTPException(
        status_code=404,
        detail=(
            f"Model '{model_id}' not found in enabled configuration. Available models: {available_models}"
        ),
    )


def _get_backend_url(model_def: ModelDefinition, endpoint: str) -> str:
    """Build backend URL for a model.

    Args:
        model_def: Model definition.
        endpoint: API endpoint (e.g., '/v1/chat/completions').

    Returns:
        Full URL to backend server.
    """
    return f"http://localhost:{model_def.port}{endpoint}"


def _filtered_forward_headers(request: Request) -> dict[str, str]:
    """Headers safe to forward to a backend (httpx sets Host/Content-Length as needed)."""
    skip = {
        "content-length",
        "host",
        "connection",
        "transfer-encoding",
    }
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """Filter response headers to remove ones that should be recalculated.

    Args:
        headers: Headers from backend response.

    Returns:
        Filtered headers dict.
    """
    # Headers that should not be forwarded (will be recalculated)
    skip_headers = {
        "content-length",  # FastAPI/Starlette will recalculate
        "transfer-encoding",  # Will be recalculated
        "connection",  # Connection-specific, not relevant
    }
    return {k: v for k, v in headers.items() if k.lower() not in skip_headers}


def _sse_headers(headers: httpx.Headers) -> dict[str, str]:
    """Build headers for SSE streaming responses.

    Belt-and-suspenders set to prevent proxy/Cloudflare buffering that would
    delay the first token by 100+ seconds.
    """
    result = _filter_response_headers(headers)
    result["cache-control"] = "no-cache"
    result["x-accel-buffering"] = "no"  # nginx-style hint, honored by many proxies
    result["connection"] = "keep-alive"  # stripped by _filter_response_headers, restore for SSE
    return result


def _convert_responses_to_chat(body: dict) -> dict:
    """Convert /v1/responses request format to /v1/chat/completions format.

    The responses API (LM Studio) uses 'input' field which can be:
    - A string (simple prompt)
    - A list of input items (for tool results, etc.)

    We convert this to the chat/completions 'messages' format.

    Args:
        body: Request body in /v1/responses format.

    Returns:
        Request body in /v1/chat/completions format.
    """
    chat_body = body.copy()

    # Handle 'input' field (LM Studio /v1/responses format)
    if "input" in body:
        input_value = body["input"]

        if isinstance(input_value, str):
            # Simple string input -> single user message
            chat_body["messages"] = [{"role": "user", "content": input_value}]
        elif isinstance(input_value, list):
            # List of input items (e.g., function_call_output items)
            # Convert to tool role messages
            messages = []
            for item in input_value:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "function_call_output":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": item.get("call_id", ""),
                                "content": item.get("output", ""),
                            }
                        )
                    elif item_type == "message":
                        # Generic message item
                        messages.append(
                            {
                                "role": item.get("role", "user"),
                                "content": item.get("content", ""),
                            }
                        )
            if messages:
                chat_body["messages"] = messages
            else:
                # Fallback: empty messages
                chat_body["messages"] = [{"role": "user", "content": ""}]

        del chat_body["input"]

    # Handle 'prompt' field (alternative format)
    elif "prompt" in body:
        chat_body["messages"] = [{"role": "user", "content": body["prompt"]}]
        del chat_body["prompt"]

    # Remove responses-specific fields that chat/completions doesn't understand
    fields_to_remove = ["previous_response_id", "reasoning"]
    for field in fields_to_remove:
        if field in chat_body:
            del chat_body[field]

    # Ensure messages exists (fallback)
    if "messages" not in chat_body:
        chat_body["messages"] = [{"role": "user", "content": ""}]

    return chat_body


def _convert_chat_to_responses(response_data: dict) -> dict:
    """Convert /v1/chat/completions response to /v1/responses format.

    Args:
        response_data: Response data in /v1/chat/completions format.

    Returns:
        Response data in /v1/responses format (same structure for most fields).
    """
    # For most backends, the response format is already compatible
    # Just ensure the structure matches what personal_agent expects
    return response_data


def _build_error_response(
    status_code: int,
    message: str,
    model_id: str | None = None,
    backend_port: int | None = None,
    error_type: str = "server_error",
) -> JSONResponse:
    """Build a structured error response compatible with OpenAI format.

    Args:
        status_code: HTTP status code.
        message: Human-readable error message.
        model_id: Model identifier (if known).
        backend_port: Backend port (if known).
        error_type: Error type (e.g., "server_error", "invalid_request_error").

    Returns:
        JSONResponse with structured error data.
    """
    content: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type if status_code < 500 else "server_error",
            "param": None,
            "code": None,
        }
    }

    # Add debug info for troubleshooting
    if model_id or backend_port:
        content["slm_server_debug"] = {}
        if model_id:
            content["slm_server_debug"]["model_id"] = model_id
        if backend_port:
            content["slm_server_debug"]["backend_port"] = backend_port
        content["slm_server_debug"]["suggestion"] = "Check /v1/backends/health for backend status"

    return JSONResponse(status_code=status_code, content=content)


def _parse_sse_telemetry(content: bytes) -> tuple[dict | None, dict | None]:
    """Extract usage and timings dicts from a buffered SSE body."""
    usage: dict | None = None
    timings: dict | None = None
    for line in content.decode(errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        if "timings" in chunk:
            timings = chunk["timings"]
    return usage, timings


def _build_request_telemetry(
    *,
    trace_id: str | None,
    span_id: str | None,
    session_id: str | None,
    model_id: str,
    backend: str,
    port: int,
    usage: dict | None,
    timings: dict | None,
    total_ms: float,
    status: int,
    ttfb_ms: float | None = None,
    heartbeat_count: int | None = None,
    client_disconnected: bool | None = None,
) -> dict[str, object]:
    """Build the request_complete telemetry doc.

    Single source of the schema so every slm-server request_complete event is identical
    regardless of model or backend (chat, rerank, ...). Fields that don't apply to a given
    request type (e.g. decode/predicted for a reranker) are left None, never dropped.

    The streaming fields default to None so non-streaming callers are unchanged.
    `ttfb_ms` is the router-observed time to the first content byte; comparing it
    against `prefill_ms` approximates how long a request waited for a backend slot,
    which `total_ms` alone cannot distinguish from slow compute.
    """
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "session_id": session_id,
        "model_id": model_id,
        "backend": backend,
        "port": port,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "prefill_ms": timings.get("prompt_ms") if timings else None,
        "decode_ms": timings.get("predicted_ms") if timings else None,
        "prompt_n": timings.get("prompt_n") if timings else None,
        "predicted_n": timings.get("predicted_n") if timings else None,
        "cache_reuse": timings.get("cache_n") if timings else None,
        "total_ms": round(total_ms, 1),
        "ttfb_ms": round(ttfb_ms, 1) if ttfb_ms is not None else None,
        "heartbeat_count": heartbeat_count,
        "client_disconnected": client_disconnected,
        "status": status,
        "ts": datetime.now(UTC).isoformat(),
    }


_SSE_HEARTBEAT_INTERVAL_SECONDS = 15.0
_SSE_HEARTBEAT = b": keep-alive\n\n"


async def _iter_with_heartbeat(response: httpx.Response) -> AsyncIterator[bytes]:
    """Yield backend chunks, filling silent gaps with SSE comment keep-alives.

    A backend prefilling a large context emits nothing for the duration, so an
    intermediary proxy sees an idle connection and severs it. SSE comments are
    ignored by conformant clients, so they hold the connection open without
    perturbing the stream.

    Args:
        response: An open streaming response from the backend.

    Yields:
        Backend chunks verbatim, interleaved with keep-alive comments.
    """
    iterator = response.aiter_bytes().__aiter__()
    pending: asyncio.Future[bytes] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=_SSE_HEARTBEAT_INTERVAL_SECONDS)
            if not done:
                yield _SSE_HEARTBEAT
                continue
            settled, pending = pending, None
            try:
                chunk = settled.result()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        if pending is not None:
            pending.cancel()


async def _open_backend_stream(
    *,
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: httpx.Timeout,
) -> httpx.Response:
    """Send a request and return the response with its body still unread.

    Status and headers are available immediately, so a caller can branch on the
    status (e.g. the /v1/responses fallback) before committing to the body.

    Args:
        client: Shared pooled HTTP client.
        url: Resolved backend URL.
        body: Request body to forward.
        headers: Headers to forward to the backend.
        timeout: Per-request httpx timeout.

    Returns:
        An open streaming response. The caller owns closing it.
    """
    request = client.build_request("POST", url, json=body, headers=headers, timeout=timeout)
    return await client.send(request, stream=True)


async def _stream_backend_response(
    *,
    response: httpx.Response,
    t0: float,
    model_id: str,
    model_def: ModelDefinition,
    trace_id: str | None = None,
    span_id: str | None = None,
    session_id: str | None = None,
    emit_telemetry: bool = True,
) -> JSONResponse | StreamingResponse:
    """Forward an open backend stream to the caller without buffering it.

    The buffering path (`client.post`) held the whole generation in memory before
    emitting a byte, so the caller's connection stayed silent for the entire turn
    and an upstream proxy timed it out (FRE-980). Here chunks are forwarded as
    they arrive, and telemetry is emitted once the stream ends rather than before
    it starts.

    Args:
        response: An open streaming response from `_open_backend_stream`.
        t0: monotonic start time, for total_ms.
        model_id: Requested model id.
        model_def: Resolved model definition.
        trace_id: Caller trace id, for telemetry.
        span_id: Caller span id, for telemetry.
        session_id: Caller session id, for telemetry.
        emit_telemetry: Whether to emit request_complete. False for endpoints
            that have never emitted it, so this fix does not silently change
            what lands in the telemetry index.

    Returns:
        A StreamingResponse over the backend SSE stream, or a JSONResponse if the
        backend answered with an error status.
    """

    def _emit_telemetry(
        usage: dict | None,
        timings: dict | None,
        status: int,
        *,
        ttfb_ms: float | None = None,
        heartbeat_count: int | None = None,
        client_disconnected: bool | None = None,
    ) -> None:
        if not emit_telemetry:
            return
        tel_doc = _build_request_telemetry(
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
            usage=usage,
            timings=timings,
            total_ms=(time.monotonic() - t0) * 1000,
            status=status,
            ttfb_ms=ttfb_ms,
            heartbeat_count=heartbeat_count,
            client_disconnected=client_disconnected,
        )
        log.info("request_complete", **tel_doc)
        asyncio.create_task(ship_request_complete(tel_doc))

    if response.status_code >= 400:
        raw = await response.aread()
        await response.aclose()
        try:
            error_detail: Any = json.loads(raw)
        except json.JSONDecodeError:
            error_detail = raw.decode(errors="replace")[:500]
        log.warning(
            "backend_error_response",
            status_code=response.status_code,
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
            error_detail=error_detail,
        )
        _emit_telemetry(None, None, response.status_code)
        return JSONResponse(
            content=error_detail if isinstance(error_detail, dict) else {"error": error_detail},
            status_code=response.status_code,
            headers=_filter_response_headers(response.headers),
        )

    async def stream_response() -> AsyncIterator[bytes]:
        # Keep the body for telemetry only — usage/timings ride in the final chunks.
        body = bytearray()
        heartbeats = 0
        ttfb_ms: float | None = None
        disconnected = False
        try:
            async for chunk in _iter_with_heartbeat(response):
                if chunk is _SSE_HEARTBEAT:
                    heartbeats += 1
                else:
                    if ttfb_ms is None:
                        ttfb_ms = (time.monotonic() - t0) * 1000
                    body.extend(chunk)
                yield chunk
        except asyncio.CancelledError:
            # The caller went away mid-stream. Before FRE-980 this was the edge
            # severing a silent connection and nothing here ever recorded it.
            disconnected = True
            log.warning(
                "stream_client_disconnected",
                model_id=model_id,
                backend=model_def.backend,
                port=model_def.port,
                trace_id=trace_id,
                bytes_forwarded=len(body),
                heartbeat_count=heartbeats,
            )
            raise
        finally:
            with contextlib.suppress(Exception):
                await response.aclose()
            usage, timings = _parse_sse_telemetry(bytes(body))
            _emit_telemetry(
                usage,
                timings,
                response.status_code,
                ttfb_ms=ttfb_ms,
                heartbeat_count=heartbeats,
                client_disconnected=disconnected,
            )

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers=_sse_headers(response.headers),
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    """Route chat completions requests to appropriate backend."""
    try:
        body = await request.json()
        model_id = body.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model' field in request body")

        model_def = _get_model_definition(model_id, request.app.state.model_config)
        backend_url = _get_backend_url(model_def, "/v1/chat/completions")

        trace_id = request.headers.get("x-trace-id")
        span_id = request.headers.get("x-span-id")
        session_id = request.headers.get("x-session-id")

        log.info(
            "routing_request",
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
        )

        filtered_headers = _filtered_forward_headers(request)

        # Use shared HTTP client with connection pooling
        client = request.app.state.http_client

        # Inject chat_template_kwargs (e.g. enable_thinking for Unsloth Qwen3.5) so backend gets it per-request
        body_forward = dict(body)
        request_timeout_seconds = _resolve_backend_timeout_seconds(body_forward, model_def)
        body_forward.pop("timeout", None)
        if (
            getattr(model_def, "chat_template_kwargs", None)
            and "chat_template_kwargs" not in body_forward
        ):
            body_forward["chat_template_kwargs"] = model_def.chat_template_kwargs

        # Override timeout for this request based on model config
        timeout = httpx.Timeout(connect=10.0, read=request_timeout_seconds, write=30.0, pool=10.0)

        # Streaming requests are proxied through unbuffered so the caller's
        # connection never goes silent for the length of a generation (FRE-980).
        if body_forward.get("stream"):
            stream_t0 = time.monotonic()
            backend_response = await _open_backend_stream(
                client=client,
                url=backend_url,
                body=body_forward,
                headers=filtered_headers,
                timeout=timeout,
            )
            return await _stream_backend_response(
                response=backend_response,
                t0=stream_t0,
                model_id=model_id,
                model_def=model_def,
                trace_id=trace_id,
                span_id=span_id,
                session_id=session_id,
            )

        t0 = time.monotonic()
        response = await client.post(
            backend_url, json=body_forward, headers=filtered_headers, timeout=timeout
        )
        total_ms = (time.monotonic() - t0) * 1000

        # Parse buffered response for telemetry (response.content is fully buffered by client.post)
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            usage, timings = _parse_sse_telemetry(response.content)
        else:
            try:
                rjson = response.json()
                usage = rjson.get("usage") or None
            except Exception:
                usage = None
            timings = None

        tel_doc = _build_request_telemetry(
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
            usage=usage,
            timings=timings,
            total_ms=total_ms,
            status=response.status_code,
        )
        log.info("request_complete", **tel_doc)
        asyncio.create_task(ship_request_complete(tel_doc))

        # Log error responses for debugging
        if response.status_code >= 400:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text[:500]  # Limit text length
            log.warning(
                "backend_error_response",
                status_code=response.status_code,
                model_id=model_id,
                backend=model_def.backend,
                port=model_def.port,
                error_detail=error_detail,
            )

        if response.headers.get("content-type", "").startswith("text/event-stream"):
            # Streaming response: use async generator to keep connection alive
            async def stream_response():
                async for chunk in response.aiter_bytes():
                    yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers=_sse_headers(response.headers),
            )
        else:
            # Non-streaming response
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code,
                headers=_filter_response_headers(response.headers),
            )

    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json()
        except Exception:
            error_detail = e.response.text[:500]

        log.error(
            "backend_http_error",
            status_code=e.response.status_code,
            model_id=model_id if "model_id" in locals() else "unknown",
            error_detail=error_detail,
        )

        return _build_error_response(
            status_code=e.response.status_code,
            message=f"Backend error: {error_detail}",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except httpx.ConnectError:
        log.error(
            "backend_unreachable",
            model_id=model_id if "model_id" in locals() else "unknown",
            port=model_def.port if "model_def" in locals() else "unknown",
        )
        return _build_error_response(
            status_code=503,
            message="Backend server unreachable. Is the model server running?",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except httpx.TimeoutException:
        log.error(
            "backend_timeout",
            model_id=model_id if "model_id" in locals() else "unknown",
            timeout=model_def.default_timeout if "model_def" in locals() else "unknown",
        )
        return _build_error_response(
            status_code=504,
            message="Backend request timeout. The model may be overloaded.",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is

    except Exception as e:
        log.error("routing_error", error=str(e), error_type=type(e).__name__)
        return _build_error_response(
            status_code=500,
            message="Internal server error while routing request.",
        )


@app.post("/v1/embeddings", response_model=None)
async def embeddings(request: Request) -> JSONResponse:
    """Route embedding requests to the appropriate backend (OpenAI /v1/embeddings)."""
    try:
        body = await request.json()
        model_id = body.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model' field in request body")

        model_def = _get_model_definition(model_id, request.app.state.model_config)
        backend_url = _get_backend_url(model_def, "/v1/embeddings")

        log.info(
            "routing_embeddings_request",
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
        )

        filtered_headers = _filtered_forward_headers(request)
        client = request.app.state.http_client
        body_forward = dict(body)
        request_timeout_seconds = _resolve_backend_timeout_seconds(body_forward, model_def)
        body_forward.pop("timeout", None)
        timeout = httpx.Timeout(
            connect=10.0,
            read=request_timeout_seconds,
            write=30.0,
            pool=10.0,
        )

        response = await client.post(
            backend_url,
            json=body_forward,
            headers=filtered_headers,
            timeout=timeout,
        )

        if response.status_code >= 400:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text[:500]
            log.warning(
                "backend_error_response",
                status_code=response.status_code,
                model_id=model_id,
                backend=model_def.backend,
                port=model_def.port,
                error_detail=error_detail,
            )

        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
            headers=_filter_response_headers(response.headers),
        )

    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json()
        except Exception:
            error_detail = e.response.text[:500]

        log.error(
            "backend_http_error",
            status_code=e.response.status_code,
            model_id=model_id if "model_id" in locals() else "unknown",
            error_detail=error_detail,
        )

        return _build_error_response(
            status_code=e.response.status_code,
            message=f"Backend error: {error_detail}",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except httpx.ConnectError:
        log.error(
            "backend_unreachable",
            model_id=model_id if "model_id" in locals() else "unknown",
            port=model_def.port if "model_def" in locals() else "unknown",
        )
        return _build_error_response(
            status_code=503,
            message="Backend server unreachable. Is the model server running?",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except httpx.TimeoutException:
        log.error(
            "backend_timeout",
            model_id=model_id if "model_id" in locals() else "unknown",
            timeout=model_def.default_timeout if "model_def" in locals() else "unknown",
        )
        return _build_error_response(
            status_code=504,
            message="Backend request timeout. The model may be overloaded.",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except HTTPException:
        raise

    except Exception as e:
        log.error("routing_error", error=str(e), error_type=type(e).__name__)
        return _build_error_response(
            status_code=500,
            message="Internal server error while routing request.",
        )


@app.post("/v1/rerank", response_model=None)
async def rerank(request: Request) -> JSONResponse:
    """Route rerank requests to the appropriate backend (llama-server /v1/rerank)."""
    try:
        body = await request.json()
        model_id = body.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model' field in request body")

        model_def = _get_model_definition(model_id, request.app.state.model_config)
        backend_url = _get_backend_url(model_def, "/v1/rerank")

        trace_id = request.headers.get("x-trace-id")
        span_id = request.headers.get("x-span-id")
        session_id = request.headers.get("x-session-id")

        log.info(
            "routing_rerank_request",
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
        )

        filtered_headers = _filtered_forward_headers(request)
        client = request.app.state.http_client
        body_forward = dict(body)
        request_timeout_seconds = _resolve_backend_timeout_seconds(body_forward, model_def)
        body_forward.pop("timeout", None)
        timeout = httpx.Timeout(
            connect=10.0,
            read=request_timeout_seconds,
            write=30.0,
            pool=10.0,
        )

        t0 = time.monotonic()
        response = await client.post(
            backend_url,
            json=body_forward,
            headers=filtered_headers,
            timeout=timeout,
        )
        total_ms = (time.monotonic() - t0) * 1000

        try:
            usage = response.json().get("usage") or None
        except Exception:
            usage = None

        # Same schema as chat so slm-requests-* is uniform across backends. A reranker has
        # no decode phase, so completion/decode/predicted/cache fields stay None (via timings=None).
        tel_doc = _build_request_telemetry(
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
            usage=usage,
            timings=None,
            total_ms=total_ms,
            status=response.status_code,
        )
        log.info("request_complete", **tel_doc)
        asyncio.create_task(ship_request_complete(tel_doc))

        if response.status_code >= 400:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text[:500]
            log.warning(
                "backend_error_response",
                status_code=response.status_code,
                model_id=model_id,
                backend=model_def.backend,
                port=model_def.port,
                error_detail=error_detail,
            )

        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
            headers=_filter_response_headers(response.headers),
        )

    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json()
        except Exception:
            error_detail = e.response.text[:500]

        log.error(
            "backend_http_error",
            status_code=e.response.status_code,
            model_id=model_id if "model_id" in locals() else "unknown",
            error_detail=error_detail,
        )

        return _build_error_response(
            status_code=e.response.status_code,
            message=f"Backend error: {error_detail}",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except httpx.ConnectError:
        log.error(
            "backend_unreachable",
            model_id=model_id if "model_id" in locals() else "unknown",
            port=model_def.port if "model_def" in locals() else "unknown",
        )
        return _build_error_response(
            status_code=503,
            message="Backend server unreachable. Is the model server running?",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except httpx.TimeoutException:
        log.error(
            "backend_timeout",
            model_id=model_id if "model_id" in locals() else "unknown",
            timeout=model_def.default_timeout if "model_def" in locals() else "unknown",
        )
        return _build_error_response(
            status_code=504,
            message="Backend request timeout. The model may be overloaded.",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except HTTPException:
        raise

    except Exception as e:
        log.error("routing_error", error=str(e), error_type=type(e).__name__)
        return _build_error_response(
            status_code=500,
            message="Internal server error while routing request.",
        )


@app.post("/v1/responses", response_model=None)
async def responses(request: Request) -> JSONResponse | StreamingResponse:
    """Route responses API requests with automatic fallback to chat/completions.

    This endpoint first tries /v1/responses on the backend. If the backend returns 404
    (endpoint not supported), it automatically converts the request to /v1/chat/completions
    format and retries. This provides compatibility with backends that don't support the
    LM Studio stateful responses API.
    """
    try:
        body = await request.json()
        model_id = body.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model' field in request body")

        model_def = _get_model_definition(model_id, request.app.state.model_config)
        backend_url = _get_backend_url(model_def, "/v1/responses")

        log.info(
            "routing_responses_request",
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
        )

        filtered_headers = _filtered_forward_headers(request)

        # Use shared HTTP client with connection pooling
        client = request.app.state.http_client

        # Override timeout for this request
        body_forward = dict(body)
        request_timeout_seconds = _resolve_backend_timeout_seconds(body_forward, model_def)
        body_forward.pop("timeout", None)
        timeout = httpx.Timeout(connect=10.0, read=request_timeout_seconds, write=30.0, pool=10.0)

        # Streaming requests are proxied through unbuffered, same as
        # /v1/chat/completions (FRE-980). The 404/422 fallback still works
        # because send(stream=True) exposes the status before the body is read.
        # Telemetry stays off here: this endpoint has never emitted
        # request_complete, and the streaming fix should not change that.
        if body_forward.get("stream"):
            stream_t0 = time.monotonic()
            probe = await _open_backend_stream(
                client=client,
                url=backend_url,
                body=body_forward,
                headers=filtered_headers,
                timeout=timeout,
            )
            if probe.status_code not in (404, 422):
                return await _stream_backend_response(
                    response=probe,
                    t0=stream_t0,
                    model_id=model_id,
                    model_def=model_def,
                    emit_telemetry=False,
                )

            await probe.aclose()
            log.info(
                "responses_fallback_to_chat",
                model_id=model_id,
                backend=model_def.backend,
                original_status=probe.status_code,
                message=(
                    "/v1/responses not supported or invalid format, "
                    "converting to /v1/chat/completions"
                ),
            )
            chat_body = _convert_responses_to_chat(body_forward)
            if (
                getattr(model_def, "chat_template_kwargs", None)
                and "chat_template_kwargs" not in chat_body
            ):
                chat_body["chat_template_kwargs"] = model_def.chat_template_kwargs
            fallback_response = await _open_backend_stream(
                client=client,
                url=_get_backend_url(model_def, "/v1/chat/completions"),
                body=chat_body,
                headers=filtered_headers,
                timeout=timeout,
            )
            return await _stream_backend_response(
                response=fallback_response,
                t0=stream_t0,
                model_id=model_id,
                model_def=model_def,
                emit_telemetry=False,
            )

        try:
            # Try /v1/responses first
            response = await client.post(
                backend_url, json=body_forward, headers=filtered_headers, timeout=timeout
            )

            # If successful (not 404/422), return response
            # 404 = endpoint doesn't exist, 422 = endpoint exists but doesn't accept format
            # Both indicate we should fall back to /v1/chat/completions
            if response.status_code not in (404, 422):
                if response.status_code >= 400:
                    try:
                        error_detail = response.json()
                    except Exception:
                        error_detail = response.text[:500]
                    log.warning(
                        "backend_error_response",
                        status_code=response.status_code,
                        model_id=model_id,
                        backend=model_def.backend,
                        port=model_def.port,
                        error_detail=error_detail,
                    )

                if response.headers.get("content-type", "").startswith("text/event-stream"):

                    async def stream_response():
                        async for chunk in response.aiter_bytes():
                            yield chunk

                    return StreamingResponse(
                        stream_response(),
                        media_type="text/event-stream",
                        headers=_sse_headers(response.headers),
                    )
                else:
                    return JSONResponse(
                        content=response.json(),
                        status_code=response.status_code,
                        headers=_filter_response_headers(response.headers),
                    )

        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (404, 422):
                raise
            # Fall through to fallback if 404 or 422

        # Backend doesn't support /v1/responses or returned validation error, fallback to /v1/chat/completions
        log.info(
            "responses_fallback_to_chat",
            model_id=model_id,
            backend=model_def.backend,
            original_status=response.status_code if "response" in locals() else "exception",
            message="/v1/responses not supported or invalid format, converting to /v1/chat/completions",
        )

        # Convert request format
        chat_body = _convert_responses_to_chat(body_forward)
        if (
            getattr(model_def, "chat_template_kwargs", None)
            and "chat_template_kwargs" not in chat_body
        ):
            chat_body["chat_template_kwargs"] = model_def.chat_template_kwargs
        fallback_url = _get_backend_url(model_def, "/v1/chat/completions")

        response = await client.post(
            fallback_url, json=chat_body, headers=filtered_headers, timeout=timeout
        )

        if response.status_code >= 400:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text[:500]
            log.warning(
                "backend_error_response",
                status_code=response.status_code,
                model_id=model_id,
                backend=model_def.backend,
                port=model_def.port,
                error_detail=error_detail,
            )

        if response.headers.get("content-type", "").startswith("text/event-stream"):

            async def stream_response():
                async for chunk in response.aiter_bytes():
                    yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers=_sse_headers(response.headers),
            )
        else:
            # Convert response back to responses format
            response_data = response.json()
            converted_data = _convert_chat_to_responses(response_data)
            return JSONResponse(
                content=converted_data,
                status_code=response.status_code,
                headers=_filter_response_headers(response.headers),
            )

    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json()
        except Exception:
            error_detail = e.response.text[:500]

        log.error(
            "backend_http_error",
            status_code=e.response.status_code,
            model_id=model_id if "model_id" in locals() else "unknown",
            error_detail=error_detail,
        )

        return _build_error_response(
            status_code=e.response.status_code,
            message=f"Backend error: {error_detail}",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except httpx.ConnectError:
        log.error(
            "backend_unreachable",
            model_id=model_id if "model_id" in locals() else "unknown",
            port=model_def.port if "model_def" in locals() else "unknown",
        )
        return _build_error_response(
            status_code=503,
            message="Backend server unreachable. Is the model server running?",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except httpx.TimeoutException:
        log.error(
            "backend_timeout",
            model_id=model_id if "model_id" in locals() else "unknown",
            timeout=model_def.default_timeout if "model_def" in locals() else "unknown",
        )
        return _build_error_response(
            status_code=504,
            message="Backend request timeout. The model may be overloaded.",
            model_id=model_id if "model_id" in locals() else None,
            backend_port=model_def.port if "model_def" in locals() else None,
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is

    except Exception as e:
        log.error("routing_error", error=str(e), error_type=type(e).__name__)
        return _build_error_response(
            status_code=500,
            message="Internal server error while routing request.",
        )


@app.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """List available models."""
    models_list = [
        {
            "id": model_def.id,
            "backend": model_def.backend,
            "port": model_def.port,
            "model_type": model_def.model_type,
            "context_length": model_def.context_length,
            "quantization": model_def.quantization,
            "supports_function_calling": model_def.supports_function_calling,
        }
        for model_def in request.app.state.model_config.models.values()
    ]

    return JSONResponse(
        content={
            "object": "list",
            "data": models_list,
        }
    )


@app.get("/v1/backends/health")
async def backends_health(request: Request) -> JSONResponse:
    """Check health of all configured backends.

    This endpoint queries the /health endpoint of each backend server to verify
    they are running and responsive. Useful for debugging startup issues and
    monitoring backend availability.
    """
    health_status = {}
    client = request.app.state.http_client

    for role, model_def in request.app.state.model_config.models.items():
        if not model_def.enabled:
            health_status[role] = {
                "status": "disabled",
                "model_id": model_def.id,
                "port": model_def.port,
            }
            continue

        try:
            # Try backend health endpoint (common for most backends)
            url = f"http://localhost:{model_def.port}/health"
            response = await client.get(url, timeout=5.0)
            health_status[role] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "port": model_def.port,
                "model_id": model_def.id,
                "backend": model_def.backend,
                "http_status": response.status_code,
            }
        except httpx.TimeoutException:
            health_status[role] = {
                "status": "timeout",
                "port": model_def.port,
                "model_id": model_def.id,
                "backend": model_def.backend,
                "error": "Health check timed out after 5s",
            }
        except httpx.ConnectError:
            health_status[role] = {
                "status": "unreachable",
                "port": model_def.port,
                "model_id": model_def.id,
                "backend": model_def.backend,
                "error": "Connection refused - backend not running",
            }
        except Exception as e:
            health_status[role] = {
                "status": "error",
                "port": model_def.port,
                "model_id": model_def.id,
                "backend": model_def.backend,
                "error": str(e),
            }

    return JSONResponse(content=health_status)


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint for the router itself."""
    return JSONResponse(content={"status": "healthy"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
