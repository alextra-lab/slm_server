"""FastAPI routing service that routes requests to backend model servers based on model ID."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from structlog import get_logger

from slm_server.config import ModelConfig, ModelDefinition, load_model_config

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
    # Search for model by ID
    for role, model_def in config.models.items():
        if model_def.id == model_id:
            return model_def

    available_models = [m.id for m in config.models.values()]
    raise HTTPException(
        status_code=404,
        detail=(
            f"Model '{model_id}' not found in configuration. "
            f"Available models: {available_models}"
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
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in skip
    }


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
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in skip_headers
    }


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
                        messages.append({
                            "role": "tool",
                            "tool_call_id": item.get("call_id", ""),
                            "content": item.get("output", ""),
                        })
                    elif item_type == "message":
                        # Generic message item
                        messages.append({
                            "role": item.get("role", "user"),
                            "content": item.get("content", ""),
                        })
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
    content = {
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

        log.info(
            "routing_request",
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
        )

        filtered_headers = _filtered_forward_headers(request)

        # Use shared HTTP client with connection pooling
        client = request.app.state.http_client

        # Inject chat_template_kwargs (e.g. enable_thinking for Unsloth Qwen3.5) so backend gets it per-request
        body_forward = dict(body)
        if getattr(model_def, "chat_template_kwargs", None) and "chat_template_kwargs" not in body_forward:
            body_forward["chat_template_kwargs"] = model_def.chat_template_kwargs
        
        # Override timeout for this request based on model config
        timeout = httpx.Timeout(
            connect=10.0,
            read=model_def.default_timeout,
            write=30.0,
            pool=10.0
        )
        
        response = await client.post(
            backend_url,
            json=body_forward,
            headers=filtered_headers,
            timeout=timeout
        )

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
                headers=_filter_response_headers(response.headers),
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
        timeout = httpx.Timeout(
            connect=10.0,
            read=model_def.default_timeout,
            write=30.0,
            pool=10.0,
        )

        response = await client.post(
            backend_url,
            json=body,
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

        log.info(
            "routing_rerank_request",
            model_id=model_id,
            backend=model_def.backend,
            port=model_def.port,
        )

        filtered_headers = _filtered_forward_headers(request)
        client = request.app.state.http_client
        timeout = httpx.Timeout(
            connect=10.0,
            read=model_def.default_timeout,
            write=30.0,
            pool=10.0,
        )

        response = await client.post(
            backend_url,
            json=body,
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
        timeout = httpx.Timeout(
            connect=10.0,
            read=model_def.default_timeout,
            write=30.0,
            pool=10.0
        )

        try:
            # Try /v1/responses first
            response = await client.post(
                backend_url,
                json=body,
                headers=filtered_headers,
                timeout=timeout
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
                        headers=_filter_response_headers(response.headers),
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
            message="/v1/responses not supported or invalid format, converting to /v1/chat/completions"
        )

        # Convert request format
        chat_body = _convert_responses_to_chat(body)
        if getattr(model_def, "chat_template_kwargs", None) and "chat_template_kwargs" not in chat_body:
            chat_body["chat_template_kwargs"] = model_def.chat_template_kwargs
        fallback_url = _get_backend_url(model_def, "/v1/chat/completions")

        response = await client.post(
            fallback_url,
            json=chat_body,
            headers=filtered_headers,
            timeout=timeout
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
                headers=_filter_response_headers(response.headers),
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
