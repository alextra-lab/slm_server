# Port Management and Model Routing

## Architecture

Each model server instance requires its own port. The routing service (port 8000) reads the `model` field from each request and forwards it to the correct backend.

```
Client Request (port 8000)
    ↓
Routing Service (FastAPI, port 8000)
    ↓ (reads model ID from request body)
Backend Model Servers (MLX/llama.cpp on configured ports)
```

- Port 8000: Routing service (FastAPI)
- Backend ports: Configured per model in `config/models.yaml` (e.g. 8501, 8502, 8503, ...)

## Port Assignment

Each model role gets its own port:

```yaml
# config/models.yaml
models:
  standard:
    id: "qwen/qwen3-4b-2507"
    backend: "mlx"
    port: 8501
  reasoning:
    id: "qwen/qwen3-8b"
    backend: "mlx"
    port: 8502
  coding:
    id: "mistralai/devstral-small-2-2512"
    backend: "mlx"
    port: 8503
```

Clients send all requests to port 8000 using the model ID — the router handles dispatch.

## Endpoints

All routing goes through port 8000:

- `POST /v1/chat/completions` — routed by `model` field
- `POST /v1/responses` — routed by `model` field, with automatic fallback to `/v1/chat/completions` if the backend returns 404
- `POST /v1/embeddings` — routed by `model` field; backend must have `model_type: embeddings`
- `POST /v1/rerank` — routed by `model` field; backend must have `model_type: rerank` and use native `llama-server`

## Embeddings

The client sends `model` in the JSON body; the router forwards to `http://localhost:{backend_port}/v1/embeddings`. Embedding models require `backend: llamacpp` and `model_type: embeddings`.

## Rerank

`POST /v1/rerank` forwards to `http://localhost:{backend_port}/v1/rerank`. Rerank backends require `backend: llamacpp`, `model_type: rerank`, and native `llama-server` on PATH (the Python `llama_cpp.server` does not expose `/v1/rerank`).

## Running Multiple Models for Benchmarking

To compare different backends or configurations, assign each to a unique port in `models.yaml` and enable all of them. The router dispatches based on the model ID in the request body, so you can test different model IDs in parallel without any additional infrastructure.

```bash
# Start all backends configured in models.yaml
uv run python -m slm_server backends
```
