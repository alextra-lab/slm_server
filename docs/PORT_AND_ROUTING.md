# Port Management and Model Routing

## Current Architecture

**Each model server instance requires its own port** because:
- Each server loads one model into memory
- OpenAI-compatible servers typically serve one model per instance
- Port conflicts occur if multiple servers try to use the same port

## Port Assignment Strategy

### Option 1: One Port Per Model (Current Approach)

Each model role gets its own port:

```yaml
# config/models.yaml
models:
  router:
    id: "liquid/lfm2.5-1.2b"
    backend: "mlx"
    port: 8500  # Router on port 8500
  reasoning:
    id: "qwen/qwen3-8b"
    backend: "mlx"
    port: 8502  # Reasoning on port 8502
  coding:
    id: "mistralai/devstral-small-2-2512"
    backend: "mlx"
    port: 8503  # Coding on port 8503
```

**Pros:**
- Simple - no routing needed
- Each model isolated
- Easy to start/stop individual models
- Works well for A/B testing (different backends on different ports)

**Cons:**
- Need to manage multiple ports
- Agent must know which port for which role

### Option 2: Single Port with Model Selection (Current Implementation)

The routing service provides a single endpoint that routes based on model ID:

```
Client Request → Routing Service (port 8000) → Route by model ID → Backend Server
```

**Pros:**
- Single endpoint for clients
- Cleaner configuration
- Automatic routing based on model ID in request body

**Cons:**
- All requests go through the routing service
- Backend servers still need separate ports

## For Benchmarking/A/B Testing

**You typically want separate ports** to:
- Run multiple backends in parallel (MLX on 8500, llama.cpp on 8501)
- Compare performance side-by-side
- Test different configurations simultaneously

## Recommendation

**For benchmarking: Use separate ports** (current approach)
- Start each backend on its own port
- Configure `models.yaml` with per-model endpoints
- No nginx needed for simple A/B testing

**Current setup:**
- Port 8000: Routing service (FastAPI) - single endpoint for all requests
- Ports 8500+ (e.g. 8501–8504 in the default `models.yaml`): Backend model servers (configurable per model)

The routing service automatically routes requests to the correct backend based on the model ID in the request body.

## Embeddings

The same pattern applies to `POST /v1/embeddings` on port **8000**: the client sends `model` in the JSON body; the router forwards to `http://localhost:{backend_port}/v1/embeddings`. Each embedding model still needs its own backend port in `config/models.yaml` (one process per model).

## Rerank

`POST /v1/rerank` on port **8000** forwards to `http://localhost:{backend_port}/v1/rerank`. Rerank backends are **llamacpp + native `llama-server` only** (see README).

## Example: Starting Multiple Models for Benchmarking

You can use the benchmark tool to start models with different backends:

```bash
# Terminal 1: Router with MLX
uv run python -m slm_server.benchmark_models start --backend mlx --model router --port 8500

# Terminal 2: Router with llama.cpp (same model, different backend)
uv run python -m slm_server.benchmark_models start --backend llamacpp --model router --port 8501

# Terminal 3: Another model with different backend
uv run python -m slm_server.benchmark_models start --backend llamacpp --model standard --port 8502
```

Or use the standard startup process which starts all configured models:

```bash
# Start all backends configured in models.yaml
uv run python -m slm_server backends
```

Then configure `models.yaml` to point to the appropriate port for testing.

The routing service handles all model selection automatically - you don't need to configure separate endpoints for each model. Simply send requests to `http://localhost:8000/v1/chat/completions` with the model ID in the request body.
