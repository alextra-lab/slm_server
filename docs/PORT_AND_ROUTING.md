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

### Option 2: Single Port with Model Selection (Requires Nginx/Reverse Proxy)

Use one port (e.g., 1234) with a reverse proxy that routes based on model ID:

```
Client Request → Nginx (port 1234) → Route by model ID → Backend Server
```

**Pros:**
- Single endpoint for agent
- Cleaner configuration
- Can load balance between instances

**Cons:**
- Requires nginx or similar reverse proxy
- More complex setup
- Still need separate backend servers (just hidden behind proxy)

## For Benchmarking/A/B Testing

**You typically want separate ports** to:
- Run multiple backends in parallel (MLX on 8500, llama.cpp on 8501, LMStudio on 8502)
- Compare performance side-by-side
- Test different configurations simultaneously

## Recommendation

**For benchmarking: Use separate ports** (current approach)
- Start each backend on its own port
- Configure `models.yaml` with per-model endpoints
- No nginx needed for simple A/B testing

**For production: Consider nginx** if you want:
- Single endpoint (`http://localhost:8000/v1`)
- Automatic routing based on model ID in request
- Load balancing between multiple instances
- Request caching

The current setup uses:
- Port 8000: Nginx (reverse proxy, optional)
- Port 8001: Routing service (FastAPI)
- Ports 8500-8503: Backend model servers (configurable per model)

## Example: Starting Multiple Models for Benchmarking

You can use the benchmark tool to start models with different backends:

```bash
# Terminal 1: Router with MLX
uv run python -m slm_server.benchmark_models start --backend mlx --model router --port 8500

# Terminal 2: Router with llama.cpp (same model, different backend)
uv run python -m slm_server.benchmark_models start --backend llamacpp --model router --port 8501

# Terminal 3: Router with LMStudio
uv run python -m slm_server.benchmark_models start --backend lmstudio --model router --port 8502
```

Or use the standard startup process which starts all configured models:

```bash
# Start all backends configured in models.yaml
uv run python -m slm_server backends
```

Then configure `models.yaml` to point to the appropriate port for testing.

## Nginx Setup (Optional, for Production)

If you want a single endpoint, you'd configure nginx to route by model ID:

```nginx
location /v1/chat/completions {
    # Route based on model parameter in request body
    # This requires custom nginx logic or Lua scripting
}
```

But for benchmarking, **separate ports are simpler and more flexible**.
