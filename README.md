# SLM Server

Unified LLM server with intelligent routing based on model ID. Optimized for Apple Silicon (M1/M2/M3) with MLX backend support.

## Architecture

```
Client Request (port 8000)
    ↓
Routing Service (FastAPI, port 8000)
    ↓ (reads model ID from request body)
Backend Model Servers (MLX/llama.cpp on ports 8500, 8501, 8502, ...)
```

## Features

- **Apple Silicon optimized**: MLX backend leverages Apple's Metal Performance Shaders for fast inference on M1/M2/M3 chips
- **Single endpoint**: All requests go to `http://localhost:8000/v1`
- **Model-based routing**: Extracts model ID from request body and routes to correct backend
- **Multi-backend support**: MLX (Apple Silicon) and llama.cpp (cross-platform)
- **Port per model**: Each model runs on its own port (configured in `config/models.yaml`)
- **OpenAI-compatible**: Full compatibility with OpenAI API format

## Requirements

- **macOS with Apple Silicon** (M1, M2, M3, or later) - Recommended for MLX backend
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

> **Note:** While llama.cpp backend works cross-platform, the MLX backend (recommended) requires Apple Silicon for optimal performance.

## Quick Start

1. **Clone and install dependencies**:
```bash
git clone <repository-url>
cd slm_server
uv sync
```

2. **Install backend dependencies** (optional, only what you need):
```bash
uv sync --extra mlx        # For MLX backend
uv sync --extra llamacpp   # For llama.cpp backend
```

3. **Set up configuration files**:
```bash
# Copy configuration template
cp config/models.yaml.example config/models.yaml

# Edit config/models.yaml with your model paths
```

4. **Configure models** (edit `config/models.yaml`):
```yaml
models:
  router:
    id: "qwen/qwen3-1.7b"
    backend: "mlx"  # or "llamacpp"
    port: 8500
    context_length: 32768
    quantization: "8bit"
    max_concurrency: 4
    default_timeout: 10
    supports_function_calling: false
    model_path: "mlx-community/LFM2.5-1.2B-Instruct-8bit"  # Hugging Face model ID (auto-downloads)
    # Or use local path: "~/.cache/huggingface/hub/models--mlx-community--LFM2.5-1.2B-Instruct-8bit"
```

5. **Start all services** (recommended):
```bash
./start.sh
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

Or start individually:

```bash
# Terminal 1: Start backend servers
uv run python -m slm_server backends

# Terminal 2: Start routing service
uv run python -m slm_server router
```

## Configuration

See `config/models.yaml.example` for a template. Copy it to `config/models.yaml` and configure your models.

Each model needs:
- `id`: Model identifier (used for routing - must match what's in request body)
- `backend`: Backend type (`mlx` or `llamacpp`)
- `port`: Port number for this model's server (must be unique per model)
- `context_length`, `quantization`, `max_concurrency`, `default_timeout`: Model-specific settings
- `model_path`: Path to model file or Hugging Face model ID

### Model Path Options

You can specify models in two ways:

1. **Hugging Face model ID** (recommended): Automatically downloads on first use
   ```yaml
   model_path: "mlx-community/LFM2.5-1.2B-Instruct-8bit"
   ```

2. **Local file path**: Full path to model directory or file
   ```yaml
   model_path: "~/.cache/huggingface/hub/models--mlx-community--LFM2.5-1.2B-Instruct-8bit"
   # Or any local directory containing the model
   ```

### Model Discovery

Models can be specified in two ways:

1. **Hugging Face model ID**: The server automatically downloads models from Hugging Face on first use. Models are cached in `~/.cache/huggingface/hub/`.

2. **Local file path**: Point directly to a model directory or file on your system. You can download models manually from Hugging Face or use any local directory.

## API

The routing service exposes OpenAI-compatible endpoints:

### `POST /v1/chat/completions`
Standard chat completions endpoint. Request body must include `model` field:
```json
{
  "model": "qwen/qwen3-1.7b",
  "messages": [{"role": "user", "content": "Hello!"}]
}
```

### `POST /v1/responses`
Responses API endpoint with automatic fallback. If the backend doesn't support `/v1/responses` (returns 404), the router automatically converts the request to `/v1/chat/completions` format. This provides compatibility with MLX and llama.cpp backends while maintaining LM Studio compatibility.

### `POST /v1/embeddings`
OpenAI-compatible embeddings. The router resolves `model` the same way as chat completions and forwards to the backend’s `/v1/embeddings`.

**GGUF text embedding models (llama.cpp):** In `config/models.yaml`, set `backend: llamacpp`, `model_type: embeddings`, and `model_path` to your `.gguf` file or a directory containing a single `.gguf`. The launcher adds native `llama-server --embedding` or `python -m llama_cpp.server --embedding true` as appropriate.

Example request:
```json
{
  "model": "nomic-ai/nomic-embed-text-v1.5",
  "input": "Hello, world"
}
```

Example:
```bash
curl -s http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-ai/nomic-embed-text-v1.5","input":"test"}' | jq
```

**MLX:** You can also run embedding models with `backend: mlx` and `model_type: embeddings` (via `mlx-openai-server`); use a non-GGUF MLX model path or Hugging Face id.

### `POST /v1/rerank`
Reranking API (llama.cpp). The router resolves `model` like other routes and forwards to the backend’s `/v1/rerank`.

**llama.cpp only:** Set `backend: llamacpp`, `model_type: rerank`, and a local GGUF path. Startup uses **native `llama-server`** with `--embedding`, `--pooling rank`, and `--reranking`. `llama-server` must be on your PATH (for example `brew install llama.cpp`). The Python `llama_cpp.server` path is not used for rerank.

Request shape follows [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (query + documents). **MLX / `mlx-openai-server` does not support rerank** in this project.

### `GET /v1/models`
List all available models and their configurations (includes `model_type`, e.g. `embeddings`, `rerank`, or `lm`).

### `GET /v1/backends/health`
Check health status of all configured backend servers. Returns status for each model:
- `healthy`: Backend responding normally
- `unreachable`: Backend not running (connection refused)
- `timeout`: Backend not responding
- `disabled`: Model disabled in config

Example response:
```json
{
  "router": {
    "status": "healthy",
    "model_id": "qwen/qwen3-1.7b",
    "backend": "mlx",
    "port": 8500
  },
  "standard": {
    "status": "unreachable",
    "error": "Connection refused - backend not running"
  }
}
```

### `GET /health`
Health check endpoint for the router itself.

## How It Works

1. **Client sends request** to `http://localhost:8000/v1/chat/completions` with model ID in body
2. **Routing service** (port 8000) extracts `model` field from JSON body
3. **Router** looks up model in `config/models.yaml` to find backend and port
4. **Router** forwards request to correct backend server (e.g., `http://localhost:8500/v1/chat/completions`)
5. **Backend** processes request and returns response
6. **Router** forwards response back to client

## Port Management

Each model **must** have its own unique port. The router uses the port from `config/models.yaml` to route requests. Example:

- `router` model → port 8500 (MLX)
- `standard` model → port 8501 (MLX)
- `reasoning` model → port 8502 (MLX)
- `coding` model → port 8503 (MLX)

The routing service runs on port 8000.

## Backend Requirements

### MLX (Recommended for Apple Silicon)
- Install: `uv sync --extra mlx`
- Requires: `mlx-openai-server` command
- Supports Hugging Face model IDs (auto-downloads) or local model paths
- **Optimized for Apple Silicon** - Uses Metal Performance Shaders for GPU acceleration
- Best performance on M1/M2/M3 Macs

### llama.cpp
- Install: `uv sync --extra llamacpp`
- Requires: `llama-cpp-python[server]` package
- Supports Hugging Face model IDs (auto-downloads) or local `.gguf` files
- Cross-platform support

## Troubleshooting

### Check Backend Health
First, check if all backends are running:
```bash
curl http://localhost:8000/v1/backends/health | jq
```

This will show the status of each backend server and help identify issues.

### Model not found
- Check that model ID in request matches `id` field in `config/models.yaml`
- Verify model file exists at the specified path, or use a Hugging Face model ID for auto-download
- Check config validation warnings on startup
- For Hugging Face models, ensure you have internet access for the first download

### Port already in use
- Each model needs a unique port
- Check what's using the port: `lsof -i :8500`
- Change port in `config/models.yaml`
- Config validation will warn about port conflicts on startup

### Backend server not starting
- Check `/v1/backends/health` endpoint to see which backends are down
- Verify backend dependencies installed: `uv sync --extra mlx` or `uv sync --extra llamacpp`
- Check logs for error messages
- Ensure model paths are correct and files exist
- For Hugging Face models, ensure you have internet access and sufficient disk space

### Connection refused / Backend unreachable
- Backend server may not be running
- Check `/v1/backends/health` to verify backend status
- Restart backend servers: `uv run python -m slm_server backends`
- Verify firewall isn't blocking local connections
