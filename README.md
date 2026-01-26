# SLM Server

Unified LLM server with nginx reverse proxy and intelligent routing based on model ID.

## Architecture

```
Client Request (port 8000)
    ↓
Nginx (reverse proxy)
    ↓
Routing Service (FastAPI, port 8001)
    ↓ (reads model ID from request body)
Backend Model Servers (MLX/llama.cpp/LMStudio on ports 1234, 8002, 8003, ...)
```

## Features

- **Single endpoint**: All requests go to `http://localhost:8000/v1`
- **Model-based routing**: Extracts model ID from request body and routes to correct backend
- **Multi-backend support**: MLX, llama.cpp, and LMStudio
- **Port per model**: Each model runs on its own port (configured in `config/models.yaml`)
- **Nginx integration**: Optional nginx for SSL termination, load balancing, and static file serving
- **OpenAI-compatible**: Full compatibility with OpenAI API format

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
# LMStudio must be installed separately
```

3. **Set up configuration files**:
```bash
# Copy configuration templates
cp config/models.yaml.example config/models.yaml
cp nginx.conf.example nginx.conf

# Edit config/models.yaml with your model paths
# Edit nginx.conf with your project path (if using nginx)
```

4. **Configure models** (edit `config/models.yaml`):
```yaml
models:
  router:
    id: "qwen/qwen3-1.7b"
    backend: "mlx"  # or "llamacpp" or "lmstudio"
    port: 8500
    context_length: 32768
    quantization: "8bit"
    max_concurrency: 4
    default_timeout: 10
    supports_function_calling: false
    model_path: "~/.cache/lm-studio/models/mlx-community/LFM2.5-1.2B-Instruct-8bit"
    # Or use Hugging Face model ID: "mlx-community/LFM2.5-1.2B-Instruct-8bit"
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

# Terminal 3: Start nginx (optional, requires sudo)
sudo nginx -c $(pwd)/nginx.conf
```

## Configuration

See `config/models.yaml.example` for a template. Copy it to `config/models.yaml` and configure your models.

Each model needs:
- `id`: Model identifier (used for routing - must match what's in request body)
- `backend`: Backend type (`mlx`, `llamacpp`, or `lmstudio`)
- `port`: Port number for this model's server (must be unique per model)
- `context_length`, `quantization`, `max_concurrency`, `default_timeout`: Model-specific settings
- `model_path`: Path to model file or Hugging Face model ID

### Model Path Options

You can specify models in two ways:

1. **Local file path**: Full path to model directory or file
   ```yaml
   model_path: "~/.cache/lm-studio/models/lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit"
   ```

2. **Hugging Face model ID**: Automatically downloads on first use
   ```yaml
   model_path: "mlx-community/LFM2.5-1.2B-Instruct-8bit"
   ```

### Model Discovery

The server automatically searches for models in:
- `~/.cache/lm-studio/models/` (LMStudio cache)
- Custom paths specified via `model_path` in config
- Hugging Face cache (`~/.cache/huggingface/hub/`) when using model IDs

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

### `GET /v1/models`
List all available models and their configurations.

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
2. **Nginx** (port 8000) forwards to routing service (port 8001)
3. **Routing service** extracts `model` field from JSON body
4. **Router** looks up model in `config/models.yaml` to find backend and port
5. **Router** forwards request to correct backend server (e.g., `http://localhost:8500/v1/chat/completions`)
6. **Backend** processes request and returns response
7. **Router** forwards response back to client

## Port Management

Each model **must** have its own unique port. The router uses the port from `config/models.yaml` to route requests. Example:

- `router` model → port 8500 (MLX)
- `standard` model → port 8501 (MLX)
- `reasoning` model → port 8502 (MLX)
- `coding` model → port 8503 (MLX)

The routing service runs on port 8001, and nginx (if used) runs on port 8000.

## Backend Requirements

### MLX
- Install: `uv sync --extra mlx`
- Requires: `mlx-openai-server` command
- Auto-discovers models from LMStudio cache

### llama.cpp
- Install: `uv sync --extra llamacpp`
- Requires: `llama-cpp-python[server]` package
- Auto-discovers `.gguf` files from LMStudio cache

### LMStudio
- Must be installed separately (GUI application)
- Models must be started manually in LMStudio GUI
- Configure port in LMStudio to match `config/models.yaml`

## Troubleshooting

### Check Backend Health
First, check if all backends are running:
```bash
curl http://localhost:8001/v1/backends/health | jq
```

This will show the status of each backend server and help identify issues.

### Model not found
- Check that model ID in request matches `id` field in `config/models.yaml`
- Verify model file exists in LMStudio cache or set `model_path` explicitly
- Check config validation warnings on startup

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
- For LMStudio, ensure it's running and configured with correct port

### Connection refused / Backend unreachable
- Backend server may not be running
- Check `/v1/backends/health` to verify backend status
- Restart backend servers: `uv run python -m slm_server backends`
- Verify firewall isn't blocking local connections
