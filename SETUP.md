# Setup Guide

Complete guide for setting up SLM Server for the first time.

## Prerequisites

- **Python 3.12+** - Required for the project
- **uv** - Fast Python package installer ([install uv](https://github.com/astral-sh/uv))
- **nginx** (optional) - For reverse proxy setup

## Initial Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd slm_server
```

### 2. Install Dependencies

```bash
# Install core dependencies
uv sync

# Install backend dependencies (choose what you need)
uv sync --extra mlx        # For MLX backend (Apple Silicon optimized)
uv sync --extra llamacpp   # For llama.cpp backend
# LMStudio must be installed separately as a GUI application
```

### 3. Set Up Configuration Files

The project includes template files that you need to copy and customize:

```bash
# Copy configuration templates
cp config/models.yaml.example config/models.yaml
cp nginx.conf.example nginx.conf
```

### 4. Configure Models

Edit `config/models.yaml` with your model paths and settings:

```yaml
models:
  router:
    id: "liquid/lfm2.5-1.2b"
    backend: "mlx"
    port: 8500
    context_length: 32768
    quantization: "8bit"
    max_concurrency: 4
    default_timeout: 10
    supports_function_calling: false
    model_path: "~/.cache/lm-studio/models/mlx-community/LFM2.5-1.2B-Instruct-8bit"
```

**Model Path Options:**

1. **Local file path**: Use full path to your model directory
   ```yaml
   model_path: "~/.cache/lm-studio/models/lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit"
   ```

2. **Hugging Face model ID**: Automatically downloads on first use
   ```yaml
   model_path: "mlx-community/LFM2.5-1.2B-Instruct-8bit"
   ```

**Important Configuration Fields:**

- `id`: Model identifier used for routing (must match request body)
- `backend`: `mlx`, `llamacpp`, or `lmstudio`
- `port`: Unique port number for this model (default: 8500-8503)
- `model_path`: Path to model or Hugging Face model ID
- `context_length`: Maximum context length (optional)
- `quantization`: Quantization level (`4bit`, `8bit`, etc.)
- `max_concurrency`: Maximum concurrent requests
- `default_timeout`: Request timeout in seconds

### 5. Configure Nginx (Optional)

If you want to use nginx as a reverse proxy:

1. Edit `nginx.conf` and update the log paths:
   ```nginx
   access_log    /path/to/slm_server/logs/nginx_access.log;
   error_log     /path/to/slm_server/logs/nginx_error.log;
   ```

2. Replace `/path/to/slm_server` with your actual project path

3. Ensure the `logs/` directory exists:
   ```bash
   mkdir -p logs
   ```

### 6. Download Models

**Option A: Using LM Studio (Recommended)**

1. Install [LM Studio](https://lmstudio.ai/)
2. Download models through the LM Studio GUI
3. Note the path where models are saved (typically `~/.cache/lm-studio/models/`)
4. Use that path in `config/models.yaml`

**Option B: Using Hugging Face Model IDs**

1. Use a Hugging Face model ID in `model_path`:
   ```yaml
   model_path: "mlx-community/LFM2.5-1.2B-Instruct-8bit"
   ```
2. The model will be automatically downloaded on first use to `~/.cache/huggingface/hub/`

**Option C: Manual Download**

1. Download models manually from Hugging Face or other sources
2. Place them in a directory
3. Use the full path in `config/models.yaml`

### 7. Start the Server

**Option 1: Using the start script (Recommended)**

```bash
./start.sh
```

This script:
- Starts all backend model servers
- Starts the routing service
- Verifies all services are running
- Handles cleanup on exit

**Option 2: Manual startup**

```bash
# Terminal 1: Start backend servers
uv run python -m slm_server backends

# Terminal 2: Start routing service
uv run python -m slm_server router

# Terminal 3: Start nginx (optional, requires sudo)
sudo nginx -c $(pwd)/nginx.conf
```

### 8. Verify Installation

Check that all services are running:

```bash
# Check router health
curl http://localhost:8001/health

# Check backend health
curl http://localhost:8001/v1/backends/health | jq

# List available models
curl http://localhost:8001/v1/models | jq
```

## Environment Variables

You can customize behavior with environment variables:

- `LMSTUDIO_CACHE`: Override LM Studio cache path (default: `~/.cache/lm-studio/models`)

Example:
```bash
export LMSTUDIO_CACHE="/custom/path/to/models"
uv run python -m slm_server backends
```

## Troubleshooting

### Models Not Found

1. Verify model paths in `config/models.yaml` are correct
2. Check that model files exist at the specified paths
3. For Hugging Face models, check `~/.cache/huggingface/hub/`
4. Review config validation warnings on startup

### Port Conflicts

1. Check which ports are in use:
   ```bash
   lsof -i :8500
   ```
2. Change port numbers in `config/models.yaml`
3. Ensure each model has a unique port

### Backend Not Starting

1. Check backend health:
   ```bash
   curl http://localhost:8001/v1/backends/health | jq
   ```
2. Verify backend dependencies are installed:
   ```bash
   uv sync --extra mlx        # For MLX
   uv sync --extra llamacpp   # For llama.cpp
   ```
3. Check logs for error messages
4. Ensure model paths are correct and files exist

### Configuration Validation Errors

The server validates configuration on startup and will warn about:
- Missing model paths
- Backend/format mismatches (e.g., MLX with GGUF files)
- Port conflicts

Review the warnings and fix the issues in `config/models.yaml`.

## Next Steps

- Read the [README.md](README.md) for API documentation
- Check [docs/PORT_AND_ROUTING.md](docs/PORT_AND_ROUTING.md) for routing details
- Test the API with a simple request:
  ```bash
  curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "liquid/lfm2.5-1.2b", "messages": [{"role": "user", "content": "Hello!"}]}'
  ```

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the `/v1/backends/health` endpoint for backend status
3. Check server logs for detailed error messages
4. Verify your configuration matches the examples in `config/models.yaml.example`
