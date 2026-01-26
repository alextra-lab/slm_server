# Setup Guide

Complete guide for setting up SLM Server for the first time.

## Prerequisites

- **macOS with Apple Silicon** (M1, M2, M3, or later) - Recommended for MLX backend
- **Python 3.12+** - Required for the project
- **uv** - Fast Python package installer ([install uv](https://github.com/astral-sh/uv))

> **Note:** While the llama.cpp backend works cross-platform, the MLX backend (recommended) requires Apple Silicon for optimal performance. This project is primarily designed for Apple Silicon Macs.

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
```

### 3. Set Up Configuration Files

The project includes template files that you need to copy and customize:

```bash
# Copy configuration template
cp config/models.yaml.example config/models.yaml
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
    model_path: "mlx-community/LFM2.5-1.2B-Instruct-8bit"  # Hugging Face model ID
```

**Model Path Options:**

1. **Hugging Face model ID** (recommended): Automatically downloads on first use
   ```yaml
   model_path: "mlx-community/LFM2.5-1.2B-Instruct-8bit"
   ```

2. **Local file path**: Use full path to your model directory
   ```yaml
   model_path: "~/.cache/huggingface/hub/models--mlx-community--LFM2.5-1.2B-Instruct-8bit"
   # Or any local directory containing the model
   ```

**Important Configuration Fields:**

- `id`: Model identifier used for routing (must match request body)
- `backend`: `mlx` or `llamacpp`
- `port`: Unique port number for this model (default: 8500-8503)
- `model_path`: Path to model or Hugging Face model ID
- `context_length`: Maximum context length (optional)
- `quantization`: Quantization level (`4bit`, `8bit`, etc.)
- `max_concurrency`: Maximum concurrent requests
- `default_timeout`: Request timeout in seconds

### 5. Download Models

**Option A: Using Hugging Face Model IDs (Recommended)**

The easiest way is to use Hugging Face model IDs - models are automatically downloaded on first use:

1. Use a Hugging Face model ID in `model_path`:
   ```yaml
   model_path: "mlx-community/LFM2.5-1.2B-Instruct-8bit"
   ```
2. The model will be automatically downloaded on first use to `~/.cache/huggingface/hub/`
3. Ensure you have internet access and sufficient disk space

**Option B: Manual Download from Hugging Face**

1. Download models manually from [Hugging Face](https://huggingface.co/)
2. Place them in a directory (e.g., `~/.cache/huggingface/hub/` or a custom location)
3. Use the full path in `config/models.yaml`:
   ```yaml
   model_path: "~/.cache/huggingface/hub/models--mlx-community--LFM2.5-1.2B-Instruct-8bit"
   ```

**Note:** The server works with any local model directory. You can download models from Hugging Face or use any other method to obtain model files.

### 6. Start the Server

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
```

### 7. Verify Installation

Check that all services are running:

```bash
# Check router health
curl http://localhost:8000/health

# Check backend health
curl http://localhost:8000/v1/backends/health | jq

# List available models
curl http://localhost:8000/v1/models | jq
```

## Environment Variables

You can customize behavior with environment variables:

- `MODEL_CACHE`: Override default model cache path for benchmark tool (default: `~/.cache/huggingface/hub`)

Example:
```bash
export MODEL_CACHE="/custom/path/to/models"
uv run python -m slm_server.benchmark_models check --backend mlx
```

## Troubleshooting

### Models Not Found

1. Verify model paths in `config/models.yaml` are correct
2. Check that model files exist at the specified paths
3. For Hugging Face models:
   - Ensure you have internet access for the first download
   - Check `~/.cache/huggingface/hub/` after download
   - Verify sufficient disk space
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
   curl http://localhost:8000/v1/backends/health | jq
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
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "liquid/lfm2.5-1.2b", "messages": [{"role": "user", "content": "Hello!"}]}'
  ```

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the `/v1/backends/health` endpoint for backend status
3. Check server logs for detailed error messages
4. Verify your configuration matches the examples in `config/models.yaml.example`
