# Setup Guide

## Prerequisites

- macOS with Apple Silicon (M1, M2, M3, M4) — required for MLX; llama.cpp works cross-platform
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

## 1. Clone the Repository

```bash
git clone <repository-url>
cd slm_server
```

## 2. Install Dependencies

```bash
# Install core dependencies
uv sync

# Install backend dependencies (choose what you need)
uv sync --extra mlx        # MLX backend (Apple Silicon)
uv sync --extra llamacpp   # llama.cpp Python server (fallback)
```

### llama.cpp: native server vs Python package

The server **prefers the native `llama-server` binary** when it is on your PATH. Install it with:

```bash
brew install llama.cpp
```

The native binary is required for:
- `model_type: rerank`
- Models with newer architectures (e.g. Qwen3.5) not yet supported by the PyPI build of `llama-cpp-python`
- `kv_unified` and `fit` config flags

If you see an error like:

```
unknown model architecture: 'qwen35'
```

install the native server so the server can use it automatically, or build `llama-cpp-python` from source:

```bash
# Apple Silicon (Metal); use -DGGML_CUDA=on for NVIDIA
CMAKE_ARGS="-DGGML_METAL=on" uv pip install "llama-cpp-python[server] @ git+https://github.com/abetlen/llama-cpp-python.git@main" --no-cache-dir --reinstall
```

Then run `uv sync --extra llamacpp` to keep the rest of the project in sync.

## 3. Configure Models

```bash
cp config/models.yaml.example config/models.yaml
```

Edit `config/models.yaml` with your model paths and settings. See the [README](README.md) for a full description of all configuration fields.

### Example entry

```yaml
models:
  standard:
    id: "qwen/qwen3-4b-2507"
    backend: "mlx"
    port: 8501
    context_length: 40960
    quantization: "8bit"
    max_concurrency: 2
    default_timeout: 45
    enable_auto_tool_choice: true
    tool_call_parser: "qwen3"
    reasoning_parser: "qwen3"
    model_path: "mlx-community/Qwen3-4b-Instruct-2507-MLX-8bit"
```

### Model path formats

- **Hugging Face model ID** (MLX only): downloaded automatically on first use
  ```yaml
  model_path: "mlx-community/Qwen3-4b-Instruct-2507-MLX-8bit"
  ```

- **Local path**: directory or file
  ```yaml
  model_path: "/path/to/models/Qwen3.5-9B-GGUF"
  ```

For llamacpp with a directory, the server picks the first `.gguf` file found. Hugging Face IDs are not supported for llamacpp.

## 4. Start the Server

**Using the start script:**

```bash
./start.sh
```

This starts all enabled backend servers, waits for them to bind to their ports, then starts the routing service on port 8000.

**Manual startup:**

```bash
# Terminal 1: backend servers
uv run python -m slm_server backends

# Terminal 2: routing service
uv run python -m slm_server router
```

## 5. Verify

```bash
# Router health
curl http://localhost:8000/health

# Backend health
curl http://localhost:8000/v1/backends/health | jq

# List configured models
curl http://localhost:8000/v1/models | jq

# Test a request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen/qwen3-4b-2507", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Environment Variables

- `MODEL_CACHE`: Override the default model cache path for the benchmark tool (default: `~/.cache/huggingface/hub`)

## Troubleshooting

### Models not found

- Verify `model_path` in `config/models.yaml` is correct
- For HF model IDs (MLX): ensure internet access on first run
- Config validation warnings are printed on startup

### Port conflicts

```bash
lsof -i :8501
```

Each model needs a unique port. Change port numbers in `config/models.yaml`.

### Backend not starting

```bash
curl http://localhost:8000/v1/backends/health | jq
```

- Verify `llama-server` is on PATH for llamacpp models: `which llama-server`
- Check that model files exist at the configured paths
- Review startup logs for error messages

### Configuration validation errors

The server validates config on startup and warns about:
- Missing model paths
- Backend/format mismatches (e.g. MLX with GGUF files)
- Port conflicts
- Invalid options for the selected model type (e.g. `tool_call_parser` on an `embeddings` model)
