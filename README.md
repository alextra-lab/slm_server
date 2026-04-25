# SLM Server

Unified LLM server with model-ID-based routing. Designed for Apple Silicon (M1/M2/M3/M4) using MLX and llama.cpp backends.

## Architecture

```
Client Request (port 8000)
    ↓
Routing Service (FastAPI, port 8000)
    ↓ (reads model ID from request body)
Backend Model Servers (MLX/llama.cpp on ports 8501, 8502, ...)
```

## Requirements

- macOS with Apple Silicon — required for MLX backend; llama.cpp works cross-platform
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- For llama.cpp: `llama-server` on PATH (e.g. `brew install llama.cpp`) — required for rerank and Qwen3.5/newer architectures

## Quick Start

1. **Install dependencies**:
```bash
git clone <repository-url>
cd slm_server
uv sync --extra mlx        # For MLX backend
uv sync --extra llamacpp   # For llama.cpp backend (Python server fallback)
```

2. **Configure models**:
```bash
cp config/models.yaml.example config/models.yaml
# Edit config/models.yaml with your model paths
```

3. **Start all services**:
```bash
./start.sh
```

Or start individually:
```bash
# Terminal 1: Start backend servers
uv run python -m slm_server backends

# Terminal 2: Start routing service
uv run python -m slm_server router
```

For detailed setup, see [SETUP.md](SETUP.md).

## Configuration

Copy `config/models.yaml.example` to `config/models.yaml` and set your model paths. Each model entry maps a role name to a server instance.

### All Configuration Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `id` | yes | — | Model identifier used for routing (must match `model` field in requests) |
| `backend` | yes | — | `mlx` or `llamacpp` |
| `port` | yes | — | Port for this model's backend server (must be unique) |
| `model_path` | yes | — | Local path to model file/directory, or Hugging Face model ID (MLX only for HF IDs) |
| `default_timeout` | yes | — | Request timeout in seconds |
| `quantization` | yes | — | Quantization level (e.g. `8bit`, `Q8_0`, `f16`) — informational for MLX; affects KV cache defaults for llamacpp |
| `model_type` | no | `lm` | `lm`, `multimodal`, `image-generation`, `image-edit`, `embeddings`, `rerank`, or `whisper` |
| `context_length` | no | model default | Maximum context length; omit to use the model's built-in default |
| `max_concurrency` | no | `1` | Maximum concurrent requests |
| `host` | no | `0.0.0.0` | Host the backend server binds to |
| `enabled` | no | `true` | Set to `false` to skip this model on startup |
| `supports_function_calling` | no | `false` | Reported in `/v1/models` response |

**MLX-only fields** (passed to `mlx-openai-server launch`):

| Field | Default | Description |
|-------|---------|-------------|
| `enable_auto_tool_choice` | `false` | Pass `--enable-auto-tool-choice` to mlx-openai-server |
| `tool_call_parser` | `null` | Parser for tool calls. Options: `qwen3`, `glm4_moe`, `qwen3_coder`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `harmony`, `minimax_m2` |
| `reasoning_parser` | `null` | Parser for reasoning/thinking tokens. Options: `qwen3`, `glm4_moe`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `harmony`, `minimax_m2` |
| `config_name` | `flux-schnell` / `flux-kontext-dev` | Config name for `image-generation` or `image-edit` model types |

**llama.cpp-only fields** (passed to `llama-server` or `llama_cpp.server`):

| Field | Default | Description |
|-------|---------|-------------|
| `chat_template_kwargs` | `null` | Dict passed as `--chat-template-kwargs` (e.g. `{enable_thinking: true}` for Qwen3.5) |
| `chat_template_file` | `null` | Optional path passed as `--chat-template-file` to override the GGUF-embedded Jinja template (resolved relative to repo root if not absolute) |
| `temp` | — | Sampling temperature |
| `top_p` | — | Top-p sampling |
| `top_k` | — | Top-k sampling |
| `min_p` | — | Min-p sampling |
| `cache_type_k` | — | KV cache type for K (e.g. `q8_0`, `f16`) |
| `cache_type_v` | — | KV cache type for V (e.g. `q8_0`, `f16`) |
| `flash_attn` | — | Flash attention (`true` / `false`) |
| `kv_unified` | — | Unified KV cache — native `llama-server` only |
| `fit` | — | `--fit` flag — native `llama-server` only |

### Model Path

Two formats are accepted:

- **Hugging Face model ID** (MLX backend only): downloaded automatically on first use
  ```yaml
  model_path: "mlx-community/Qwen3-8B-MLX-8bit"
  ```

- **Local path**: directory containing a `.gguf` (llamacpp) or model files (MLX), or a direct path to a `.gguf` file
  ```yaml
  model_path: "/path/to/models/Qwen3.5-9B-GGUF"
  ```

For llamacpp with a directory, the server picks the first `.gguf` file found (alphabetically). Hugging Face model IDs are not supported for llamacpp — use a local path.

## API

The routing service exposes OpenAI-compatible endpoints on port 8000.

### `POST /v1/chat/completions`

Standard chat completions. The `model` field in the request body selects the backend:

```json
{
  "model": "qwen/qwen3-4b-2507",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

The router also injects `chat_template_kwargs` from config into the request body if set and not already present.
Routing uses only enabled model entries. If the same model ID appears in multiple enabled entries, the router returns `409` so model IDs stay unambiguous.

### `POST /v1/responses`

Responses API with automatic fallback. The router first tries `/v1/responses` on the backend. If the backend returns 404 or 422, it converts the request to `/v1/chat/completions` format and retries.

### `POST /v1/embeddings`

OpenAI-compatible embeddings. Requires a model with `model_type: embeddings` and `backend: llamacpp`. The backend is started with `--embedding` (native `llama-server`) or `--embedding true` (Python server).

```json
{
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "input": "Hello, world"
}
```

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":"test"}' | jq
```

MLX embedding models are also supported: set `backend: mlx` and `model_type: embeddings`.

### `POST /v1/rerank`

Reranking. Requires `model_type: rerank`, `backend: llamacpp`, and native `llama-server` on PATH. The backend is started with `--embedding --pooling rank --reranking`. The Python `llama_cpp.server` does not support rerank.

Request body follows the [llama.cpp server rerank format](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (query + documents).

### `GET /v1/models`

Lists all configured models and their settings (id, backend, port, model_type, context_length, quantization, supports_function_calling).

### `GET /v1/backends/health`

Health status of all configured backends:

```json
{
  "standard": {
    "status": "healthy",
    "model_id": "qwen/qwen3-4b-2507",
    "backend": "mlx",
    "port": 8501
  },
  "reasoning": {
    "status": "unreachable",
    "error": "Connection refused - backend not running"
  }
}
```

Possible statuses: `healthy`, `unreachable`, `timeout`, `unhealthy`, `error`, `disabled`.

### `GET /health`

Router health check.

## Backend Details

### MLX

- Install: `uv sync --extra mlx`
- Requires `mlx-openai-server` command (installed via the extra)
- Accepts Hugging Face model IDs (auto-downloads) or local model directories
- Apple Silicon only

### llama.cpp

- Install: `uv sync --extra llamacpp` (installs `llama-cpp-python[server]` as fallback)
- **Native `llama-server`** (e.g. `brew install llama.cpp`) is used automatically when found on PATH and is required for:
  - `model_type: rerank`
  - Models with newer architectures (Qwen3.5, etc.) not yet supported by the PyPI build
  - `kv_unified` and `fit` flags
- When native `llama-server` is not found, falls back to `python -m llama_cpp.server`
- Requires local `.gguf` files — Hugging Face model IDs are not supported

## Troubleshooting

### Check backend health
```bash
curl http://localhost:8000/v1/backends/health | jq
```

### Model not found
- Verify the `id` in `config/models.yaml` matches the `model` field in your request exactly
- Check that `enabled` is not set to `false`

### Port already in use
```bash
lsof -i :8501
```
Each model must have a unique port. Config validation warns about port conflicts on startup.

### Backend not starting
- Check `/v1/backends/health` to see which backends are down
- Ensure model paths are correct and files exist
- For llamacpp: verify `llama-server` is on PATH (`which llama-server`)
- Check logs for error messages

### "unknown model architecture" error (llamacpp)
The PyPI build of `llama-cpp-python` may not support newer model architectures. Install native `llama-server`:
```bash
brew install llama.cpp
```
The server detects it on PATH and uses it automatically.
