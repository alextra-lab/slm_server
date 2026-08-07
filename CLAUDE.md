# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
uv sync                          # core deps
uv sync --extra mlx              # + MLX backend (Apple Silicon only)
uv sync --extra llamacpp         # + llama-cpp-python server
brew install llama.cpp           # native llama-server (required for rerank, multimodal, newer architectures)

./start.sh                       # start backends then router (port 8000)
./stop.sh                        # kill all backend and router processes

# Manual startup (two terminals):
uv run python -m slm_server backends
uv run python -m slm_server router
```

## Testing & Linting

```bash
uv run pytest                    # all tests
uv run pytest tests/test_router_chat.py            # single file
uv run pytest tests/test_router_chat.py::test_name # single test
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run mypy src/                 # type check
```

Pytest config: `pythonpath = ["src"]`, `asyncio_mode = "auto"`. Tests use monkeypatching (no unittest.mock). Router tests use FastAPI `TestClient` with faked `httpx` responses. Backend tests call command-building functions directly with `tmp_path` for fake model files.

## Architecture

SLM Server is a local LLM inference gateway for Apple Silicon. It supports two backends: **llama.cpp** (primary, via native `llama-server` binary or the `llama-cpp-python` fallback) and **MLX** (via `mlx-openai-server`). A FastAPI router on port 8000 proxies OpenAI-compatible API requests to backend model servers running on separate ports, routing by the `model` field in each request body.

```
Client → Router (FastAPI :8000) → Backend servers (llama.cpp/MLX :8501, :8502, ...)
```

### Source modules (`src/slm_server/`)

- **`__main__.py`** — CLI entry point: `python -m slm_server [router|backends|mlx-rerank]`
- **`config.py`** — `ModelDefinition` (Pydantic) and `ModelConfig` schema; YAML loading from `config/models.yaml`; startup validation (port conflicts, backend/format mismatches, path existence)
- **`mlx_rerank_server.py`** — In-repo FastAPI rerank server for `backend: mlx-rerank`. Loads an MLX Qwen3-Reranker model via `mlx_lm` and scores query/document pairs with the official yes/no-logit method; exposes `/v1/rerank` (llama-server-compatible shape, results sorted descending) and `/health`. Scoring is pure functions (`build_rerank_prompt`, `relevance_score`) + an injectable scorer for testability.
- **`router.py`** — FastAPI app with shared `httpx.AsyncClient` (connection pooling). Endpoints: `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/responses` (auto-fallback to chat/completions on 404), `/v1/models`, `/v1/backends/health`, `/v1/telemetry/effective-config`, `/health`. Errors returned in OpenAI format with `slm_server_debug` metadata.
- **`telemetry.py`** — OTLP span export to the OpenTelemetry Collector (FRE-1071, ADR-0129 D5/D8). One span per completed request, from four emit sites distinguished by a `slm.emit_path` attribute: chat, responses, rerank, streaming. Continues an inbound W3C `traceparent` so spans join the caller's trace. Fail-soft by contract — `emit_request_span` swallows everything and the network hop runs on the batch processor's thread. The tracer provider is installed from the router lifespan, never lazily on a request. See README § Telemetry.
- **`start_backends.py`** — Builds CLI commands for each backend type and launches them as subprocesses. Four command builders: `build_llama_native_command()` (preferred — native `llama-server` binary), `build_llamacpp_command()` (fallback Python server, no rerank/multimodal support), `build_mlx_command()` (`mlx-openai-server`), and `build_mlx_rerank_command()` (launches `python -m slm_server mlx-rerank` for `backend: mlx-rerank`). MLX flag support is detected dynamically via `--help` introspection. MLX backends get up to 3 retries for NSRangeException crashes.
- **`watchdog.py`** — Restarts backends that stop serving (FRE-241). `BackendHealthTracker` counts real-traffic outcomes per port inside the router (4xx never counts; a 300s stall on a streaming request counts on its own); `BackendSupervisor` runs in the `backends` launcher and does the killing and relaunching, bounded at 5 restarts per 10 minutes before abandoning a backend. The two processes are joined by a JSON restart-request file rather than a control socket. Decisions land in `logs/watchdog.jsonl`. See README § Watchdog.
- **`benchmark_models.py`** — Typer CLI for A/B testing individual models. Reuses command builders from `start_backends`.

### Key patterns

- **No class hierarchies** — backend dispatch is conditional (`if model_def.backend == "mlx"` / `"llamacpp"` / `"mlx-rerank"`). Functions, not classes, are the primary abstraction.
- **Input validation for subprocess safety** — parser names, model types, config names, and hosts are validated against explicit allowlists before being passed to subprocess commands. Path traversal is blocked.
- **Streaming** — SSE responses from backends are proxied through via `StreamingResponse`.
- **Per-request timeout** — clients can set a `timeout` field (1–3600s) in the request body; the router consumes it (not forwarded to backend).

## Configuration

Model configuration lives in `config/models.yaml` (gitignored; copy from `config/models.yaml.example`). Each model entry defines: id, backend (`mlx`/`llamacpp`/`mlx-rerank`), port, model_type, model_path, and optional sampling/template/parser settings. See README.md for the full field reference.

Ruff and mypy are configured in `pyproject.toml`. Ruff: line-length 100, py312, rules E/F/I/N/W/UP (E501 ignored). mypy: `disallow_untyped_defs = false`.

## Dependencies

Managed via `pyproject.toml` + `uv.lock`. Python >=3.12,<3.13. Optional extras: `mlx`, `llamacpp`, `dev`. Security-pinned transitive deps in `[tool.uv] constraint-dependencies`.
