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
- For llama.cpp: run `./scripts/build_llama.sh`, then set `SLM_LLAMA_SERVER_BIN` in `.env` — required for rerank and Qwen3.5/newer architectures. A `llama-server` on PATH is used as a fallback, but Homebrew's build trails upstream and cannot load newer architectures such as `qwen4exp`.

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
| `backend` | yes | — | `mlx`, `llamacpp`, or `mlx-rerank` (in-repo MLX reranker server) |
| `port` | yes | — | Port for this model's backend server (must be unique) |
| `model_path` | yes | — | Local path to model file/directory, or Hugging Face model ID (MLX only for HF IDs) |
| `default_timeout` | yes | — | Request timeout in seconds |
| `quantization` | yes | — | Quantization level (e.g. `8bit`, `Q8_0`, `f16`) — informational for MLX; affects KV cache defaults for llamacpp |
| `model_type` | no | `lm` | `lm`, `multimodal`, `image-generation`, `image-edit`, `embeddings`, `rerank`, or `whisper` |
| `context_length` | no | model default | Maximum context length; omit to use the model's built-in default |
| `max_concurrency` | no | `1` | Maximum concurrent requests (MLX maps this to the server's supported queue/concurrency flag at runtime) |
| `host` | no | `0.0.0.0` | Host the backend server binds to |
| `enabled` | no | `true` | Set to `false` to skip this model on startup |
| `supports_function_calling` | no | `false` | Reported in `/v1/models` response |

**MLX-only fields** (passed to `mlx-openai-server launch`):

| Field | Default | Description |
|-------|---------|-------------|
| `enable_auto_tool_choice` | `false` | Pass `--enable-auto-tool-choice` to mlx-openai-server |
| `tool_call_parser` | `null` | Parser for tool calls. See current `mlx-openai-server --help` for the full parser list supported by your installed version |
| `reasoning_parser` | `null` | Parser for reasoning/thinking tokens. Set to `null` (or omit) to disable thinking mode |
| `config_name` | `flux-schnell` / `flux-kontext-dev` | Config name for `image-generation` or `image-edit` model types |

**llama.cpp-only fields** (passed to `llama-server` or `llama_cpp.server`):

| Field | Default | Description |
|-------|---------|-------------|
| `chat_template_kwargs` | `null` | Dict passed as `--chat-template-kwargs` (e.g. `{enable_thinking: true}` for Qwen3.5) |
| `chat_template_file` | `null` | Path to a Jinja template file passed as `--jinja --chat-template-file` (overrides the GGUF-embedded template; resolved relative to repo root if not absolute) — native `llama-server` only |
| `temp` | — | Sampling temperature |
| `top_p` | — | Top-p sampling |
| `top_k` | — | Top-k sampling |
| `min_p` | — | Min-p sampling |
| `presence_penalty` | — | Presence penalty (discourages already-seen tokens) |
| `repetition_penalty` | — | Repeat penalty multiplier (`1.0` = disabled) |
| `n_predict` | — | Maximum tokens to generate per request |
| `cache_type_k` | — | KV cache type for K (e.g. `q8_0`, `f16`) |
| `cache_type_v` | — | KV cache type for V (e.g. `q8_0`, `f16`) |
| `cache_ram` | — | Max host context/state cache in MiB (`--cache-ram`; `0` disables, `-1` unlimited) — native `llama-server` only |
| `kv_offload` | — | Offload KV cache to GPU (`true` → `--kv-offload`, `false` → `--no-kv-offload`; default enabled) — native `llama-server` only |
| `flash_attn` | — | Flash attention (`true` / `false`) |
| `kv_unified` | — | Unified KV cache — native `llama-server` only |
| `fit` | — | `--fit` flag — native `llama-server` only |
| `cont_batching` | — | Enable continuous batching (`--cont-batching`) — native `llama-server` only |
| `cache_prompt` | — | Enable/disable prompt prefix caching (`true` → `--cache-prompt`, `false` → `--no-cache-prompt`) — native `llama-server` only |
| `spec_type` | — | Speculative decoding type (e.g. `draft-mtp`) — native `llama-server` only |
| `spec_draft_n_max` | — | Max draft tokens for speculative decoding (e.g. `2`) — native `llama-server` only |
| `spec_model_path` | — | Path to a sidecar draft-head GGUF (native `-md`) — needed when the MTP head ships beside the model rather than inside it, as for Qwen3.8-Flash-Next |
| `verbose` | — | Enable verbose `llama-server` logging (`--verbose`); stderr is redirected to `logs/llama-<id>-<port>.log` — native `llama-server` only |
| `mmproj_path` | `null` | Path to multimodal projector `.gguf` — required when `model_type: multimodal` |

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

### Qwen3.5 MLX Note

For `mlx-community/Qwen3.5-9B-8bit` local checkpoints, use:

```yaml
model_type: "multimodal"
tool_call_parser: "qwen3_coder"
reasoning_parser: null   # disables thinking
```

This model's config uses the Qwen 3.5 multimodal architecture, so `model_type: multimodal` is required.

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

Reranking. Requires `model_type: rerank` with one of two backends:

- **`backend: llamacpp`** — native `llama-server` on PATH (GGUF models). Started with `--embedding --pooling rank --reranking`. The Python `llama_cpp.server` does not support rerank.
- **`backend: mlx-rerank`** — in-repo MLX reranker server (`python -m slm_server mlx-rerank`) for MLX-format Qwen3-Reranker models (e.g. `mxfp8`/`bf16` safetensors). Scores each query/document pair with the official Qwen3-Reranker yes/no-logit method and returns the same response shape as `llama-server`.

Both return results sorted by `relevance_score` descending; an optional `top_n` limits the count. Request body follows the [llama.cpp server rerank format](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (query + documents).

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
- **Native `llama-server`** (built by `./scripts/build_llama.sh`, selected via `SLM_LLAMA_SERVER_BIN`, else found on PATH) is required for:
  - `model_type: rerank`
  - Models with newer architectures (Qwen3.5, etc.) not yet supported by the PyPI build
  - `kv_unified`, `fit`, `cont_batching`, and `cache_prompt` flags
  - `chat_template_file` (with `--jinja` for Jinja template processing)
- When native `llama-server` is not found, falls back to `python -m llama_cpp.server`
- Requires local `.gguf` files — Hugging Face model IDs are not supported

## Watchdog

Backends that stop serving are restarted automatically (FRE-241). The hard case is
not a dead process but a **stuck** one — alive, still holding its port, unable to
produce a token. Neither health path can see that state: `/health` reports on the
router itself, and `/v1/backends/health` only proves a socket answered. So the
watchdog judges backends by the router's own errors on real traffic instead.

A restart is triggered by:

| Signal | Trips after |
|---|---|
| Backend unreachable (503, connection refused) | 2 consecutive |
| Backend timeout (504, 408) | 2 consecutive |
| Backend 5xx | 2 consecutive |
| Backend saturated (429) or too-early (425) | 2 consecutive |
| No first byte on a streaming request for 300s | 1 — see below |
| Backend process exited | immediately |

Statuses are classified explicitly rather than by a `status < 500` threshold,
because some 4xx describe the *backend's condition* and some describe the
*caller's request*:

| Verdict | Statuses | Effect |
|---|---|---|
| health | 2xx, 400, 404, 422 | resets the failure streak |
| failure | 408, 425, 429, all 5xx | counts toward a restart |
| ignore | everything else | neither — an unknown code is not evidence of health |

A threshold got this wrong in a way that mattered. Scoring 429 or 408 as health
did not merely fail to count them: recording health **resets** the consecutive
failure streak, so a backend degrading into 429s could answer them indefinitely
and never trip, while also erasing the record of genuine failures on either side
of it. The third bucket exists for the same reason — an unrecognised status must
not reset a streak either.

That hole was invisible to the telemetry, and could only ever have been
invisible: the router's error paths emitted nothing, so all 798 recorded
requests are status 200 and no 4xx or 5xx had ever been observed at all. The
watchdog's own detection is the first thing that makes those paths visible, so
its classification could not be validated against history — only reasoned about
and then tested. The emit gap was live until this change landed.

The 300s stall threshold is calibrated, not guessed. Across 798 recorded requests,
router-observed time to first byte peaked at 30.4s and backend-reported prefill at
243.2s, so 300s of silence falls outside the observed distribution entirely — which
is what makes a single occurrence conclusive. Stall tracking applies only to
streaming requests; a non-streaming call buffers the whole generation and has no
first byte until it finishes (observed `total_ms` reaches 890.9s).

Restarts are bounded at 5 per 10 minutes per backend. Past that the backend is
abandoned with the reason recorded, so a configuration that cannot start fails
visibly instead of churning forever.

A restart request is also discarded if it predates the target backend's start plus
a 180s grace period. That one rule closes three separate ways to restart a healthy
backend: a request file left behind by a previous run; a trip raised against the
old process while its replacement is still starting; and — the worst of the three
— the minutes a 35B spends loading, during which every request is refused and
would otherwise trip restart after restart until a perfectly good backend hit its
bound and was abandoned.

Two limitations, accepted rather than hidden:

- Stray reaping selects by listening port, not by ownership. An unrelated
  application listening on 8502/8503 on another local interface would be killed
  during a restart. Those ports belong to this project on this machine.
- `check_requests` clears a request file before acting on it, so a launcher that
  dies mid-restart loses that request. The next run starts backends fresh, which
  is the outcome the request was asking for anyway.
- A missing model path is never waited on. Remounting an external volume is a
  manual human action with unbounded latency, so no timer can be right: every
  candidate duration is a guess about when someone will notice, spent looking
  busy while reporting nothing. The absence is recorded, the launch fails at
  once, the bound abandons it, and the launcher exits non-zero.

### Restart log

`logs/watchdog.jsonl` — one JSON object per line, appended:

```json
{"ts":"2026-08-06T09:14:02.117Z","event":"restart_succeeded","port":8502,
 "role":"reasoning","model_id":"unsloth/qwen3.6-35-A3B","reason":"stall",
 "old_pid":35559,"new_pid":41220,"attempt":1}
```

Events: `backend_failure`, `unclassified_status`, `restart_requested`, `stale_request_discarded`,
`backend_exited`, `backend_not_running`, `restart_started`, `restart_succeeded`,
`restart_failed`, `port_stray_killed`, `stray_kill_refused`, `model_path_missing`,
`backend_abandoned`, `supervisor_started`, `supervisor_stopped`.

`supervisor_started` carries every tunable actually in effect, so the running
configuration is answerable from the log alone rather than inferred from when
the process happened to start.

Of the three classification verdicts, `failure` and `ignore` are logged and
`health` is not — health is every successful request and would swamp the file.
That asymmetry is deliberate and it fixes a directional blind spot: previously
only failures were recorded, so a status wrongly classified as health produced
silence, and silence is indistinguishable from nothing having gone wrong. The
log could reveal a restart that should not have happened but never a failure
that should have been counted and was not. `unclassified_status` gives the next
surprising status somewhere to appear.

### Server output

`./start.sh` tees its whole run to `logs/start.out` (rotated past 10MB), while
still printing live to the terminal. Before this, the router's stdout was the
only place backend errors appeared at all and it was persisted nowhere — the
sole copy was the scrollback of whichever terminal started the server, lost on
the next restart. Elasticsearch cannot substitute: it holds 815 documents for
the month to 2026-08-06, of which 814 are status 200, because the error paths
emitted nothing.

```bash
./scripts/slm-logs.sh              # server output, prettified
./scripts/slm-logs.sh watchdog     # restart decisions only
./scripts/slm-logs.sh all          # both, interleaved
./scripts/slm-logs.sh raw          # unformatted
```

Because the router's error paths never emitted telemetry, backend failures were not
recorded anywhere durable before this — this log is the first measurement of how
often a model actually wedges.

### Start at login — withdrawn

There is deliberately no LaunchAgent, plist or installer. Automatic start at
login was specified, built, and then **withdrawn by the owner** — not left
unverified. The distinction matters: unverifiable would mean we could not check
it, withdrawn means it is not wanted. The server is started by hand.

The watchdog's capability is unaffected by this. Only the boot-time apparatus
went; detection, restart and the bound are untouched and proven live.

One thing the removal changed rather than deleted. The launcher used to exit 0
when every backend was abandoned, chosen so a LaunchAgent's `KeepAlive` would
not relaunch it into churn. With launchd gone that rationale evaporated, and
the behaviour it left behind was worse than neutral: the consumer of that exit
code became a human's shell, where exit 0 reports success while no model is
running. It now exits non-zero. The bound is unchanged — attempts still stop
and the reason is still recorded — only the reporting of that outcome changed.

### Tuning

All optional, via environment (`.env` is loaded by `start.sh`):

`SLM_WATCHDOG_ENABLED` · `SLM_WATCHDOG_FAILURE_THRESHOLD` · `SLM_WATCHDOG_STALL_SECONDS` ·
`SLM_WATCHDOG_SWEEP_SECONDS` · `SLM_WATCHDOG_MAX_RESTARTS` · `SLM_WATCHDOG_RESTART_WINDOW_SECONDS` ·
`SLM_WATCHDOG_RESTART_COOLDOWN_SECONDS` · `SLM_WATCHDOG_STARTUP_GRACE_SECONDS` ·
`SLM_WATCHDOG_REQUEST_DIR` · `SLM_WATCHDOG_LOG_PATH`

Set `SLM_WATCHDOG_ENABLED=false` to fall back to the previous behaviour, where
backends are started and never supervised.

## Telemetry

Every completed request becomes one OpenTelemetry span, exported over OTLP/HTTP
to the OpenTelemetry Collector. The Collector is the single egress point for
traces (ADR-0129 D5); nothing here writes to Elasticsearch, and no index name is
formatted client-side any more.

| Variable | Default | Meaning |
|---|---|---|
| `SLM_OTLP_ENDPOINT` | `http://localhost:4318` | Base OTLP/HTTP endpoint of the Collector |
| `SLM_OTEL_ENABLED` | `true` | Set to `false` to export nothing |
| `SLM_OTEL_SERVICE_NAME` | `slm-server` | Published as the `service.name` resource attribute |

**Export is on by default**, unlike the Elasticsearch shipper it replaces, which
did nothing until a URL was set. Telemetry that has to be remembered to switch on
is the failure mode this change exists to end. The cost is that while the
Collector is not running, the router logs a batch-export warning every few
seconds; export is fail-soft, so requests are unaffected either way.

**The caller's trace is continued, not replaced.** A W3C `traceparent` on the
inbound request makes the span a child of the caller's span, inside the caller's
trace — which is what puts SLM work inside the calling turn. A caller that sends
no `traceparent` gets its own trace, and its legacy `X-Trace-Id` rides along as
the `slm.client.trace_id` attribute so the join stays recoverable.

Spans carry `gen_ai.*` semantic conventions (`gen_ai.request.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`) plus a `slm.emit_path`
attribute naming which of the four emit sites produced them — `chat`,
`responses`, `rerank` or `streaming`. Without a distinct value per site, a path
that stopped emitting would be indistinguishable from one that is merely idle.

### `GET /v1/telemetry/effective-config`

Reports the export configuration this process actually resolved:

```bash
curl -s localhost:8000/v1/telemetry/effective-config
```

```json
{
  "service_name": "slm-server",
  "otlp_endpoint": "http://localhost:4318",
  "otlp_traces_endpoint": "http://localhost:4318/v1/traces",
  "otlp_protocol": "http/protobuf",
  "telemetry_enabled": true,
  "tracer_provider_initialised": true,
  "emit_paths": ["chat", "responses", "rerank", "streaming"],
  "elasticsearch_export": false
}
```

It exists because the ADR-0129 acceptance verifier runs on a host that cannot
otherwise inspect this one, and it is generated from live state rather than
hand-written — so it cannot name an endpoint the exporter is not using.

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
- `start.sh` now checks process liveness in addition to open ports, so stale listeners on old PIDs no longer produce false "ready" status
- Check logs for error messages

### "unknown model architecture" error (llamacpp)
Neither the PyPI `llama-cpp-python` build nor Homebrew's `llama.cpp` tracks upstream closely enough for the
newest architectures. Homebrew v0.3.0 (build 10621) cannot load Qwen3.8-Flash-Next at all, failing with
`unknown model architecture: 'qwen4exp'`. Build the pinned native binary instead:
```bash
./scripts/build_llama.sh
```
Then point the launcher at it in `.env`:
```
SLM_LLAMA_SERVER_BIN=/Users/Alex/Dev/llama.cpp/build/bin/llama-server
```
`SLM_LLAMA_SERVER_BIN` takes precedence; a `llama-server` on PATH is the fallback. The pinned commit lives in
`config/llama.cpp.pin`.
