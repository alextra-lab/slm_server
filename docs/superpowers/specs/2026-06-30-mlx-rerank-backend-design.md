# MLX Rerank Backend — Design

**Date:** 2026-06-30
**Status:** Approved (pending spec review)

## Problem

The llamacpp 4B reranker works well, but we want to serve the MLX `mxfp8`
Qwen3-Reranker models (already downloaded under LM Studio's models dir) through
the same OpenAI-style `/v1/rerank` API the router exposes.

Investigation ruled out every "reuse an existing server" path:

- `mlx-openai-server` has **no** `/v1/rerank` and can't distinguish reranker
  from embedding payloads (upstream issue #306).
- **LM Studio cannot serve this reranker over HTTP in any usable form**
  (verified against a running instance):
  - `/v1/rerank` → endpoint does not exist.
  - `/v1/embeddings` → rejects it; LM Studio classifies the model as
    `type: llm`, not an embedding model.
  - `/v1/chat/completions` → returns empty content and no logprobs → no
    yes/no score signal.
  LM Studio's reranking is reachable only through its SDK (scoring done
  internally), never on the wire.

Therefore the rerank scoring must be implemented in-repo. The only backend that
can already rerank (`llama-server --reranking`) needs **GGUF**, but these files
are MLX `safetensors`. So we build a small self-contained MLX rerank server.

## Goal

A new backend, managed like every other backend (launched by `start_backends`,
proxied by the router, health-checked via `/v1/backends/health`), that serves
the Qwen3-Reranker `mxfp8` models and returns the exact response shape the
router already proxies for `/v1/rerank`.

Non-goal: changing the router. It routes by port and forwards the `/v1/rerank`
body verbatim, returning the backend response as-is — so **no router changes
are needed**.

## Architecture

### New module: `src/slm_server/mlx_rerank_server.py`

A FastAPI app (run under uvicorn) that loads one Qwen3-Reranker MLX model and
scores query/document pairs.

Dependencies are already present — `fastapi` + `uvicorn[standard]` (core),
`mlx-lm` (the `mlx` extra). **No new dependencies.**

- **Startup:** `model, tokenizer = mlx_lm.load(model_path)` once. Resolve the
  `"yes"` and `"no"` token ids once and cache them.
- **`GET /health`** → `{"status": "ok"}`, 200. (Used by the router's
  `/v1/backends/health`, which GETs `http://localhost:{port}/health`.)
- **`POST /v1/rerank`** — accepts:
  ```json
  {"model": "...", "query": "...", "documents": ["...", "..."], "top_n": 3, "instruction": "optional"}
  ```
  For each document, score it (see Scoring), then return — matching
  `llama-server`'s shape exactly:
  ```json
  {
    "model": "<id>",
    "object": "list",
    "usage": {"prompt_tokens": N, "total_tokens": N},
    "results": [{"index": 0, "relevance_score": 0.97}, {"index": 1, "relevance_score": 2.5e-05}]
  }
  ```
  Results are returned in input order (matching the live llamacpp reranker).
  If `top_n` is provided, return the top-N by score (sorted descending).

### Scoring (official Qwen3-Reranker method)

Qwen3-Reranker is a causal LM fine-tuned to answer "yes"/"no" to a relevance
question. The relevance score is `P(yes)` at the final position.

- **System:** `Judge whether the Document meets the requirements based on the
  Query and the Instruct provided. Note that the answer can only be "yes" or
  "no".`
- **User:** `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
  with default instruction `Given a web search query, retrieve relevant
  passages that answer the query`.
- **Assistant prefix (suffix of the prompt):** ends with
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- Run one forward pass; take the last-token logits; extract the `yes` and `no`
  logits; `score = softmax([no_logit, yes_logit])[yes]` → a value in `[0, 1]`.

Prompt-building and score-normalization are **pure functions** (no model
needed) so they can be unit-tested directly. Exact `mlx_lm` logits call
(`model(mx.array([ids]))[:, -1, :]`) and token-id resolution are confirmed
during implementation.

### Integration points

- **Config schema** (`config.py`): add `"mlx-rerank"` to the
  `ModelDefinition.backend` Literal.
- **Validator** (`config.py`): relax the rerank check — `model_type: rerank` is
  allowed for `backend in {"llamacpp", "mlx-rerank"}` (currently llamacpp only).
- **Command builder** (`start_backends.py`): `build_mlx_rerank_command()` →
  `[sys.executable, "-m", "slm_server", "mlx-rerank", "--model-path", <path>,
  "--port", <port>, "--host", <host>, "--served-model-name", <id>,
  ("--context-length", <ctx>)?]`.
- **Dispatch** (`start_backends.py`, `start_model_server`): when
  `backend == "mlx-rerank"`, build/launch via the new builder. Same
  Popen/health-wait/verbose-log handling as other backends.
- **CLI** (`__main__.py`): new `mlx-rerank` subcommand that parses
  `--model-path/--port/--host/--context-length/--served-model-name` and runs
  the FastAPI app with uvicorn.

### Config entries (`config/models.yaml`)

```yaml
  rerank_mlx_4b:
    id: "mlx-community/Qwen3-Reranker-4B-mxfp8"
    backend: "mlx-rerank"
    port: 8508
    model_type: "rerank"
    context_length: 40960
    quantization: "mxfp8"
    max_concurrency: 1
    host: "0.0.0.0"
    default_timeout: 300
    model_path: "/Volumes/EnvoyUltra/lm-studio/models/mlx-community/Qwen3-Reranker-4B-mxfp8"
    enabled: true

  rerank_mlx_8b:
    id: "mlx-community/Qwen3-Reranker-8B-mxfp8"
    backend: "mlx-rerank"
    port: 8509
    model_type: "rerank"
    context_length: 40960
    quantization: "mxfp8"
    max_concurrency: 1
    host: "0.0.0.0"
    default_timeout: 300
    model_path: "/Volumes/EnvoyUltra/lm-studio/models/mlx-community/Qwen3-Reranker-8B-mxfp8"
    enabled: false   # still downloading — flip to true when complete
```

The existing llamacpp `rerank` (4B GGUF) stays unchanged.

## Performance

v1 scores documents **sequentially** (one forward pass per document) — simple
and correct. The router already serializes via `max_concurrency`. Batching all
documents of a request into one padded forward pass is a clear follow-up
optimization (padding/masking adds bug surface) and is **out of scope for v1**.

## Testing

- Unit: `build_mlx_rerank_command()` emits the expected argv.
- Unit: prompt builder produces the exact Qwen3-Reranker prompt string.
- Unit: score normalization (`softmax` of two logits) returns expected `P(yes)`.
- Unit: validator accepts `rerank` + `mlx-rerank`, still rejects `rerank` + `mlx`.
- Smoke (skipped if the model dir is absent): load + rerank the Paris example;
  assert the relevant doc scores far above the distractor.
- **Correctness gate:** compare MLX scores against the known-good llamacpp 4B on
  the same query/doc pairs (Paris example → ~0.97, distractor → ~0). Wild
  divergence means the prompt/token-id mapping is wrong — catch it before trust.

## Risks

- **Scoring correctness** — wrong prompt format or yes/no token ids produce
  garbage scores (as bad GGUF conversions did). Mitigated by the correctness
  gate above and pure-function unit tests.
- **`mlx_lm` logits API / `mxfp8` load** — confirmed during implementation; the
  model loads in LM Studio's MLX engine (4.16 GB, ctx 40960), so MLX support is
  expected.
- **Volume access** — model loading reads `/Volumes/EnvoyUltra`; the backend
  must be launched from an authorized context (same macOS TCC constraint as the
  other backends).

## Out of scope

- Batched scoring (future optimization).
- Enabling the 8B entry (until its download completes).
- Streaming responses.
- Any router changes.

## File summary

| File | Change |
|---|---|
| `src/slm_server/mlx_rerank_server.py` | **new** — FastAPI rerank server + scoring |
| `src/slm_server/__main__.py` | new `mlx-rerank` subcommand |
| `src/slm_server/start_backends.py` | `build_mlx_rerank_command()` + dispatch |
| `src/slm_server/config.py` | `mlx-rerank` backend type + validator relax |
| `config/models.yaml` | `rerank_mlx_4b` (enabled) + `rerank_mlx_8b` (disabled) |
| `tests/` | command builder, prompt, scoring, validator, smoke tests |
