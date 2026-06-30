# MLX Rerank Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve Qwen3-Reranker `mxfp8` MLX models through an in-repo FastAPI `/v1/rerank` backend that the router proxies like any other backend.

**Architecture:** A new `mlx_rerank_server.py` loads one reranker via `mlx_lm` and scores query/document pairs with the official Qwen3-Reranker yes/no-logit method. A new `mlx-rerank` backend type wires it into the existing config/launcher; the router needs no changes (it proxies `/v1/rerank` by port). Scoring is split into pure functions (unit-tested) and a model-backed scorer injected into the app (so endpoint logic is testable without MLX).

**Tech Stack:** Python 3.12, FastAPI, uvicorn, `mlx_lm`/`mlx` (the `mlx` extra), pytest.

## Global Constraints

- Python `>=3.12,<3.13`; line-length 100; ruff rules E/F/I/N/W/UP (E501 ignored).
- No new dependencies — use `fastapi`/`uvicorn[standard]` (core) and `mlx-lm`/`mlx` (the `mlx` extra) only.
- Tests use monkeypatching (no `unittest.mock`); `pythonpath = ["src"]`, `asyncio_mode = "auto"`. Router/endpoint tests use FastAPI `TestClient`.
- Input validation for subprocess safety: validate paths/hosts before passing to subprocess commands (follow existing `validate_path`/`validate_host` usage).
- `/v1/rerank` response MUST match llama-server exactly: `{"model","object":"list","usage":{"prompt_tokens","total_tokens"},"results":[{"index","relevance_score"}]}`, results in input order unless `top_n` is given.
- Run `uv run ruff check src/ tests/` and `uv run mypy src/` clean before each commit.

---

### Task 1: Scoring pure functions

**Files:**
- Create: `src/slm_server/mlx_rerank_server.py`
- Test: `tests/test_mlx_rerank_scoring.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DEFAULT_INSTRUCTION: str`
  - `build_rerank_prompt(query: str, document: str, instruction: str) -> str`
  - `relevance_score(yes_logit: float, no_logit: float) -> float` (returns P(yes) in `[0,1]`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mlx_rerank_scoring.py
import math

from slm_server.mlx_rerank_server import (
    DEFAULT_INSTRUCTION,
    build_rerank_prompt,
    relevance_score,
)


def test_build_rerank_prompt_exact_format():
    p = build_rerank_prompt("What is the capital of France?", "Paris is the capital.", "Find the answer")
    assert p == (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        '<|im_end|>\n<|im_start|>user\n'
        '<Instruct>: Find the answer\n'
        '<Query>: What is the capital of France?\n'
        '<Document>: Paris is the capital.'
        '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    )


def test_default_instruction_used_when_blank():
    p = build_rerank_prompt("q", "d", DEFAULT_INSTRUCTION)
    assert f"<Instruct>: {DEFAULT_INSTRUCTION}" in p


def test_relevance_score_monotonic_and_bounded():
    assert relevance_score(10.0, -10.0) > 0.99
    assert relevance_score(-10.0, 10.0) < 0.01
    assert math.isclose(relevance_score(0.0, 0.0), 0.5, rel_tol=1e-9)
    s = relevance_score(2.0, 1.0)
    assert 0.0 < s < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mlx_rerank_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (module or names not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# src/slm_server/mlx_rerank_server.py
"""Self-contained MLX reranker server (Qwen3-Reranker yes/no-logit scoring)."""

import math

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

_PROMPT_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
_PROMPT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def build_rerank_prompt(query: str, document: str, instruction: str) -> str:
    """Build the official Qwen3-Reranker prompt for one (query, document) pair."""
    body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    return _PROMPT_PREFIX + body + _PROMPT_SUFFIX


def relevance_score(yes_logit: float, no_logit: float) -> float:
    """Softmax over [no, yes] logits; return P(yes) in [0, 1]."""
    m = max(yes_logit, no_logit)
    ey = math.exp(yes_logit - m)
    en = math.exp(no_logit - m)
    return ey / (ey + en)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mlx_rerank_scoring.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/slm_server/mlx_rerank_server.py tests/test_mlx_rerank_scoring.py
uv run ruff format src/slm_server/mlx_rerank_server.py tests/test_mlx_rerank_scoring.py
git add src/slm_server/mlx_rerank_server.py tests/test_mlx_rerank_scoring.py
git commit -m "feat(mlx-rerank): prompt builder and score normalization"
```

---

### Task 2: Config — `mlx-rerank` backend type + validator relax

**Files:**
- Modify: `src/slm_server/config.py:13` (backend Literal), `src/slm_server/config.py:188-192` (validator)
- Test: `tests/test_config_mlx_rerank.py`

**Interfaces:**
- Consumes: `ModelDefinition`, `ModelConfig`, `validate_model_config` (existing).
- Produces: `backend` accepts `"mlx-rerank"`; `validate_model_config` allows `model_type: rerank` for `backend in {"llamacpp","mlx-rerank"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_mlx_rerank.py
from slm_server.config import ModelConfig, ModelDefinition, validate_model_config


def _md(**kw):
    base = dict(id="x", backend="mlx-rerank", port=8508, quantization="mxfp8",
               default_timeout=300, model_type="rerank", model_path="/tmp/x")
    base.update(kw)
    return ModelDefinition(**base)


def test_mlx_rerank_backend_accepted_by_schema():
    md = _md()
    assert md.backend == "mlx-rerank"


def test_validator_allows_rerank_on_mlx_rerank():
    cfg = ModelConfig(models={"r": _md(model_path="/tmp/does-not-exist")})
    issues = validate_model_config(cfg)
    assert not any("only supported with backend llamacpp" in i for i in issues)


def test_validator_still_rejects_rerank_on_mlx():
    cfg = ModelConfig(models={"r": _md(backend="mlx", model_path="/tmp/does-not-exist")})
    issues = validate_model_config(cfg)
    assert any("rerank" in i and "llamacpp" in i for i in issues)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_mlx_rerank.py -v`
Expected: FAIL — `ValidationError` (Literal rejects `"mlx-rerank"`).

- [ ] **Step 3: Implement — widen Literal and validator**

In `src/slm_server/config.py` line 13, change:

```python
    backend: Literal["mlx", "llamacpp"] = Field(..., description="Backend type")
```

to:

```python
    backend: Literal["mlx", "llamacpp", "mlx-rerank"] = Field(..., description="Backend type")
```

In `src/slm_server/config.py`, replace the validator block at lines 188-192:

```python
        if model_def.model_type == "rerank" and model_def.backend != "llamacpp":
            issues.append(
                f"{role}: model_type rerank is only supported with backend llamacpp "
                "(native llama-server with --reranking; mlx-openai-server has no rerank mode)"
            )
```

with:

```python
        if model_def.model_type == "rerank" and model_def.backend not in ("llamacpp", "mlx-rerank"):
            issues.append(
                f"{role}: model_type rerank is only supported with backend llamacpp "
                "(native llama-server --reranking) or mlx-rerank (in-repo MLX server)"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_mlx_rerank.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/slm_server/config.py tests/test_config_mlx_rerank.py
uv run mypy src/slm_server/config.py
git add src/slm_server/config.py tests/test_config_mlx_rerank.py
git commit -m "feat(mlx-rerank): add mlx-rerank backend type and allow rerank model_type"
```

---

### Task 3: Command builder + dispatch

**Files:**
- Modify: `src/slm_server/start_backends.py` (new function near other builders; dispatch branch at line 760)
- Test: `tests/test_mlx_rerank_command.py`

**Interfaces:**
- Consumes: `validate_path`, `validate_host` (existing in `start_backends.py`).
- Produces: `build_mlx_rerank_command(model_path, port, host="0.0.0.0", served_model_name=None, context_length=None) -> list[str]`. Argv: `[sys.executable, "-m", "slm_server", "mlx-rerank", "--model-path", <path>, "--port", <port>, "--host", <host>, ("--served-model-name", <name>)?, ("--context-length", <ctx>)?]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mlx_rerank_command.py
import sys

from slm_server.start_backends import build_mlx_rerank_command


def test_build_mlx_rerank_command_basic(tmp_path):
    d = tmp_path / "Qwen3-Reranker-4B-mxfp8"
    d.mkdir()
    cmd = build_mlx_rerank_command(
        model_path=d, port=8508, host="0.0.0.0",
        served_model_name="mlx-community/Qwen3-Reranker-4B-mxfp8", context_length=40960,
    )
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "slm_server", "mlx-rerank"]
    assert "--model-path" in cmd and str(d) in cmd
    assert cmd[cmd.index("--port") + 1] == "8508"
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
    assert cmd[cmd.index("--served-model-name") + 1] == "mlx-community/Qwen3-Reranker-4B-mxfp8"
    assert cmd[cmd.index("--context-length") + 1] == "40960"


def test_build_mlx_rerank_command_omits_optional(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    cmd = build_mlx_rerank_command(model_path=d, port=8509)
    assert "--served-model-name" not in cmd
    assert "--context-length" not in cmd


def test_build_mlx_rerank_command_rejects_bad_port(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    try:
        build_mlx_rerank_command(model_path=d, port=80)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mlx_rerank_command.py -v`
Expected: FAIL — `ImportError` (function not defined).

- [ ] **Step 3: Implement the builder**

Add to `src/slm_server/start_backends.py` (place after `build_mlx_command`, before `find_native_llama_server`). `sys`, `Path`, `cast`, `validate_path`, `validate_host` are already imported.

```python
def build_mlx_rerank_command(
    model_path: Path | str,
    port: int,
    host: str = "0.0.0.0",
    served_model_name: str | None = None,
    context_length: int | None = None,
) -> list[str]:
    """Build command to launch the in-repo MLX rerank server (python -m slm_server mlx-rerank)."""
    resolved = cast(Path, validate_path(model_path, allow_hf_model=False))
    host = validate_host(host)
    if not (1024 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}. Must be between 1024 and 65535")
    if context_length is not None and context_length <= 0:
        raise ValueError(f"Invalid context_length: {context_length}. Must be positive")
    cmd = [
        sys.executable,
        "-m",
        "slm_server",
        "mlx-rerank",
        "--model-path",
        str(resolved),
        "--port",
        str(port),
        "--host",
        host,
    ]
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    if context_length is not None:
        cmd.extend(["--context-length", str(context_length)])
    return cmd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mlx_rerank_command.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire the dispatch branch**

In `src/slm_server/start_backends.py`, immediately before the `elif model_def.backend == "llamacpp":` branch (line 781), insert:

```python
        elif model_def.backend == "mlx-rerank":
            cmd = build_mlx_rerank_command(
                model_path=Path(model_path) if not is_hf_model else model_path,
                port=model_def.port,
                host=getattr(model_def, "host", "0.0.0.0"),
                served_model_name=model_def.id,
                context_length=model_def.context_length,
            )
```

- [ ] **Step 6: Add a dispatch test (monkeypatched Popen)**

```python
# append to tests/test_mlx_rerank_command.py
def test_start_model_server_uses_mlx_rerank_builder(tmp_path, monkeypatch):
    import slm_server.start_backends as sb
    from slm_server.config import ModelConfig, ModelDefinition

    d = tmp_path / "Qwen3-Reranker-4B-mxfp8"
    d.mkdir()
    md = ModelDefinition(id="mlx-community/Qwen3-Reranker-4B-mxfp8", backend="mlx-rerank",
                         port=8508, quantization="mxfp8", default_timeout=300,
                         model_type="rerank", model_path=str(d))
    captured = {}

    class FakeProc:
        pid = 4242
        stderr = None
        def poll(self):
            return None

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(sb.subprocess, "Popen", fake_popen)
    proc = sb.start_model_server(md, ModelConfig(models={"r": md}))
    assert proc is not None
    assert captured["cmd"][3] == "mlx-rerank"
    assert "--served-model-name" in captured["cmd"]
```

- [ ] **Step 7: Run, lint, type-check, commit**

```bash
uv run pytest tests/test_mlx_rerank_command.py -v
uv run ruff check src/slm_server/start_backends.py tests/test_mlx_rerank_command.py
uv run mypy src/slm_server/start_backends.py
git add src/slm_server/start_backends.py tests/test_mlx_rerank_command.py
git commit -m "feat(mlx-rerank): command builder and start_model_server dispatch"
```

---

### Task 4: FastAPI app with injected scorer

**Files:**
- Modify: `src/slm_server/mlx_rerank_server.py`
- Test: `tests/test_mlx_rerank_app.py`

**Interfaces:**
- Consumes: `build_rerank_prompt`, `relevance_score` (Task 1).
- Produces:
  - Type alias `Scorer = Callable[[str, list[str], str], tuple[list[float], int]]` (returns `(scores_in_input_order, prompt_tokens_total)`).
  - `create_app(scorer: Scorer, served_model_name: str) -> fastapi.FastAPI` exposing `GET /health` and `POST /v1/rerank`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mlx_rerank_app.py
from fastapi.testclient import TestClient

from slm_server.mlx_rerank_server import create_app


def _client(scores):
    def scorer(query, documents, instruction):
        return list(scores), 42
    return TestClient(create_app(scorer, "mlx-community/Qwen3-Reranker-4B-mxfp8"))


def test_health():
    c = _client([0.9])
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_rerank_input_order_and_shape():
    c = _client([0.9, 0.1])
    r = c.post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a", "b"]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "mlx-community/Qwen3-Reranker-4B-mxfp8"
    assert body["usage"] == {"prompt_tokens": 42, "total_tokens": 42}
    assert body["results"] == [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.1},
    ]


def test_rerank_top_n_sorts_desc_and_limits():
    c = _client([0.1, 0.9, 0.5])
    r = c.post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a", "b", "c"], "top_n": 2})
    idxs = [x["index"] for x in r.json()["results"]]
    assert idxs == [1, 2]


def test_rerank_missing_query_400():
    c = _client([0.9])
    r = c.post("/v1/rerank", json={"model": "m", "documents": ["a"]})
    assert r.status_code == 400


def test_rerank_missing_documents_400():
    c = _client([0.9])
    r = c.post("/v1/rerank", json={"model": "m", "query": "q"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mlx_rerank_app.py -v`
Expected: FAIL — `ImportError` (`create_app` not defined).

- [ ] **Step 3: Implement `create_app`**

Append to `src/slm_server/mlx_rerank_server.py` (add imports at top: `from collections.abc import Callable`; `from fastapi import FastAPI, HTTPException, Request`; `from fastapi.responses import JSONResponse`):

```python
Scorer = Callable[[str, list[str], str], tuple[list[float], int]]


def create_app(scorer: Scorer, served_model_name: str) -> FastAPI:
    """Build the rerank FastAPI app around an injected scorer (model-agnostic)."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/v1/rerank", response_model=None)
    async def rerank(request: Request) -> JSONResponse:
        body = await request.json()
        query = body.get("query")
        documents = body.get("documents")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query'")
        if not documents or not isinstance(documents, list):
            raise HTTPException(status_code=400, detail="Missing or invalid 'documents'")
        instruction = body.get("instruction") or DEFAULT_INSTRUCTION
        top_n = body.get("top_n")

        scores, prompt_tokens = scorer(query, documents, instruction)
        results = [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]
        if top_n is not None:
            results = sorted(results, key=lambda r: r["relevance_score"], reverse=True)[: int(top_n)]
        return JSONResponse(
            {
                "model": served_model_name,
                "object": "list",
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
                "results": results,
            }
        )

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mlx_rerank_app.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/slm_server/mlx_rerank_server.py tests/test_mlx_rerank_app.py
uv run mypy src/slm_server/mlx_rerank_server.py
git add src/slm_server/mlx_rerank_server.py tests/test_mlx_rerank_app.py
git commit -m "feat(mlx-rerank): FastAPI app with injected scorer"
```

---

### Task 5: Model-backed scorer + CLI entry

**Files:**
- Modify: `src/slm_server/mlx_rerank_server.py` (real scorer + `run` entrypoint)
- Modify: `src/slm_server/__main__.py` (new `mlx-rerank` subcommand)
- Test: `tests/test_mlx_rerank_smoke.py`

**Interfaces:**
- Consumes: `build_rerank_prompt`, `relevance_score`, `create_app`, `Scorer`.
- Produces:
  - `load_scorer(model_path: str, context_length: int | None = None) -> Scorer` (loads the MLX model, returns a scorer).
  - `run(model_path: str, port: int, host: str, served_model_name: str, context_length: int | None) -> None` (loads scorer, runs uvicorn).

- [ ] **Step 1: Write the smoke test (skips without the model)**

```python
# tests/test_mlx_rerank_smoke.py
import os

import pytest

MODEL = "/Volumes/EnvoyUltra/lm-studio/models/mlx-community/Qwen3-Reranker-4B-mxfp8"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL), reason="MLX reranker model not present"
)


def test_load_scorer_ranks_relevant_above_distractor():
    from slm_server.mlx_rerank_server import load_scorer

    scorer = load_scorer(MODEL)
    scores, tokens = scorer(
        "What is the capital of France?",
        ["Paris is the capital of France.", "Bananas are yellow."],
        "Given a web search query, retrieve relevant passages that answer the query",
    )
    assert tokens > 0
    assert scores[0] > 0.5 > scores[1]
```

- [ ] **Step 2: Run test to verify it fails (or skips)**

Run: `uv run pytest tests/test_mlx_rerank_smoke.py -v`
Expected: FAIL with `ImportError` if the model dir is present; otherwise SKIPPED. (Either outcome is acceptable to proceed; the import must exist by Step 4.)

- [ ] **Step 3: Implement `load_scorer` and `run`**

Append to `src/slm_server/mlx_rerank_server.py` (add `import argparse` at top):

```python
def _resolve_token_id(tokenizer, text: str) -> int:
    """Resolve the single token id for 'yes'/'no', tolerant of tokenizer wrappers."""
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    tid = hf.convert_tokens_to_ids(text)
    if isinstance(tid, int) and tid >= 0:
        return tid
    ids = tokenizer.encode(text)
    return ids[-1]


def load_scorer(model_path: str, context_length: int | None = None) -> Scorer:
    """Load the MLX reranker and return a scorer closure."""
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(model_path)
    yes_id = _resolve_token_id(tokenizer, "yes")
    no_id = _resolve_token_id(tokenizer, "no")

    def scorer(query: str, documents: list[str], instruction: str) -> tuple[list[float], int]:
        scores: list[float] = []
        total = 0
        for doc in documents:
            ids = tokenizer.encode(build_rerank_prompt(query, doc, instruction))
            if context_length is not None and len(ids) > context_length:
                ids = ids[:context_length]
            total += len(ids)
            logits = model(mx.array([ids]))[0, -1, :]
            scores.append(relevance_score(float(logits[yes_id].item()), float(logits[no_id].item())))
        return scores, total

    return scorer


def run(model_path: str, port: int, host: str, served_model_name: str,
        context_length: int | None) -> None:
    """Load the scorer and serve the rerank app with uvicorn."""
    import uvicorn

    scorer = load_scorer(model_path, context_length)
    uvicorn.run(create_app(scorer, served_model_name), host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="slm_server mlx-rerank")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--context-length", type=int, default=None)
    args = parser.parse_args(argv)
    run(
        model_path=args.model_path,
        port=args.port,
        host=args.host,
        served_model_name=args.served_model_name or args.model_path,
        context_length=args.context_length,
    )
```

- [ ] **Step 4: Wire the `__main__.py` subcommand**

In `src/slm_server/__main__.py`, add a branch before the `else:` (line 18):

```python
    elif len(sys.argv) > 1 and sys.argv[1] == "mlx-rerank":
        # Start the in-repo MLX rerank server
        from slm_server.mlx_rerank_server import main as mlx_rerank_main

        mlx_rerank_main(sys.argv[2:])
```

Also update the usage string at line 19:

```python
        print("Usage: python -m slm_server [router|backends|mlx-rerank]")
```

- [ ] **Step 5: Run smoke test + full suite**

Run: `uv run pytest tests/test_mlx_rerank_smoke.py -v` (PASS if model present, else SKIPPED)
Run: `uv run pytest -q` (Expected: all PASS, smoke possibly skipped)

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src/slm_server/mlx_rerank_server.py src/slm_server/__main__.py tests/test_mlx_rerank_smoke.py
uv run mypy src/slm_server/mlx_rerank_server.py src/slm_server/__main__.py
git add src/slm_server/mlx_rerank_server.py src/slm_server/__main__.py tests/test_mlx_rerank_smoke.py
git commit -m "feat(mlx-rerank): model-backed scorer and CLI entrypoint"
```

---

### Task 6: Config entries

**Files:**
- Modify: `config/models.yaml`
- Test: (covered by existing `load_model_config` validation; manual check below)

**Interfaces:**
- Consumes: `mlx-rerank` backend type + validator (Task 2).
- Produces: `rerank_mlx_4b` (enabled) and `rerank_mlx_8b` (disabled) model entries.

- [ ] **Step 1: Add the entries**

Append to `config/models.yaml` (after the existing `rerank` block):

```yaml
  # Qwen3-Reranker-4B mxfp8 (MLX) — OpenAI /v1/rerank via in-repo mlx-rerank server
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

  # Qwen3-Reranker-8B mxfp8 (MLX) — still downloading; flip enabled to true when complete
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
    enabled: false
```

- [ ] **Step 2: Verify config loads and validates**

Run:
```bash
uv run python -c "from slm_server.config import load_model_config; c=load_model_config(); print({k:(v.backend,v.port,v.enabled) for k,v in c.models.items() if 'rerank_mlx' in k})"
```
Expected: prints `rerank_mlx_4b` (`mlx-rerank`, 8508, True) and `rerank_mlx_8b` (`mlx-rerank`, 8509, False) with no `model_type rerank is only supported` warning for them. (A `model_path does not exist` warning for the 8B is acceptable while it downloads.)

- [ ] **Step 3: Commit**

```bash
git add config/models.yaml
git commit -m "feat(mlx-rerank): add rerank_mlx_4b (enabled) and rerank_mlx_8b (disabled) config"
```

---

### Task 7: Live correctness gate

**Files:** none (verification only).

**Interfaces:** Consumes the running backend (launched from an authorized shell with `/Volumes` access).

- [ ] **Step 1: Launch the backend manually (authorized shell)**

Run (from a shell with volume access):
```bash
uv run python -m slm_server mlx-rerank \
  --model-path /Volumes/EnvoyUltra/lm-studio/models/mlx-community/Qwen3-Reranker-4B-mxfp8 \
  --port 8508 --served-model-name mlx-community/Qwen3-Reranker-4B-mxfp8 &
sleep 30
curl -s http://localhost:8508/health
```
Expected: `{"status":"ok"}`.

- [ ] **Step 2: Compare scores against the known-good llamacpp 4B**

Run:
```bash
curl -s http://localhost:8508/v1/rerank -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-Reranker-4B-mxfp8","query":"What is the capital of France?","documents":["Paris is the capital of France.","Bananas are yellow.","Berlin is in Germany."]}'
```
Expected: index 0 scores near 1.0 (e.g. > 0.9); indices 1 and 2 near 0. This mirrors the llamacpp 4B (`Voodisss/Qwen3-Reranker-4B` → ~0.97 for Paris). If index 0 is NOT highest, the prompt/token-id mapping is wrong — fix `_resolve_token_id`/`build_rerank_prompt` before trusting the backend.

- [ ] **Step 3: Stop the manual instance**

```bash
kill %1
```

- [ ] **Step 4: Final full suite**

Run: `uv run pytest -q`
Expected: all PASS (smoke skipped if model absent in CI).
```
```
