"""Self-contained MLX reranker server (Qwen3-Reranker yes/no-logit scoring)."""

import argparse
import math
from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


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
        if top_n is not None:
            try:
                top_n = int(top_n)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="Invalid 'top_n'")
        scores, prompt_tokens = scorer(query, documents, instruction)
        results = [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return JSONResponse(
            {
                "model": served_model_name,
                "object": "list",
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
                "results": results,
            }
        )

    return app


def _resolve_token_id(tokenizer, text: str) -> int:
    """Resolve the single token id for 'yes'/'no', tolerant of tokenizer wrappers."""
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    tid = hf.convert_tokens_to_ids(text)
    if isinstance(tid, int) and tid >= 0:
        return tid
    ids = tokenizer.encode(text)
    return int(ids[-1])


def load_scorer(model_path: str, context_length: int | None = None) -> Scorer:
    """Load the MLX reranker and return a scorer closure."""
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(model_path)  # type: ignore[misc]
    yes_id = _resolve_token_id(tokenizer, "yes")
    no_id = _resolve_token_id(tokenizer, "no")

    suffix_ids = tokenizer.encode(_PROMPT_SUFFIX)

    def scorer(query: str, documents: list[str], instruction: str) -> tuple[list[float], int]:
        scores: list[float] = []
        total = 0
        for doc in documents:
            ids = tokenizer.encode(build_rerank_prompt(query, doc, instruction))
            if context_length is not None and len(ids) > context_length:
                keep = max(0, context_length - len(suffix_ids))
                ids = ids[:keep] + suffix_ids
            total += len(ids)
            logits = model(mx.array([ids]))[0, -1, :]
            scores.append(relevance_score(float(logits[yes_id].item()), float(logits[no_id].item())))
        return scores, total

    return scorer


def run(
    model_path: str, port: int, host: str, served_model_name: str, context_length: int | None
) -> None:
    """Load the scorer and serve the rerank app with uvicorn."""
    import uvicorn

    scorer = load_scorer(model_path, context_length)
    uvicorn.run(create_app(scorer, served_model_name), host=host, port=port, access_log=False)


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
