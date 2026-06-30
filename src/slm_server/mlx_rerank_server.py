"""Self-contained MLX reranker server (Qwen3-Reranker yes/no-logit scoring)."""

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

        scores, prompt_tokens = scorer(query, documents, instruction)
        results = [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]
        if top_n is not None:
            try:
                top_n = int(top_n)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="Invalid 'top_n'")
            results = sorted(results, key=lambda r: r["relevance_score"], reverse=True)[:top_n]
        return JSONResponse(
            {
                "model": served_model_name,
                "object": "list",
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
                "results": results,
            }
        )

    return app
