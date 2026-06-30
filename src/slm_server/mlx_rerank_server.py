"""Self-contained MLX reranker server (Qwen3-Reranker yes/no-logit scoring)."""

import math

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
