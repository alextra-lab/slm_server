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
