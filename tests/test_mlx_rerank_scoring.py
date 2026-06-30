import math

from slm_server.mlx_rerank_server import (
    DEFAULT_INSTRUCTION,
    build_rerank_prompt,
    relevance_score,
)


def test_build_rerank_prompt_exact_format():
    p = build_rerank_prompt(
        "What is the capital of France?", "Paris is the capital.", "Find the answer"
    )
    assert p == (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
        "<Instruct>: Find the answer\n"
        "<Query>: What is the capital of France?\n"
        "<Document>: Paris is the capital."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
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
