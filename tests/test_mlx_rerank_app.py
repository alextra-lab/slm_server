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
