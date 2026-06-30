from slm_server.config import ModelConfig, ModelDefinition, validate_model_config


def _md(**kw):
    base = dict(
        id="x",
        backend="mlx-rerank",
        port=8508,
        quantization="mxfp8",
        default_timeout=300,
        model_type="rerank",
        model_path="/tmp/x",
    )
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


def test_validator_rejects_mlx_rerank_with_non_rerank_model_type():
    cfg = ModelConfig(models={"r": _md(model_type="lm", model_path="/tmp/x")})
    issues = validate_model_config(cfg)
    assert any("mlx-rerank requires model_type rerank" in i for i in issues)
