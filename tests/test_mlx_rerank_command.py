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
