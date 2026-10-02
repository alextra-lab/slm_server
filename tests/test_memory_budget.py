"""One heavy engine at a time: the declared memory budget (spec §4)."""

from __future__ import annotations

import signal

import pytest

from slm_server import start_backends as sb
from slm_server.config import ModelConfig, ModelDefinition, check_memory_budget


def _entry(
    model_id: str,
    port: int,
    *,
    peak: float | None = None,
    model_type: str = "lm",
    enabled: bool = True,
) -> ModelDefinition:
    return ModelDefinition(
        id=model_id,
        backend="llamacpp",
        port=port,
        quantization="Q4",
        default_timeout=600,
        model_type=model_type,
        model_path="/path/to/models/model.gguf",
        peak_memory_gib=peak,
        enabled=enabled,
    )


def _config(*entries: ModelDefinition) -> ModelConfig:
    return ModelConfig(models={f"role{i}": e for i, e in enumerate(entries)})


def test_single_heavy_entry_without_peak_only_warns() -> None:
    result = check_memory_budget(_config(_entry("a", 8502)), budget_gib=100)
    assert result.errors == []
    assert any("peak_memory_gib" in w for w in result.warnings)


def test_two_heavy_entries_with_one_missing_peak_is_an_error() -> None:
    result = check_memory_budget(
        _config(_entry("a", 8502, peak=23.0), _entry("b", 8503)), budget_gib=100
    )
    assert any("peak_memory_gib" in e and "role1" in e for e in result.errors)


def test_under_budget_passes() -> None:
    result = check_memory_budget(
        _config(_entry("a", 8502, peak=23.0), _entry("b", 8503, peak=40.0)), budget_gib=100
    )
    assert result.errors == []
    assert result.warnings == []


def test_over_budget_names_each_entry() -> None:
    result = check_memory_budget(
        _config(_entry("a", 8502, peak=87.0), _entry("b", 8503, peak=85.0)), budget_gib=100
    )
    assert len(result.errors) == 1
    assert "role0=87.0" in result.errors[0] and "role1=85.0" in result.errors[0]


def test_rerank_entry_is_not_heavy() -> None:
    result = check_memory_budget(
        _config(_entry("a", 8502, peak=87.0), _entry("r", 8506, model_type="rerank")),
        budget_gib=100,
    )
    assert result.errors == []


def test_disabled_entries_are_ignored() -> None:
    result = check_memory_budget(
        _config(_entry("a", 8502, peak=87.0), _entry("b", 8503, peak=85.0, enabled=False)),
        budget_gib=100,
    )
    assert result.errors == []


def test_budget_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_MEMORY_BUDGET_GIB", "50")
    result = check_memory_budget(_config(_entry("a", 8502, peak=60.0)))
    assert result.errors


def test_invalid_budget_environment_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLM_MEMORY_BUDGET_GIB", "lots")
    result = check_memory_budget(_config(_entry("a", 8502, peak=60.0)))
    assert result.errors == []
    assert any("SLM_MEMORY_BUDGET_GIB" in w for w in result.warnings)


def test_launcher_exits_before_starting_any_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(_entry("a", 8502, peak=87.0), _entry("b", 8503, peak=85.0))
    started: list[str] = []
    monkeypatch.setenv("SLM_MEMORY_BUDGET_GIB", "100")
    monkeypatch.setattr(sb, "load_model_config", lambda: cfg)
    monkeypatch.setattr(
        sb, "start_model_server", lambda model_def, config: started.append(model_def.id)
    )
    monkeypatch.setattr(signal, "signal", lambda sig, handler: None)
    with pytest.raises(SystemExit) as exc:
        sb.main()
    assert exc.value.code == 1
    assert started == []
