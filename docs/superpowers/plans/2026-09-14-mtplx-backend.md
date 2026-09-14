# MTPLX Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a managed `backend: "mtplx"` to slm_server so the launcher starts `mtplx serve`, the router routes to it by model id, and the watchdog restarts it, with llama.cpp unchanged as the default.

**Architecture:** Follow the existing per-backend pattern. `config.py` gains MTPLX fields, MTPLX validation, and a memory-budget check. `start_backends.py` gains `find_mtplx_binary()`, `build_mtplx_command()` and one `elif` branch in `start_model_server()`. `router.py` gains one helper that applies `chat_template_kwargs` per backend at the three existing sites, a 404 for disabled ids, and usage-based telemetry fields. `watchdog.py` treats 507 as "ignore". `start.sh` gets a configurable readiness timeout.

**Tech Stack:** Python 3.12, FastAPI, httpx, pydantic, structlog, pytest (asyncio auto mode), ruff, mypy, bash.

**Spec:** `docs/superpowers/specs/2026-09-14-mtplx-backend-design.md` (commit 88aaffc). Read it before any task.

## Global Constraints

- Work on branch `mtplx-backend`. Do not push. Do not change the running server or the live `config/models.yaml`.
- One task per commit. Each commit message ends with these two lines:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL`
- Before each commit run, from the repo root: `uv run pytest`, `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`, `uv run mypy src/`. All must pass.
- Ruff: line length 100, rules E/F/I/N/W/UP (E501 ignored). mypy: `disallow_untyped_defs = false`.
- If `ruff format --check` fails, run `uv run ruff format src/ tests/`, then run the checks again. Code blocks in this plan are not pre-formatted.
- New tests use `monkeypatch` and small fake classes, never `unittest.mock`. Fake packs live in `tmp_path`.
- MTPLX binds `127.0.0.1` with `--no-auth`, always.
- MTPLX default for `preserve_thinking` is `off`.
- `mtp_depth` range: 1 to the pack's `mtp_depth_max` from `mtplx_runtime.json`; 6 when that key is absent.
- `SLM_BACKEND_READY_TIMEOUT` default: 180 seconds. `SLM_MEMORY_BUDGET_GIB` default: 100.
- Heavy entry: enabled, `model_type` `lm` or `multimodal`.
- The repo is public: no local volume paths, emails, or secrets in code, tests, or docs. Use placeholder paths such as `/path/to/models/...`.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/slm_server/config.py` | Modify | New fields; MTPLX validation helpers; `check_memory_budget()` |
| `src/slm_server/start_backends.py` | Modify | `find_mtplx_binary()`, `build_mtplx_command()`, MTPLX branch, memory check in `main()` |
| `src/slm_server/watchdog.py` | Modify | 507 → `ignore` |
| `src/slm_server/router.py` | Modify | Disabled id → 404; `_prepare_chat_template_kwargs()`; usage-based telemetry fields |
| `src/slm_server/telemetry.py` | Modify | `slm.reasoning_tokens` span attribute |
| `start.sh` | Modify | `SLM_BACKEND_READY_TIMEOUT` |
| `tests/test_config_mtplx.py` | Create | MTPLX validation |
| `tests/test_start_backends_mtplx.py` | Create | Builder, binary lookup, launcher branch |
| `tests/test_memory_budget.py` | Create | Memory guard and launcher exit |
| `tests/test_watchdog.py` | Modify | 507 classification |
| `tests/test_router_model_selection.py` | Modify | Disabled id 404 |
| `tests/test_router_watchdog.py` | Modify | Unknown/disabled ids are not scored |
| `tests/test_router_mtplx.py` | Create | Kwargs translation, headers, stream pass-through, telemetry |
| `README.md`, `config/models.yaml.example`, `CLAUDE.md` | Modify | Documentation |

---

### Task 1: Config fields and MTPLX validation

**Files:**
- Modify: `src/slm_server/config.py`
- Test: `tests/test_config_mtplx.py`

**Interfaces:**
- Produces:
  - `ModelDefinition.backend: Literal["mlx", "llamacpp", "mlx-rerank", "mtplx"]`
  - New `ModelDefinition` fields: `mtp_depth: int | None`, `reasoning_effort: Literal["low","medium","high","xhigh","auto"] | None`, `preserve_thinking: Literal["off","on","auto","scoped"] | None`, `mtplx_profile: Literal["stable","performance-cold","sustained","turbo","exact","max-diagnostic"] | None`, `mtplx_batching_preset: Literal["solo","latency","agent","throughput"] | None`, `mtplx_fan_mode: Literal["default","smart","max"] | None`, `peak_memory_gib: float | None`
  - `MTPLX_DEPTH_CEILING: int = 6`
  - `THERMALFORGE_SOCKET: Path = Path("/tmp/thermalforge.sock")`
  - `read_mtp_depth_max(pack_dir: Path) -> int`
  - `mtplx_config_errors(model_def: ModelDefinition) -> list[str]` (hard errors, no role prefix)
  - `mtplx_config_warnings(model_def: ModelDefinition) -> list[str]` (warnings, no role prefix)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_mtplx.py`:

```python
"""Validation rules for backend: mtplx entries (spec §1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_server import config as config_module
from slm_server.config import (
    ModelConfig,
    ModelDefinition,
    mtplx_config_errors,
    mtplx_config_warnings,
    read_mtp_depth_max,
    validate_model_config,
)


def _pack(tmp_path: Path, runtime: dict | None = None) -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "config.json").write_text("{}")
    (pack / "mtplx_runtime.json").write_text(json.dumps(runtime or {"mtp_depth_max": 3}))
    return pack


def _mtplx(pack: Path | str, **overrides: object) -> ModelDefinition:
    fields: dict[str, object] = {
        "id": "mtplx-test",
        "backend": "mtplx",
        "port": 8600,
        "quantization": "4bit",
        "default_timeout": 600,
        "model_path": str(pack),
        "mtp_depth": 3,
    }
    fields.update(overrides)
    return ModelDefinition(**fields)


def _issues(model_def: ModelDefinition) -> list[str]:
    return validate_model_config(ModelConfig(models={"m": model_def}))


def test_valid_mtplx_entry_has_no_issues(tmp_path: Path) -> None:
    assert _issues(_mtplx(_pack(tmp_path))) == []


def test_missing_depth_is_an_error(tmp_path: Path) -> None:
    errors = mtplx_config_errors(_mtplx(_pack(tmp_path), mtp_depth=None))
    assert any("mtp_depth" in e for e in errors)


def test_depth_above_pack_limit_is_an_error(tmp_path: Path) -> None:
    errors = mtplx_config_errors(_mtplx(_pack(tmp_path, {"mtp_depth_max": 3}), mtp_depth=4))
    assert any("mtp_depth_max" in e for e in errors)


def test_depth_limit_defaults_to_ceiling_without_the_key(tmp_path: Path) -> None:
    pack = _pack(tmp_path, {"other": 1})
    assert read_mtp_depth_max(pack) == 6
    assert mtplx_config_errors(_mtplx(pack, mtp_depth=6)) == []


def test_depth_above_ceiling_rejected_by_model(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _mtplx(_pack(tmp_path), mtp_depth=7)


def test_pack_without_runtime_file_is_an_error(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "config.json").write_text("{}")
    errors = mtplx_config_errors(_mtplx(pack))
    assert any("mtplx_runtime.json" in e for e in errors)


def test_huggingface_id_is_an_error() -> None:
    errors = mtplx_config_errors(_mtplx("Youssofal/Some-Pack"))
    assert any("local directory" in e for e in errors)


@pytest.mark.parametrize("key", ["reasoning_effort", "preserve_thinking"])
def test_effort_keys_in_config_kwargs_are_an_error(tmp_path: Path, key: str) -> None:
    model_def = _mtplx(_pack(tmp_path), chat_template_kwargs={key: "medium"})
    assert any(key in e for e in mtplx_config_errors(model_def))


def test_unknown_enum_value_rejected_by_model(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _mtplx(_pack(tmp_path), reasoning_effort="extreme")


def test_llamacpp_only_field_warns(tmp_path: Path) -> None:
    warnings = mtplx_config_warnings(_mtplx(_pack(tmp_path), min_p=0.0))
    assert any("min_p" in w for w in warnings)


def test_concurrency_above_one_with_serial_preset_warns(tmp_path: Path) -> None:
    warnings = mtplx_config_warnings(_mtplx(_pack(tmp_path), max_concurrency=3))
    assert any("max_concurrency" in w for w in warnings)


def test_concurrency_with_agent_preset_does_not_warn(tmp_path: Path) -> None:
    model_def = _mtplx(_pack(tmp_path), max_concurrency=3, mtplx_batching_preset="agent")
    assert not any("max_concurrency" in w for w in mtplx_config_warnings(model_def))


def test_unusual_host_warns(tmp_path: Path) -> None:
    warnings = mtplx_config_warnings(_mtplx(_pack(tmp_path), host="192.168.1.5"))
    assert any("host" in w for w in warnings)


def test_fan_mode_without_daemon_socket_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "THERMALFORGE_SOCKET", tmp_path / "absent.sock")
    warnings = mtplx_config_warnings(_mtplx(_pack(tmp_path), mtplx_fan_mode="smart"))
    assert any("thermalforge" in w for w in warnings)


def test_mtplx_only_field_on_llamacpp_warns(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    model_def = ModelDefinition(
        id="llama-test",
        backend="llamacpp",
        port=8502,
        quantization="Q4",
        default_timeout=600,
        model_path=str(gguf),
        mtp_depth=3,
    )
    assert any("mtp_depth" in issue for issue in _issues(model_def))


def test_validate_prefixes_mtplx_issues_with_role(tmp_path: Path) -> None:
    issues = _issues(_mtplx(_pack(tmp_path), mtp_depth=None))
    assert any(issue.startswith("m: ") and "mtp_depth" in issue for issue in issues)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_mtplx.py -v`
Expected: FAIL at import with `ImportError: cannot import name 'mtplx_config_errors'`.

- [ ] **Step 3: Implement the fields and helpers**

In `src/slm_server/config.py`, change the imports at the top to:

```python
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
```

Change the `backend` field to:

```python
    backend: Literal["mlx", "llamacpp", "mlx-rerank", "mtplx"] = Field(
        ..., description="Backend type"
    )
```

Insert these fields immediately after the `mmproj_path` field and before `enabled`:

```python
    # MTPLX-only fields (backend: mtplx). See docs/superpowers/specs/2026-09-14-mtplx-backend-design.md.
    mtp_depth: int | None = Field(
        None,
        ge=1,
        le=6,
        description="MTP draft depth passed to `mtplx serve --depth`. Required for backend mtplx: "
        "`mtplx serve` ignores saved tuning. Tune with `mtplx tune` and copy the best depth here.",
    )
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "auto"] | None = Field(
        None, description="`mtplx serve --reasoning-effort`. Only used when backend is mtplx."
    )
    preserve_thinking: Literal["off", "on", "auto", "scoped"] | None = Field(
        None,
        description="`mtplx serve --preserve-thinking`. Defaults to off when unset. Only used when "
        "backend is mtplx.",
    )
    mtplx_profile: (
        Literal["stable", "performance-cold", "sustained", "turbo", "exact", "max-diagnostic"] | None
    ) = Field(None, description="`mtplx serve --profile`. Only used when backend is mtplx.")
    mtplx_batching_preset: Literal["solo", "latency", "agent", "throughput"] | None = Field(
        None,
        description="`mtplx serve --batching-preset`. Unset means serial (one request at a time). "
        "Only used when backend is mtplx.",
    )
    mtplx_fan_mode: Literal["default", "smart", "max"] | None = Field(
        None,
        description="`mtplx serve --fan-mode`. smart and max need the thermalforge daemon. Only "
        "used when backend is mtplx.",
    )
    peak_memory_gib: float | None = Field(
        None,
        gt=0,
        description="Declared peak resident memory in GiB, for the launcher's memory budget. "
        "Applies to every backend.",
    )
```

Insert these module-level definitions immediately after the `ModelConfig` class:

```python
MTPLX_DEPTH_CEILING = 6
THERMALFORGE_SOCKET = Path("/tmp/thermalforge.sock")

_MTPLX_ONLY_FIELDS = (
    "mtp_depth",
    "reasoning_effort",
    "preserve_thinking",
    "mtplx_profile",
    "mtplx_batching_preset",
    "mtplx_fan_mode",
)

_LLAMACPP_ONLY_FIELDS = (
    "min_p",
    "repetition_penalty",
    "n_predict",
    "ubatch_size",
    "kv_unified",
    "cache_type_k",
    "cache_type_v",
    "cache_ram",
    "kv_offload",
    "flash_attn",
    "fit",
    "cont_batching",
    "cache_prompt",
    "spec_type",
    "spec_draft_n_max",
    "spec_model_path",
    "verbose",
    "chat_template_file",
    "mmproj_path",
)

_MTPLX_CONFIG_KWARG_KEYS_FORBIDDEN = ("reasoning_effort", "preserve_thinking")


def read_mtp_depth_max(pack_dir: Path) -> int:
    """Read the pack's declared maximum MTP depth.

    Args:
        pack_dir: MTPLX pack directory.

    Returns:
        `mtp_depth_max` from `mtplx_runtime.json`, or `MTPLX_DEPTH_CEILING` when the file or
        the key is missing or invalid.
    """
    try:
        data = json.loads((pack_dir / "mtplx_runtime.json").read_text())
    except (OSError, ValueError):
        return MTPLX_DEPTH_CEILING
    value = data.get("mtp_depth_max") if isinstance(data, dict) else None
    if isinstance(value, int) and 1 <= value <= MTPLX_DEPTH_CEILING:
        return value
    return MTPLX_DEPTH_CEILING


def mtplx_config_errors(model_def: ModelDefinition) -> list[str]:
    """Hard errors for a backend: mtplx entry. The builder refuses to start on any of these."""
    errors: list[str] = []
    path_str = model_def.model_path or ""
    if not path_str or not path_str.startswith("/"):
        errors.append(
            "backend mtplx requires model_path to be a local directory (absolute path), "
            f"got: {path_str!r}"
        )
    else:
        pack = Path(path_str)
        if not pack.is_dir():
            errors.append(f"backend mtplx model_path is not a directory: {pack}")
        else:
            for required in ("config.json", "mtplx_runtime.json"):
                if not (pack / required).is_file():
                    errors.append(f"backend mtplx pack is missing {required}: {pack}")
            if model_def.mtp_depth is not None:
                depth_max = read_mtp_depth_max(pack)
                if model_def.mtp_depth > depth_max:
                    errors.append(
                        f"mtp_depth {model_def.mtp_depth} exceeds the pack's mtp_depth_max "
                        f"{depth_max}"
                    )
    if model_def.mtp_depth is None:
        errors.append("backend mtplx requires mtp_depth (mtplx serve ignores saved tuning)")
    for key in _MTPLX_CONFIG_KWARG_KEYS_FORBIDDEN:
        if model_def.chat_template_kwargs and key in model_def.chat_template_kwargs:
            errors.append(
                f"backend mtplx: set {key} as a top-level field, not inside chat_template_kwargs"
            )
    return errors


def mtplx_config_warnings(model_def: ModelDefinition) -> list[str]:
    """Warnings for a backend: mtplx entry. The server still starts."""
    warnings: list[str] = []
    for field in _LLAMACPP_ONLY_FIELDS:
        if getattr(model_def, field) is not None:
            warnings.append(f"backend mtplx ignores {field}; remove for clarity")
    if model_def.max_concurrency > 1 and model_def.mtplx_batching_preset in (
        None,
        "solo",
        "latency",
    ):
        warnings.append(
            f"max_concurrency {model_def.max_concurrency} with a serial MTPLX scheduler: "
            "requests queue one at a time; set mtplx_batching_preset to agent or throughput"
        )
    if model_def.host not in ("0.0.0.0", "127.0.0.1"):
        warnings.append(
            f"host {model_def.host} is ignored: mtplx binds 127.0.0.1 (non-localhost binds need "
            "an API key)"
        )
    if model_def.mtplx_fan_mode in ("smart", "max") and not THERMALFORGE_SOCKET.exists():
        warnings.append(
            f"mtplx_fan_mode {model_def.mtplx_fan_mode} set but the thermalforge daemon socket "
            f"{THERMALFORGE_SOCKET} is missing; fans stay on Apple automatic control"
        )
    return warnings
```

In `validate_model_config`, insert as the first statements inside the `for role, model_def in config.models.items():` loop:

```python
        if model_def.backend == "mtplx":
            issues.extend(f"{role}: {e}" for e in mtplx_config_errors(model_def))
            issues.extend(f"{role}: {w}" for w in mtplx_config_warnings(model_def))
        else:
            for field in _MTPLX_ONLY_FIELDS:
                if getattr(model_def, field) is not None:
                    issues.append(
                        f"{role}: {field} only applies to backend mtplx; ignored for "
                        f"{model_def.backend}"
                    )
```

Also in `validate_model_config`, the existing `is_hf_model` check runs for MTPLX too. That is harmless: `mtplx_config_errors` already reported it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_mtplx.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full checks**

Run: `uv run pytest && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/`
Expected: all pass. If `ruff format --check` fails, run `uv run ruff format src/ tests/` and re-run.

- [ ] **Step 6: Commit**

```bash
git add src/slm_server/config.py tests/test_config_mtplx.py
git commit -m "feat(config): add backend mtplx fields and validation

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL"
```

---

### Task 2: MTPLX command builder and launcher branch

**Files:**
- Modify: `src/slm_server/start_backends.py`
- Test: `tests/test_start_backends_mtplx.py`

**Interfaces:**
- Consumes: `ModelDefinition`, `mtplx_config_errors(model_def) -> list[str]` (Task 1)
- Produces:
  - `ALLOWED_MTPLX_REASONING_PARSERS: set[str]`
  - `find_mtplx_binary() -> str | None`
  - `build_mtplx_command(model_def: ModelDefinition, mtplx_bin: str) -> list[str]`
  - `start_model_server(...)` handles `backend == "mtplx"`; log file `logs/mtplx-<safe id>-<port>.log`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_start_backends_mtplx.py`:

```python
"""build_mtplx_command, find_mtplx_binary and the mtplx launcher branch (spec §1, §3)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from slm_server import start_backends as sb
from slm_server.config import ModelConfig, ModelDefinition


def _pack(tmp_path: Path, depth_max: int = 3) -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "config.json").write_text("{}")
    (pack / "mtplx_runtime.json").write_text(json.dumps({"mtp_depth_max": depth_max}))
    return pack


def _mtplx(pack: Path | str, **overrides: object) -> ModelDefinition:
    fields: dict[str, object] = {
        "id": "mtplx-qwen38-27b",
        "backend": "mtplx",
        "port": 8600,
        "quantization": "4bit",
        "default_timeout": 600,
        "model_path": str(pack),
        "mtp_depth": 3,
    }
    fields.update(overrides)
    return ModelDefinition(**fields)


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_minimal_command(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    cmd = sb.build_mtplx_command(_mtplx(pack), "/opt/mtplx")
    assert cmd[:2] == ["/opt/mtplx", "serve"]
    assert _flag(cmd, "--model") == str(pack)
    assert _flag(cmd, "--model-id") == "mtplx-qwen38-27b"
    assert _flag(cmd, "--host") == "127.0.0.1"
    assert _flag(cmd, "--port") == "8600"
    assert "--no-auth" in cmd
    assert _flag(cmd, "--depth") == "3"
    assert _flag(cmd, "--preserve-thinking") == "off"


def test_optional_flags(tmp_path: Path) -> None:
    model_def = _mtplx(
        _pack(tmp_path),
        context_length=131072,
        reasoning_effort="medium",
        preserve_thinking="scoped",
        reasoning_parser="qwen3",
        mtplx_profile="turbo",
        mtplx_batching_preset="agent",
        mtplx_fan_mode="default",
        temp=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=0.0,
    )
    cmd = sb.build_mtplx_command(model_def, "/opt/mtplx")
    assert _flag(cmd, "--context-window") == "131072"
    assert _flag(cmd, "--reasoning-effort") == "medium"
    assert _flag(cmd, "--preserve-thinking") == "scoped"
    assert _flag(cmd, "--reasoning-parser") == "qwen3"
    assert _flag(cmd, "--profile") == "turbo"
    assert _flag(cmd, "--batching-preset") == "agent"
    assert _flag(cmd, "--fan-mode") == "default"
    assert _flag(cmd, "--default-temperature") == "0.6"
    assert _flag(cmd, "--default-top-p") == "0.95"
    assert _flag(cmd, "--default-top-k") == "20"
    assert _flag(cmd, "--default-presence-penalty") == "0.0"


def test_unset_optional_flags_are_absent(tmp_path: Path) -> None:
    cmd = sb.build_mtplx_command(_mtplx(_pack(tmp_path)), "/opt/mtplx")
    for flag in (
        "--context-window",
        "--reasoning-effort",
        "--reasoning-parser",
        "--profile",
        "--batching-preset",
        "--fan-mode",
        "--default-temperature",
    ):
        assert flag not in cmd


def test_host_field_never_changes_bind_address(tmp_path: Path) -> None:
    cmd = sb.build_mtplx_command(_mtplx(_pack(tmp_path), host="0.0.0.0"), "/opt/mtplx")
    assert _flag(cmd, "--host") == "127.0.0.1"


def test_missing_depth_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mtp_depth"):
        sb.build_mtplx_command(_mtplx(_pack(tmp_path), mtp_depth=None), "/opt/mtplx")


def test_depth_above_pack_limit_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mtp_depth_max"):
        sb.build_mtplx_command(_mtplx(_pack(tmp_path, depth_max=3), mtp_depth=5), "/opt/mtplx")


def test_huggingface_id_raises() -> None:
    with pytest.raises(ValueError, match="local directory"):
        sb.build_mtplx_command(_mtplx("Youssofal/Some-Pack"), "/opt/mtplx")


def test_unknown_reasoning_parser_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reasoning_parser"):
        sb.build_mtplx_command(_mtplx(_pack(tmp_path), reasoning_parser="glm4_moe"), "/opt/mtplx")


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_binary_lookup_prefers_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = _executable(tmp_path / "mtplx-override")
    monkeypatch.setenv("SLM_MTPLX_BIN", str(override))
    monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/local/bin/mtplx")
    assert sb.find_mtplx_binary() == str(override)


def test_binary_lookup_uses_path_next(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_MTPLX_BIN", raising=False)
    monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/local/bin/mtplx")
    assert sb.find_mtplx_binary() == "/usr/local/bin/mtplx"


def test_binary_lookup_falls_back_to_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_MTPLX_BIN", raising=False)
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    home = tmp_path / "home"
    (home / ".mtplx" / "bin").mkdir(parents=True)
    fallback = _executable(home / ".mtplx" / "bin" / "mtplx")
    monkeypatch.setattr(sb.Path, "home", classmethod(lambda cls: home))
    assert sb.find_mtplx_binary() == str(fallback)


def test_binary_lookup_returns_none_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLM_MTPLX_BIN", raising=False)
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    monkeypatch.setattr(sb.Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))
    assert sb.find_mtplx_binary() is None


class _FakeProcess:
    pid = 4242
    returncode = None

    def poll(self) -> None:
        return None


def test_start_model_server_launches_mtplx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack = _pack(tmp_path)
    model_def = _mtplx(pack)
    launched: dict[str, object] = {}

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakeProcess:
        launched["cmd"] = cmd
        launched["stderr_name"] = getattr(kwargs.get("stderr"), "name", None)
        return _FakeProcess()

    monkeypatch.setattr(sb, "find_mtplx_binary", lambda: "/opt/mtplx")
    monkeypatch.setattr(sb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sb.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sb, "LOG_DIR", tmp_path / "logs")

    process = sb.start_model_server(model_def, ModelConfig(models={"m": model_def}))

    assert process is not None
    cmd = launched["cmd"]
    assert isinstance(cmd, list) and cmd[:2] == ["/opt/mtplx", "serve"]
    assert str(launched["stderr_name"]).endswith(os.sep + "mtplx-mtplx-qwen38-27b-8600.log")


def test_start_model_server_without_binary_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_def = _mtplx(_pack(tmp_path))
    monkeypatch.setattr(sb, "find_mtplx_binary", lambda: None)
    monkeypatch.setattr(sb, "LOG_DIR", tmp_path / "logs")
    assert sb.start_model_server(model_def, ModelConfig(models={"m": model_def})) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_start_backends_mtplx.py -v`
Expected: FAIL with `AttributeError: module 'slm_server.start_backends' has no attribute 'build_mtplx_command'`.

- [ ] **Step 3: Implement**

In `src/slm_server/start_backends.py`, change the config import to:

```python
from slm_server.config import ModelConfig, ModelDefinition, load_model_config, mtplx_config_errors
```

Add after `ALLOWED_REASONING_PARSERS`:

```python
# Reasoning parsers `mtplx serve --reasoning-parser` accepts (mtplx 2.11.2).
ALLOWED_MTPLX_REASONING_PARSERS = {"qwen3", "step3p5", "gemma4", "poolside_v1", "none"}
```

Add after `find_native_llama_server()`:

```python
def find_mtplx_binary() -> str | None:
    """Return the `mtplx` CLI path, or None if there is none.

    Order: SLM_MTPLX_BIN, then `mtplx` on PATH, then ~/.mtplx/bin/mtplx (where the MTPLX
    app installs its wrapper). The wrapper execs the runtime's Python, so the launched PID
    is the server's PID and the watchdog can stop it directly.
    """
    override = os.environ.get("SLM_MTPLX_BIN")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        log.warning(
            "mtplx_binary_override_unusable",
            path=override,
            message="SLM_MTPLX_BIN is not an executable file; falling back to PATH",
        )
    on_path = shutil.which("mtplx")
    if on_path:
        return on_path
    default = Path.home() / ".mtplx" / "bin" / "mtplx"
    if default.is_file() and os.access(default, os.X_OK):
        return str(default)
    return None


def build_mtplx_command(model_def: ModelDefinition, mtplx_bin: str) -> list[str]:
    """Build the `mtplx serve` command for a backend: mtplx entry.

    Always binds 127.0.0.1 with --no-auth: the router is the only client, and a non-localhost
    bind would require an API key. preserve_thinking defaults to off, because MTPLX's own
    default (auto) resolved to on and kept earlier reasoning where llama.cpp strips it.
    Never sets MTPLX_SSE_HEARTBEAT=0: the pre-first-token keep-alives stop a long cold prefill
    from looking like a stall.

    Raises:
        ValueError: If the entry fails mtplx_config_errors or a CLI value is unsafe.
    """
    errors = mtplx_config_errors(model_def)
    if errors:
        raise ValueError("; ".join(errors))
    if not (1024 <= model_def.port <= 65535):
        raise ValueError(f"Invalid port: {model_def.port}. Must be between 1024 and 65535")
    pack = cast(Path, validate_path(cast(str, model_def.model_path), allow_hf_model=False))
    served_id = cast(str, validate_served_model_name(model_def.id))
    cmd = [
        mtplx_bin,
        "serve",
        "--model",
        str(pack),
        "--model-id",
        served_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(model_def.port),
        "--no-auth",
        "--depth",
        str(model_def.mtp_depth),
        "--preserve-thinking",
        model_def.preserve_thinking or "off",
    ]
    if model_def.context_length is not None:
        cmd.extend(["--context-window", str(model_def.context_length)])
    if model_def.reasoning_effort is not None:
        cmd.extend(["--reasoning-effort", model_def.reasoning_effort])
    parser = validate_parser_name(
        model_def.reasoning_parser, ALLOWED_MTPLX_REASONING_PARSERS, "reasoning_parser"
    )
    if parser is not None:
        cmd.extend(["--reasoning-parser", parser])
    if model_def.mtplx_profile is not None:
        cmd.extend(["--profile", model_def.mtplx_profile])
    if model_def.mtplx_batching_preset is not None:
        cmd.extend(["--batching-preset", model_def.mtplx_batching_preset])
    if model_def.mtplx_fan_mode is not None:
        cmd.extend(["--fan-mode", model_def.mtplx_fan_mode])
    if model_def.temp is not None:
        cmd.extend(["--default-temperature", str(model_def.temp)])
    if model_def.top_p is not None:
        cmd.extend(["--default-top-p", str(model_def.top_p)])
    if model_def.top_k is not None:
        cmd.extend(["--default-top-k", str(model_def.top_k)])
    if model_def.presence_penalty is not None:
        cmd.extend(["--default-presence-penalty", str(model_def.presence_penalty)])
    return cmd
```

In `start_model_server`, add this branch immediately before `elif model_def.backend == "llamacpp":`:

```python
        elif model_def.backend == "mtplx":
            mtplx_bin = find_mtplx_binary()
            if mtplx_bin is None:
                log.error(
                    "mtplx_binary_not_found",
                    model_id=model_def.id,
                    searched=["SLM_MTPLX_BIN", "PATH:mtplx", "~/.mtplx/bin/mtplx"],
                )
                return None
            cmd = build_mtplx_command(model_def, mtplx_bin)
```

The existing log-file code already names the file `mtplx-<id>-<port>.log`, because its prefix is `model_def.backend` for every backend except llamacpp.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_start_backends_mtplx.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full checks**

Run: `uv run pytest && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/slm_server/start_backends.py tests/test_start_backends_mtplx.py
git commit -m "feat(launcher): start mtplx serve for backend mtplx entries

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL"
```

---

### Task 3: Memory budget guard

**Files:**
- Modify: `src/slm_server/config.py`, `src/slm_server/start_backends.py` (`main`)
- Test: `tests/test_memory_budget.py`

**Interfaces:**
- Consumes: `ModelDefinition.peak_memory_gib` (Task 1)
- Produces:
  - `DEFAULT_MEMORY_BUDGET_GIB: float = 100.0`
  - `HEAVY_MODEL_TYPES: tuple[str, ...] = ("lm", "multimodal")`
  - `MemoryBudgetResult` dataclass with `errors: list[str]` and `warnings: list[str]`
  - `check_memory_budget(config: ModelConfig, budget_gib: float | None = None) -> MemoryBudgetResult`
  - `start_backends.main()` exits with status 1 before starting any backend when `errors` is non-empty

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_budget.py`:

```python
"""One heavy engine at a time: the declared memory budget (spec §4)."""

from __future__ import annotations

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
    monkeypatch.setattr(sb, "start_model_server", lambda model_def, config: started.append(model_def.id))
    with pytest.raises(SystemExit) as exc:
        sb.main()
    assert exc.value.code == 1
    assert started == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_memory_budget.py -v`
Expected: FAIL with `ImportError: cannot import name 'check_memory_budget'`.

- [ ] **Step 3: Implement `check_memory_budget`**

In `src/slm_server/config.py`, add to the imports:

```python
import os
from dataclasses import dataclass, field
```

Add after `mtplx_config_warnings`:

```python
DEFAULT_MEMORY_BUDGET_GIB = 100.0
HEAVY_MODEL_TYPES = ("lm", "multimodal")


@dataclass
class MemoryBudgetResult:
    """Outcome of the launcher's memory check. Any error blocks startup."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_memory_budget(config: ModelConfig, budget_gib: float | None = None) -> MemoryBudgetResult:
    """Refuse configs that would load more model memory than the Mac can hold.

    One heavy engine at a time is the intended mode: memory pressure and heat change decode
    speed, and callers share one concurrency pool across every local model.

    Args:
        config: Loaded model configuration.
        budget_gib: Budget in GiB. None reads SLM_MEMORY_BUDGET_GIB, default 100.

    Returns:
        Errors (startup must stop) and warnings (startup continues).
    """
    result = MemoryBudgetResult()
    if budget_gib is None:
        raw = os.environ.get("SLM_MEMORY_BUDGET_GIB")
        budget_gib = DEFAULT_MEMORY_BUDGET_GIB
        if raw is not None:
            try:
                budget_gib = float(raw)
            except ValueError:
                result.warnings.append(
                    f"SLM_MEMORY_BUDGET_GIB={raw!r} is not a number; using "
                    f"{DEFAULT_MEMORY_BUDGET_GIB}"
                )
    enabled = [(role, m) for role, m in config.models.items() if m.enabled]
    heavy = [(role, m) for role, m in enabled if m.model_type in HEAVY_MODEL_TYPES]
    missing = [role for role, m in heavy if m.peak_memory_gib is None]
    if len(heavy) >= 2 and missing:
        result.errors.append(
            f"{len(heavy)} heavy entries are enabled and these lack peak_memory_gib: "
            f"{', '.join(missing)}; declare it on every heavy entry or enable only one"
        )
    elif missing:
        result.warnings.append(
            f"{missing[0]} has no peak_memory_gib; the memory budget cannot check it"
        )
    declared = [(role, m.peak_memory_gib) for role, m in enabled if m.peak_memory_gib is not None]
    total = sum(peak for _role, peak in declared)
    if total > budget_gib:
        detail = ", ".join(f"{role}={peak}" for role, peak in declared)
        result.errors.append(
            f"declared peak memory {total:.1f} GiB exceeds the budget {budget_gib:.1f} GiB "
            f"({detail})"
        )
    return result
```

- [ ] **Step 4: Call it from the launcher**

In `src/slm_server/start_backends.py`, change the config import to:

```python
from slm_server.config import (
    ModelConfig,
    ModelDefinition,
    check_memory_budget,
    load_model_config,
    mtplx_config_errors,
)
```

In `main()`, insert immediately after the `try: config = load_model_config() ... sys.exit(1)` block and before `watchdog_settings = load_watchdog_settings()`:

```python
    # Refuse before anything loads: two heavy engines in one Mac corrupt each other's decode
    # speed, and a start that swaps under pressure is worse than a clear refusal.
    budget = check_memory_budget(config)
    for warning in budget.warnings:
        log.warning("memory_budget_warning", detail=warning)
    if budget.errors:
        log.error("memory_budget_exceeded", errors=budget.errors)
        sys.exit(1)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_memory_budget.py tests/test_start_backends_main.py -v`
Expected: all PASS. The existing `test_start_backends_main.py` tests use one heavy entry without `peak_memory_gib`, so the guard logs a warning and startup continues.

- [ ] **Step 6: Run the full checks**

Run: `uv run pytest && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/slm_server/config.py src/slm_server/start_backends.py tests/test_memory_budget.py
git commit -m "feat(launcher): refuse to start past the declared memory budget

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL"
```

---

### Task 4: HTTP 507 is ignored by the watchdog

**Files:**
- Modify: `src/slm_server/watchdog.py` (`classify_status`)
- Test: `tests/test_watchdog.py`

**Interfaces:**
- Produces: `classify_status(507) == ("ignore", None)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_watchdog.py`:

```python
def test_507_is_ignored() -> None:
    """MTPLX refuses a request that does not fit in memory with 507.

    Not a failure (restarting cannot make the request fit) and not health (a backend
    refusing everything under memory pressure must not reset the failure streak).
    """
    assert wd.classify_status(507) == ("ignore", None)


def test_507_does_not_erase_a_failure_streak() -> None:
    tracker = wd.BackendHealthTracker(failure_threshold=2)
    assert tracker.record_failure(8502, "timeout") is False
    assert wd.classify_status(507)[0] == "ignore"
    assert tracker.record_failure(8502, "timeout") is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_watchdog.py -k 507 -v`
Expected: `test_507_is_ignored` FAILS with `('failure', 'server_error') != ('ignore', None)`.

- [ ] **Step 3: Implement**

In `src/slm_server/watchdog.py`, in `classify_status`, insert immediately before `if status >= 500:`:

```python
    if status == 507:
        # MTPLX's memory governor refuses a request that cannot fit. Restarting cannot make
        # it fit, and counting it as health would let a backend refusing everything under
        # memory pressure look healthy.
        return "ignore", None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_watchdog.py -v`
Expected: all PASS, including the existing `test_5xx_is_a_failure` cases.

- [ ] **Step 5: Run the full checks and commit**

Run: `uv run pytest && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/`

```bash
git add src/slm_server/watchdog.py tests/test_watchdog.py
git commit -m "fix(watchdog): ignore HTTP 507 instead of counting it as a failure

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL"
```

---

### Task 5: Disabled model id returns 404 and is never scored

**Files:**
- Modify: `src/slm_server/router.py` (`_get_model_definition`)
- Modify: `tests/test_router_model_selection.py`, `tests/test_router_watchdog.py`

**Interfaces:**
- Produces: `_get_model_definition` raises `HTTPException(status_code=404, detail="Model '<id>' is configured but currently disabled.")` for a disabled-only id.

- [ ] **Step 1: Change the existing test and add the scoring tests**

In `tests/test_router_model_selection.py`, replace the body assertion of `test_get_model_definition_returns_disabled_message_when_only_disabled`:

```python
    assert exc.value.status_code == 404
    assert "currently disabled" in str(exc.value.detail)
```

Append to `tests/test_router_watchdog.py`:

```python
def _disabled_model_def() -> ModelDefinition:
    return ModelDefinition(
        id="mtplx-idle",
        backend="llamacpp",
        port=8600,
        context_length=32768,
        quantization="4bit",
        max_concurrency=1,
        default_timeout=120,
        model_path="hf/stub",
        enabled=False,
    )


@pytest.mark.parametrize(
    ("model_id", "status"), [("mtplx-idle", 404), ("no-such-model", 404)]
)
def test_router_raised_lookup_errors_are_not_scored(
    monkeypatch: pytest.MonkeyPatch,
    watchdog_settings: wd.WatchdogSettings,
    model_id: str,
    status: int,
) -> None:
    """A request for the engine that is not loaded is normal during swaps.

    It must return a 4xx (clients do not retry it) and never reach the watchdog, which
    only scores responses after an endpoint sets request.state.backend_port.
    """
    cfg = ModelConfig(models={"reasoning": _model_def(), "idle": _disabled_model_def()})
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    monkeypatch.setattr(router_module, "load_watchdog_settings", lambda: watchdog_settings)
    with TestClient(app) as test_client:
        recorded: list[str] = []
        watchdog = app.state.watchdog
        for name in ("record_success", "record_failure", "record_unclassified", "record_request_error"):
            monkeypatch.setattr(watchdog, name, lambda *a, _n=name, **k: recorded.append(_n))
        response = test_client.post(
            "/v1/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == status
    assert recorded == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_router_model_selection.py tests/test_router_watchdog.py -v`
Expected: the disabled cases FAIL with `503 != 404`. The unknown-id case passes already.

- [ ] **Step 3: Implement**

In `src/slm_server/router.py`, in `_get_model_definition`, replace:

```python
    if disabled_match_found:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' is configured but currently disabled.",
        )
```

with:

```python
    if disabled_match_found:
        # 404, not 503: with one heavy engine loaded at a time, a request for the engine
        # that is not loaded is normal during a swap, and clients retry a 5xx.
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' is configured but currently disabled.",
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_router_model_selection.py tests/test_router_watchdog.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full checks and commit**

Run: `uv run pytest && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/`

```bash
git add src/slm_server/router.py tests/test_router_model_selection.py tests/test_router_watchdog.py
git commit -m "fix(router): return 404 for a disabled model id instead of 503

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL"
```

---

### Task 6: Router chat_template_kwargs translation for MTPLX

**Files:**
- Modify: `src/slm_server/router.py` (new helper; three call sites)
- Test: `tests/test_router_mtplx.py`

**Interfaces:**
- Consumes: `ModelDefinition.backend == "mtplx"` (Task 1)
- Produces: `_prepare_chat_template_kwargs(body: dict[str, Any], model_def: ModelDefinition) -> dict[str, Any]` (returns a new dict; never mutates `body`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_router_mtplx.py`:

```python
"""Router behaviour for backend: mtplx entries (spec §2)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from slm_server import router as router_module  # type: ignore[import-untyped]
from slm_server.config import ModelConfig, ModelDefinition  # type: ignore[import-untyped]

app = router_module.app
prepare = router_module._prepare_chat_template_kwargs


def _mtplx_def(**overrides: object) -> ModelDefinition:
    fields: dict[str, object] = {
        "id": "mtplx-test",
        "backend": "mtplx",
        "port": 8600,
        "quantization": "4bit",
        "default_timeout": 600,
        "model_path": "/path/to/pack",
        "mtp_depth": 3,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    fields.update(overrides)
    return ModelDefinition(**fields)


def _llama_def(**overrides: object) -> ModelDefinition:
    fields: dict[str, object] = {
        "id": "llama-test",
        "backend": "llamacpp",
        "port": 8502,
        "quantization": "Q4",
        "default_timeout": 600,
        "model_path": "/path/to/model.gguf",
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
    }
    fields.update(overrides)
    return ModelDefinition(**fields)


# ── helper ────────────────────────────────────────────────────────────────


def test_llamacpp_injects_config_kwargs_only_when_request_has_none() -> None:
    model_def = _llama_def()
    assert prepare({"model": "x"}, model_def)["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    body = {"model": "x", "chat_template_kwargs": {"enable_thinking": False}}
    assert prepare(body, model_def) == body


def test_mtplx_merges_config_and_request_kwargs_per_key() -> None:
    body = {"model": "x", "chat_template_kwargs": {"enable_thinking": False, "foo": 1}}
    out = prepare(body, _mtplx_def())
    assert out["chat_template_kwargs"] == {"enable_thinking": False, "foo": 1}


def test_mtplx_uses_config_kwargs_when_request_has_none() -> None:
    out = prepare({"model": "x"}, _mtplx_def())
    assert out["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in out


def test_mtplx_lifts_request_reasoning_effort_to_top_level() -> None:
    body = {"model": "x", "chat_template_kwargs": {"reasoning_effort": "low"}}
    out = prepare(body, _mtplx_def())
    assert out["reasoning_effort"] == "low"
    assert out["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_mtplx_top_level_reasoning_effort_wins() -> None:
    body = {
        "model": "x",
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    assert prepare(body, _mtplx_def())["reasoning_effort"] == "xhigh"


def test_mtplx_does_not_lift_from_config_kwargs() -> None:
    model_def = _mtplx_def(chat_template_kwargs={"reasoning_effort": "low"})
    assert "reasoning_effort" not in prepare({"model": "x"}, model_def)


def test_prepare_does_not_mutate_the_request_body() -> None:
    body = {"model": "x", "chat_template_kwargs": {"reasoning_effort": "low"}}
    snapshot = {"model": "x", "chat_template_kwargs": {"reasoning_effort": "low"}}
    prepare(body, _mtplx_def())
    assert body == snapshot


# ── through the router ─────────────────────────────────────────────────────


@pytest.fixture
def mtplx_client(monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(models={"mtplx": _mtplx_def()})
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    with TestClient(app) as client:
        yield client


def test_chat_forwards_translated_body_and_headers(mtplx_client: TestClient) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["headers"] = {k.lower(): v for k, v in (kwargs.get("headers") or {}).items()}
        return httpx.Response(200, json={"id": "ok", "choices": []})

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    response = mtplx_client.post(
        "/v1/chat/completions",
        json={
            "model": "mtplx-test",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
        headers={
            "X-Session-Id": "sess-1",
            "X-Trace-Id": "trace-1",
            "X-Span-Id": "span-1",
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        },
    )

    assert response.status_code == 200
    assert captured["url"] == "http://localhost:8600/v1/chat/completions"
    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["json"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }
    for header in ("x-session-id", "x-trace-id", "x-span-id", "traceparent"):
        assert header in captured["headers"]


def test_responses_fallback_translates_the_chat_body(mtplx_client: TestClient) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/v1/responses"):
            return httpx.Response(404, json={"error": "not found"})
        captured["json"] = kwargs.get("json")
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "hi"}}]},
        )

    app.state.http_client.post = fake_post  # type: ignore[method-assign]

    response = mtplx_client.post(
        "/v1/responses",
        json={
            "model": "mtplx-test",
            "input": "hi",
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
    )

    assert response.status_code == 200
    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["json"]["chat_template_kwargs"]["enable_thinking"] is True


class _FakeStreamingClient:
    """Stands in for the shared httpx.AsyncClient on the streaming path.

    `bodies` maps a URL suffix to (status, SSE bytes). `sent` records each forwarded body.
    """

    def __init__(self, bodies: dict[str, tuple[int, bytes]]) -> None:
        self.bodies = bodies
        self.sent: list[tuple[str, Any]] = []

    def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
        self.sent.append((url, kwargs.get("json")))
        return httpx.Request(method, url)

    async def send(self, request: httpx.Request, stream: bool = False) -> httpx.Response:
        for suffix, (status, body) in self.bodies.items():
            if str(request.url).endswith(suffix):
                return httpx.Response(
                    status,
                    headers={"content-type": "text/event-stream"},
                    content=body,
                    request=request,
                )
        raise AssertionError(f"unexpected backend URL: {request.url}")


async def test_streaming_responses_fallback_translates_the_chat_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ModelConfig(models={"mtplx": _mtplx_def()})
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    app.state.model_config = cfg
    fake = _FakeStreamingClient(
        {
            "/v1/responses": (404, b""),
            "/v1/chat/completions": (200, b"data: [DONE]\n\n"),
        }
    )
    app.state.http_client = fake

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "mtplx-test",
                "input": "hi",
                "stream": True,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
        )

    assert response.status_code == 200
    url, body = fake.sent[-1]
    assert url == "http://localhost:8600/v1/chat/completions"
    assert body["reasoning_effort"] == "low"
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_router_mtplx.py -v`
Expected: FAIL at collection with `AttributeError: module 'slm_server.router' has no attribute '_prepare_chat_template_kwargs'`.

- [ ] **Step 3: Implement the helper**

In `src/slm_server/router.py`, add after `_filtered_forward_headers`:

```python
def _prepare_chat_template_kwargs(
    body: dict[str, Any], model_def: ModelDefinition
) -> dict[str, Any]:
    """Return a copy of `body` with this backend's chat_template_kwargs applied.

    llama.cpp and MLX: unchanged behaviour. The config kwargs are injected only when the
    request carries none; llama-server merges request kwargs over its launch kwargs itself.

    MTPLX: the config and request kwargs merge per key (request wins), matching llama-server.
    MTPLX reads chat_template_kwargs.enable_thinking but not .reasoning_effort, so a
    reasoning_effort the REQUEST put in its kwargs is lifted to the top level. Config kwargs
    never hold that key on an MTPLX entry (validation), so the configured effort comes only
    from the launch flag.
    """
    out = dict(body)
    config_kwargs = model_def.chat_template_kwargs or {}
    if model_def.backend != "mtplx":
        if config_kwargs and "chat_template_kwargs" not in out:
            out["chat_template_kwargs"] = config_kwargs
        return out
    raw_request_kwargs = body.get("chat_template_kwargs")
    request_kwargs = raw_request_kwargs if isinstance(raw_request_kwargs, dict) else {}
    merged = {**config_kwargs, **request_kwargs}
    if merged:
        out["chat_template_kwargs"] = merged
    if "reasoning_effort" in request_kwargs and "reasoning_effort" not in out:
        out["reasoning_effort"] = request_kwargs["reasoning_effort"]
    return out
```

If `ModelDefinition` is not already imported in `router.py`, the existing import line `from slm_server.config import ModelConfig, ModelDefinition, load_model_config` already provides it.

- [ ] **Step 4: Replace the three call sites**

In `chat_completions`, replace:

```python
        if (
            getattr(model_def, "chat_template_kwargs", None)
            and "chat_template_kwargs" not in body_forward
        ):
            body_forward["chat_template_kwargs"] = model_def.chat_template_kwargs
```

with:

```python
        body_forward = _prepare_chat_template_kwargs(body_forward, model_def)
```

In the streaming `/v1/responses` fallback, replace:

```python
            chat_body = _convert_responses_to_chat(body_forward)
            if (
                getattr(model_def, "chat_template_kwargs", None)
                and "chat_template_kwargs" not in chat_body
            ):
                chat_body["chat_template_kwargs"] = model_def.chat_template_kwargs
```

with:

```python
            chat_body = _prepare_chat_template_kwargs(
                _convert_responses_to_chat(body_forward), model_def
            )
```

In the non-streaming `/v1/responses` fallback, replace:

```python
        chat_body = _convert_responses_to_chat(body_forward)
        if (
            getattr(model_def, "chat_template_kwargs", None)
            and "chat_template_kwargs" not in chat_body
        ):
            chat_body["chat_template_kwargs"] = model_def.chat_template_kwargs
```

with:

```python
        chat_body = _prepare_chat_template_kwargs(
            _convert_responses_to_chat(body_forward), model_def
        )
```

After the edits, `grep -n 'model_def.chat_template_kwargs' src/slm_server/router.py` must print only the line inside `_prepare_chat_template_kwargs`. `_convert_responses_to_chat` starts from `body.copy()`, so a request's `chat_template_kwargs` reaches the helper.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_router_mtplx.py tests/test_router_chat.py -v`
Expected: all PASS. The existing chat tests prove the llama.cpp and MLX behaviour did not change.

- [ ] **Step 6: Run the full checks and commit**

Run: `uv run pytest && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/`

```bash
git add src/slm_server/router.py tests/test_router_mtplx.py
git commit -m "feat(router): translate chat_template_kwargs for backend mtplx

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL"
```

---

### Task 7: Stream pass-through and usage-based telemetry

**Files:**
- Modify: `src/slm_server/router.py` (`_build_request_telemetry`)
- Modify: `src/slm_server/telemetry.py` (`_ATTRIBUTE_SOURCES`)
- Test: `tests/test_router_mtplx.py`

**Interfaces:**
- Consumes: `_prepare_chat_template_kwargs` (Task 6)
- Produces: telemetry doc fields `cache_reuse` (from `timings.cache_n`, else `usage.prompt_tokens_details.cached_tokens`) and `reasoning_tokens` (from `usage.completion_tokens_details.reasoning_tokens`); span attribute `slm.reasoning_tokens`

- [ ] **Step 1: Write the failing tests**

In `tests/test_router_mtplx.py`, add these lines to the top import block (ruff E402 rejects imports below code):

```python
import asyncio
import json
from collections.abc import Mapping
```

Then append this code to the end of the file. It reuses `_FakeStreamingClient` from Task 6:

```python
_MTPLX_SSE = (
    b": keep-alive\n\n"
    + b"data: "
    + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q":"x"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    ).encode()
    + b"\n\n"
    + b"data: "
    + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}).encode()
    + b"\n\n"
    + b"data: "
    + json.dumps(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 8422,
                "completion_tokens": 24,
                "prompt_tokens_details": {"cached_tokens": 6388},
                "completion_tokens_details": {"reasoning_tokens": 35},
            },
        }
    ).encode()
    + b"\n\ndata: [DONE]\n\n"
)


async def test_stream_passes_mtplx_chunks_through_and_records_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ModelConfig(models={"mtplx": _mtplx_def()})
    monkeypatch.setattr(
        router_module, "load_model_config", lambda config_path=None, validate=True: cfg
    )
    app.state.model_config = cfg
    app.state.http_client = _FakeStreamingClient({"/v1/chat/completions": (200, _MTPLX_SSE)})
    docs: list[dict] = []

    def fake_emit(doc: dict, *, emit_path: str, headers: Mapping[str, str]) -> None:
        docs.append(doc)

    monkeypatch.setattr(router_module, "emit_request_span", fake_emit)

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=True), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mtplx-test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        text = response.text

    await asyncio.sleep(0)

    assert response.status_code == 200
    assert ": keep-alive" in text
    assert '"finish_reason": "length"' in text
    assert '"tool_calls"' in text
    assert '"cached_tokens": 6388' in text
    assert '"reasoning_tokens": 35' in text
    assert docs, "a telemetry doc was emitted"
    assert docs[-1]["cache_reuse"] == 6388
    assert docs[-1]["reasoning_tokens"] == 35


def test_llamacpp_cache_reuse_still_comes_from_timings() -> None:
    doc = router_module._build_request_telemetry(
        trace_id=None,
        span_id=None,
        session_id=None,
        model_id="llama-test",
        backend="llamacpp",
        port=8502,
        usage={"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 1}},
        timings={"cache_n": 80},
        total_ms=1.0,
        status=200,
    )
    assert doc["cache_reuse"] == 80
    assert doc["reasoning_tokens"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_router_mtplx.py -k "stream_passes or cache_reuse_still" -v`
Expected: FAIL with `KeyError: 'reasoning_tokens'`. The pass-through assertions on `text` already pass, because the router forwards bytes unchanged; they stay as regression guards.

- [ ] **Step 3: Implement the telemetry fields**

In `src/slm_server/router.py`, in `_build_request_telemetry`, insert before the `return {`:

```python
    prompt_details = (usage or {}).get("prompt_tokens_details") or {}
    completion_details = (usage or {}).get("completion_tokens_details") or {}
```

In the returned dict, replace the `cache_reuse` line with:

```python
        # llama.cpp reports cache hits in timings; MTPLX only in usage.prompt_tokens_details.
        "cache_reuse": (
            timings.get("cache_n") if timings else prompt_details.get("cached_tokens")
        ),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
```

In `src/slm_server/telemetry.py`, in `_ATTRIBUTE_SOURCES`, add after `("slm.cache_reuse", "cache_reuse"),`:

```python
    ("slm.reasoning_tokens", "reasoning_tokens"),
```

- [ ] **Step 4: Update the exact telemetry key set**

`tests/test_router_rerank.py` asserts that every endpoint emits an identical key set. In `_TELEMETRY_KEYS`, replace:

```python
    "prompt_n", "predicted_n", "cache_reuse", "total_ms", "status", "ts",
```

with:

```python
    "prompt_n", "predicted_n", "cache_reuse", "reasoning_tokens", "total_ms", "status", "ts",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_router_mtplx.py tests/test_router_chat.py tests/test_router_rerank.py tests/test_telemetry.py -v`
Expected: all PASS.

- [ ] **Step 6: Run the full checks and commit**

Run: `uv run pytest && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/`

```bash
git add src/slm_server/router.py src/slm_server/telemetry.py tests/test_router_mtplx.py tests/test_router_rerank.py
git commit -m "feat(telemetry): record usage-based cache and reasoning tokens

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL"
```

---

### Task 8: Configurable backend readiness timeout in start.sh

**Files:**
- Modify: `start.sh` (`verify_backend_ports`)

**Interfaces:**
- Produces: `SLM_BACKEND_READY_TIMEOUT` (seconds, default 180) controls how long `start.sh` waits for backend ports.

- [ ] **Step 1: Implement**

In `start.sh`, inside `verify_backend_ports`, replace:

```bash
    local max_attempts=30
```

with:

```bash
    # One check per second. MTPLX opens its port only after the model load and the
    # foreground warm-up, so a cold start can take minutes; llama-server opens its port
    # first and loads afterwards. A healthy start still finishes as soon as ports answer.
    local max_attempts="${SLM_BACKEND_READY_TIMEOUT:-180}"
    case "$max_attempts" in
        ''|*[!0-9]*)
            echo "⚠️  SLM_BACKEND_READY_TIMEOUT='$max_attempts' is not a whole number; using 180"
            max_attempts=180
            ;;
    esac
```

- [ ] **Step 2: Verify**

Run: `bash -n start.sh && grep -n 'SLM_BACKEND_READY_TIMEOUT' start.sh`
Expected: no syntax error output; the grep prints the new lines.

Run: `bash -c 'SLM_BACKEND_READY_TIMEOUT=abc; max_attempts="${SLM_BACKEND_READY_TIMEOUT:-180}"; case "$max_attempts" in ""|*[!0-9]*) max_attempts=180;; esac; echo $max_attempts'`
Expected: `180`

Do NOT run `./start.sh` in this task. Live startup timing is checked only in the post-arm-4 verification window.

- [ ] **Step 3: Commit**

```bash
git add start.sh
git commit -m "feat(start): make the backend readiness wait configurable

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL"
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md`, `config/models.yaml.example`, `CLAUDE.md`

- [ ] **Step 1: Add the README section**

Run: `grep -n '^## ' README.md`. Insert the section below immediately before the `## Watchdog` heading. If there is no `## Watchdog` heading, insert it immediately before `## Telemetry`.

````markdown
## MTPLX backend

`backend: mtplx` runs a Youssofal MTPLX pack with `mtplx serve` (Apple Silicon, MLX, native MTP
speculative decoding). llama.cpp stays the default backend.

Requirements:

- The `mtplx` CLI: `SLM_MTPLX_BIN`, `mtplx` on `PATH`, or `~/.mtplx/bin/mtplx`.
- `model_path`: a local pack directory containing `config.json` and `mtplx_runtime.json`.
- `mtp_depth`: required. `mtplx serve` ignores saved tuning, so tune first and copy the result:
  `mtplx tune --model /path/to/pack --retune` (tune tests depths 1–3 only).

MTPLX-only fields: `mtp_depth`, `reasoning_effort`, `preserve_thinking` (default `off`),
`mtplx_profile`, `mtplx_batching_preset` (unset = serial, one request at a time),
`mtplx_fan_mode` (`smart` and `max` need the thermalforge daemon at `/tmp/thermalforge.sock`).
Set effort only as the top-level `reasoning_effort` field, never inside `chat_template_kwargs`.

Behaviour to know:

- The server binds `127.0.0.1` with `--no-auth`; the router is its only client.
- The port opens after the model load and the foreground warm-up. An extended warm-up then runs in
  the background and yields to requests. Timing-sensitive clients wait until `/health`
  `warmup.background.state` leaves `running`. `SLM_BACKEND_READY_TIMEOUT` (default 180 s) sets how
  long `start.sh` waits for ports.
- Streaming requests get MTPLX keep-alive comments before the first token. Non-streaming requests
  get no bytes until MTPLX finishes, so proxy read timeouts still apply.
- MTPLX answers HTTP 507 when a request does not fit in memory. The watchdog ignores 507.
- The router's first-byte stall rule cannot detect a wedged backend that still sends keep-alives
  (MTPLX) or pings (llama.cpp). MTPLX's own 300 s stream-stall deadline is the partial guard.

### Memory budget

Run one heavy engine at a time. Declare `peak_memory_gib` on each `lm` or `multimodal` entry.
The backend launcher refuses to start when two or more heavy entries are enabled and any lacks
`peak_memory_gib`, or when the declared total exceeds `SLM_MEMORY_BUDGET_GIB` (default 100).
````

- [ ] **Step 2: Add the example entry**

Append to `config/models.yaml.example`, inside the `models:` mapping, keeping two-space indentation:

```yaml
  # MTPLX backend (Apple Silicon, MLX, native MTP). Tune first:
  #   mtplx tune --model /path/to/models/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed --retune
  # then copy the best depth into mtp_depth.
  mtplx_27b_example:
    id: "mtplx-qwen38-27b-optimized-speed"
    backend: "mtplx"
    port: 8600
    model_type: "lm"
    context_length: 131072
    quantization: "4bit"
    max_concurrency: 1
    default_timeout: 600
    reasoning_parser: "qwen3"
    reasoning_effort: "medium"
    preserve_thinking: "off"
    mtp_depth: 3
    peak_memory_gib: 24
    model_path: "/path/to/models/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    enabled: false
```

Run: `uv run python -c "import yaml; from slm_server.config import ModelConfig; ModelConfig(**yaml.safe_load(open('config/models.yaml.example')))"`
Expected: no output (the example still parses).

- [ ] **Step 3: Update CLAUDE.md**

In `CLAUDE.md`, replace the sentence:

```
It supports two backends: **llama.cpp** (primary, via native `llama-server` binary or the `llama-cpp-python` fallback) and **MLX** (via `mlx-openai-server`).
```

with:

```
It supports three backends: **llama.cpp** (primary, via native `llama-server` binary or the `llama-cpp-python` fallback), **MLX** (via `mlx-openai-server`), and **MTPLX** (via `mtplx serve`).
```

In the `start_backends.py` bullet, replace `Four command builders:` with `Five command builders:` and append this sentence to the end of that bullet:

```
`build_mtplx_command()` launches `mtplx serve` for `backend: mtplx` (binary from `SLM_MTPLX_BIN`, PATH, or `~/.mtplx/bin/mtplx`); `main()` refuses to start when `check_memory_budget()` reports an error.
```

In the `config.py` bullet, append:

```
Also MTPLX validation (`mtplx_config_errors`/`mtplx_config_warnings`) and the memory budget (`check_memory_budget`, `SLM_MEMORY_BUDGET_GIB`).
```

In the `router.py` bullet, append:

```
For `backend: mtplx` entries, `_prepare_chat_template_kwargs()` merges config and request `chat_template_kwargs` and lifts a request's `reasoning_effort` to the top level. A disabled model id returns 404.
```

In the `watchdog.py` bullet, append:

```
HTTP 507 (MTPLX memory refusal) is ignored: it neither counts as a failure nor resets the failure streak.
```

In the "Key patterns" list, replace:

```
(`if model_def.backend == "mlx"` / `"llamacpp"` / `"mlx-rerank"`)
```

with:

```
(`if model_def.backend == "mlx"` / `"llamacpp"` / `"mlx-rerank"` / `"mtplx"`)
```

In the Configuration section, replace `backend (\`mlx\`/\`llamacpp\`/\`mlx-rerank\`)` with `backend (\`mlx\`/\`llamacpp\`/\`mlx-rerank\`/\`mtplx\`)`.

If any quoted sentence above is not found verbatim, run `grep -n 'backends' CLAUDE.md` and apply the same meaning to the current wording; keep the change minimal.

- [ ] **Step 4: Run the full checks and commit**

Run: `uv run pytest && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/`

```bash
git add README.md config/models.yaml.example CLAUDE.md
git commit -m "docs: document the MTPLX backend and the memory budget

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HKZSVJuS3ehHkGcCt6msQL"
```

---

## After Task 9

Live verification (spec §8 step 3) is NOT part of this plan's execution. It runs only in the owner-approved window after FRE-1517 arm 4, together with the FRE-1519 and FRE-1474 live checks.
