"""Backend stderr logs must not grow without bound.

llama-server logs every request at INFO, and the launcher rotates a log only
when its backend restarts. The supervisor trims an oversized log on its sweep.
"""

import json
from pathlib import Path

import pytest

from slm_server import watchdog as wd


def _log(log_dir: Path, name: str, size: int) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_bytes(b"x" * size)
    return path


def test_an_oversized_backend_log_is_kept_as_prev_and_truncated(tmp_path: Path) -> None:
    path = _log(tmp_path, "llama-unsloth_qwen3.8-flash-next-8502.log", 200)
    assert wd.trim_backend_logs(tmp_path, 100) == [path]
    assert path.stat().st_size == 0
    assert (tmp_path / (path.name + ".prev")).read_bytes() == b"x" * 200


def test_a_log_under_the_cap_is_untouched(tmp_path: Path) -> None:
    path = _log(tmp_path, "mlx-rerank-mlx-community_Qwen3-Reranker-4B-8506.log", 50)
    assert wd.trim_backend_logs(tmp_path, 100) == []
    assert path.stat().st_size == 50


@pytest.mark.parametrize("name", ["start.out", "watchdog.jsonl", "router.log", "notes.log"])
def test_files_that_are_not_backend_logs_are_never_trimmed(tmp_path: Path, name: str) -> None:
    path = _log(tmp_path, name, 200)
    assert wd.trim_backend_logs(tmp_path, 100) == []
    assert path.stat().st_size == 200


def test_a_zero_cap_disables_trimming(tmp_path: Path) -> None:
    path = _log(tmp_path, "llama-test_model-8502.log", 200)
    assert wd.trim_backend_logs(tmp_path, 0) == []
    assert path.stat().st_size == 200


def test_an_append_mode_writer_continues_at_the_new_end(tmp_path: Path) -> None:
    """The backend keeps its file handle open across the trim."""
    path = tmp_path / "llama-test_model-8502.log"
    with open(path, "a") as writer:
        writer.write("before trim\n" * 20)
        writer.flush()
        wd.trim_backend_logs(tmp_path, 10)
        writer.write("after trim\n")
        writer.flush()
    assert path.read_text() == "after trim\n"


def test_the_supervisor_sweep_trims_and_records_it(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    path = _log(log_dir, "llama-test_model-8502.log", 200)
    settings = wd.WatchdogSettings(
        request_dir=tmp_path / "requests",
        log_path=tmp_path / "watchdog.jsonl",
        backend_log_dir=log_dir,
        backend_log_max_bytes=100,
    )
    supervisor = wd.BackendSupervisor(settings, start_fn=lambda _m: None, sleep_fn=lambda _s: None)

    supervisor.poll_once()

    assert path.stat().st_size == 0
    events = [json.loads(line) for line in settings.log_path.read_text().splitlines()]
    assert {"event": "backend_log_trimmed", "path": str(path)}.items() <= events[-1].items()


def test_settings_built_directly_never_trim(tmp_path: Path) -> None:
    """Tests construct WatchdogSettings directly and must not touch the live logs/."""
    assert wd.WatchdogSettings().backend_log_dir is None


def test_load_settings_reads_the_log_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SLM_BACKEND_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_BACKEND_LOG_MAX_MB", "7")
    settings = wd.load_settings()
    assert settings.backend_log_dir == tmp_path
    assert settings.backend_log_max_bytes == 7 * 1024 * 1024
