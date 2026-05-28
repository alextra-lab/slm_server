#!/usr/bin/env python3
"""Benchmark spec_draft_n_max values for MTP speculative decoding.

Starts llama-server for each model × draft-n value combination
(0 = no-MTP baseline, 1+ with draft-mtp), collects mean tok/s per run,
then prints a per-model detail table and a final cross-model comparison.

Usage:
    # Two configured roles + one ad-hoc 27B model
    uv run python scripts/benchmark_mtp_draft_n.py \\
        --models sub_agent_qwen36,reasoning \\
        --model-path /Volumes/.../Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf \\
        --thinking

    # Just one role
    uv run python scripts/benchmark_mtp_draft_n.py --models sub_agent_qwen36
"""

import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_server.config import ModelDefinition, load_model_config
from slm_server.start_backends import build_llama_native_command, find_native_llama_server

console = Console()

BENCH_PORT = 8599
HEALTH_TIMEOUT = 120

BENCH_PROMPT = (
    "Write a detailed technical explanation of how transformer attention mechanisms work, "
    "covering multi-head attention, key-query-value projections, scaled dot-product attention, "
    "and how positional encodings interact with the attention computation."
)

# Fixed per-request budget — same for all models so results are comparable.
# Independent of the server's --n-predict cap (passed from models.yaml at startup).
BENCH_MAX_TOKENS = 512

# Sampling fixed for all runs — determines MTP acceptance rate; must reflect real usage.
BENCH_TEMP = 0.6
BENCH_TOP_P = 0.85
BENCH_TOP_K = 10


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def wait_for_ready(port: int, timeout: int = HEALTH_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_completion(port: int, thinking: bool) -> float | None:
    """Return predicted tok/s from llama.cpp timings, or None on failure."""
    payload = {
        "model": "bench",
        "messages": [{"role": "user", "content": BENCH_PROMPT}],
        "max_tokens": BENCH_MAX_TOKENS,
        "temperature": BENCH_TEMP,
        "top_p": BENCH_TOP_P,
        "top_k": BENCH_TOP_K,
        "stream": False,
    }
    if thinking:
        payload["thinking"] = {"type": "enabled", "budget_tokens": BENCH_MAX_TOKENS // 2}
    try:
        r = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=600)
        r.raise_for_status()
        timings = r.json().get("timings", {})
        tps = timings.get("predicted_per_second")
        return float(tps) if tps is not None else None
    except Exception as e:
        console.print(f"  [red]request failed: {e}[/red]")
        return None


def build_server_cmd(
    model_path: Path,
    model_def: ModelDefinition,
    native_bin: str,
    spec_type: str | None,
    spec_draft_n_max: int | None,
) -> list[str]:
    return build_llama_native_command(
        model_path,
        BENCH_PORT,
        model_def.context_length,
        model_def.quantization,
        model_def.max_concurrency,
        chat_template_kwargs=getattr(model_def, "chat_template_kwargs", None),
        chat_template_file=getattr(model_def, "chat_template_file", None),
        mmproj_path=getattr(model_def, "mmproj_path", None),
        model_alias="bench",
        llama_server_bin=native_bin,
        model_type=getattr(model_def, "model_type", "lm"),
        temp=BENCH_TEMP,
        top_p=BENCH_TOP_P,
        top_k=BENCH_TOP_K,
        min_p=None,
        presence_penalty=None,
        kv_unified=getattr(model_def, "kv_unified", None),
        cache_type_k=getattr(model_def, "cache_type_k", None),
        cache_type_v=getattr(model_def, "cache_type_v", None),
        flash_attn=getattr(model_def, "flash_attn", None),
        fit=getattr(model_def, "fit", None),
        n_predict=getattr(model_def, "n_predict", None),
        cont_batching=getattr(model_def, "cont_batching", None),
        cache_prompt=getattr(model_def, "cache_prompt", None),
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
    )


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def benchmark_one_value(
    draft_n: int,
    model_def: ModelDefinition,
    model_path: Path,
    native_bin: str,
    reps: int,
    thinking: bool,
) -> dict:
    """Run one server instance for draft_n, return result dict."""
    if draft_n == 0:
        cmd = build_server_cmd(model_path, model_def, native_bin, None, None)
        label = "no-MTP"
    else:
        cmd = build_server_cmd(model_path, model_def, native_bin, "draft-mtp", draft_n)
        label = f"n={draft_n}"

    console.print(f"\n  [cyan]▶ {label}[/cyan]  [dim]{' '.join(cmd[-4:])}…[/dim]")

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for_ready(BENCH_PORT):
            console.print(f"  [red]server did not become ready within {HEALTH_TIMEOUT}s[/red]")
            return {"draft_n": draft_n, "label": label, "error": "timeout"}

        console.print("    warming up…")
        run_completion(BENCH_PORT, thinking)

        samples = []
        for i in range(reps):
            tps = run_completion(BENCH_PORT, thinking)
            if tps is not None:
                samples.append(tps)
                console.print(f"    run {i+1}/{reps}: {tps:.1f} tok/s")

        return {
            "draft_n": draft_n,
            "label": label,
            "mean_tps": mean(samples) if samples else None,
            "stdev_tps": stdev(samples) if len(samples) > 1 else 0.0,
            "samples": len(samples),
        }
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)


def benchmark_model(
    label: str,
    model_def: ModelDefinition,
    model_path: Path,
    native_bin: str,
    values: list[int],
    reps: int,
    thinking: bool,
) -> list[dict]:
    """Benchmark all draft-n values for one model; print per-model detail table."""
    server_n_predict = getattr(model_def, "n_predict", None)
    console.print(f"\n[bold underline]{label}[/bold underline]  ({model_def.id})")
    console.print(
        f"  thinking: {'on' if thinking else 'off'} | "
        f"server n_predict cap: {server_n_predict or 'unlimited'} | "
        f"bench max_tokens: {BENCH_MAX_TOKENS}"
    )

    results = []
    for draft_n in values:
        result = benchmark_one_value(draft_n, model_def, model_path, native_bin, reps, thinking)
        results.append(result)

    # Per-model detail table
    valid = [r for r in results if r.get("mean_tps") is not None]
    baseline_tps = next((r["mean_tps"] for r in valid if r["draft_n"] == 0), None)
    best = max(valid, key=lambda r: r["mean_tps"]) if valid else None

    tbl = Table(title=f"\n{label} results")
    tbl.add_column("draft_n", style="cyan", justify="center")
    tbl.add_column("mean tok/s", style="green", justify="right")
    tbl.add_column("±stdev", style="dim", justify="right")
    tbl.add_column("vs baseline", style="yellow", justify="right")

    for r in results:
        is_best = best and r["draft_n"] == best["draft_n"]
        style = "bold green" if is_best else ""
        def cell(s, _s=style):
            return f"[{_s}]{s}[/{_s}]" if _s else str(s)

        if r.get("mean_tps") is None:
            tbl.add_row(str(r["draft_n"]), "[red]error[/red]", "—", "—")
        else:
            pct = ((r["mean_tps"] / baseline_tps) - 1) * 100 if baseline_tps else 0
            tbl.add_row(cell(r["draft_n"]), cell(f"{r['mean_tps']:.1f}"),
                        f"{r['stdev_tps']:.1f}", cell(f"{pct:+.1f}%"))

    console.print(tbl)
    return results


# ---------------------------------------------------------------------------
# Cross-model comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    all_results: list[tuple[str, list[dict]]],
    values: list[int],
) -> None:
    """Print a final table: rows = draft_n, columns = models, cells = mean tok/s (+% vs own baseline)."""
    tbl = Table(title="\n[bold]Cross-model comparison[/bold]  (tok/s, +% vs own no-MTP baseline)")
    tbl.add_column("draft_n", style="cyan", justify="center")
    for label, _ in all_results:
        tbl.add_column(label, justify="right")

    # Collect per-model baselines and bests
    baselines: dict[str, float | None] = {}
    bests: dict[str, int | None] = {}
    for label, results in all_results:
        valid = [r for r in results if r.get("mean_tps") is not None]
        baselines[label] = next((r["mean_tps"] for r in valid if r["draft_n"] == 0), None)
        best = max(valid, key=lambda r: r["mean_tps"]) if valid else None
        bests[label] = best["draft_n"] if best else None

    for draft_n in values:
        row = [str(draft_n) if draft_n > 0 else "0 (baseline)"]
        for label, results in all_results:
            r = next((x for x in results if x["draft_n"] == draft_n), None)
            if r is None or r.get("mean_tps") is None:
                row.append("—")
                continue
            baseline = baselines[label]
            pct = ((r["mean_tps"] / baseline) - 1) * 100 if baseline else 0
            is_best = bests[label] == draft_n
            text = f"{r['mean_tps']:.1f}  ({pct:+.1f}%)"
            row.append(f"[bold green]{text}[/bold green]" if is_best else text)
        tbl.add_row(*row)

    console.print(tbl)

    # Recommendation per model
    console.print()
    for label, results in all_results:
        valid = [r for r in results if r.get("mean_tps") is not None]
        baseline_tps = baselines[label]
        best = max(valid, key=lambda r: r["mean_tps"]) if valid else None
        if not best or not baseline_tps:
            continue
        gain = ((best["mean_tps"] / baseline_tps) - 1) * 100
        if best["draft_n"] == 0:
            console.print(f"  [yellow]{label}[/yellow]: MTP provides no benefit — remove spec_type")
        else:
            console.print(
                f"  [green]{label}[/green]: optimal spec_draft_n_max=[bold]{best['draft_n']}[/bold] "
                f"({best['mean_tps']:.1f} tok/s, {gain:+.1f}% vs baseline)"
            )


# ---------------------------------------------------------------------------
# Ad-hoc model construction
# ---------------------------------------------------------------------------

def make_adhoc_model_def(
    model_path: Path,
    thinking: bool,
    template_ref: ModelDefinition,
) -> ModelDefinition:
    quant = model_path.stem.split("-")[-1] if "-" in model_path.stem else "UD-Q4_K_XL"
    chat_kw = {"enable_thinking": thinking, "preserve_thinking": thinking}
    model_type = "multimodal" if (model_path.parent / "mmproj-F32.gguf").exists() else "lm"
    mmproj = str(model_path.parent / "mmproj-F32.gguf") if model_type == "multimodal" else None

    return ModelDefinition(
        id=model_path.parent.name,
        backend="llamacpp",
        port=BENCH_PORT,
        model_type=model_type,
        model_path=str(model_path),
        mmproj_path=mmproj,
        context_length=getattr(template_ref, "context_length", 32768),
        quantization=quant,
        max_concurrency=1,
        default_timeout=600,
        chat_template_kwargs=chat_kw,
        chat_template_file=getattr(template_ref, "chat_template_file", None),
        kv_unified=getattr(template_ref, "kv_unified", None),
        cache_type_k=getattr(template_ref, "cache_type_k", None),
        cache_type_v=getattr(template_ref, "cache_type_v", None),
        flash_attn=getattr(template_ref, "flash_attn", None),
        fit=getattr(template_ref, "fit", None),
        cont_batching=getattr(template_ref, "cont_batching", None),
        cache_prompt=getattr(template_ref, "cache_prompt", None),
        n_predict=getattr(template_ref, "n_predict", None),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="Benchmark spec_draft_n_max for MTP speculative decoding")


@app.command()
def main(
    models: Annotated[str | None, typer.Option("--models", "-m", help="Comma-separated model roles from models.yaml")] = None,
    model_path_arg: Annotated[Path | None, typer.Option("--model-path", help="Ad-hoc .gguf path (not in models.yaml)")] = None,
    thinking: Annotated[bool | None, typer.Option("--thinking/--no-thinking", help="Thinking mode for ad-hoc model (default: on)")] = None,
    reps: Annotated[int, typer.Option("--reps", "-r", help="Measured runs per draft-n value")] = 5,
    draft_values: Annotated[str, typer.Option("--draft-values", help="Comma-separated values to test; 0 = no-MTP baseline")] = "0,1,2,3,4",
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    if models is None and model_path_arg is None:
        console.print("[red]Specify --models <role[,role]> and/or --model-path <path>[/red]")
        raise typer.Exit(1)

    values = [int(v.strip()) for v in draft_values.split(",")]
    config = load_model_config(config_path)
    native_bin = find_native_llama_server()
    if not native_bin:
        console.print("[red]llama-server not found on PATH (brew install llama.cpp)[/red]")
        raise typer.Exit(1)

    template_role = "reasoning" if "reasoning" in config.models else next(iter(config.models))
    template = config.models[template_role]

    # Build ordered list of (label, model_def, model_path, thinking)
    targets: list[tuple[str, ModelDefinition, Path, bool]] = []

    if models:
        for role in [r.strip() for r in models.split(",")]:
            if role not in config.models:
                console.print(f"[red]Role '{role}' not found. Available: {', '.join(config.models)}[/red]")
                raise typer.Exit(1)
            md = config.models[role]
            if not md.model_path:
                console.print(f"[red]model_path not set for role '{role}'[/red]")
                raise typer.Exit(1)
            mp = Path(md.model_path)
            if not mp.exists():
                console.print(f"[red]model_path does not exist: {mp}[/red]")
                raise typer.Exit(1)
            ckw = getattr(md, "chat_template_kwargs", {}) or {}
            enable_thinking = bool(ckw.get("enable_thinking", False))
            targets.append((role, md, mp, enable_thinking))

    if model_path_arg is not None:
        if not model_path_arg.exists():
            console.print(f"[red]model_path does not exist: {model_path_arg}[/red]")
            raise typer.Exit(1)
        enable_thinking = thinking if thinking is not None else True
        md = make_adhoc_model_def(model_path_arg, enable_thinking, template)
        targets.append((model_path_arg.parent.name, md, model_path_arg, enable_thinking))

    console.print(f"\n[bold]MTP spec_draft_n_max benchmark[/bold]")
    console.print(f"sampling: temp={BENCH_TEMP}  top_p={BENCH_TOP_P}  top_k={BENCH_TOP_K}  max_tokens={BENCH_MAX_TOKENS}")
    console.print(f"draft values: {values}  |  reps per value: {reps}")
    console.print(f"models: {', '.join(t[0] for t in targets)}")

    all_results: list[tuple[str, list[dict]]] = []
    for label, md, mp, enable_thinking in targets:
        results = benchmark_model(label, md, mp, native_bin, values, reps, enable_thinking)
        all_results.append((label, results))

    print_comparison_table(all_results, values)


if __name__ == "__main__":
    app()
