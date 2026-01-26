#!/usr/bin/env python3
"""CLI tool for A/B testing models with different backends.

This script reads models.yaml and can start models using:
- MLX (Apple Silicon optimized) - supports concurrency
- llama.cpp (via llama-cpp-python server) - supports concurrency
- LMStudio (using model cache) - concurrency managed internally

**Backend Dependencies**: Backend servers are optional and can be installed via:
- `uv sync --extra mlx` (for MLX backend)
- `uv sync --extra llamacpp` (for llama.cpp backend)
- Or install manually: `pip install mlx-openai-server` or `pip install 'llama-cpp-python[server]'`

**Port Management**: Each model server instance requires its own port. For A/B testing,
start different backends on different ports (e.g., MLX on 1234, llama.cpp on 8001).
Configure `models.yaml` with per-model `port` fields to specify the port.

All backends are configured to provide OpenAI-compatible endpoints, ensuring seamless
integration with any OpenAI-compatible client (including httpx-based clients).

Performance parameters configured:
- Context length: Set from models.yaml
- Quantization: Used for GPU layer optimization (llama.cpp)
- Max concurrency: Set for MLX and llama.cpp (LMStudio manages internally)
- GPU layers: Optimized for Apple Silicon based on quantization

Usage:
    # Using uv (recommended):
    uv run python -m slm_server.benchmark_models --backend mlx --model router
    
    # Or with activated virtual environment:
    source .venv/bin/activate
    python -m slm_server.benchmark_models --backend mlx --model router
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
import typer
from rich.console import Console
from rich.table import Table

from slm_server.config import ModelConfig, ModelDefinition, load_model_config
from slm_server.start_backends import (
    BackendType,
    build_llamacpp_command,
    build_mlx_command,
    find_command_in_venv,
    find_model_path,
)

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

app = typer.Typer(help="Benchmark models using different backends")
console = Console()
log = structlog.get_logger(__name__)

# LMStudio model cache path - can be overridden with LMSTUDIO_CACHE environment variable
LMSTUDIO_CACHE = Path(
    os.getenv("LMSTUDIO_CACHE", str(Path.home() / ".cache" / "lm-studio" / "models"))
)


def build_lmstudio_command(model_path: Path, port: int) -> list[str] | None:
    """Build command to start LMStudio server.

    Note: LMStudio typically runs as a GUI app, but can be started via CLI.
    This assumes LMStudio CLI is available or provides instructions.

    Args:
        model_path: Path to model in LMStudio cache
        port: Server port

    Returns:
        Command list for subprocess or None if not available.
    """
    # LMStudio may need to be started via GUI or CLI
    # Check if lmstudio CLI is available
    lmstudio_cli = Path("/Applications/LM Studio.app/Contents/MacOS/LM Studio")
    if lmstudio_cli.exists():
        return [
            str(lmstudio_cli),
            "--server",
            "--port",
            str(port),
            "--model",
            str(model_path),
        ]
    # Alternative: use LMStudio's API if available
    # For now, return None and provide instructions
    return None


def start_model_server(
    backend: BackendType,
    model_path: Path,
    port: int,
    model_config: ModelDefinition,
) -> subprocess.Popen[str] | None:
    """Start a model server using the specified backend.

    All backends are configured to provide OpenAI-compatible endpoints:
    - Base URL: http://localhost:{port}/v1
    - Endpoints: /v1/chat/completions, /v1/completions, /v1/embeddings

    Args:
        backend: Backend type (mlx, llamacpp, lmstudio)
        model_path: Path to model file or directory
        port: Server port
        model_config: Model configuration

    Returns:
        Subprocess.Popen object if started successfully, None otherwise.
    """
    cmd: list[str] | None = None
    if backend == "mlx":
        cmd = build_mlx_command(
            model_path=model_path,
            port=port,
            context_length=model_config.context_length,
            max_concurrency=model_config.max_concurrency,
            model_type=getattr(model_config, "model_type", "lm"),
            host=getattr(model_config, "host", "0.0.0.0"),
            enable_auto_tool_choice=getattr(model_config, "enable_auto_tool_choice", False),
            tool_call_parser=getattr(model_config, "tool_call_parser", None),
            reasoning_parser=getattr(model_config, "reasoning_parser", None),
            config_name=getattr(model_config, "config_name", None),
        )
    elif backend == "llamacpp":
        cmd = build_llamacpp_command(
            model_path,
            port,
            model_config.context_length,
            model_config.quantization,
            model_config.max_concurrency,
        )
    elif backend == "lmstudio":
        cmd = build_lmstudio_command(model_path, port)
        if cmd is None:
            console.print(
                "[yellow]LMStudio CLI not found. Please start LMStudio manually:[/yellow]"
            )
            console.print("  1. Open LM Studio")
            console.print(f"  2. Load model: {model_path}")
            console.print(f"  3. Start server on port {port}")
            console.print(f"  4. Ensure server exposes: http://localhost:{port}/v1")
            return None

    # At this point, cmd is guaranteed to be list[str] (not None)
    # because all branches either set cmd or return early
    console.print(f"[green]Starting {backend} server with OpenAI-compatible API...[/green]")
    console.print(f"  Command: {' '.join(cmd)}")
    console.print(f"  Base URL: http://localhost:{port}/v1")
    console.print("  Endpoints: /v1/chat/completions, /v1/completions")
    console.print(f"  Model: {model_path}")
    console.print(f"  Context Length: {model_config.context_length}")
    console.print(f"  Quantization: {model_config.quantization}")
    if backend in ["mlx", "llamacpp"]:
        console.print(f"  Max Concurrency: {model_config.max_concurrency}")
    elif backend == "lmstudio":
        console.print("  Max Concurrency: N/A (LMStudio manages concurrency internally)")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return process
    except FileNotFoundError as e:
        console.print(f"[red]Command not found: {cmd[0]}[/red]")
        if backend == "mlx":
            console.print("[yellow]Install with one of:[/yellow]")
            console.print("  [cyan]uv sync --extra mlx[/cyan]  (recommended)")
            console.print("  [cyan]pip install mlx-openai-server[/cyan]")
        elif backend == "llamacpp":
            console.print("[yellow]Install with one of:[/yellow]")
            console.print("  [cyan]uv sync --extra llamacpp[/cyan]  (recommended)")
            console.print("  [cyan]pip install 'llama-cpp-python[server]'[/cyan]")
        else:
            console.print(f"[yellow]Please ensure {backend} is installed and in PATH[/yellow]")
        log.error("command_not_found", command=cmd[0], backend=backend, error=str(e))
        return None
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")
        log.error("server_start_failed", backend=backend, error=str(e))
        return None


@app.command()
def list_models(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to models.yaml (default: config/models.yaml)"),
    ] = None,
) -> None:
    """List all models configured in models.yaml."""
    try:
        config = load_model_config(config_path)
        table = Table(title="Configured Models")
        table.add_column("Role", style="cyan")
        table.add_column("Model ID", style="green")
        table.add_column("Backend", style="magenta")
        table.add_column("Port", style="blue")
        table.add_column("Context Length", style="yellow")
        table.add_column("Quantization", style="magenta")
        table.add_column("Timeout", style="blue")

        for role, model_def in config.models.items():
            table.add_row(
                role,
                model_def.id,
                model_def.backend,
                str(model_def.port),
                str(model_def.context_length),
                model_def.quantization,
                f"{model_def.default_timeout}s",
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]Failed to load model config: {e}[/red]")
        sys.exit(1)


@app.command()
def start(
    backend: Annotated[
        BackendType, typer.Option("--backend", "-b", help="Backend: mlx, llamacpp, or lmstudio")
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model role (e.g., router, reasoning)")
    ],
    port: Annotated[
        int | None, typer.Option("--port", "-p", help="Server port (overrides config)")
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to models.yaml (default: config/models.yaml)"),
    ] = None,
    model_file: Annotated[
        Path | None,
        typer.Option("--model-file", "-f", help="Override model file path (skips auto-discovery)"),
    ] = None,
) -> None:
    """Start a model server using the specified backend."""
    try:
        config = load_model_config(config_path)

        if model not in config.models:
            console.print(f"[red]Model role '{model}' not found in config[/red]")
            console.print(f"[yellow]Available roles: {', '.join(config.models.keys())}[/yellow]")
            sys.exit(1)

        model_def = config.models[model]
        
        # Use port from config if not provided
        if port is None:
            port = model_def.port
        
        # Verify backend matches
        if model_def.backend != backend:
            console.print(
                f"[yellow]Warning: Model '{model}' is configured for backend '{model_def.backend}', "
                f"but you specified '{backend}'[/yellow]"
            )
            console.print(f"[yellow]Using backend '{backend}' as specified[/yellow]")

        console.print(f"[cyan]Starting model: {model_def.id} ({model})[/cyan]")

        # Find or use provided model path
        if model_file:
            model_path: Path = model_file
        else:
            found_path = find_model_path(model_def.id, backend, LMSTUDIO_CACHE)
            if found_path is None:
                console.print(
                    f"[red]Model file not found for {model_def.id} with backend {backend}[/red]"
                )
                console.print(
                    "[yellow]Use --model-file to specify the path manually[/yellow]"
                )
                sys.exit(1)
            model_path = found_path

        # Start server
        process = start_model_server(backend, model_path, port, model_def)

        if process is None:
            sys.exit(1)

        console.print(f"[green]Server started (PID: {process.pid})[/green]")
        console.print(f"[cyan]OpenAI-compatible endpoint: http://localhost:{port}/v1[/cyan]")
        console.print(
            f"[cyan]For models.yaml, use endpoint: http://localhost:{port}/v1[/cyan]"
        )
        console.print("[yellow]Press Ctrl+C to stop[/yellow]")

        # Wait for process
        try:
            process.wait()
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping server...[/yellow]")
            process.terminate()
            process.wait()
            console.print("[green]Server stopped[/green]")

    except Exception as e:
        console.print(f"[red]Failed to load model config: {e}[/red]")
        log.exception("unexpected_error", error=str(e))
        sys.exit(1)


@app.command()
def check(
    backend: Annotated[
        BackendType, typer.Option("--backend", "-b", help="Backend: mlx, llamacpp, or lmstudio")
    ],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to models.yaml (default: config/models.yaml)"),
    ] = None,
) -> None:
    """Check if models are available for the specified backend."""
    try:
        config = load_model_config(config_path)

        table = Table(title=f"Model Availability Check ({backend})")
        table.add_column("Role", style="cyan")
        table.add_column("Model ID", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Path", style="blue")

        for role, model_def in config.models.items():
            model_path = find_model_path(model_def.id, backend, LMSTUDIO_CACHE)
            if model_path:
                table.add_row(role, model_def.id, "[green]✓ Found[/green]", str(model_path))
            else:
                table.add_row(role, model_def.id, "[red]✗ Not Found[/red]", "-")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to load model config: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
