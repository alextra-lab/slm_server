"""Script to start backend model servers based on config/models.yaml."""

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import structlog
from structlog import get_logger

from slm_server.config import ModelConfig, load_model_config

log = get_logger(__name__)

BackendType = Literal["mlx", "llamacpp", "lmstudio"]


def find_model_path(model_id: str, backend: BackendType, cache_dir: Path) -> Path | None:
    """Find model file for the given model ID and backend.

    Args:
        model_id: Model identifier (e.g., "qwen/qwen3-1.7b")
        backend: Backend type (mlx, llamacpp, lmstudio)
        cache_dir: Base cache directory to search

    Returns:
        Path to model file/directory, or None if not found.
    """
    # Extract model name from ID (e.g., "qwen/qwen3-1.7b" -> "qwen3-1.7b")
    model_name = model_id.split("/")[-1]

    # Search patterns by backend
    if backend == "mlx":
        # MLX models are directories with model files
        patterns = [
            f"*{model_name}*MLX*",
            f"*{model_name}*mlx*",
            f"*{model_name.upper()}*MLX*",
        ]
        for pattern in patterns:
            matches = list(cache_dir.glob(f"**/{pattern}"))
            if matches:
                return matches[0]

    elif backend == "llamacpp":
        # llama.cpp needs .gguf files
        patterns = [
            f"*{model_name}*.gguf",
            f"*{model_name.upper()}*.gguf",
            f"*{model_name}*GGUF*/*.gguf",
        ]
        for pattern in patterns:
            matches = list(cache_dir.glob(f"**/{pattern}"))
            if matches:
                return matches[0]

    elif backend == "lmstudio":
        # LMStudio can use either format
        for pattern in [f"*{model_name}*"]:
            matches = list(cache_dir.glob(f"**/{pattern}"))
            if matches:
                return matches[0]

    return None


def find_command_in_venv(command: str) -> str:
    """Find command in virtual environment or system PATH.

    Args:
        command: Command name to find.

    Returns:
        Full path to command, or command name if not found.
    """
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        venv_bin = Path(sys.prefix) / "bin" / command
        if venv_bin.exists():
            return str(venv_bin)
    which_path = shutil.which(command)
    if which_path:
        return which_path
    return command


def build_mlx_command(
    model_path: Path | str,
    port: int,
    context_length: int | None,
    max_concurrency: int,
    model_type: str = "lm",
    host: str = "0.0.0.0",
    enable_auto_tool_choice: bool = False,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    config_name: str | None = None,
) -> list[str]:
    """Build command to start MLX OpenAI server.

    Args:
        model_path: Path to model file.
        port: Server port.
        context_length: Maximum context length (passed to mlx-openai-server). None uses model default.
        max_concurrency: Maximum concurrent requests.
        model_type: Type of model (lm, multimodal, image-generation, image-edit, embeddings, whisper).
        host: Host to run the server on.
        enable_auto_tool_choice: Enable automatic tool choice.
        tool_call_parser: Tool call parser to use (e.g., "qwen3", "qwen3_coder").
        reasoning_parser: Reasoning parser to use (e.g., "qwen3", "harmony").
        config_name: Model configuration name (for image-generation/image-edit).

    Returns:
        Command list for subprocess.
    """
    cmd = [
        find_command_in_venv("mlx-openai-server"),
        "launch",
        "--model-path",
        str(model_path),
        "--model-type",
        model_type,
        "--port",
        str(port),
        "--host",
        host,
    ]
    
    # Context length (optional - only add if specified)
    if context_length is not None and context_length > 0:
        cmd.extend(["--context-length", str(context_length)])
    
    # Max concurrency (always add, default is 1)
    cmd.extend(["--max-concurrency", str(max_concurrency)])
    
    # Enable auto tool choice if requested
    if enable_auto_tool_choice:
        cmd.extend(["--enable-auto-tool-choice"])
    
    # Tool call parser (optional)
    if tool_call_parser:
        cmd.extend(["--tool-call-parser", tool_call_parser])
    
    # Reasoning parser (optional)
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    
    # Config name for image-generation and image-edit models
    if model_type in ("image-generation", "image-edit") and config_name:
        cmd.extend(["--config-name", config_name])
    elif model_type == "image-generation" and not config_name:
        # Default for image-generation
        cmd.extend(["--config-name", "flux-schnell"])
    elif model_type == "image-edit" and not config_name:
        # Default for image-edit
        cmd.extend(["--config-name", "flux-kontext-dev"])
    
    return cmd


def build_llamacpp_command(
    model_path: Path, port: int, context_length: int | None, quantization: str, max_concurrency: int
) -> list[str]:
    """Build command to start llama.cpp server.

    Args:
        model_path: Path to model file.
        port: Server port.
        context_length: Maximum context length. Defaults to 4096 if None.
        quantization: Quantization level (affects GPU layers).
        max_concurrency: Maximum concurrent requests.

    Returns:
        Command list for subprocess.
    """
    # llama.cpp requires a context length, so provide a default if None
    if context_length is None:
        context_length = 4096
    gpu_layers = 35 if quantization in ["8bit", "8-bit"] else 40
    return [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        str(model_path),
        "--host",
        "localhost",
        "--port",
        str(port),
        "--ctx-size",
        str(context_length),
        "--n-gpu-layers",
        str(gpu_layers),
        "--threads",
        "4",
        "--n-parallel",
        str(max_concurrency),
    ]


def start_model_server(model_def, config: ModelConfig) -> subprocess.Popen | None:
    """Start a single model server.

    Args:
        model_def: ModelDefinition instance.
        config: Full ModelConfig (for reference).

    Returns:
        subprocess.Popen instance, or None if failed.
    """
    # Use model_path from config (required)
    if not model_def.model_path:
        log.error("model_path_not_specified", model_id=model_def.id)
        return None
    
    # Check if model_path is a local path or Hugging Face model ID
    # Hugging Face IDs don't start with "/" and contain "/"
    is_hf_model = "/" in model_def.model_path and not model_def.model_path.startswith("/")
    
    if is_hf_model:
        # It's a Hugging Face model ID - will be downloaded automatically
        model_path = model_def.model_path
        log.info("using_huggingface_model", model_id=model_def.id, hf_model=model_path)
    else:
        # It's a local path - verify it exists
        model_path = Path(model_def.model_path)
        if not model_path.exists():
            log.error("model_path_not_found", path=str(model_path), model_id=model_def.id)
            return None
        model_path = str(model_path)

    log.info(
        "starting_model_server",
        model_id=model_def.id,
        backend=model_def.backend,
        port=model_def.port,
        model_path=model_path,
    )

    # Build command based on backend
    if model_def.backend == "mlx":
        cmd = build_mlx_command(
            model_path=Path(model_path) if not is_hf_model else model_path,
            port=model_def.port,
            context_length=model_def.context_length,
            max_concurrency=model_def.max_concurrency,
            model_type=getattr(model_def, "model_type", "lm"),
            host=getattr(model_def, "host", "0.0.0.0"),
            enable_auto_tool_choice=getattr(model_def, "enable_auto_tool_choice", False),
            tool_call_parser=getattr(model_def, "tool_call_parser", None),
            reasoning_parser=getattr(model_def, "reasoning_parser", None),
            config_name=getattr(model_def, "config_name", None),
        )
    elif model_def.backend == "llamacpp":
        cmd = build_llamacpp_command(
            model_path, model_def.port, model_def.context_length, model_def.quantization, model_def.max_concurrency
        )
    elif model_def.backend == "lmstudio":
        log.warning(
            "lmstudio_requires_manual_start",
            model_id=model_def.id,
            port=model_def.port,
            message="LMStudio must be started manually via GUI. Configure model to use port.",
        )
        return None
    else:
        log.error("unknown_backend", backend=model_def.backend, model_id=model_def.id)
        return None

    # Start process
    try:
        # Redirect stdout/stderr to devnull to avoid blocking on pipe buffers
        # For long-running servers, we don't need to capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # Give the process a moment to fail fast if there are startup errors
        time.sleep(0.5)
        
        # Check if process is still running
        if process.poll() is not None:
            # Process exited immediately - startup failure
            log.error(
                "server_exited_immediately",
                model_id=model_def.id,
                port=model_def.port,
                return_code=process.returncode,
            )
            return None
        
        log.info(
            "model_server_started",
            model_id=model_def.id,
            backend=model_def.backend,
            port=model_def.port,
            pid=process.pid,
        )
        return process
    except Exception as e:
        log.error("failed_to_start_server", model_id=model_def.id, error=str(e))
        return None


def main() -> None:
    """Main entry point: start all backend servers."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )

    # Load config
    try:
        config = load_model_config()
    except Exception as e:
        log.error("failed_to_load_config", error=str(e))
        sys.exit(1)

    # Start all model servers
    processes: list[tuple[str, subprocess.Popen]] = []
    attempted = 0
    failed = 0
    for role, model_def in config.models.items():
        # Skip disabled models
        if not model_def.enabled:
            log.info("skipping_disabled_model", model_id=model_def.id, role=role, port=model_def.port)
            continue
        
        if model_def.backend == "lmstudio":
            log.info("skipping_lmstudio", model_id=model_def.id, role=role, message="Start manually in LMStudio GUI")
            continue

        attempted += 1
        log.info("attempting_to_start_server", model_id=model_def.id, role=role, port=model_def.port, backend=model_def.backend)
        process = start_model_server(model_def, config)
        if process:
            processes.append((model_def.id, process))
            log.info("server_started_successfully", model_id=model_def.id, port=model_def.port, pid=process.pid)
        else:
            failed += 1
            log.warning("server_failed_to_start", model_id=model_def.id, role=role, port=model_def.port)

    log.info("startup_summary", attempted=attempted, started=len(processes), failed=failed)
    
    if not processes:
        log.error("no_servers_started", attempted=attempted)
        sys.exit(1)

    log.info("all_servers_started", count=len(processes), total_attempted=attempted)

    # Wait for all processes
    try:
        for model_id, process in processes:
            process.wait()
    except KeyboardInterrupt:
        log.info("shutting_down_servers")
        for model_id, process in processes:
            process.terminate()
            process.wait(timeout=5)
            if process.poll() is None:
                process.kill()
        log.info("all_servers_stopped")


if __name__ == "__main__":
    main()
