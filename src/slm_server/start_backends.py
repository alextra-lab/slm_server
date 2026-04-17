"""Script to start backend model servers based on config/models.yaml."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, cast

import structlog
from structlog import get_logger

from slm_server.config import ModelConfig, load_model_config

log = get_logger(__name__)

BackendType = Literal["mlx", "llamacpp"]

# Whitelist of allowed parser names to prevent injection
ALLOWED_TOOL_CALL_PARSERS = {
    "qwen3",
    "glm4_moe",
    "qwen3_coder",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_vl",
    "harmony",
    "minimax_m2",
}

ALLOWED_REASONING_PARSERS = {
    "qwen3",
    "glm4_moe",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_vl",
    "harmony",
    "minimax_m2",
}

ALLOWED_MODEL_TYPES = {
    "lm",
    "multimodal",
    "image-generation",
    "image-edit",
    "embeddings",
    "rerank",
    "whisper",
}

ALLOWED_CONFIG_NAMES = {"flux-schnell", "flux-kontext-dev"}


def validate_path(path: Path | str, allow_hf_model: bool = False) -> Path | str:
    """Validate and sanitize a file path to prevent path traversal attacks.

    Args:
        path: Path to validate.
        allow_hf_model: If True, allow Hugging Face model IDs (format: "org/model").

    Returns:
        Validated path.

    Raises:
        ValueError: If path contains dangerous characters or path traversal sequences.
    """
    path_str = str(path)

    # Allow Hugging Face model IDs (format: "org/model")
    if allow_hf_model and "/" in path_str and not path_str.startswith("/"):
        # Validate HF model ID format: alphanumeric, hyphens, underscores, and forward slashes only
        if not re.match(r"^[a-zA-Z0-9._/-]+$", path_str):
            raise ValueError(f"Invalid Hugging Face model ID format: {path_str}")
        # Check for path traversal attempts even in HF IDs
        if ".." in path_str or path_str.startswith("/") or "//" in path_str:
            raise ValueError(
                f"Invalid path: contains path traversal or invalid characters: {path_str}"
            )
        return path_str

    # For local paths, resolve to absolute path and validate
    path_obj = Path(path)

    # Check for path traversal sequences
    if ".." in path_str or path_str.startswith("~"):
        # Resolve to absolute path to normalize
        try:
            path_obj = path_obj.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: cannot resolve: {path_str}") from e

    # Validate path doesn't contain null bytes or other dangerous characters
    if "\x00" in path_str:
        raise ValueError(f"Invalid path: contains null byte: {path_str}")

    # Check for shell metacharacters that could be dangerous
    dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r"]
    for char in dangerous_chars:
        if char in path_str:
            raise ValueError(f"Invalid path: contains dangerous character '{char}': {path_str}")

    return path_obj


def validate_host(host: str) -> str:
    """Validate host value to prevent injection attacks.

    Args:
        host: Host value to validate.

    Returns:
        Validated host string.

    Raises:
        ValueError: If host contains invalid characters.
    """
    # Allow localhost, 0.0.0.0, and valid IP addresses
    if host in ("localhost", "0.0.0.0", "127.0.0.1"):
        return host

    # Validate IP address format (basic check)
    ip_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    if re.match(ip_pattern, host):
        # Validate each octet is 0-255
        parts = host.split(".")
        if all(0 <= int(part) <= 255 for part in parts):
            return host

    # Validate hostname format (alphanumeric, hyphens, dots)
    hostname_pattern = r"^[a-zA-Z0-9.-]+$"
    if re.match(hostname_pattern, host):
        # Check for dangerous characters
        if any(
            char in host
            for char in [";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r", "\x00", " "]
        ):
            raise ValueError(f"Invalid host: contains dangerous characters: {host}")
        return host

    raise ValueError(f"Invalid host format: {host}")


def validate_parser_name(
    parser: str | None, allowed_parsers: set[str], parser_type: str
) -> str | None:
    """Validate parser name against whitelist.

    Args:
        parser: Parser name to validate.
        allowed_parsers: Set of allowed parser names.
        parser_type: Type of parser (for error messages).

    Returns:
        Validated parser name or None.

    Raises:
        ValueError: If parser name is not in whitelist.
    """
    if parser is None:
        return None

    if parser not in allowed_parsers:
        raise ValueError(
            f"Invalid {parser_type}: '{parser}'. Allowed values: {sorted(allowed_parsers)}"
        )

    return parser


def validate_model_type(model_type: str) -> str:
    """Validate model type against whitelist.

    Args:
        model_type: Model type to validate.

    Returns:
        Validated model type.

    Raises:
        ValueError: If model type is not in whitelist.
    """
    if model_type not in ALLOWED_MODEL_TYPES:
        raise ValueError(
            f"Invalid model_type: '{model_type}'. Allowed values: {sorted(ALLOWED_MODEL_TYPES)}"
        )
    return model_type


def validate_config_name(config_name: str | None) -> str | None:
    """Validate config name against whitelist or allow None.

    Args:
        config_name: Config name to validate.

    Returns:
        Validated config name or None.

    Raises:
        ValueError: If config name is not in whitelist.
    """
    if config_name is None:
        return None

    if config_name not in ALLOWED_CONFIG_NAMES:
        raise ValueError(
            f"Invalid config_name: '{config_name}'. Allowed values: {sorted(ALLOWED_CONFIG_NAMES)}"
        )

    return config_name


def find_model_path(model_id: str, backend: BackendType, cache_dir: Path) -> Path | None:
    """Find model file for the given model ID and backend.

    Args:
        model_id: Model identifier (e.g., "qwen/qwen3-1.7b")
        backend: Backend type (mlx, llamacpp)
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

    return None


def find_command_in_venv(command: str) -> str:
    """Find command in virtual environment or system PATH.

    Args:
        command: Command name to find.

    Returns:
        Full path to command, or command name if not found.
    """
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
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
        model_type: Type of model (lm, multimodal, image-generation, image-edit, embeddings, rerank, whisper).
        host: Host to run the server on.
        enable_auto_tool_choice: Enable automatic tool choice.
        tool_call_parser: Tool call parser to use (e.g., "qwen3", "qwen3_coder").
        reasoning_parser: Reasoning parser to use (e.g., "qwen3", "harmony").
        config_name: Model configuration name (for image-generation/image-edit).

    Returns:
        Command list for subprocess.

    Raises:
        ValueError: If any input validation fails.
    """
    # Validate and sanitize all inputs
    model_path = validate_path(model_path, allow_hf_model=True)
    host = validate_host(host)
    model_type = validate_model_type(model_type)
    tool_call_parser = validate_parser_name(
        tool_call_parser, ALLOWED_TOOL_CALL_PARSERS, "tool_call_parser"
    )
    reasoning_parser = validate_parser_name(
        reasoning_parser, ALLOWED_REASONING_PARSERS, "reasoning_parser"
    )
    config_name = validate_config_name(config_name)

    # Validate port range
    if not (1024 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}. Must be between 1024 and 65535")

    # Validate context_length if provided
    if context_length is not None and context_length <= 0:
        raise ValueError(f"Invalid context_length: {context_length}. Must be positive")

    # Validate max_concurrency
    if max_concurrency <= 0:
        raise ValueError(f"Invalid max_concurrency: {max_concurrency}. Must be positive")

    mlx_cmd = find_command_in_venv("mlx-openai-server")
    if mlx_cmd == "mlx-openai-server" or not Path(mlx_cmd).exists():
        raise ValueError("mlx-openai-server not found. Install the MLX extra: uv sync --extra mlx")

    cmd = [
        mlx_cmd,
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


def find_native_llama_server() -> str | None:
    """Return path to native llama-server binary (e.g. from brew install llama.cpp), or None."""
    path = shutil.which("llama-server")
    return path


def build_llama_native_command(
    model_path: Path,
    port: int,
    context_length: int | None,
    quantization: str,
    max_concurrency: int,
    chat_template_kwargs: dict | None,
    model_alias: str | None,
    llama_server_bin: str,
    model_type: str = "lm",
    *,
    temp: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    kv_unified: bool | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    flash_attn: bool | str | None = None,
    fit: bool | str | None = None,
) -> list[str]:
    """Build command for native llama-server (e.g. from brew install llama.cpp).

    Uses Unsloth-recommended flags and supports --chat-template-kwargs (qwen35, qwen35moe).
    Optional sampling/cache flags are only added when present (config-driven).
    For model_type "embeddings", adds --embedding (OpenAI /v1/embeddings).
    For model_type "rerank", adds --embedding, --pooling rank, --reranking (OpenAI /v1/rerank).
    """
    model_type = validate_model_type(model_type)
    if not (1024 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}. Must be between 1024 and 65535")
    if context_length is None:
        context_length = 4096
    elif context_length <= 0:
        raise ValueError(f"Invalid context_length: {context_length}")
    cmd = [
        llama_server_bin,
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_length),
        "--n-gpu-layers",
        "999",
        "--threads",
        "-1",
    ]
    if model_alias:
        cmd.extend(["--alias", model_alias])
    if model_type == "embeddings":
        cmd.append("--embedding")
    elif model_type == "rerank":
        cmd.extend(["--embedding", "--pooling", "rank", "--reranking"])
    elif chat_template_kwargs:
        cmd.extend(["--chat-template-kwargs", json.dumps(chat_template_kwargs)])
    # Optional config-driven flags (only add when present)
    if temp is not None:
        cmd.extend(["--temp", str(temp)])
    if top_p is not None:
        cmd.extend(["--top-p", str(top_p)])
    if top_k is not None:
        cmd.extend(["--top-k", str(top_k)])
    if min_p is not None:
        cmd.extend(["--min-p", str(min_p)])
    if kv_unified is True:
        cmd.append("--kv-unified")
    if cache_type_k is not None:
        cmd.extend(["--cache-type-k", cache_type_k])
    if cache_type_v is not None:
        cmd.extend(["--cache-type-v", cache_type_v])
    if flash_attn is not None:
        cmd.extend(["--flash-attn", "on" if flash_attn in (True, "on", "true") else "off"])
    if fit is not None:
        cmd.extend(["--fit", "on" if fit in (True, "on", "true") else "off"])
    return cmd


def _cache_type_to_ggml_type(s: str) -> int:
    """Map cache type string (e.g. q8_0) to GGML type for llama-cpp-python server."""
    m = {"q8_0": 8, "f16": 1}
    return m.get(s.lower(), 8)


def build_llamacpp_command(
    model_path: Path,
    port: int,
    context_length: int | None,
    quantization: str,
    max_concurrency: int,
    chat_template_kwargs: dict | None = None,
    model_alias: str | None = None,
    model_type: str = "lm",
    *,
    temp: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    flash_attn: bool | str | None = None,
) -> list[str]:
    """Build command to start llama.cpp server.

    Args:
        model_path: Path to model file.
        port: Server port.
        context_length: Maximum context length. Defaults to 4096 if None.
        quantization: Quantization level (affects GPU layers).
        max_concurrency: Maximum concurrent requests.
        chat_template_kwargs: Optional dict for --chat_template_kwargs (e.g. {"enable_thinking": true} for Qwen3.5).
        model_alias: Model name advertised by the server (for Claude Code / Unsloth; e.g. unsloth/Qwen3.5-35B-A3B).
        model_type: When "embeddings", enables embedding mode (--embedding true).
        rerank is not supported (use native llama-server only).
        temp, top_p, top_k, min_p, cache_type_k, cache_type_v, flash_attn: Optional; only added when present.

    Returns:
        Command list for subprocess.

    Raises:
        ValueError: If any input validation fails.
    """
    model_type = validate_model_type(model_type)
    if model_type == "rerank":
        raise ValueError(
            "model_type rerank requires native llama-server (llama_cpp.server has no /v1/rerank); "
            "install llama.cpp and ensure llama-server is on PATH"
        )

    # Validate and sanitize model path (local path only; HF IDs use allow_hf_model=True)
    model_path = cast(Path, validate_path(model_path, allow_hf_model=False))

    # Validate port range
    if not (1024 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}. Must be between 1024 and 65535")

    # Validate context_length
    if context_length is None:
        context_length = 4096
    elif context_length <= 0:
        raise ValueError(f"Invalid context_length: {context_length}. Must be positive")

    # Validate quantization (basic check for dangerous characters)
    if not isinstance(quantization, str) or any(
        char in quantization
        for char in [";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r", "\x00"]
    ):
        raise ValueError(f"Invalid quantization: contains dangerous characters: {quantization}")

    # Validate max_concurrency
    if max_concurrency <= 0:
        raise ValueError(f"Invalid max_concurrency: {max_concurrency}. Must be positive")

    # Unsloth guidance: offload all layers to GPU (999), use all CPU threads (-1)
    n_gpu_layers = 999
    type_k_val = _cache_type_to_ggml_type(cache_type_k) if cache_type_k else 8
    type_v_val = _cache_type_to_ggml_type(cache_type_v) if cache_type_v else 8
    flash_val = (
        "true"
        if flash_attn in (True, "on", "true")
        else "false"
        if flash_attn is not None
        else "true"
    )
    cmd = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        str(model_path),
        "--host",
        "localhost",
        "--port",
        str(port),
        "--n_ctx",
        str(context_length),
        "--n_gpu_layers",
        str(n_gpu_layers),
        "--n_threads",
        "-1",
        "--type_k",
        str(type_k_val),
        "--type_v",
        str(type_v_val),
        "--flash_attn",
        flash_val,
    ]
    # Alias so Claude Code / clients can use the same model id as in config (per Unsloth docs)
    if model_alias:
        cmd.extend(["--model_alias", model_alias])
    if model_type == "embeddings":
        cmd.extend(["--embedding", "true"])
    # Optional config-driven flags (only add when present)
    if temp is not None:
        cmd.extend(["--temp", str(temp)])
    if top_p is not None:
        cmd.extend(["--top_p", str(top_p)])
    if top_k is not None:
        cmd.extend(["--top_k", str(top_k)])
    if min_p is not None:
        cmd.extend(["--min_p", str(min_p)])
    return cmd


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

    model_type = getattr(model_def, "model_type", "lm")
    skip_chat_template = model_type in ("embeddings", "rerank")
    chat_template_kwargs = (
        None if skip_chat_template else getattr(model_def, "chat_template_kwargs", None)
    )

    # Build command based on backend (llamacpp may use native binary or Python server)
    used_native_llama_server = False
    try:
        if model_def.backend == "mlx":
            if model_type == "rerank":
                log.error(
                    "mlx_does_not_support_rerank",
                    model_id=model_def.id,
                    message="model_type rerank requires backend llamacpp and native llama-server",
                )
                return None
            cmd = build_mlx_command(
                model_path=Path(model_path) if not is_hf_model else model_path,
                port=model_def.port,
                context_length=model_def.context_length,
                max_concurrency=model_def.max_concurrency,
                model_type=model_type,
                host=getattr(model_def, "host", "0.0.0.0"),
                enable_auto_tool_choice=getattr(model_def, "enable_auto_tool_choice", False),
                tool_call_parser=getattr(model_def, "tool_call_parser", None),
                reasoning_parser=getattr(model_def, "reasoning_parser", None),
                config_name=getattr(model_def, "config_name", None),
            )
        elif model_def.backend == "llamacpp":
            # llama.cpp doesn't support Hugging Face model IDs - must be local path
            if is_hf_model:
                log.error(
                    "llamacpp_does_not_support_hf_models",
                    model_id=model_def.id,
                    model_path=model_path,
                    message="llama.cpp backend requires local .gguf files, not Hugging Face model IDs",
                )
                return None
            model_path_obj = Path(model_path)
            # If path is a directory, resolve to first .gguf file (Unsloth GGUF in folder)
            if model_path_obj.is_dir():
                gguf_files = sorted(model_path_obj.glob("*.gguf"))
                if not gguf_files:
                    log.error(
                        "no_gguf_in_directory",
                        model_id=model_def.id,
                        path=str(model_path_obj),
                    )
                    return None
                model_path_obj = gguf_files[0]
                log.info(
                    "resolved_gguf_from_directory", model_id=model_def.id, gguf=str(model_path_obj)
                )
            # Prefer native llama-server (e.g. brew install llama.cpp) for Qwen3.5 and full Unsloth support
            native_bin = find_native_llama_server()
            if model_type == "rerank" and not native_bin:
                log.error(
                    "rerank_requires_native_llama_server",
                    model_id=model_def.id,
                    message="model_type rerank needs llama-server on PATH; llama_cpp.server does not expose /v1/rerank",
                )
                return None
            if native_bin:
                used_native_llama_server = True
                log.info("using_native_llama_server", path=native_bin, model_id=model_def.id)
                cmd = build_llama_native_command(
                    model_path_obj,
                    model_def.port,
                    model_def.context_length,
                    model_def.quantization,
                    model_def.max_concurrency,
                    chat_template_kwargs=chat_template_kwargs,
                    model_alias=model_def.id,
                    llama_server_bin=native_bin,
                    model_type=model_type,
                    temp=getattr(model_def, "temp", None),
                    top_p=getattr(model_def, "top_p", None),
                    top_k=getattr(model_def, "top_k", None),
                    min_p=getattr(model_def, "min_p", None),
                    kv_unified=getattr(model_def, "kv_unified", None),
                    cache_type_k=getattr(model_def, "cache_type_k", None),
                    cache_type_v=getattr(model_def, "cache_type_v", None),
                    flash_attn=getattr(model_def, "flash_attn", None),
                    fit=getattr(model_def, "fit", None),
                )
            else:
                cmd = build_llamacpp_command(
                    model_path_obj,
                    model_def.port,
                    model_def.context_length,
                    model_def.quantization,
                    model_def.max_concurrency,
                    chat_template_kwargs=chat_template_kwargs,
                    model_alias=model_def.id,
                    model_type=model_type,
                    temp=getattr(model_def, "temp", None),
                    top_p=getattr(model_def, "top_p", None),
                    top_k=getattr(model_def, "top_k", None),
                    min_p=getattr(model_def, "min_p", None),
                    cache_type_k=getattr(model_def, "cache_type_k", None),
                    cache_type_v=getattr(model_def, "cache_type_v", None),
                    flash_attn=getattr(model_def, "flash_attn", None),
                )
        else:
            log.error("unknown_backend", backend=model_def.backend, model_id=model_def.id)
            return None
    except ValueError as e:
        log.error("invalid_config_parameters", model_id=model_def.id, error=str(e))
        return None

    # Start process (capture stderr so we can log it on startup failure)
    # For llamacpp Python server only: set env for enable_thinking. Native llama-server gets --chat-template-kwargs on CLI.
    run_env = os.environ.copy()
    if model_def.backend == "llamacpp" and not used_native_llama_server and chat_template_kwargs:
        kw = json.dumps(chat_template_kwargs)
        run_env["LLAMA_CHAT_TEMPLATE_KWARGS"] = kw
        run_env["LLAMA_ARG_CHAT_TEMPLATE_KWARGS"] = kw
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=run_env,
        )

        # Give the process a moment to fail fast if there are startup errors
        time.sleep(0.5)

        # Check if process is still running
        if process.poll() is not None:
            # Process exited immediately - startup failure; log stderr for debugging
            stderr_out = ""
            if process.stderr:
                try:
                    stderr_out = (process.stderr.read() or "").strip()
                except Exception:
                    pass
            log.error(
                "server_exited_immediately",
                model_id=model_def.id,
                port=model_def.port,
                return_code=process.returncode,
                stderr=stderr_out or None,
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
            log.info(
                "skipping_disabled_model", model_id=model_def.id, role=role, port=model_def.port
            )
            continue

        attempted += 1
        log.info(
            "attempting_to_start_server",
            model_id=model_def.id,
            role=role,
            port=model_def.port,
            backend=model_def.backend,
        )
        process = start_model_server(model_def, config)
        if process:
            processes.append((model_def.id, process))
            log.info(
                "server_started_successfully",
                model_id=model_def.id,
                port=model_def.port,
                pid=process.pid,
            )
        else:
            failed += 1
            log.warning(
                "server_failed_to_start", model_id=model_def.id, role=role, port=model_def.port
            )

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
