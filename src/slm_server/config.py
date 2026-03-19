"""Configuration management for SLM Server."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ModelDefinition(BaseModel):
    """Configuration for a single model server instance."""

    id: str = Field(..., description="Model identifier (used for routing)")
    backend: Literal["mlx", "llamacpp"] = Field(..., description="Backend type")
    port: int = Field(..., ge=1024, le=65535, description="Port number for this model server")
    context_length: int | None = Field(
        None, ge=1, description="Maximum context length. Default: None (uses model's default context length)"
    )
    quantization: str = Field(..., description="Quantization level")
    max_concurrency: int = Field(1, ge=1, description="Maximum concurrent requests (default: 1)")
    default_timeout: int = Field(..., ge=1, description="Default timeout in seconds")
    model_type: Literal["lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"] = Field(
        "lm", description="Type of model to run (default: lm)"
    )
    host: str = Field("0.0.0.0", description="Host to run the server on (default: 0.0.0.0)")
    enable_auto_tool_choice: bool = Field(
        False,
        description="Enable automatic tool choice. Only works with language models (lm or multimodal model types)."
    )
    tool_call_parser: str | None = Field(
        None,
        description="Tool call parser for mlx-openai-server. Available options: qwen3, glm4_moe, qwen3_coder, qwen3_moe, qwen3_next, qwen3_vl, harmony, minimax_m2. Only works with language models (lm or multimodal model types)."
    )
    reasoning_parser: str | None = Field(
        None,
        description="Reasoning parser for mlx-openai-server. Available options: qwen3, glm4_moe, qwen3_moe, qwen3_next, qwen3_vl, harmony, minimax_m2. Only works with language models (lm or multimodal model types)."
    )
    config_name: str | None = Field(
        None,
        description="Model configuration name. Default: flux-schnell for image-generation, flux-kontext-dev for image-edit"
    )
    supports_function_calling: bool = Field(
        False, description="Whether model supports native function calling via mlx-openai-server"
    )
    # For llamacpp: Unsloth Qwen3.5 thinking/reasoning. E.g. {"enable_thinking": true} for 35B-A3B, false/default for 9B.
    # See https://unsloth.ai/docs/models/qwen3.5#how-to-enable-or-disable-reasoning-and-thinking
    chat_template_kwargs: dict | None = Field(
        None,
        description="Optional chat template kwargs for llama.cpp (e.g. enable_thinking for Qwen3.5). Only used when backend is llamacpp.",
    )
    # Optional llamacpp-only CLI options; only applied when present (no defaults in code).
    temp: float | None = Field(None, description="Sampling temperature (llamacpp). Only used when backend is llamacpp.")
    top_p: float | None = Field(None, description="Top-p sampling (llamacpp). Only used when backend is llamacpp.")
    top_k: int | None = Field(None, description="Top-k sampling (llamacpp). Only used when backend is llamacpp.")
    min_p: float | None = Field(None, description="Min-p sampling (llamacpp). Only used when backend is llamacpp.")
    kv_unified: bool | None = Field(None, description="Use unified KV cache (llamacpp native). Only used when backend is llamacpp.")
    cache_type_k: str | None = Field(None, description="KV cache type for K (e.g. q8_0). Only used when backend is llamacpp.")
    cache_type_v: str | None = Field(None, description="KV cache type for V (e.g. q8_0). Only used when backend is llamacpp.")
    flash_attn: bool | str | None = Field(
        None,
        description="Flash attention on/off (llamacpp; true or 'on'). Only used when backend is llamacpp.",
    )
    fit: bool | str | None = Field(
        None,
        description="Fit option (llamacpp native; true or 'on'). Only used when backend is llamacpp.",
    )
    model_path: str | None = Field(None, description="Optional path to model file (auto-discovered if not set)")
    enabled: bool = Field(True, description="Whether this model server should be started")


class ModelConfig(BaseModel):
    """Complete model configuration loaded from YAML."""

    models: dict[str, ModelDefinition]


def validate_model_config(config: ModelConfig) -> list[str]:
    """Validate model configuration and return list of warnings/errors.

    Args:
        config: ModelConfig instance to validate.

    Returns:
        List of validation issues (warnings and errors). Empty list means valid.
    """
    issues = []

    for role, model_def in config.models.items():
        # Check if model_path is provided
        if not model_def.model_path:
            issues.append(f"{role}: model_path is required")
            continue

        # Check if model_path is a Hugging Face model ID or local path
        is_hf_model = "/" in model_def.model_path and not model_def.model_path.startswith("/")
        
        if is_hf_model:
            # It's a Hugging Face model ID - will be downloaded on first use
            # No validation needed
            continue
        
        path = Path(model_def.model_path)
        
        # Check if path exists (only for local paths)
        if not path.exists():
            issues.append(f"{role}: model_path does not exist: {path}")
            continue

        # Check backend/path compatibility
        if model_def.backend == "llamacpp":
            # llamacpp requires .gguf files
            if path.is_dir():
                # Check if directory contains .gguf files
                gguf_files = list(path.glob("*.gguf"))
                if not gguf_files:
                    issues.append(
                        f"{role}: llamacpp backend requires .gguf file, but directory has no .gguf files: {path}"
                    )
            elif not str(path).lower().endswith(".gguf"):
                issues.append(
                    f"{role}: llamacpp backend requires .gguf file, got: {path}"
                )

        elif model_def.backend == "mlx":
            # MLX cannot use .gguf files
            if str(path).lower().endswith(".gguf"):
                issues.append(
                    f"{role}: mlx backend cannot use .gguf files: {path}"
                )
            elif path.is_file() and not any(ext in str(path).lower() for ext in [".mlx", ".safetensors"]):
                issues.append(
                    f"{role}: mlx backend typically needs .mlx directory or safetensors files: {path}"
                )

        # Check port conflicts
        ports_seen = {}
        for other_role, other_def in config.models.items():
            if other_role != role and other_def.enabled and model_def.enabled:
                if other_def.port == model_def.port:
                    if model_def.port not in ports_seen:
                        issues.append(
                            f"{role} and {other_role}: port conflict - both use port {model_def.port}"
                        )
                        ports_seen[model_def.port] = True

    return issues


def load_model_config(config_path: Path | None = None, validate: bool = True) -> ModelConfig:
    """Load model configuration from YAML file.

    Args:
        config_path: Path to models.yaml. Defaults to config/models.yaml relative to project root.
        validate: If True, validate config and log warnings (default: True).

    Returns:
        Validated ModelConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    import yaml
    from structlog import get_logger

    log = get_logger(__name__)

    if config_path is None:
        # Find project root (parent of src/)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "models.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with config_path.open() as f:
        data = yaml.safe_load(f)

    try:
        config = ModelConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid model config: {e}") from e

    # Validate config if requested
    if validate:
        issues = validate_model_config(config)
        if issues:
            log.warning(
                "model_config_validation_issues",
                count=len(issues),
                issues=issues,
            )
            for issue in issues:
                log.warning("config_issue", issue=issue)

    return config
