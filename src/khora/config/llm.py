"""LiteLLM configuration for unified LLM access.

Provides a unified interface to all LLM providers (OpenAI, Anthropic, Google, etc.)
with fallbacks and routing. Based on the memoryman/potemkin pattern.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field


class LiteLLMConfig(BaseModel):
    """Configuration for LiteLLM unified model access.

    This configuration provides a unified interface to all LLM providers
    including OpenAI, Anthropic, Google, and others through LiteLLM.
    """

    # Primary model configuration
    model: str = Field(
        default="gpt-4o-mini",
        description="Primary model to use (e.g., gpt-4o-mini, claude-sonnet-4-20250514, gemini-2.0-flash)",
    )
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable name for API key",
    )

    # Model parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation",
    )
    max_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum tokens to generate",
    )

    # Request configuration
    timeout: int = Field(
        default=30,
        gt=0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries on failure",
    )
    retry_wait: int = Field(
        default=2,
        ge=0,
        description="Wait time between retries in seconds",
    )

    # Concurrency
    max_concurrent_llm_calls: int = Field(
        default=20,
        gt=0,
        description="Maximum concurrent LLM API calls",
    )

    # Router configuration for fallbacks
    model_list: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of model configurations for router fallbacks",
    )
    router_settings: dict[str, Any] | None = Field(
        default=None,
        description="Router settings (routing_strategy, num_retries, etc.)",
    )

    # Embedding model configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model to use for embeddings",
    )
    embedding_api_key_env: str | None = Field(
        default=None,
        description="Environment variable for embedding API key (defaults to api_key_env)",
    )
    embedding_dimension: int = Field(
        default=1536,
        gt=0,
        description="Embedding vector dimension",
    )
    embed_concurrency: int = Field(
        default=50,
        gt=0,
        description="Maximum concurrent embedding API calls",
    )
    embed_batch_size: int = Field(
        default=200,
        gt=0,
        description="Maximum texts per embedding API sub-batch (hard cap)",
    )
    embed_batch_tokens: int = Field(
        default=50_000,
        gt=0,
        description="Maximum estimated tokens per embedding sub-batch. "
        "Dynamically sizes batches so short texts get large batches "
        "and long texts get small batches, keeping API response "
        "payloads under ~2MB.",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> LiteLLMConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            LiteLLMConfig instance
        """
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> LiteLLMConfig:
        """Create configuration from a dictionary.

        Args:
            config: Configuration dictionary (can be nested under 'llm' key)

        Returns:
            LiteLLMConfig instance
        """
        # Handle nested configuration
        if "llm" in config:
            config = config["llm"]

        # Handle config_file path
        if "config_file" in config:
            return cls.from_yaml(config["config_file"])

        return cls.model_validate(config)

    def get_api_key(self) -> str:
        """Get the API key from environment variable."""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            logger.warning(f"API key environment variable {self.api_key_env} not set")
        return key

    def get_embedding_api_key(self) -> str:
        """Get the embedding API key from environment variable."""
        env_var = self.embedding_api_key_env or self.api_key_env
        key = os.environ.get(env_var, "")
        if not key:
            logger.warning(f"Embedding API key environment variable {env_var} not set")
        return key


_shared_aiohttp_session: Any = None


def configure_litellm(config: LiteLLMConfig | None = None) -> None:
    """Configure LiteLLM with the given configuration.

    This function should be called once at application startup to configure
    LiteLLM's global settings.

    Args:
        config: LiteLLM configuration (uses defaults if None)
    """
    try:
        import litellm
    except ImportError:
        logger.warning("litellm package not installed, skipping configuration")
        return

    if config is None:
        config = LiteLLMConfig()

    # Critical for compatibility across providers
    litellm.drop_params = True

    # Disable verbose logging, telemetry, and "Give Feedback" debug messages
    litellm.set_verbose = False
    litellm.telemetry = False  # type: ignore[assignment]
    litellm.suppress_debug_info = True

    # Set up API keys from environment
    api_key = config.get_api_key()
    if api_key:
        # LiteLLM uses provider-specific env vars, so we ensure they're set
        if "openai" in config.model.lower() or config.model.startswith("gpt"):
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        elif "claude" in config.model.lower() or "anthropic" in config.model.lower():
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        elif "gemini" in config.model.lower():
            os.environ.setdefault("GOOGLE_API_KEY", api_key)

    logger.info(f"LiteLLM configured with model: {config.model}")


async def _init_shared_session() -> None:
    """Create the shared aiohttp session for litellm calls if not already created.

    Must be called from an async context (e.g. engine connect()) so that
    aiohttp.ClientSession is instantiated inside a running coroutine.
    The guard prevents session leaks on repeated connect() calls.
    """
    global _shared_aiohttp_session
    if _shared_aiohttp_session is not None:
        return
    try:
        import aiohttp

        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10, keepalive_timeout=30)
        _shared_aiohttp_session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=600),
        )
        logger.debug("Shared aiohttp session created for litellm calls")
    except ImportError:
        logger.debug("aiohttp not available, skipping shared session creation")


def get_shared_session() -> Any:
    """Return the shared aiohttp session for litellm calls, or None if not initialised."""
    return _shared_aiohttp_session


async def close_shared_session() -> None:
    """Close the shared aiohttp session. Call on engine/app shutdown."""
    global _shared_aiohttp_session
    if _shared_aiohttp_session is not None:
        await _shared_aiohttp_session.close()
        _shared_aiohttp_session = None
        logger.debug("Shared aiohttp session closed")


def create_litellm_router(config: LiteLLMConfig) -> Any:
    """Create a LiteLLM router for fallback handling.

    Args:
        config: LiteLLM configuration with model_list

    Returns:
        LiteLLM Router instance
    """
    try:
        from litellm import Router
    except ImportError:
        logger.warning("litellm package not installed")
        return None

    if not config.model_list:
        logger.warning("No model_list configured, router not created")
        return None

    router_settings = config.router_settings or {
        "routing_strategy": "simple-shuffle",
        "num_retries": config.max_retries,
    }

    router = Router(
        model_list=config.model_list,
        **router_settings,
    )

    logger.info(f"LiteLLM router created with {len(config.model_list)} models")
    return router


async def acompletion(
    prompt: str,
    config: LiteLLMConfig | None = None,
    *,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> str:
    """Async completion with LiteLLM.

    Args:
        prompt: User prompt
        config: LiteLLM configuration (uses defaults if None)
        system_prompt: Optional system prompt
        **kwargs: Additional arguments passed to litellm.acompletion

    Returns:
        Generated text response
    """
    try:
        import litellm
    except ImportError:
        raise RuntimeError("litellm package not installed. Run: pip install litellm")

    if config is None:
        config = LiteLLMConfig()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    import time as _time

    _t0 = _time.perf_counter()
    response = await litellm.acompletion(
        model=config.model,
        messages=messages,
        temperature=kwargs.pop("temperature", config.temperature),
        max_tokens=kwargs.pop("max_tokens", config.max_tokens),
        timeout=kwargs.pop("timeout", config.timeout),
        num_retries=kwargs.pop("num_retries", config.max_retries),
        **kwargs,
    )
    _latency = (_time.perf_counter() - _t0) * 1000

    # Record telemetry
    from khora.telemetry import get_collector

    usage = getattr(response, "usage", None)
    _prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    _completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    _total_tokens = getattr(usage, "total_tokens", 0) or 0
    _operation = kwargs.get("_telemetry_op", "completion")
    get_collector().record_llm_call(
        operation=_operation,
        model=config.model,
        prompt_tokens=_prompt_tokens,
        completion_tokens=_completion_tokens,
        total_tokens=_total_tokens,
        latency_ms=_latency,
    )

    # DYT-645: Extractors call litellm.acompletion() directly, not this helper.
    # If they switch to this helper, remove their own record_usage() calls to
    # avoid double-counting.
    from khora.memory_lake import LLMUsage
    from khora.telemetry.context import record_usage

    record_usage(
        LLMUsage(
            operation=_operation,
            model=config.model,
            prompt_tokens=_prompt_tokens,
            completion_tokens=_completion_tokens,
            total_tokens=_total_tokens,
            latency_ms=_latency,
        )
    )

    return response.choices[0].message.content


async def aembedding(
    text: str | list[str],
    config: LiteLLMConfig | None = None,
    **kwargs: Any,
) -> list[list[float]]:
    """Async embedding generation with LiteLLM.

    Args:
        text: Text or list of texts to embed
        config: LiteLLM configuration (uses defaults if None)
        **kwargs: Additional arguments passed to litellm.aembedding

    Returns:
        List of embedding vectors
    """
    try:
        import litellm
    except ImportError:
        raise RuntimeError("litellm package not installed. Run: pip install litellm")

    if config is None:
        config = LiteLLMConfig()

    # Ensure text is a list
    if isinstance(text, str):
        text = [text]

    import time as _time

    _t0 = _time.perf_counter()
    response = await litellm.aembedding(
        model=config.embedding_model,
        input=text,
        timeout=kwargs.pop("timeout", config.timeout),
        **kwargs,
    )
    _latency = (_time.perf_counter() - _t0) * 1000

    # Record telemetry
    from khora.telemetry import get_collector

    usage = getattr(response, "usage", None)
    _prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    _total_tokens = getattr(usage, "total_tokens", 0) or 0
    get_collector().record_llm_call(
        operation="embedding",
        model=config.embedding_model,
        prompt_tokens=_prompt_tokens,
        total_tokens=_total_tokens,
        latency_ms=_latency,
        metadata={"batch_size": len(text)},
    )

    # DYT-645: Embedders call litellm.aembedding() directly, not this helper.
    # If they switch to this helper, remove their own record_usage() calls to
    # avoid double-counting.
    from khora.memory_lake import LLMUsage
    from khora.telemetry.context import record_usage

    record_usage(
        LLMUsage(
            operation="embedding",
            model=config.embedding_model,
            prompt_tokens=_prompt_tokens,
            completion_tokens=0,
            total_tokens=_total_tokens,
            latency_ms=_latency,
            batch_size=len(text),
        )
    )

    return [item["embedding"] for item in response.data]
