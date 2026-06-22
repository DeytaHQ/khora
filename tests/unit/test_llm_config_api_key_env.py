"""Unit tests for api_key_env auto-derivation from the model prefix.

When a non-OpenAI model is configured but ``api_key_env`` is left at the
OpenAI default, the env-var pointer must be derived from the model prefix so
the OpenAI key is not silently read and copied into the wrong provider slot.
"""

from __future__ import annotations

import pytest

from khora.config.llm import (
    DEFAULT_API_KEY_ENV,
    LiteLLMConfig,
    derive_api_key_env,
)
from khora.config.schema import LLMSettings


class TestDeriveApiKeyEnv:
    """The shared prefix -> env-var mapping helper."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemini/gemini-2.5-flash", "GOOGLE_API_KEY"),
            ("gemini-2.0-flash", "GOOGLE_API_KEY"),
            ("vertex_ai/gemini-1.5-pro", "GOOGLE_API_KEY"),
            ("claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
            ("anthropic/claude-3-5-haiku", "ANTHROPIC_API_KEY"),
            ("gpt-4o-mini", "OPENAI_API_KEY"),
            ("openai/gpt-4o", "OPENAI_API_KEY"),
        ],
    )
    def test_known_prefixes(self, model: str, expected: str) -> None:
        assert derive_api_key_env(model) == expected

    def test_unknown_prefix_returns_none(self) -> None:
        # Unknown provider: we cannot derive, so the helper returns None and the
        # caller keeps whatever was configured.
        assert derive_api_key_env("some-local-llm") is None


class TestSharedDefaultConstant:
    """Both config models must share one default so they cannot drift."""

    def test_constant_value(self) -> None:
        assert DEFAULT_API_KEY_ENV == "OPENAI_API_KEY"

    def test_litellm_default_uses_constant(self) -> None:
        assert LiteLLMConfig().api_key_env == DEFAULT_API_KEY_ENV

    def test_llmsettings_default_uses_constant(self) -> None:
        assert LLMSettings().api_key_env == DEFAULT_API_KEY_ENV


class TestLiteLLMConfigAutoDerive:
    def test_gemini_model_derives_google_key_env(self) -> None:
        cfg = LiteLLMConfig(model="gemini/gemini-2.5-flash")
        assert cfg.api_key_env == "GOOGLE_API_KEY"

    def test_anthropic_model_derives_anthropic_key_env(self) -> None:
        cfg = LiteLLMConfig(model="claude-sonnet-4-20250514")
        assert cfg.api_key_env == "ANTHROPIC_API_KEY"

    def test_openai_model_keeps_openai_default(self) -> None:
        cfg = LiteLLMConfig(model="gpt-4o-mini")
        assert cfg.api_key_env == "OPENAI_API_KEY"

    def test_explicit_override_is_honored(self) -> None:
        # User deliberately pointed at a custom env var: do not overwrite it.
        cfg = LiteLLMConfig(model="gemini/gemini-2.5-flash", api_key_env="MY_GEMINI_KEY")
        assert cfg.api_key_env == "MY_GEMINI_KEY"

    def test_unknown_model_keeps_default(self) -> None:
        cfg = LiteLLMConfig(model="some-local-llm")
        assert cfg.api_key_env == "OPENAI_API_KEY"


class TestLLMSettingsAutoDerive:
    def test_gemini_model_derives_google_key_env(self) -> None:
        cfg = LLMSettings(model="gemini/gemini-2.5-flash")
        assert cfg.api_key_env == "GOOGLE_API_KEY"

    def test_anthropic_model_derives_anthropic_key_env(self) -> None:
        cfg = LLMSettings(model="anthropic/claude-3-5-haiku")
        assert cfg.api_key_env == "ANTHROPIC_API_KEY"

    def test_openai_model_keeps_openai_default(self) -> None:
        cfg = LLMSettings(model="gpt-4o-mini")
        assert cfg.api_key_env == "OPENAI_API_KEY"

    def test_explicit_override_is_honored(self) -> None:
        cfg = LLMSettings(model="gemini/gemini-2.5-flash", api_key_env="MY_GEMINI_KEY")
        assert cfg.api_key_env == "MY_GEMINI_KEY"
