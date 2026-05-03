"""Unit tests for LiteLLM shared aiohttp connector configuration (DYT-3599)."""

from __future__ import annotations

import pytest

from khora.config.llm import (
    LiteLLMConfig,
    _init_shared_session,
    close_shared_session,
    configure_litellm,
)


@pytest.fixture
async def fresh_session_state():
    """Reset module-global session state around each test."""
    import khora.config.llm as llm_mod

    # Tear down any session left behind by another test.
    if llm_mod._shared_aiohttp_session is not None:
        await close_shared_session()
    llm_mod._connector_settings = None
    yield
    if llm_mod._shared_aiohttp_session is not None:
        await close_shared_session()
    llm_mod._connector_settings = None


class TestLiteLLMConfigConnectorFields:
    """LiteLLMConfig exposes the new connector fields with correct defaults."""

    def test_defaults_match_pre_v090_throughput(self) -> None:
        config = LiteLLMConfig()
        assert config.max_total_connections == 200
        assert config.max_connections_per_host == 0  # 0 = unlimited per aiohttp
        assert config.keepalive_timeout_s == 30.0

    def test_overrides_via_kwargs(self) -> None:
        config = LiteLLMConfig(
            max_total_connections=500,
            max_connections_per_host=50,
            keepalive_timeout_s=60.0,
        )
        assert config.max_total_connections == 500
        assert config.max_connections_per_host == 50
        assert config.keepalive_timeout_s == 60.0

    def test_invalid_values_rejected(self) -> None:
        with pytest.raises(ValueError):
            LiteLLMConfig(max_total_connections=0)  # gt=0
        with pytest.raises(ValueError):
            LiteLLMConfig(max_connections_per_host=-1)  # ge=0
        with pytest.raises(ValueError):
            LiteLLMConfig(keepalive_timeout_s=0)  # gt=0


class TestConfigureLiteLLMCachesConnectorSettings:
    """configure_litellm caches connector settings for _init_shared_session."""

    def test_first_call_caches_values(self, fresh_session_state) -> None:
        import khora.config.llm as llm_mod

        configure_litellm(
            LiteLLMConfig(
                max_total_connections=300,
                max_connections_per_host=25,
                keepalive_timeout_s=15.0,
            )
        )
        assert llm_mod._connector_settings == {
            "limit": 300,
            "limit_per_host": 25,
            "keepalive_timeout": 15.0,
        }

    def test_second_call_with_matching_values_is_silent(
        self, fresh_session_state, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = LiteLLMConfig(max_total_connections=300)
        configure_litellm(cfg)
        configure_litellm(cfg)
        # No warning when settings match.
        assert not any("differ from the cached" in r.message for r in caplog.records)

    def test_second_call_with_different_values_warns(
        self, fresh_session_state, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Loguru intercept — capture via a custom sink.
        from loguru import logger as loguru_logger

        records: list[str] = []
        sink_id = loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
        try:
            configure_litellm(LiteLLMConfig(max_total_connections=200))
            configure_litellm(LiteLLMConfig(max_total_connections=999))
        finally:
            loguru_logger.remove(sink_id)
        assert any("differ from the cached" in r for r in records), records


class TestInitSharedSessionUsesCachedSettings:
    """_init_shared_session reads connector settings from configure_litellm cache."""

    @pytest.mark.asyncio
    async def test_session_built_with_cached_settings(self, fresh_session_state) -> None:
        import khora.config.llm as llm_mod

        configure_litellm(
            LiteLLMConfig(
                max_total_connections=400,
                max_connections_per_host=50,
                keepalive_timeout_s=10.0,
            )
        )
        await _init_shared_session()
        connector = llm_mod._shared_aiohttp_session.connector
        assert connector.limit == 400
        assert connector.limit_per_host == 50
        # aiohttp stores keepalive_timeout as float on the connector.
        assert connector._keepalive_timeout == 10.0

    @pytest.mark.asyncio
    async def test_session_built_with_defaults_when_no_config_registered(self, fresh_session_state) -> None:
        # Skip configure_litellm entirely — _init_shared_session must still
        # produce a session, falling back to LiteLLMConfig defaults.
        import khora.config.llm as llm_mod

        await _init_shared_session()
        defaults = LiteLLMConfig()
        connector = llm_mod._shared_aiohttp_session.connector
        assert connector.limit == defaults.max_total_connections
        assert connector.limit_per_host == defaults.max_connections_per_host
        assert connector._keepalive_timeout == defaults.keepalive_timeout_s

    @pytest.mark.asyncio
    async def test_second_init_is_noop(self, fresh_session_state) -> None:
        import khora.config.llm as llm_mod

        await _init_shared_session()
        first = llm_mod._shared_aiohttp_session
        await _init_shared_session()
        assert llm_mod._shared_aiohttp_session is first

    @pytest.mark.asyncio
    async def test_close_clears_cache_for_next_lifecycle(self, fresh_session_state) -> None:
        import khora.config.llm as llm_mod

        configure_litellm(LiteLLMConfig(max_total_connections=100))
        await _init_shared_session()
        await close_shared_session()
        assert llm_mod._shared_aiohttp_session is None
        assert llm_mod._connector_settings is None


class TestLLMSettingsForwardsConnectorFieldsToLiteLLMConfig:
    """LLMSettings (KhoraConfig.llm) mirrors the connector fields so YAML/env
    configuration reaches the shared session via the engine translation."""

    def test_llm_settings_has_connector_fields(self) -> None:
        from khora.config.schema import LLMSettings

        settings = LLMSettings()
        assert settings.max_total_connections == 200
        assert settings.max_connections_per_host == 0
        assert settings.keepalive_timeout_s == 30.0

    def test_llm_settings_overrides(self) -> None:
        from khora.config.schema import LLMSettings

        settings = LLMSettings(
            max_total_connections=500,
            max_connections_per_host=100,
            keepalive_timeout_s=45.0,
        )
        assert settings.max_total_connections == 500
        assert settings.max_connections_per_host == 100
        assert settings.keepalive_timeout_s == 45.0

    def test_env_var_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # KHORA_LLM_* env prefix per LLMSettings.model_config. pydantic-settings
        # reads env vars at instantiation time so no module reload is needed
        # (and would pollute global state for sibling tests).
        from khora.config.schema import LLMSettings

        monkeypatch.setenv("KHORA_LLM_MAX_TOTAL_CONNECTIONS", "750")
        monkeypatch.setenv("KHORA_LLM_MAX_CONNECTIONS_PER_HOST", "33")
        monkeypatch.setenv("KHORA_LLM_KEEPALIVE_TIMEOUT_S", "12.5")

        settings = LLMSettings()
        assert settings.max_total_connections == 750
        assert settings.max_connections_per_host == 33
        assert settings.keepalive_timeout_s == 12.5
