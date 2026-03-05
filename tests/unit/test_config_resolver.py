"""Tests for ConfigResolver."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.config.resolver import ConfigResolver, ResolvedConfig


@dataclass
class _FakeNamespace:
    """Minimal namespace stub for testing."""

    id: Any = None
    name: str = "test-ns"
    config_overrides: dict[str, Any] = field(default_factory=dict)


@pytest.mark.unit
class TestResolvedConfig:
    """ResolvedConfig data access."""

    def test_get_existing_key(self):
        rc = ResolvedConfig(values={"a": 1}, sources={"a": "global"})
        assert rc.get("a") == 1

    def test_get_missing_key_returns_default(self):
        rc = ResolvedConfig()
        assert rc.get("missing") is None
        assert rc.get("missing", 42) == 42

    def test_get_source(self):
        rc = ResolvedConfig(values={"a": 1}, sources={"a": "namespace"})
        assert rc.get_source("a") == "namespace"
        assert rc.get_source("missing") is None


@pytest.mark.unit
class TestConfigResolverGlobalOnly:
    """Resolution with global config only (no storage)."""

    async def test_global_config_returned_when_no_storage(self):
        resolver = ConfigResolver(global_config={"chunk_size": 256, "model": "gpt-4o"})
        ns_id = uuid4()

        result = await resolver.resolve_for_namespace(ns_id)

        assert result.get("chunk_size") == 256
        assert result.get("model") == "gpt-4o"
        assert result.get_source("chunk_size") == "global"
        assert result.get_source("model") == "global"

    async def test_empty_global_config(self):
        resolver = ConfigResolver()
        ns_id = uuid4()

        result = await resolver.resolve_for_namespace(ns_id)

        assert result.values == {}
        assert result.sources == {}


@pytest.mark.unit
class TestConfigResolverNamespaceOverride:
    """Resolution with namespace overrides from storage."""

    async def test_namespace_overrides_global(self):
        fake_ns = _FakeNamespace(config_overrides={"chunk_size": 1024, "custom_key": "ns_val"})
        storage = AsyncMock()
        storage.get_namespace = AsyncMock(return_value=fake_ns)
        resolver = ConfigResolver(storage=storage, global_config={"chunk_size": 256, "model": "gpt-4o"})
        ns_id = uuid4()

        result = await resolver.resolve_for_namespace(ns_id)

        assert result.get("chunk_size") == 1024
        assert result.get_source("chunk_size") == "namespace"
        assert result.get("model") == "gpt-4o"
        assert result.get_source("model") == "global"
        assert result.get("custom_key") == "ns_val"
        assert result.get_source("custom_key") == "namespace"

    async def test_namespace_not_found_falls_back_to_global(self):
        storage = AsyncMock()
        storage.get_namespace = AsyncMock(return_value=None)
        resolver = ConfigResolver(storage=storage, global_config={"chunk_size": 256})
        ns_id = uuid4()

        result = await resolver.resolve_for_namespace(ns_id)

        assert result.get("chunk_size") == 256
        assert result.get_source("chunk_size") == "global"

    async def test_namespace_with_no_overrides(self):
        fake_ns = _FakeNamespace(config_overrides={})
        storage = AsyncMock()
        storage.get_namespace = AsyncMock(return_value=fake_ns)
        resolver = ConfigResolver(storage=storage, global_config={"model": "gpt-4o"})
        ns_id = uuid4()

        result = await resolver.resolve_for_namespace(ns_id)

        assert result.get("model") == "gpt-4o"
        assert result.get_source("model") == "global"

    async def test_keys_filter_limits_resolved_keys(self):
        fake_ns = _FakeNamespace(config_overrides={"chunk_size": 1024, "model": "claude"})
        storage = AsyncMock()
        storage.get_namespace = AsyncMock(return_value=fake_ns)
        resolver = ConfigResolver(storage=storage, global_config={"chunk_size": 256, "model": "gpt-4o", "extra": True})
        ns_id = uuid4()

        result = await resolver.resolve_for_namespace(ns_id, keys=["chunk_size"])

        assert result.get("chunk_size") == 1024
        assert result.get("model") is None
        assert result.get("extra") is None


@pytest.mark.unit
class TestConfigResolverImmediate:
    """resolve_immediate() without storage lookup."""

    def test_global_only(self):
        resolver = ConfigResolver(global_config={"a": 1})
        result = resolver.resolve_immediate()

        assert result.get("a") == 1
        assert result.get_source("a") == "global"

    def test_namespace_overrides_global(self):
        resolver = ConfigResolver(global_config={"a": 1, "b": 2})
        result = resolver.resolve_immediate(namespace_config={"a": 10, "c": 3})

        assert result.get("a") == 10
        assert result.get_source("a") == "namespace"
        assert result.get("b") == 2
        assert result.get_source("b") == "global"
        assert result.get("c") == 3
        assert result.get_source("c") == "namespace"

    def test_explicit_global_overrides_init_global(self):
        resolver = ConfigResolver(global_config={"a": 1})
        result = resolver.resolve_immediate(global_config={"a": 99})

        assert result.get("a") == 99

    def test_empty_inputs(self):
        resolver = ConfigResolver()
        result = resolver.resolve_immediate()

        assert result.values == {}
        assert result.sources == {}


@pytest.mark.unit
class TestConfigResolverHelpers:
    """get_pipeline_config() and get_llm_config() helpers."""

    def test_pipeline_config_defaults(self):
        resolver = ConfigResolver()
        resolved = ResolvedConfig()
        pipeline = resolver.get_pipeline_config(resolved)

        assert pipeline["chunking_strategy"] == "semantic"
        assert pipeline["chunk_size"] == 512
        assert pipeline["chunk_overlap"] == 50
        assert pipeline["embedding_model"] == "text-embedding-3-small"
        assert pipeline["extraction_model"] == "gpt-4o-mini"
        assert pipeline["extraction_skill"] == "general_entities"

    def test_pipeline_config_overridden(self):
        resolver = ConfigResolver()
        resolved = ResolvedConfig(values={"chunk_size": 1024, "embedding_model": "custom-embed"})
        pipeline = resolver.get_pipeline_config(resolved)

        assert pipeline["chunk_size"] == 1024
        assert pipeline["embedding_model"] == "custom-embed"

    def test_llm_config_defaults(self):
        resolver = ConfigResolver()
        resolved = ResolvedConfig()
        llm = resolver.get_llm_config(resolved)

        assert llm["model"] == "gpt-4o-mini"
        assert llm["temperature"] == 0.7
        assert llm["max_tokens"] == 2000
        assert llm["timeout"] == 30

    def test_llm_config_overridden(self):
        resolver = ConfigResolver()
        resolved = ResolvedConfig(values={"llm_model": "claude", "llm_temperature": 0.0})
        llm = resolver.get_llm_config(resolved)

        assert llm["model"] == "claude"
        assert llm["temperature"] == 0.0
