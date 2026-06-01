"""Unit tests for LiteLLM Router wiring in khora.config.llm.acompletion (#936)."""

from __future__ import annotations

import sys
import types

import pytest

from khora.config.llm import LiteLLMConfig, acompletion


def _fake_response(model: str):
    message = types.SimpleNamespace(content=f"ok:{model}")
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    return types.SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def reset_router_cache():
    import khora.config.llm as llm_mod

    saved = dict(llm_mod._router_cache)
    llm_mod._router_cache.clear()
    yield
    llm_mod._router_cache.clear()
    llm_mod._router_cache.update(saved)


@pytest.fixture
def fake_litellm(monkeypatch):
    """Install a stub litellm module with recording acompletion + Router."""
    state = {"direct_calls": [], "router_calls": [], "routers_built": 0, "router_init_kwargs": []}

    async def fake_acompletion(**kwargs):
        state["direct_calls"].append(kwargs)
        return _fake_response(kwargs.get("model"))

    class FakeRouter:
        def __init__(self, **kwargs):
            state["routers_built"] += 1
            state["router_init_kwargs"].append(kwargs)

        async def acompletion(self, **kwargs):
            state["router_calls"].append(kwargs)
            return _fake_response(kwargs.get("model"))

    fake_module = types.ModuleType("litellm")
    fake_module.acompletion = fake_acompletion
    fake_module.Router = FakeRouter
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    return state


@pytest.mark.asyncio
async def test_no_model_list_uses_direct_acompletion(fake_litellm, reset_router_cache):
    cfg = LiteLLMConfig(model="gpt-4o-mini")
    result = await acompletion("hello", config=cfg)

    assert result == "ok:gpt-4o-mini"
    assert len(fake_litellm["direct_calls"]) == 1
    assert fake_litellm["direct_calls"][0]["model"] == "gpt-4o-mini"
    assert fake_litellm["routers_built"] == 0
    assert fake_litellm["router_calls"] == []


@pytest.mark.asyncio
async def test_model_list_routes_through_router(fake_litellm, reset_router_cache):
    cfg = LiteLLMConfig(
        model="primary",
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "gpt-4o-mini"}},
            {"model_name": "primary", "litellm_params": {"model": "claude-sonnet-4-20250514"}},
        ],
        router_settings={"num_retries": 3},
    )
    result = await acompletion("hello", config=cfg)

    assert result == "ok:primary"
    # Router built and used; direct litellm.acompletion not called.
    assert fake_litellm["routers_built"] == 1
    assert len(fake_litellm["router_calls"]) == 1
    assert fake_litellm["router_calls"][0]["model"] == "primary"
    assert fake_litellm["direct_calls"] == []
    # Router init received the configured model_list + settings.
    init = fake_litellm["router_init_kwargs"][0]
    assert len(init["model_list"]) == 2
    assert init["num_retries"] == 3


@pytest.mark.asyncio
async def test_router_is_cached_across_calls(fake_litellm, reset_router_cache):
    cfg = LiteLLMConfig(
        model="primary",
        model_list=[{"model_name": "primary", "litellm_params": {"model": "gpt-4o-mini"}}],
    )
    await acompletion("a", config=cfg)
    await acompletion("b", config=cfg)

    # Two completions, one router build.
    assert fake_litellm["routers_built"] == 1
    assert len(fake_litellm["router_calls"]) == 2
