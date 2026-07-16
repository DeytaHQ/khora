"""Sandbox-rendering tests for prompt templates (SSTI -> RCE hardening).

Extraction/expertise prompts (``ExpertiseComposer.render_prompt``) and persona
chat prompts (``PromptGenerator``) render through a Jinja
``ImmutableSandboxedEnvironment``. Unsafe constructs must raise
``jinja2.exceptions.SecurityError`` rather than failing open.
"""

from __future__ import annotations

import pytest
from jinja2.exceptions import SecurityError

from khora.chat.persona import ChatConfig, PersonaConfig
from khora.chat.prompt import PromptGenerator
from khora.extraction.extractors.llm import LLMEntityExtractor
from khora.extraction.skills import (
    EntityTypeConfig,
    ExpertiseConfig,
    ExpertiseLoader,
)
from khora.extraction.skills.composer import ExpertiseComposer

# ---------------------------------------------------------------------------
# 1 + 4. Unsafe constructs are blocked (raise, never fail open)
# ---------------------------------------------------------------------------

_SSTI_PAYLOAD = "UID=={{ cycler.__init__.__globals__.os.popen('id -u').read().strip() }}"


class TestSandboxBlocksSSTI:
    def test_render_prompt_raises_security_error_on_ssti(self) -> None:
        """A dunder-traversal SSTI payload raises SecurityError."""
        composer = ExpertiseComposer()
        with pytest.raises(SecurityError):
            composer.render_prompt(_SSTI_PAYLOAD)

    def test_render_prompt_does_not_return_interpolated_output(self) -> None:
        """The blocked payload never fails open by returning the raw template."""
        composer = ExpertiseComposer()
        try:
            result = composer.render_prompt(_SSTI_PAYLOAD)
        except SecurityError:
            # Expected: raised rather than returning anything.
            return
        # If it somehow did not raise, it must NOT have returned the payload
        # string (a fail-open regression) nor command output.
        pytest.fail(f"expected SecurityError, got a returned value: {result!r}")


# ---------------------------------------------------------------------------
# 2. Legitimate expertise templates render identically under the sandbox
# ---------------------------------------------------------------------------


class TestLegitimateExpertiseRendering:
    def test_for_loop_and_join_filter_render(self) -> None:
        """A normal loop + ``| join`` filter renders to the expected string."""
        expertise = ExpertiseConfig(
            name="test",
            entity_types=[
                EntityTypeConfig(name="PERSON", description="A human individual"),
                EntityTypeConfig(name="ORG", description="An organization"),
            ],
            tool_schemas={"web_search": {}, "calculator": {}},
        )
        template = (
            "{% for e in entity_types %}{{ e.name }}: {{ e.description }}\n{% endfor %}Tools: {{ tools | join(', ') }}"
        )
        expected = "PERSON: A human individual\nORG: An organization\nTools: web_search, calculator"

        composer = ExpertiseComposer()
        assert composer.render_prompt(template, expertise=expertise) == expected

    def test_non_security_render_error_fails_open(self) -> None:
        """A benign template error still warns + returns the raw template."""
        composer = ExpertiseComposer()
        # Unclosed block -> TemplateSyntaxError (not a SecurityError).
        template = "{% for x in %}"
        assert composer.render_prompt(template) == template


# ---------------------------------------------------------------------------
# 3. Persona chat template renders correctly under the sandbox
# ---------------------------------------------------------------------------


def _make_persona() -> PersonaConfig:
    return PersonaConfig(
        name="Jane Doe",
        title="VP Engineering",
        company="Acme Inc",
        background="Experienced engineering leader.",
        chat=ChatConfig(system_prompt_template=""),
    )


class TestPersonaChatRendering:
    def test_default_template_without_history(self) -> None:
        gen = PromptGenerator(_make_persona())
        prompt = gen.build_system_prompt()
        assert "Jane Doe" in prompt
        assert "VP Engineering" in prompt
        assert "Acme Inc" in prompt
        assert "Previous conversation context" not in prompt

    def test_default_template_with_history(self) -> None:
        gen = PromptGenerator(_make_persona())
        prompt = gen.build_system_prompt(history_summary="We discussed roadmap.")
        assert "Previous conversation context" in prompt
        assert "We discussed roadmap." in prompt

    def test_persona_template_rejects_ssti(self) -> None:
        """A persona template carrying an SSTI payload raises at render time."""
        persona = PersonaConfig(
            name="Jane Doe",
            title="VP",
            company="Acme",
            background="bg",
            chat=ChatConfig(system_prompt_template=("{{ cycler.__init__.__globals__.os.popen('id').read() }}")),
        )
        gen = PromptGenerator(persona)
        with pytest.raises(SecurityError):
            gen.build_system_prompt()


# ---------------------------------------------------------------------------
# 5. All builtin expertise skills render without error and stably
# ---------------------------------------------------------------------------


class TestBuiltinExpertiseRendering:
    @pytest.mark.parametrize("builtin_name", ["general", "meetings", "slack"])
    def test_builtin_prompts_render_without_raising(self, builtin_name: str) -> None:
        loader = ExpertiseLoader()
        config = loader.load_builtin(builtin_name)
        composer = ExpertiseComposer()

        for template in (config.system_prompt, config.extraction_prompt):
            first = composer.render_prompt(template, expertise=config)
            second = composer.render_prompt(template, expertise=config)
            # Stable / idempotent under the sandbox.
            assert first == second

        # The system prompt is non-empty for every builtin.
        assert composer.render_prompt(config.system_prompt, expertise=config).strip()


# ---------------------------------------------------------------------------
# Caller-level regression guard: SecurityError must propagate through the
# LLM extractor rather than silently downgrading to a fallback prompt.
# ---------------------------------------------------------------------------


class TestExtractorFailsClosed:
    def test_render_system_prompt_propagates_security_error(self) -> None:
        """An SSTI payload in expertise.system_prompt raises, not fallback."""
        expertise = ExpertiseConfig(
            name="malicious",
            system_prompt=_SSTI_PAYLOAD,
        )
        extractor = LLMEntityExtractor(model="test-model")
        with pytest.raises(SecurityError):
            extractor._render_system_prompt(expertise, None)

    def test_render_extraction_prompt_propagates_security_error(self) -> None:
        """An SSTI payload in expertise.extraction_prompt raises, not fallback."""
        expertise = ExpertiseConfig(
            name="malicious",
            extraction_prompt=_SSTI_PAYLOAD,
        )
        extractor = LLMEntityExtractor(model="test-model")
        with pytest.raises(SecurityError):
            extractor._render_extraction_prompt("some text", ["PERSON"], expertise, None)

    async def test_extract_multi_batch_propagates_security_error(self) -> None:
        """A poisoned extraction_prompt fails closed on the batch path too."""
        expertise = ExpertiseConfig(
            name="malicious",
            extraction_prompt=_SSTI_PAYLOAD,
        )
        extractor = LLMEntityExtractor(model="test-model")
        # SecurityError surfaces during prompt render, before any LLM call.
        with pytest.raises(SecurityError):
            await extractor._extract_multi_batch(
                ["some text"],
                ["PERSON"],
                None,
                expertise=expertise,
            )
