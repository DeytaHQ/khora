"""Unit tests for khora.chat.persona — PersonaConfig and load_persona_config."""

from __future__ import annotations

from khora.chat.persona import (
    ChatConfig,
    CommunicationStyle,
    CompressionConfig,
    KeyRelationship,
    PersonaConfig,
    PersonaContext,
    ResponseConfig,
    load_persona_config,
)

# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------


class TestCommunicationStyle:
    """Tests for CommunicationStyle dataclass."""

    def test_defaults(self) -> None:
        style = CommunicationStyle()
        assert style.tone == "professional"
        assert style.perspective == "balanced"
        assert style.preferences == []

    def test_custom(self) -> None:
        style = CommunicationStyle(
            tone="friendly",
            perspective="first-person",
            preferences=["concise", "actionable"],
        )
        assert style.tone == "friendly"
        assert style.perspective == "first-person"
        assert style.preferences == ["concise", "actionable"]


class TestKeyRelationship:
    """Tests for KeyRelationship dataclass."""

    def test_required_fields(self) -> None:
        rel = KeyRelationship(name="Alice", role="CTO")
        assert rel.name == "Alice"
        assert rel.role == "CTO"
        assert rel.context == ""

    def test_with_context(self) -> None:
        rel = KeyRelationship(name="Bob", role="VP Eng", context="Direct report")
        assert rel.context == "Direct report"


class TestPersonaContext:
    """Tests for PersonaContext dataclass."""

    def test_defaults(self) -> None:
        ctx = PersonaContext()
        assert ctx.current_priorities == []
        assert ctx.key_relationships == []

    def test_with_data(self) -> None:
        rel = KeyRelationship(name="Alice", role="CTO")
        ctx = PersonaContext(
            current_priorities=["ship v2", "hire"],
            key_relationships=[rel],
        )
        assert len(ctx.current_priorities) == 2
        assert ctx.key_relationships[0].name == "Alice"


class TestCompressionConfig:
    """Tests for CompressionConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = CompressionConfig()
        assert cfg.enabled is True
        assert cfg.compress_after_turns == 10
        assert cfg.keep_recent_turns == 3
        assert cfg.strategy == "summarize"

    def test_custom(self) -> None:
        cfg = CompressionConfig(
            enabled=False,
            compress_after_turns=20,
            keep_recent_turns=5,
            strategy="truncate",
        )
        assert cfg.enabled is False
        assert cfg.compress_after_turns == 20
        assert cfg.keep_recent_turns == 5
        assert cfg.strategy == "truncate"


class TestResponseConfig:
    """Tests for ResponseConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = ResponseConfig()
        assert cfg.max_tokens == 1000
        assert cfg.temperature == 0.7
        assert cfg.include_sources is True
        assert cfg.cite_search_results is True

    def test_custom(self) -> None:
        cfg = ResponseConfig(
            max_tokens=2000,
            temperature=0.0,
            include_sources=False,
            cite_search_results=False,
        )
        assert cfg.max_tokens == 2000
        assert cfg.temperature == 0.0
        assert cfg.include_sources is False


class TestChatConfig:
    """Tests for ChatConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = ChatConfig()
        assert cfg.max_history_turns == 20
        assert isinstance(cfg.compression, CompressionConfig)
        assert isinstance(cfg.response, ResponseConfig)
        assert cfg.system_prompt_template == ""

    def test_custom(self) -> None:
        cfg = ChatConfig(
            max_history_turns=50,
            system_prompt_template="You are {{ persona.name }}.",
        )
        assert cfg.max_history_turns == 50
        assert "persona.name" in cfg.system_prompt_template


# ---------------------------------------------------------------------------
# PersonaConfig
# ---------------------------------------------------------------------------


class TestPersonaConfig:
    """Tests for PersonaConfig dataclass."""

    def test_all_fields(self) -> None:
        persona = PersonaConfig(
            name="Test Bot",
            title="Chief Bot",
            company="BotCorp",
            email="bot@corp.com",
            background="A helpful assistant.",
            expertise=["testing", "automation"],
        )
        assert persona.name == "Test Bot"
        assert persona.title == "Chief Bot"
        assert persona.company == "BotCorp"
        assert persona.email == "bot@corp.com"
        assert persona.background == "A helpful assistant."
        assert persona.expertise == ["testing", "automation"]

    def test_defaults(self) -> None:
        persona = PersonaConfig(name="Bot", title="Assistant", company="Corp")
        assert persona.email == ""
        assert persona.background == ""
        assert persona.expertise == []
        assert isinstance(persona.communication_style, CommunicationStyle)
        assert isinstance(persona.context, PersonaContext)
        assert isinstance(persona.chat, ChatConfig)


# ---------------------------------------------------------------------------
# load_persona_config
# ---------------------------------------------------------------------------


class TestLoadPersonaConfig:
    """Tests for load_persona_config()."""

    def test_full_yaml(self, tmp_path) -> None:
        """Loads a full YAML persona config."""
        yaml_content = """\
persona:
  name: Jane Doe
  title: VP Engineering
  company: Acme Inc
  email: jane@acme.com
  background: Engineering leader with 15 years experience.
  expertise:
    - distributed systems
    - team management
  communication_style:
    tone: direct
    perspective: first-person
    preferences:
      - be concise
      - use examples
  context:
    current_priorities:
      - ship v2
      - hire senior engineers
    key_relationships:
      - name: Alice
        role: CTO
        context: Direct manager

chat:
  max_history_turns: 30
  compression:
    enabled: true
    compress_after_turns: 15
    keep_recent_turns: 5
    strategy: summarize
  response:
    max_tokens: 2000
    temperature: 0.3
    include_sources: true
    cite_search_results: false
  system_prompt_template: "Custom template for {{ persona.name }}"
"""
        config_file = tmp_path / "persona.yaml"
        config_file.write_text(yaml_content)

        persona = load_persona_config(config_file)

        assert persona.name == "Jane Doe"
        assert persona.title == "VP Engineering"
        assert persona.company == "Acme Inc"
        assert persona.email == "jane@acme.com"
        assert "15 years" in persona.background
        assert "distributed systems" in persona.expertise

        assert persona.communication_style.tone == "direct"
        assert persona.communication_style.perspective == "first-person"
        assert len(persona.communication_style.preferences) == 2

        assert len(persona.context.current_priorities) == 2
        assert persona.context.key_relationships[0].name == "Alice"
        assert persona.context.key_relationships[0].role == "CTO"

        assert persona.chat.max_history_turns == 30
        assert persona.chat.compression.enabled is True
        assert persona.chat.compression.compress_after_turns == 15
        assert persona.chat.compression.keep_recent_turns == 5
        assert persona.chat.response.max_tokens == 2000
        assert persona.chat.response.temperature == 0.3
        assert persona.chat.response.cite_search_results is False
        assert "Custom template" in persona.chat.system_prompt_template

    def test_minimal_yaml(self, tmp_path) -> None:
        """Loads a minimal YAML with only required fields."""
        yaml_content = """\
persona:
  name: Bot
  title: Assistant
  company: Corp
"""
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text(yaml_content)

        persona = load_persona_config(config_file)

        assert persona.name == "Bot"
        assert persona.title == "Assistant"
        assert persona.company == "Corp"
        assert persona.email == ""
        assert persona.background == ""
        assert persona.expertise == []
        assert persona.communication_style.tone == "professional"
        assert persona.chat.max_history_turns == 20
        assert persona.chat.compression.enabled is True
        assert persona.chat.response.max_tokens == 1000

    def test_empty_persona_section(self, tmp_path) -> None:
        """Handles YAML with empty persona section."""
        yaml_content = """\
persona: {}
"""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text(yaml_content)

        persona = load_persona_config(config_file)

        assert persona.name == "Assistant"
        assert persona.title == ""
        assert persona.company == ""

    def test_string_path(self, tmp_path) -> None:
        """Accepts string path."""
        yaml_content = """\
persona:
  name: StringBot
  title: Bot
  company: Corp
"""
        config_file = tmp_path / "string_path.yaml"
        config_file.write_text(yaml_content)

        persona = load_persona_config(str(config_file))

        assert persona.name == "StringBot"
