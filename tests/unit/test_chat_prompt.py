"""Unit tests for khora.chat.prompt — PromptGenerator."""

from __future__ import annotations

from khora.chat.history import ChatMessage
from khora.chat.persona import ChatConfig, PersonaConfig
from khora.chat.prompt import PromptGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_persona(**overrides) -> PersonaConfig:
    """Create a PersonaConfig with sensible test defaults."""
    chat = ChatConfig(
        system_prompt_template=overrides.pop("system_prompt_template", ""),
    )
    return PersonaConfig(
        name=overrides.pop("name", "Jane Doe"),
        title=overrides.pop("title", "VP Engineering"),
        company=overrides.pop("company", "Acme Inc"),
        background=overrides.pop("background", "Experienced engineering leader."),
        chat=chat,
        **overrides,
    )


# ---------------------------------------------------------------------------
# PromptGenerator.__init__
# ---------------------------------------------------------------------------


class TestPromptGeneratorInit:
    """Tests for PromptGenerator initialization."""

    def test_stores_persona(self) -> None:
        """Init stores persona reference."""
        persona = _make_persona()
        gen = PromptGenerator(persona)
        assert gen.persona is persona

    def test_uses_default_template(self) -> None:
        """Uses default template when persona has no custom one."""
        persona = _make_persona()
        gen = PromptGenerator(persona)
        # Render it to verify it works
        prompt = gen.build_system_prompt()
        assert "Jane Doe" in prompt
        assert "VP Engineering" in prompt

    def test_uses_custom_template(self) -> None:
        """Uses custom template from persona."""
        persona = _make_persona(system_prompt_template="Hello, I am {{ persona.name }} of {{ persona.company }}.")
        gen = PromptGenerator(persona)
        prompt = gen.build_system_prompt()
        assert prompt == "Hello, I am Jane Doe of Acme Inc."


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for PromptGenerator.build_system_prompt()."""

    def test_includes_persona_info(self) -> None:
        """System prompt includes persona name, title, and background."""
        persona = _make_persona(
            name="Alice",
            title="CTO",
            company="TechCo",
            background="Specialist in distributed systems.",
        )
        gen = PromptGenerator(persona)
        prompt = gen.build_system_prompt()

        assert "Alice" in prompt
        assert "CTO" in prompt
        assert "TechCo" in prompt
        assert "distributed systems" in prompt

    def test_includes_history_summary(self) -> None:
        """System prompt includes history summary when provided."""
        persona = _make_persona()
        gen = PromptGenerator(persona)
        prompt = gen.build_system_prompt("We discussed the Q3 roadmap.")

        assert "We discussed the Q3 roadmap." in prompt
        assert "Previous conversation context" in prompt

    def test_excludes_history_section_when_empty(self) -> None:
        """System prompt omits history section when summary is empty."""
        persona = _make_persona()
        gen = PromptGenerator(persona)
        prompt = gen.build_system_prompt("")

        assert "Previous conversation context" not in prompt


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Tests for PromptGenerator.build_messages()."""

    def test_basic_structure(self) -> None:
        """Returns list with system message and user message."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        messages = gen.build_messages("What is X?", [], "", [])

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "What is X?" in messages[-1]["content"]

    def test_includes_search_results(self) -> None:
        """User message includes search results when provided."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        search_results = [
            {"content": "X is a framework for building apps.", "source": "docs", "score": 0.95},
            {"content": "X was released in 2023.", "source": "blog", "score": 0.85},
        ]

        messages = gen.build_messages("What is X?", search_results, "", [])

        user_content = messages[-1]["content"]
        assert "Relevant context" in user_content
        assert "X is a framework" in user_content
        assert "(docs)" in user_content
        assert "(blog)" in user_content
        assert "Question: What is X?" in user_content

    def test_no_search_results(self) -> None:
        """User message without search results has no context section."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        messages = gen.build_messages("What is X?", [], "", [])

        user_content = messages[-1]["content"]
        assert "Relevant context" not in user_content
        assert "Question: What is X?" in user_content

    def test_includes_history_messages(self) -> None:
        """Recent history messages are included between system and user."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        history = [
            ChatMessage(role="user", content="What is Y?"),
            ChatMessage(role="assistant", content="Y is a library."),
        ]

        messages = gen.build_messages("What is X?", [], "", history)

        assert len(messages) == 4  # system + 2 history + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is Y?"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Y is a library."
        assert messages[3]["role"] == "user"
        assert "What is X?" in messages[3]["content"]

    def test_includes_history_summary_in_system(self) -> None:
        """History summary is included in the system prompt."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        messages = gen.build_messages(
            "Follow up question",
            [],
            "We discussed project timelines.",
            [],
        )

        system_content = messages[0]["content"]
        assert "We discussed project timelines." in system_content

    def test_search_results_truncated_to_five(self) -> None:
        """Only first 5 search results are included."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        search_results = [{"content": f"Result {i}", "source": "src", "score": 0.9} for i in range(10)]

        messages = gen.build_messages("query", search_results, "", [])

        user_content = messages[-1]["content"]
        assert "Result 0" in user_content
        assert "Result 4" in user_content
        assert "Result 5" not in user_content

    def test_combined_history_search_and_query(self) -> None:
        """Full message list with history, search results, and query."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        history = [
            ChatMessage(role="user", content="Tell me about A."),
            ChatMessage(role="assistant", content="A is important."),
        ]
        search_results = [
            {"content": "B is related to A.", "source": "wiki", "score": 0.9},
        ]

        messages = gen.build_messages(
            "How does B relate to A?",
            search_results,
            "Earlier discussion about A.",
            history,
        )

        assert len(messages) == 4  # system + 2 history + user
        assert "Earlier discussion about A." in messages[0]["content"]
        assert messages[1]["content"] == "Tell me about A."
        assert messages[2]["content"] == "A is important."
        assert "B is related to A." in messages[3]["content"]
        assert "How does B relate to A?" in messages[3]["content"]


# ---------------------------------------------------------------------------
# _format_user_message
# ---------------------------------------------------------------------------


class TestFormatUserMessage:
    """Tests for PromptGenerator._format_user_message()."""

    def test_query_only(self) -> None:
        """Formats just the query when no search results."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        result = gen._format_user_message("What is X?", [])
        assert result == "Question: What is X?"

    def test_with_results(self) -> None:
        """Includes numbered search results."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        results = [
            {"content": "First result.", "source": "docs"},
            {"content": "Second result.", "source": "wiki"},
        ]

        result = gen._format_user_message("query", results)

        assert "[1] (docs): First result." in result
        assert "[2] (wiki): Second result." in result
        assert "---" in result
        assert "Question: query" in result

    def test_content_truncation(self) -> None:
        """Long content is truncated to 500 chars."""
        persona = _make_persona()
        gen = PromptGenerator(persona)

        long_content = "A" * 1000
        results = [{"content": long_content, "source": "src"}]

        result = gen._format_user_message("q", results)

        # The content portion should be at most 500 chars
        assert "A" * 500 in result
        assert "A" * 501 not in result
