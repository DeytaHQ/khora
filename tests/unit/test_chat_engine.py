"""Unit tests for khora.chat.engine — ChatEngine and ChatResponse."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from khora.chat.engine import ChatEngine, ChatResponse
from khora.chat.persona import ChatConfig, CompressionConfig, PersonaConfig, ResponseConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_persona(**overrides) -> PersonaConfig:
    """Create a PersonaConfig with sensible test defaults."""
    compression = CompressionConfig(
        enabled=overrides.pop("compression_enabled", True),
        compress_after_turns=overrides.pop("compress_after", 10),
        keep_recent_turns=overrides.pop("keep_recent", 3),
    )
    response = ResponseConfig(
        max_tokens=overrides.pop("max_tokens", 500),
        temperature=overrides.pop("temperature", 0.5),
    )
    chat = ChatConfig(
        max_history_turns=overrides.pop("max_history_turns", 20),
        compression=compression,
        response=response,
    )
    return PersonaConfig(
        name=overrides.pop("name", "Test Bot"),
        title=overrides.pop("title", "Test Assistant"),
        company=overrides.pop("company", "TestCorp"),
        chat=chat,
        **overrides,
    )


def _make_mock_kb() -> MagicMock:
    """Create a mock Khora."""
    kb = MagicMock()

    # recall returns a result with chunks
    mock_chunk = MagicMock()
    mock_chunk.content = "relevant document content"
    mock_chunk.document_id = uuid4()
    mock_chunk.score = 0.95

    mock_recall = MagicMock()
    mock_recall.chunks = [mock_chunk]
    mock_recall.entities = []
    mock_recall.engine_info = {}
    kb.recall = AsyncMock(return_value=mock_recall)

    # get_document returns a doc with metadata
    mock_doc = MagicMock()
    mock_doc.metadata = {"source_system": "confluence"}
    mock_doc.source = "confluence/page"
    kb.get_document = AsyncMock(return_value=mock_doc)

    return kb


def _make_mock_recall_result(
    *,
    chunks: list[MagicMock] | None = None,
    entities: list[MagicMock] | None = None,
    engine_info: dict | None = None,
) -> MagicMock:
    """Create a mock recall result."""
    mock_recall = MagicMock()
    mock_recall.chunks = chunks or []
    mock_recall.entities = entities or []
    mock_recall.engine_info = engine_info or {}
    return mock_recall


def _make_litellm_response(content: str = "Generated response") -> MagicMock:
    """Create a mock litellm response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150
    return response


# ---------------------------------------------------------------------------
# ChatResponse dataclass
# ---------------------------------------------------------------------------


class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        conv_id = uuid4()
        msg_id = uuid4()
        r = ChatResponse(
            content="Hello",
            conversation_id=conv_id,
            message_id=msg_id,
            sources=[{"content": "src", "score": 0.9}],
            metadata={"model": "gpt-4o"},
        )
        assert r.content == "Hello"
        assert r.conversation_id == conv_id
        assert r.message_id == msg_id
        assert len(r.sources) == 1
        assert r.metadata["model"] == "gpt-4o"

    def test_defaults(self) -> None:
        """Default values for optional fields."""
        r = ChatResponse(
            content="Hello",
            conversation_id=uuid4(),
            message_id=uuid4(),
        )
        assert r.sources == []
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# ChatEngine.__init__
# ---------------------------------------------------------------------------


class TestChatEngineInit:
    """Tests for ChatEngine initialization."""

    def test_stores_persona_and_kb(self) -> None:
        """Init stores persona, khora, and model."""
        persona = _make_persona()
        kb = _make_mock_kb()

        engine = ChatEngine(persona, kb, llm_model="gpt-4o-mini")

        assert engine.persona is persona
        assert engine.khora is kb
        assert engine.llm_model == "gpt-4o-mini"

    def test_default_model(self) -> None:
        """Default LLM model is gpt-4o."""
        persona = _make_persona()
        kb = _make_mock_kb()

        engine = ChatEngine(persona, kb)

        assert engine.llm_model == "gpt-4o"

    def test_creates_history_manager(self) -> None:
        """Init creates HistoryManager with persona settings."""
        persona = _make_persona(max_history_turns=30, compress_after=15, keep_recent=5)
        kb = _make_mock_kb()

        engine = ChatEngine(persona, kb)

        assert engine.history_manager.max_turns == 30
        assert engine.history_manager.compress_after == 15
        assert engine.history_manager.keep_recent == 5

    def test_creates_prompt_generator(self) -> None:
        """Init creates PromptGenerator with persona."""
        persona = _make_persona()
        kb = _make_mock_kb()

        engine = ChatEngine(persona, kb)

        assert engine.prompt_generator.persona is persona


# ---------------------------------------------------------------------------
# ChatEngine.chat()
# ---------------------------------------------------------------------------


class TestChatEngineChat:
    """Tests for ChatEngine.chat()."""

    async def test_chat_returns_chat_response(self) -> None:
        """chat() returns a ChatResponse with correct fields."""
        persona = _make_persona(compression_enabled=False)
        kb = _make_mock_kb()
        engine = ChatEngine(persona, kb)

        ns_id = uuid4()
        mock_response = _make_litellm_response("The answer is 42.")

        with (
            patch("khora.chat.engine.litellm") as mock_litellm,
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.telemetry.get_collector") as mock_get_collector,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_get_collector.return_value = MagicMock()

            result = await engine.chat("What is the answer?", namespace_id=ns_id)

        assert isinstance(result, ChatResponse)
        assert result.content == "The answer is 42."
        assert isinstance(result.conversation_id, UUID)
        assert isinstance(result.message_id, UUID)
        assert result.metadata["model"] == "gpt-4o"

    async def test_chat_calls_recall(self) -> None:
        """chat() calls kb.recall() with the query."""
        persona = _make_persona(compression_enabled=False)
        kb = _make_mock_kb()
        engine = ChatEngine(persona, kb)

        ns_id = uuid4()
        mock_response = _make_litellm_response()

        with (
            patch("khora.chat.engine.litellm") as mock_litellm,
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.telemetry.get_collector") as mock_get_collector,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_get_collector.return_value = MagicMock()

            await engine.chat("test query", namespace_id=ns_id)

        kb.recall.assert_awaited_once_with(
            "test query",
            namespace=ns_id,
            limit=10,
        )

    async def test_chat_calls_litellm(self) -> None:
        """chat() calls litellm.acompletion with correct params."""
        persona = _make_persona(
            max_tokens=800,
            temperature=0.3,
            compression_enabled=False,
        )
        kb = _make_mock_kb()
        engine = ChatEngine(persona, kb, llm_model="gpt-4o-mini")

        ns_id = uuid4()
        mock_response = _make_litellm_response()

        with (
            patch("khora.chat.engine.litellm") as mock_litellm,
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.telemetry.get_collector") as mock_get_collector,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_get_collector.return_value = MagicMock()

            await engine.chat("test", namespace_id=ns_id)

        mock_litellm.acompletion.assert_awaited_once()
        call_kwargs = mock_litellm.acompletion.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs.kwargs["max_tokens"] == 800
        assert call_kwargs.kwargs["temperature"] == 0.3

    async def test_chat_records_telemetry(self) -> None:
        """chat() records telemetry via get_collector()."""
        persona = _make_persona(compression_enabled=False)
        kb = _make_mock_kb()
        engine = ChatEngine(persona, kb)

        ns_id = uuid4()
        mock_response = _make_litellm_response()

        with (
            patch("khora.chat.engine.litellm") as mock_litellm,
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.telemetry.get_collector") as mock_get_collector,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_collector = MagicMock()
            mock_get_collector.return_value = mock_collector

            await engine.chat("test", namespace_id=ns_id)

        mock_collector.record_llm_call.assert_called_once()
        call_kwargs = mock_collector.record_llm_call.call_args
        assert call_kwargs.kwargs["operation"] == "chat"
        assert call_kwargs.kwargs["prompt_tokens"] == 100
        assert call_kwargs.kwargs["completion_tokens"] == 50
        assert call_kwargs.kwargs["total_tokens"] == 150

    async def test_chat_with_existing_conversation_id(self) -> None:
        """chat() uses provided conversation_id."""
        persona = _make_persona(compression_enabled=False)
        kb = _make_mock_kb()
        engine = ChatEngine(persona, kb)

        ns_id = uuid4()
        conv_id = uuid4()
        mock_response = _make_litellm_response()

        with (
            patch("khora.chat.engine.litellm") as mock_litellm,
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.telemetry.get_collector") as mock_get_collector,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_get_collector.return_value = MagicMock()

            result = await engine.chat(
                "test",
                namespace_id=ns_id,
                conversation_id=conv_id,
            )

        assert result.conversation_id == conv_id

    async def test_chat_sources_in_response(self) -> None:
        """chat() includes search results as sources."""
        persona = _make_persona(compression_enabled=False)
        kb = _make_mock_kb()
        engine = ChatEngine(persona, kb)

        ns_id = uuid4()
        mock_response = _make_litellm_response()

        with (
            patch("khora.chat.engine.litellm") as mock_litellm,
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.telemetry.get_collector") as mock_get_collector,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_get_collector.return_value = MagicMock()

            result = await engine.chat("test", namespace_id=ns_id)

        assert len(result.sources) == 1
        assert result.sources[0]["source"] == "confluence"
        assert result.sources[0]["content"] == "relevant document content"

    async def test_chat_returns_question_card_when_recall_abstains(self) -> None:
        """Weak retrieval returns question-card metadata and skips the LLM."""
        persona = _make_persona(compression_enabled=False)
        kb = _make_mock_kb()
        kb.recall = AsyncMock(
            return_value=_make_mock_recall_result(
                engine_info={"abstention_signals": {"should_abstain": True}},
            )
        )
        engine = ChatEngine(persona, kb)

        ns_id = uuid4()

        with (
            patch("khora.chat.engine.litellm") as mock_litellm,
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await engine.chat("Who's most into running?", namespace_id=ns_id)

        mock_litellm.acompletion.assert_not_called()
        assert result.metadata["abstained"] is True
        assert result.metadata["show_question_card"] is True
        assert result.metadata["question_card"]["question"] == "Who's most into running?"
        assert result.metadata["question_card_reason"] == "retrieval_abstained"

    async def test_chat_passes_understanding_to_prompt(self) -> None:
        """Prompt generation receives query-understanding metadata from recall."""
        persona = _make_persona(compression_enabled=False)
        kb = _make_mock_kb()
        kb.recall.return_value.engine_info = {
            "understanding": {"intent": "QUESTION", "answer_type": "SUMMARY", "entities": ["running"]}
        }
        engine = ChatEngine(persona, kb)
        engine.prompt_generator.build_messages = MagicMock(
            return_value=[{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
        )

        ns_id = uuid4()
        mock_response = _make_litellm_response()

        with (
            patch("khora.chat.engine.litellm") as mock_litellm,
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.telemetry.get_collector") as mock_get_collector,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_get_collector.return_value = MagicMock()
            await engine.chat("Who's most into running?", namespace_id=ns_id)

        assert engine.prompt_generator.build_messages.call_args.kwargs["understanding"] == {
            "intent": "QUESTION",
            "answer_type": "SUMMARY",
            "entities": ["running"],
        }


# ---------------------------------------------------------------------------
# ChatEngine.clear_conversation()
# ---------------------------------------------------------------------------


class TestChatEngineClearConversation:
    """Tests for ChatEngine.clear_conversation()."""

    def test_delegates_to_history_manager(self) -> None:
        """clear_conversation() delegates to history manager."""
        persona = _make_persona()
        kb = _make_mock_kb()
        engine = ChatEngine(persona, kb)

        conv_id = uuid4()
        engine.history_manager.clear = MagicMock()

        engine.clear_conversation(conv_id)

        engine.history_manager.clear.assert_called_once_with(conv_id)
