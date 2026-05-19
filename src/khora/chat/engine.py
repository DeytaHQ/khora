"""Chat engine for conversational Khora interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import litellm
from loguru import logger

from .history import HistoryManager
from .prompt import PromptGenerator

if TYPE_CHECKING:
    from khora.khora import Khora

    from .persona import PersonaConfig


@dataclass
class ChatResponse:
    """Response from chat engine."""

    content: str
    conversation_id: UUID
    message_id: UUID
    sources: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ChatEngine:
    """Main chat engine orchestrating persona, history, and responses."""

    def __init__(
        self,
        persona: PersonaConfig,
        khora: Khora,
        llm_model: str = "gpt-4o",
        agentic_search: bool = False,
    ) -> None:
        """Initialize the chat engine.

        Args:
            persona: Persona configuration for response generation
            khora: Khora instance for search
            llm_model: LLM model to use for response generation
            agentic_search: Whether to use multi-step agentic search
        """
        self.persona = persona
        self.khora = khora
        self.llm_model = llm_model
        self.agentic_search = agentic_search

        self.history_manager = HistoryManager(
            max_turns=persona.chat.max_history_turns,
            compress_after=persona.chat.compression.compress_after_turns,
            keep_recent=persona.chat.compression.keep_recent_turns,
        )

        self.prompt_generator = PromptGenerator(persona)

    async def chat(
        self,
        query: str,
        *,
        namespace_id: UUID,
        conversation_id: UUID | None = None,
    ) -> ChatResponse:
        """Process a chat message and generate response.

        Args:
            query: User's question
            namespace_id: Namespace to search in
            conversation_id: Optional conversation ID for history

        Returns:
            ChatResponse with generated answer
        """
        from khora.telemetry.context import clear_trace_id, ensure_trace_id

        ensure_trace_id()

        # Get or create conversation
        conv_id = conversation_id or uuid4()
        self.history_manager.get_or_create(conv_id, namespace_id)

        logger.debug(f"Processing chat query: {query[:50]}...")

        # 1. Search Khora for relevant context
        recall_result = await self.khora.recall(
            query,
            namespace=namespace_id,
            limit=10,
            agentic=self.agentic_search,
        )

        # Convert to simple dict format for prompt
        # Resolve source_system from parent documents (chunks don't store it directly)
        search_results = []
        doc_cache: dict[str, str] = {}

        # Build chunk provenance map from recall metadata
        chunk_provenance_map: dict[str, str] = {}
        search_methods = recall_result.engine_info.get("search_methods", {})
        chunk_overlap = search_methods.get("chunk_overlap", {})
        for method_key, method_data in chunk_overlap.items():
            # method_key is like "vector_only", "graph_only", "vector_and_graph"
            method_label = method_key.replace("_only", "").replace("_and_", "+")
            for chunk_id in method_data.get("ids", []):
                chunk_provenance_map[chunk_id] = method_label

        for chunk in recall_result.chunks[:5]:
            source = "unknown"
            if chunk.document_id:
                doc_id_str = str(chunk.document_id)
                if doc_id_str in doc_cache:
                    source = doc_cache[doc_id_str]
                else:
                    try:
                        doc = await self.khora.get_document(chunk.document_id)
                        if doc:
                            source = (doc.metadata or {}).get("source_system", "")
                            if not source:
                                source = doc.source.split("/")[0] if doc.source else "unknown"
                            doc_cache[doc_id_str] = source
                    except Exception as e:
                        logger.debug(f"Failed to fetch document source for {doc_id_str}: {e}")

            # Look up search provenance from recall metadata
            source_method = chunk_provenance_map.get(str(chunk.id), "")

            result_entry: dict[str, Any] = {
                "content": chunk.content,
                "source": source,
                "score": chunk.score,
            }
            if source_method:
                result_entry["found_via"] = source_method

            search_results.append(result_entry)

        # Build entity context from recall entities (top 10 with descriptions)
        entity_context: list[dict[str, Any]] = []
        if recall_result.entities:
            for entity in recall_result.entities[:10]:
                if not entity.description:
                    continue
                entry: dict[str, Any] = {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "description": entity.description,
                    "attributes": {k: v for k, v in (entity.attributes or {}).items() if v},
                }
                entity_context.append(entry)

        logger.debug(
            f"Found {len(search_results)} relevant search results and {len(entity_context)} entity context entries"
        )

        # 2. Get conversation context
        history_summary, recent_messages = self.history_manager.get_context_messages(conv_id)

        # 3. Build prompt
        messages = self.prompt_generator.build_messages(
            query,
            search_results,
            history_summary,
            recent_messages,
            entity_context=entity_context,
        )

        # 4. Generate response
        logger.debug(f"Generating response with {self.llm_model}")
        import time as _time

        _t0 = _time.perf_counter()
        response = await litellm.acompletion(
            model=self.llm_model,
            messages=messages,
            max_tokens=self.persona.chat.response.max_tokens,
            temperature=self.persona.chat.response.temperature,
        )
        _latency = (_time.perf_counter() - _t0) * 1000

        # Record telemetry
        from khora.telemetry import get_collector

        usage = getattr(response, "usage", None)
        get_collector().record_llm_call(
            operation="chat",
            model=self.llm_model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            latency_ms=_latency,
            namespace_id=namespace_id,
        )

        response_content = response.choices[0].message.content

        # 5. Add messages to history
        self.history_manager.add_message(conv_id, "user", query, search_results)
        assistant_msg = self.history_manager.add_message(conv_id, "assistant", response_content)

        # 6. Compress history if needed
        if self.persona.chat.compression.enabled:
            compressed = await self.history_manager.compress_if_needed(conv_id)
            if compressed:
                logger.debug("Compressed conversation history")

        clear_trace_id()

        return ChatResponse(
            content=response_content,
            conversation_id=conv_id,
            message_id=assistant_msg.id,
            sources=search_results,
            metadata={
                "model": self.llm_model,
                "search_count": len(recall_result.chunks),
            },
        )

    def clear_conversation(self, conversation_id: UUID) -> None:
        """Clear a conversation's history.

        Args:
            conversation_id: Conversation to clear
        """
        self.history_manager.clear(conversation_id)
