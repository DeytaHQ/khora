"""Prompt generation for chat responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Template

if TYPE_CHECKING:
    from .history import ChatMessage
    from .persona import PersonaConfig


class PromptGenerator:
    """Generates prompts for chat responses."""

    DEFAULT_SYSTEM_TEMPLATE = """You are {{ persona.name }}, {{ persona.title }} of {{ persona.company }}.

{{ persona.background }}

When answering questions:
1. Draw from the provided search results and company context
2. Be direct and actionable
3. Acknowledge when you don't have specific information

{% if history_summary %}
Previous conversation context:
{{ history_summary }}
{% endif %}"""

    def __init__(self, persona: PersonaConfig) -> None:
        """Initialize the prompt generator.

        Args:
            persona: Persona configuration
        """
        self.persona = persona
        template_str = persona.chat.system_prompt_template or self.DEFAULT_SYSTEM_TEMPLATE
        self._system_template = Template(template_str)

    def build_system_prompt(self, history_summary: str = "") -> str:
        """Build the system prompt with persona and history context.

        Args:
            history_summary: Compressed summary of conversation history

        Returns:
            System prompt string
        """
        return self._system_template.render(
            persona=self.persona,
            history_summary=history_summary,
        )

    def build_messages(
        self,
        user_query: str,
        search_results: list[dict],
        history_summary: str,
        recent_messages: list[ChatMessage],
    ) -> list[dict]:
        """Build the complete message list for LLM.

        Args:
            user_query: Current user query
            search_results: Relevant search results
            history_summary: Compressed history summary
            recent_messages: Recent conversation messages

        Returns:
            List of message dicts for LLM
        """
        messages = []

        # System prompt with persona
        messages.append(
            {
                "role": "system",
                "content": self.build_system_prompt(history_summary),
            }
        )

        # Add recent conversation history
        for msg in recent_messages:
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        # Build user message with search context
        user_content = self._format_user_message(user_query, search_results)
        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        return messages

    def _format_user_message(
        self,
        query: str,
        search_results: list[dict],
    ) -> str:
        """Format user query with search context.

        Args:
            query: User's question
            search_results: Relevant search results

        Returns:
            Formatted user message
        """
        parts = []

        if search_results:
            parts.append("Relevant context from company knowledge base:\n")
            for i, result in enumerate(search_results[:5], 1):
                content = result.get("content", "")[:500]
                source = result.get("source", "unknown")
                parts.append(f"[{i}] ({source}): {content}\n")
            parts.append("\n---\n")

        parts.append(f"Question: {query}")

        return "".join(parts)
