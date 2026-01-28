"""Chat module for Khora Memory Lake.

Provides conversational interface with persona-based responses,
history management, and memory lake integration.
"""

from __future__ import annotations

from .engine import ChatEngine, ChatResponse
from .history import ChatMessage, ConversationHistory, HistoryManager
from .persona import PersonaConfig, load_persona_config
from .prompt import PromptGenerator

__all__ = [
    "ChatEngine",
    "ChatResponse",
    "ChatMessage",
    "ConversationHistory",
    "HistoryManager",
    "PersonaConfig",
    "load_persona_config",
    "PromptGenerator",
]
