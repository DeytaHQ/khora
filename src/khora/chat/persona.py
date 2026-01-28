"""Persona configuration for chat mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CommunicationStyle:
    """Communication style preferences."""

    tone: str = "professional"
    perspective: str = "balanced"
    preferences: list[str] = field(default_factory=list)


@dataclass
class KeyRelationship:
    """Key relationship in the persona's network."""

    name: str
    role: str
    context: str = ""


@dataclass
class PersonaContext:
    """Context about the persona's current situation."""

    current_priorities: list[str] = field(default_factory=list)
    key_relationships: list[KeyRelationship] = field(default_factory=list)


@dataclass
class CompressionConfig:
    """History compression configuration."""

    enabled: bool = True
    compress_after_turns: int = 10
    keep_recent_turns: int = 3
    strategy: str = "summarize"


@dataclass
class ResponseConfig:
    """Response generation configuration."""

    max_tokens: int = 1000
    temperature: float = 0.7
    include_sources: bool = True
    cite_search_results: bool = True


@dataclass
class ChatConfig:
    """Chat behavior configuration."""

    max_history_turns: int = 20
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    system_prompt_template: str = ""


@dataclass
class PersonaConfig:
    """Complete persona configuration."""

    name: str
    title: str
    company: str
    email: str = ""
    background: str = ""
    expertise: list[str] = field(default_factory=list)
    communication_style: CommunicationStyle = field(default_factory=CommunicationStyle)
    context: PersonaContext = field(default_factory=PersonaContext)
    chat: ChatConfig = field(default_factory=ChatConfig)


def load_persona_config(path: Path | str) -> PersonaConfig:
    """Load persona configuration from YAML file.

    Args:
        path: Path to the persona YAML file

    Returns:
        PersonaConfig instance
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse nested structures
    persona_data = data.get("persona", {})
    chat_data = data.get("chat", {})

    # Build communication style
    style_data = persona_data.get("communication_style", {})
    communication_style = CommunicationStyle(
        tone=style_data.get("tone", "professional"),
        perspective=style_data.get("perspective", "balanced"),
        preferences=style_data.get("preferences", []),
    )

    # Build context
    ctx_data = persona_data.get("context", {})
    relationships = [KeyRelationship(**r) for r in ctx_data.get("key_relationships", [])]
    context = PersonaContext(
        current_priorities=ctx_data.get("current_priorities", []),
        key_relationships=relationships,
    )

    # Build chat config
    comp_data = chat_data.get("compression", {})
    compression = CompressionConfig(
        enabled=comp_data.get("enabled", True),
        compress_after_turns=comp_data.get("compress_after_turns", 10),
        keep_recent_turns=comp_data.get("keep_recent_turns", 3),
        strategy=comp_data.get("strategy", "summarize"),
    )

    resp_data = chat_data.get("response", {})
    response = ResponseConfig(
        max_tokens=resp_data.get("max_tokens", 1000),
        temperature=resp_data.get("temperature", 0.7),
        include_sources=resp_data.get("include_sources", True),
        cite_search_results=resp_data.get("cite_search_results", True),
    )

    chat_config = ChatConfig(
        max_history_turns=chat_data.get("max_history_turns", 20),
        compression=compression,
        response=response,
        system_prompt_template=chat_data.get("system_prompt_template", ""),
    )

    return PersonaConfig(
        name=persona_data.get("name", "Assistant"),
        title=persona_data.get("title", ""),
        company=persona_data.get("company", ""),
        email=persona_data.get("email", ""),
        background=persona_data.get("background", ""),
        expertise=persona_data.get("expertise", []),
        communication_style=communication_style,
        context=context,
        chat=chat_config,
    )
