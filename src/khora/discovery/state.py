"""State models for the interactive discovery agent.

Defines the phase enum, data classes for discovered sources and fetch
results, and the mutable session state carried through the discovery loop.
All models are serializable to JSON for session persistence / resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

# ---------------------------------------------------------------------------
# Agent phase state machine
# ---------------------------------------------------------------------------


class AgentPhase(str, Enum):
    """Phases of the interactive discovery state machine.

    Transitions:
        GATHER_INTENT → SEARCH → PRESENT_RESULTS → SELECT_SOURCES
        → FETCH → REVIEW → INGEST → DONE

    Retry loops:
        PRESENT_RESULTS (no results) → SEARCH
        FETCH (all failed)           → SEARCH
        REVIEW (retry)               → SEARCH
        REVIEW (refine)              → GATHER_INTENT
    """

    GATHER_INTENT = "gather_intent"
    SEARCH = "search"
    PRESENT_RESULTS = "present_results"
    SELECT_SOURCES = "select_sources"
    FETCH = "fetch"
    REVIEW = "review"
    AUGMENT = "augment"
    INGEST = "ingest"
    DONE = "done"


# ---------------------------------------------------------------------------
# Source type / status enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """Type of a discovered data source."""

    WEBPAGE = "webpage"
    API = "api"
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    PDF = "pdf"
    REPO = "repo"
    RSS = "rss"
    DATASET = "dataset"
    OTHER = "other"


class SourceStatus(str, Enum):
    """Lifecycle status of a discovered source."""

    DISCOVERED = "discovered"
    SELECTED = "selected"
    FETCHING = "fetching"
    FETCHED = "fetched"
    VALIDATED = "validated"
    FAILED = "failed"


class FetchMethod(str, Enum):
    """How data was fetched from a source."""

    FIRECRAWL_SCRAPE = "firecrawl_scrape"
    FIRECRAWL_CRAWL = "firecrawl_crawl"
    DIRECT_DOWNLOAD = "direct_download"
    GENERATED_SCRIPT = "generated_script"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiscoveredSource:
    """A single data source found during discovery."""

    url: str
    title: str
    description: str = ""
    source_type: SourceType = SourceType.OTHER
    status: SourceStatus = SourceStatus.DISCOVERED
    relevance_score: float = 0.0
    access_method: str = ""  # "direct_download", "api_call", "scrape", etc.
    requires_auth: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # Discovery provenance
    discovered_via: str = ""  # "perplexity", "firecrawl", "user"
    discovery_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "source_type": self.source_type.value,
            "status": self.status.value,
            "relevance_score": self.relevance_score,
            "access_method": self.access_method,
            "requires_auth": self.requires_auth,
            "metadata": self.metadata,
            "discovered_via": self.discovered_via,
            "discovery_query": self.discovery_query,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveredSource:
        return cls(
            url=data["url"],
            title=data["title"],
            description=data.get("description", ""),
            source_type=SourceType(data.get("source_type", "other")),
            status=SourceStatus(data.get("status", "discovered")),
            relevance_score=data.get("relevance_score", 0.0),
            access_method=data.get("access_method", ""),
            requires_auth=data.get("requires_auth", False),
            metadata=data.get("metadata", {}),
            discovered_via=data.get("discovered_via", ""),
            discovery_query=data.get("discovery_query", ""),
        )


@dataclass(slots=True)
class FetchAttempt:
    """Record of a single fetch attempt against a source."""

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    method: FetchMethod = FetchMethod.DIRECT_DOWNLOAD
    success: bool = False
    error: str | None = None
    bytes_fetched: int = 0
    duration_seconds: float = 0.0
    script_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "method": self.method.value,
            "success": self.success,
            "error": self.error,
            "bytes_fetched": self.bytes_fetched,
            "duration_seconds": self.duration_seconds,
            "script_used": self.script_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FetchAttempt:
        return cls(
            timestamp=data.get("timestamp", ""),
            method=FetchMethod(data.get("method", "direct_download")),
            success=data.get("success", False),
            error=data.get("error"),
            bytes_fetched=data.get("bytes_fetched", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            script_used=data.get("script_used"),
        )


@dataclass(slots=True)
class FetchResult:
    """Result of fetching data from a discovered source."""

    source: DiscoveredSource
    local_path: str  # where content was saved
    content_type: str = ""  # "text/html", "application/json", etc.
    size_bytes: int = 0
    success: bool = False
    error: str | None = None
    attempts: list[FetchAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "local_path": self.local_path,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "success": self.success,
            "error": self.error,
            "attempts": [a.to_dict() for a in self.attempts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FetchResult:
        return cls(
            source=DiscoveredSource.from_dict(data["source"]),
            local_path=data.get("local_path", ""),
            content_type=data.get("content_type", ""),
            size_bytes=data.get("size_bytes", 0),
            success=data.get("success", False),
            error=data.get("error"),
            attempts=[FetchAttempt.from_dict(a) for a in data.get("attempts", [])],
        )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """Mutable state bag for an entire discovery session.

    Serializable to JSON for persistence / resume across crashes.
    """

    # Identity
    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Current phase
    phase: AgentPhase = AgentPhase.GATHER_INTENT

    # User intent
    user_intent: str = ""
    search_queries: list[str] = field(default_factory=list)

    # Discovery results
    discovered: list[DiscoveredSource] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)

    # Fetch results
    fetched: list[FetchResult] = field(default_factory=list)
    output_dir: str = "./khora_discovery_data"

    # Conversation history (for LLM context)
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    # Loop control
    iteration: int = 0
    max_iterations: int = 5

    # Cost tracking
    total_cost_usd: float = 0.0
    max_cost_usd: float = 2.0

    @property
    def selected_sources(self) -> list[DiscoveredSource]:
        """Return the sources the user has selected."""
        return [self.discovered[i] for i in self.selected_indices if i < len(self.discovered)]

    @property
    def successful_fetches(self) -> list[FetchResult]:
        """Return only successful fetch results."""
        return [f for f in self.fetched if f.success]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "phase": self.phase.value,
            "user_intent": self.user_intent,
            "search_queries": self.search_queries,
            "discovered": [s.to_dict() for s in self.discovered],
            "selected_indices": self.selected_indices,
            "fetched": [f.to_dict() for f in self.fetched],
            "output_dir": self.output_dir,
            "conversation_history": self.conversation_history,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "total_cost_usd": self.total_cost_usd,
            "max_cost_usd": self.max_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        return cls(
            session_id=data.get("session_id", str(uuid4())),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            phase=AgentPhase(data.get("phase", "gather_intent")),
            user_intent=data.get("user_intent", ""),
            search_queries=data.get("search_queries", []),
            discovered=[DiscoveredSource.from_dict(s) for s in data.get("discovered", [])],
            selected_indices=data.get("selected_indices", []),
            fetched=[FetchResult.from_dict(f) for f in data.get("fetched", [])],
            output_dir=data.get("output_dir", "./khora_discovery_data"),
            conversation_history=data.get("conversation_history", []),
            iteration=data.get("iteration", 0),
            max_iterations=data.get("max_iterations", 5),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            max_cost_usd=data.get("max_cost_usd", 2.0),
        )

    def save(self, path: str | Path) -> None:
        """Persist session to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SessionState:
        """Load session from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
