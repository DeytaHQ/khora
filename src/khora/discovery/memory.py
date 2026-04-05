"""Persistent memory for the discovery agent.

Stores fetch outcomes (successes, failures, methods used) in a SurrealDB
embedded database so the agent can learn from previous attempts and avoid
repeating failed approaches.

Optional -- gracefully degrades to no-op when surrealdb is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

_HAS_SURREALDB = False
try:
    from surrealdb import Surreal

    _HAS_SURREALDB = True
except ImportError:
    pass


@dataclass(slots=True)
class MemoryEntry:
    """A single memory record."""

    url: str
    method: str  # "perplexity_search", "firecrawl_scrape", "direct_download", "generated_script"
    outcome: str  # "success", "failure", "partial"
    error: str = ""
    content_type: str = ""
    file_size: int = 0
    quality_score: float = 0.0
    query: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class DiscoveryMemory:
    """Persistent memory backed by SurrealDB embedded mode.

    Stores and retrieves discovery outcomes so the agent can:
    - Avoid retrying URLs that previously failed
    - Use methods that worked for similar source types
    - Track what has been tried for each URL

    All operations are no-ops when surrealdb is not installed.

    Usage::

        memory = DiscoveryMemory("./output/data/.chronicle.db")
        await memory.connect()

        await memory.remember(MemoryEntry(url="...", method="direct_download", outcome="success"))

        failures = await memory.recall_failures("example.com")
        methods = await memory.recall_successful_methods("csv")
        tried = await memory.has_been_tried("https://example.com/data.csv", "direct_download")

        await memory.close()
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._client: Any = None
        self._connected = False

    @property
    def available(self) -> bool:
        """Whether the SurrealDB SDK is installed."""
        return _HAS_SURREALDB

    async def connect(self) -> None:
        """Connect to the embedded SurrealDB instance and initialize schema."""
        if not _HAS_SURREALDB:
            logger.debug("SurrealDB not installed -- discovery memory disabled")
            return

        try:
            self._client = Surreal(f"surrealkv://{self._db_path}")
            await self._client.connect()
            await self._client.use("discovery", "memory")
            await self._client.signin({"username": "root", "password": "root"})
            # Create table schema
            await self._client.query("""
                DEFINE TABLE IF NOT EXISTS memory SCHEMAFULL;
                DEFINE FIELD IF NOT EXISTS url ON memory TYPE string;
                DEFINE FIELD IF NOT EXISTS method ON memory TYPE string;
                DEFINE FIELD IF NOT EXISTS outcome ON memory TYPE string;
                DEFINE FIELD IF NOT EXISTS error ON memory TYPE string;
                DEFINE FIELD IF NOT EXISTS content_type ON memory TYPE string;
                DEFINE FIELD IF NOT EXISTS file_size ON memory TYPE int;
                DEFINE FIELD IF NOT EXISTS quality_score ON memory TYPE float;
                DEFINE FIELD IF NOT EXISTS query ON memory TYPE string;
                DEFINE FIELD IF NOT EXISTS timestamp ON memory TYPE string;
                DEFINE INDEX IF NOT EXISTS idx_memory_url ON memory FIELDS url;
                DEFINE INDEX IF NOT EXISTS idx_memory_method ON memory FIELDS method;
            """)
            self._connected = True
            logger.debug(f"Discovery memory connected: {self._db_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize discovery memory: {e}")
            self._client = None

    async def close(self) -> None:
        """Close the SurrealDB connection."""
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False

    async def remember(self, entry: MemoryEntry) -> None:
        """Store a discovery outcome."""
        if not self._connected:
            return
        try:
            await self._client.query(
                "CREATE memory CONTENT $data",
                {
                    "data": {
                        "url": entry.url,
                        "method": entry.method,
                        "outcome": entry.outcome,
                        "error": entry.error,
                        "content_type": entry.content_type,
                        "file_size": entry.file_size,
                        "quality_score": entry.quality_score,
                        "query": entry.query,
                        "timestamp": entry.timestamp,
                    }
                },
            )
        except Exception as e:
            logger.debug(f"Failed to store memory: {e}")

    async def recall_failures(self, url_pattern: str) -> list[dict[str, Any]]:
        """Get previous failures for URLs containing the pattern."""
        if not self._connected:
            return []
        try:
            result = await self._client.query(
                "SELECT * FROM memory WHERE url CONTAINS $pattern AND outcome = 'failure'"
                " ORDER BY timestamp DESC LIMIT 10",
                {"pattern": url_pattern},
            )
            return result[0] if result else []
        except Exception as e:
            logger.debug(f"Failed to recall failures: {e}")
            return []

    async def recall_successful_methods(self, source_type: str) -> list[str]:
        """Get methods that worked for a source type (from query text)."""
        if not self._connected:
            return []
        try:
            result = await self._client.query(
                "SELECT DISTINCT method FROM memory WHERE outcome = 'success' AND query CONTAINS $type",
                {"type": source_type},
            )
            return [r.get("method", "") for r in (result[0] if result else []) if r.get("method")]
        except Exception as e:
            logger.debug(f"Failed to recall methods: {e}")
            return []

    async def has_been_tried(self, url: str, method: str) -> bool:
        """Check if a URL+method combo was already attempted."""
        if not self._connected:
            return False
        try:
            result = await self._client.query(
                "SELECT count() AS c FROM memory WHERE url = $url AND method = $method GROUP ALL",
                {"url": url, "method": method},
            )
            rows = result[0] if result else []
            return bool(rows and rows[0].get("c", 0) > 0)
        except Exception as e:
            logger.debug(f"Failed to check tried: {e}")
            return False

    async def get_stats(self) -> dict[str, int]:
        """Get summary stats of stored memories."""
        if not self._connected:
            return {"total": 0, "successes": 0, "failures": 0}
        try:
            result = await self._client.query("SELECT outcome, count() AS c FROM memory GROUP BY outcome")
            rows = result[0] if result else []
            stats: dict[str, int] = {"total": 0, "successes": 0, "failures": 0}
            for row in rows:
                count = row.get("c", 0)
                stats["total"] += count
                if row.get("outcome") == "success":
                    stats["successes"] = count
                elif row.get("outcome") == "failure":
                    stats["failures"] = count
            return stats
        except Exception:
            return {"total": 0, "successes": 0, "failures": 0}
