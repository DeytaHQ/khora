"""Khora - Deyta's memory lake and materialization of knowledge.

Khora provides a unified interface for:
- Storing and retrieving knowledge artifacts
- Materializing data transformations
- Building memory graphs and relationships

Example usage:
    from khora import MemoryLake

    async with MemoryLake() as lake:
        await lake.remember("Important information to store")
        results = await lake.recall("query about information")
"""

from .cli import main
from .memory_lake import MemoryLake, RecallResult, RememberResult
from .query import SearchMode

__version__ = "0.0.12"

__all__ = [
    "main",
    "MemoryLake",
    "RememberResult",
    "RecallResult",
    "SearchMode",
]
