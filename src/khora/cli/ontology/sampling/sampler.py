"""Data sampler for ontology construction.

Implements stratified sampling across multiple data sources, ensuring
diversity in file types and source proportions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from loguru import logger

from ..sources.base import DataSource, SampleChunk, SourceSummary


@dataclass
class DataSampler:
    """Stratified sampler across multiple data sources.

    Allocates a character budget proportionally across sources using
    sqrt-weighting (prevents large sources from dominating while ensuring
    small sources get meaningful allocation).
    """

    sources: list[DataSource] = field(default_factory=list)
    summaries: list[SourceSummary] = field(default_factory=list)
    samples: list[SampleChunk] = field(default_factory=list)

    def add_source(self, source: DataSource) -> SourceSummary:
        """Add a source, scan it, and return its summary."""
        summary = source.scan()
        self.sources.append(source)
        self.summaries.append(summary)
        logger.debug(
            f"Added source: {summary.source_id} "
            f"({summary.source_type}, {summary.file_count} files, {summary.size_human})"
        )
        return summary

    def sample_all(self, budget_chars: int = 30_000) -> list[SampleChunk]:
        """Sample from all sources, allocating budget proportionally.

        Uses sqrt-weighting so small sources are not starved while large
        sources do not dominate.

        Args:
            budget_chars: Total character budget across all sources.

        Returns:
            List of SampleChunk objects with combined text content.
        """
        if not self.sources:
            return []

        # Calculate sqrt-weighted allocation
        sizes = [max(1, s.total_bytes) for s in self.summaries]
        sqrt_sizes = [math.sqrt(s) for s in sizes]
        sqrt_total = sum(sqrt_sizes)

        budgets = [max(500, int(budget_chars * sq / sqrt_total)) for sq in sqrt_sizes]

        # Sample from each source
        self.samples = []
        for source, budget, summary in zip(self.sources, budgets, self.summaries):
            try:
                chunks = source.sample(budget)
                self.samples.extend(chunks)
                total_chars = sum(c.char_count for c in chunks)
                logger.debug(f"Sampled {total_chars} chars from {summary.source_id} " f"(budget: {budget})")
            except Exception:
                logger.warning(f"Failed to sample from {summary.source_id}")

        return self.samples

    @property
    def total_chars(self) -> int:
        """Total characters across all samples."""
        return sum(c.char_count for c in self.samples)

    @property
    def source_count(self) -> int:
        """Number of configured sources."""
        return len(self.sources)

    def format_samples_for_llm(self, max_chars: int = 30_000) -> str:
        """Format all samples into a single string for LLM consumption.

        Each sample is delimited with source metadata for context.

        Args:
            max_chars: Maximum total characters in the output.

        Returns:
            Formatted string with all samples.
        """
        parts: list[str] = []
        total = 0

        for chunk in self.samples:
            if total >= max_chars:
                break

            # Build header with metadata
            meta_parts = [f"source: {chunk.source_id}"]
            if "file" in chunk.metadata:
                meta_parts.append(f"file: {chunk.metadata['file']}")
            if "region" in chunk.metadata:
                meta_parts.append(f"region: {chunk.metadata['region']}")
            if "extension" in chunk.metadata:
                meta_parts.append(f"type: {chunk.metadata['extension']}")

            header = " | ".join(meta_parts)
            remaining = max_chars - total
            content = chunk.content[:remaining]

            parts.append(f"--- [{header}] ---\n{content}")
            total += len(content) + len(header) + 10  # ~10 chars for delimiters

        return "\n\n".join(parts)
