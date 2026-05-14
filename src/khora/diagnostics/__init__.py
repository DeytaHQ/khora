"""Diagnostic helpers (one-shot reporters for engineering decisions).

These modules are intentionally simple and reach into storage adapters
directly. They are not part of khora's stable public API and may be
renamed or removed without a major-version bump.
"""

from khora.diagnostics.graph_density import GraphStats, compute_graph_stats

__all__ = ["GraphStats", "compute_graph_stats"]
