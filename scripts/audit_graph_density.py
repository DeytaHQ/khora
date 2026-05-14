"""Print per-namespace graph-density stats for the PPR decision gate (#598).

Connects to khora with the ambient ``KHORA_*`` configuration, walks every
namespace it can see (or a list passed via ``--namespace``), and prints
a CSV-ready table plus a one-line verdict per namespace.

Usage::

    # All namespaces in the ambient configuration
    uv run python scripts/audit_graph_density.py

    # A single namespace
    uv run python scripts/audit_graph_density.py --namespace 11111111-2222-...

    # JSON output (machine-readable)
    uv run python scripts/audit_graph_density.py --format json

Decision criteria (from #598): the median namespace must have ≥3
connected components OR mean degree ≥5 in the largest connected
component. If neither holds, PPR converges near-uniform and the
#542 BFS+RRF → PPR swap isn't worth the complexity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict
from uuid import UUID

from khora import Khora
from khora.diagnostics.graph_density import GraphStats, compute_graph_stats


async def _run(namespace_filters: list[UUID] | None, output_format: str) -> int:
    async with Khora() as kb:
        # ``list_namespaces`` is on the storage coordinator; if the caller
        # supplied an explicit filter we don't need it.
        if namespace_filters:
            target_ids = namespace_filters
        else:
            namespaces = await kb.storage.list_namespaces(limit=10_000)
            target_ids = [ns.namespace_id for ns in namespaces]

        rows: list[GraphStats] = []
        for ns_id in target_ids:
            stats = await compute_graph_stats(kb.storage, ns_id)
            rows.append(stats)

    if not rows:
        print("No namespaces found.", file=sys.stderr)
        return 1

    if output_format == "json":
        # Stringify the UUID so the JSON is round-trippable.
        json.dump(
            [{**asdict(r), "namespace_id": str(r.namespace_id)} for r in rows],
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        _print_csv(rows)
    _print_verdict(rows)
    return 0


def _print_csv(rows: list[GraphStats]) -> None:
    headers = [
        "namespace_id",
        "num_entities",
        "num_relationships",
        "mean_degree",
        "median_degree",
        "num_components",
        "largest_cc_size",
        "largest_cc_fraction",
        "mean_degree_largest_cc",
        "meets_ppr_threshold",
    ]
    print(",".join(headers))
    for r in rows:
        print(
            ",".join(
                str(v)
                for v in (
                    r.namespace_id,
                    r.num_entities,
                    r.num_relationships,
                    f"{r.mean_degree:.3f}",
                    f"{r.median_degree:.3f}",
                    r.num_components,
                    r.largest_cc_size,
                    f"{r.largest_cc_fraction:.3f}",
                    f"{r.mean_degree_largest_cc:.3f}",
                    r.meets_ppr_threshold,
                )
            )
        )


def _print_verdict(rows: list[GraphStats]) -> None:
    """One-line summary the operator can paste into the decision issue."""
    median_components = statistics.median(r.num_components for r in rows)
    median_largest_cc_degree = statistics.median(r.mean_degree_largest_cc for r in rows)
    meets_rate = sum(1 for r in rows if r.meets_ppr_threshold) / len(rows)

    print("", file=sys.stderr)
    print(
        f"# Across {len(rows)} namespaces: median num_components={median_components:.1f}, "
        f"median mean_degree_largest_cc={median_largest_cc_degree:.2f}, "
        f"meets_ppr_threshold={meets_rate:.0%} of namespaces.",
        file=sys.stderr,
    )
    print(
        "# #598 decision: land #542 swap iff this rate is meaningfully >50% on a representative sample.",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        action="append",
        type=UUID,
        help="Restrict to a specific namespace UUID (may be repeated). Default: all namespaces.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output format. CSV is the default so the result pastes into a spreadsheet.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.namespace, args.format))


if __name__ == "__main__":
    raise SystemExit(main())
