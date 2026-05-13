"""Measure ACTION_ITEM / DECISION / BLOCKER / RISK extraction quality.

Reads labeled examples from ``tests/fixtures/action_item_labels.jsonl``,
runs the extraction skill against each, and reports precision / recall /
F1 per typed entity.

Issue #569: this is the scaffolding so that once a larger labeled set
exists, the measurement infrastructure is ready. Today the fixture has
only 5 hand-written examples — not enough to declare GA.

# TODO: expand fixture to 50 labeled meetings before declaring
# ACTION_ITEM extraction GA. Devil's Advocate gating thresholds:
# precision >= 0.7, recall >= 0.6 on the 50-example labeled set.

Usage:
    uv run python scripts/measure_action_item_extraction.py
    uv run python scripts/measure_action_item_extraction.py \\
        --fixture tests/fixtures/action_item_labels.jsonl \\
        --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from khora.extraction.extractors.llm import LLMEntityExtractor
from khora.extraction.skills.loader import ExpertiseLoader

# Typed entities we measure. Anything outside this set is ignored —
# we only care about the high-precision typed work artifacts.
TYPED_ENTITY_TYPES = {"ACTION_ITEM", "DECISION", "BLOCKER", "RISK"}


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    """Load JSONL fixture into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    examples: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _normalize_name(name: str) -> str:
    """Loose name comparison — lowercase + whitespace-collapse."""
    return " ".join((name or "").lower().split())


def _match(predicted: dict[str, Any], gold: dict[str, Any]) -> bool:
    """A predicted entity matches a gold entity when the type matches and
    the normalized name overlaps. Permissive on purpose — exact-string
    match is too strict for free-text labels.
    """
    if predicted.get("entity_type") != gold.get("entity_type"):
        return False
    p_name = _normalize_name(predicted.get("name", ""))
    g_name = _normalize_name(gold.get("name", ""))
    if not p_name or not g_name:
        return False
    return p_name in g_name or g_name in p_name


async def _extract(extractor: LLMEntityExtractor, text: str, expertise: Any) -> list[dict[str, Any]]:
    """Run the extractor on one example and return predicted typed entities."""
    result = await extractor.extract(text, expertise=expertise)
    out: list[dict[str, Any]] = []
    for e in result.entities:
        if e.entity_type not in TYPED_ENTITY_TYPES:
            continue
        out.append(
            {
                "entity_type": e.entity_type,
                "name": e.name,
                "attributes": dict(e.attributes),
            }
        )
    return out


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from counts. Guards against div-by-zero."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


async def _main(args: argparse.Namespace) -> int:
    fixture_path = Path(args.fixture)
    examples = _load_fixture(fixture_path)
    print(f"Loaded {len(examples)} labeled examples from {fixture_path}")

    loader = ExpertiseLoader()
    expertise = loader.load_builtin("meetings", use_cache=False)
    extractor = LLMEntityExtractor(model=args.model)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for example in examples:
        text = example["text"]
        gold_labels = example.get("labels", [])
        try:
            predicted = await _extract(extractor, text, expertise)
        except Exception as exc:
            print(f"  [{example.get('id')}] extraction failed: {exc}")
            continue

        # Track which gold labels have been matched so we don't double-count.
        gold_matched = [False] * len(gold_labels)
        for p in predicted:
            matched = False
            for i, g in enumerate(gold_labels):
                if gold_matched[i]:
                    continue
                if _match(p, g):
                    counts[p["entity_type"]]["tp"] += 1
                    gold_matched[i] = True
                    matched = True
                    break
            if not matched:
                counts[p["entity_type"]]["fp"] += 1

        for i, g in enumerate(gold_labels):
            if not gold_matched[i]:
                counts[g["entity_type"]]["fn"] += 1

    print()
    print(f"{'Type':<14} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 56)
    total_tp = total_fp = total_fn = 0
    for entity_type in sorted(TYPED_ENTITY_TYPES):
        c = counts.get(entity_type, {"tp": 0, "fp": 0, "fn": 0})
        p, r, f1 = _prf(c["tp"], c["fp"], c["fn"])
        print(f"{entity_type:<14} {c['tp']:>4} {c['fp']:>4} {c['fn']:>4} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
        total_tp += c["tp"]
        total_fp += c["fp"]
        total_fn += c["fn"]
    p, r, f1 = _prf(total_tp, total_fp, total_fn)
    print("-" * 56)
    print(f"{'OVERALL':<14} {total_tp:>4} {total_fp:>4} {total_fn:>4} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")

    if len(examples) < 50:
        print()
        print(
            f"WARNING: fixture has {len(examples)} examples — well below the 50-example "
            "threshold required to declare ACTION_ITEM extraction GA. See module "
            "docstring for the Devil's Advocate gating criteria (P>=0.7, R>=0.6)."
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default="tests/fixtures/action_item_labels.jsonl",
        help="Path to JSONL fixture of labeled examples.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LiteLLM model name for extraction (default: gpt-4o-mini).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parse_args())))
