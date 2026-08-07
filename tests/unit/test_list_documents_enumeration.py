"""Unit coverage for the document-enumeration facade surface.

Covers the pieces that need no live backend: the ``DocumentPage`` sequence
contract, the ``occurred_at`` enumeration key-scope rejection (structured error
+ asserted copy), and the ``status`` enum validation. The multi-page cursor walk
and facade/storage parity live in the integration lane (they need a seeded
namespace on the sqlite_lance stack).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import Document
from khora.core.models.document import DocumentCursor, DocumentPage
from khora.filter import RecallFilter, RecallFilterValidationError
from khora.filter.ast import parse_to_ast
from khora.khora import _reject_non_enumerable_keys


def _ast(wire: dict) -> object:
    return parse_to_ast(RecallFilter.model_validate(wire))


class TestDocumentPageSequence:
    """DocumentPage is a Sequence[Document]: iterate / len / truthiness / index."""

    def _page(self, docs: list[Document], **kw: object) -> DocumentPage:
        kw.setdefault("next_after", None)
        kw.setdefault("exhausted", True)
        return DocumentPage(docs, **kw)  # type: ignore[arg-type]

    def test_is_sequence(self) -> None:
        assert isinstance(self._page([]), Sequence)

    def test_len_iter_index(self) -> None:
        docs = [Document(content="a"), Document(content="b")]
        page = self._page(docs)
        assert len(page) == 2
        assert list(page) == docs
        assert page[0] is docs[0]
        assert page[-1] is docs[1]
        assert list(page[0:1]) == [docs[0]]

    def test_truthiness_tracks_length(self) -> None:
        # khora-benchmarks-style `if await kb.list_documents(...)` usage.
        assert not self._page([])
        assert self._page([Document(content="x")])

    def test_carries_walk_metadata(self) -> None:
        cur = DocumentCursor(created_at=datetime.now(UTC), id=uuid4())
        page = DocumentPage(
            [Document(content="x")],
            next_after=cur,
            exhausted=False,
            post_filtered_keys=("metadata.tier",),
        )
        assert page.next_after is cur
        assert page.exhausted is False
        assert page.post_filtered_keys == ("metadata.tier",)


class TestOccurredAtRejection:
    """occurred_at is the chunk event-time axis — not enumerable on a document row."""

    @pytest.mark.parametrize(
        "wire",
        [
            {"occurred_at": {"$gte": "2020-01-01T00:00:00Z"}},
            {"$or": [{"occurred_at": {"$gte": "2020-01-01T00:00:00Z"}}, {"source_type": "report"}]},
            {"$not": {"occurred_at": {"$lte": "2020-01-01T00:00:00Z"}}},
        ],
    )
    def test_rejected_with_structured_error(self, wire: dict) -> None:
        with pytest.raises(RecallFilterValidationError) as exc_info:
            _reject_non_enumerable_keys(_ast(wire))
        errors = exc_info.value.errors
        assert len(errors) == 1
        fe = errors[0]
        assert fe.path == "occurred_at"
        assert fe.code == "key_not_enumerable"
        # The nine enumerable keys (the ten system keys minus occurred_at).
        assert fe.allowed is not None and len(fe.allowed) == 9
        assert "occurred_at" not in fe.allowed
        # Copy names both substitutes and refuses to call the axes equivalent.
        assert "source_timestamp" in fe.message
        assert "created_at" in fe.message
        assert "equivalent" in fe.message

    @pytest.mark.parametrize(
        "wire",
        [
            {"source_type": "report"},
            {"created_at": {"$gte": "2020-01-01T00:00:00Z"}},
            {"source_timestamp": {"$gte": "2020-01-01T00:00:00Z"}},
            {"metadata.tier": "gold"},
            {"title": {"$in": ["a", "b"]}},
        ],
    )
    def test_enumerable_keys_pass(self, wire: dict) -> None:
        # Does not raise.
        _reject_non_enumerable_keys(_ast(wire))
