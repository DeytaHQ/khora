"""Self-tests for the two document-enumeration test helpers.

Both helpers fail the same way when they are wrong — silently, by passing — and
both are about to be depended on from four backend modules plus the property-based walk
fuzzer, so each assertion they make gets a NEGATIVE case here proving it can
actually fail. Precedent: ``tests/unit/test_diagnostics_helper.py``.

Nothing here touches a store. These are pure-function tests over hand-built pages
and hand-built parameter logs, which is the point: the per-backend modules prove
the helpers say the right thing about a real scan, and this module proves they say
*anything at all*.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from khora.core.models import Document
from khora.core.models.document import DocumentCursor, DocumentPage
from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from tests.test_helpers.document_page_oracle import (
    assert_documents_satisfy,
    assert_page_compliant,
    assert_total_order,
    assert_walk_compliant,
    assert_walk_matches_expected,
)
from tests.test_helpers.document_scan_spy import (
    PROBE_VALUE,
    assert_split_honest,
    flatten_params,
)

_BASE = datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)


def _ast(wire: dict):
    return parse_to_ast(RecallFilter.model_validate(wire))


def _doc(seconds: int, hex_tail: str = "0001", **fields) -> Document:
    """A document at ``_BASE + seconds`` with a pinned id, so order is decidable."""
    return Document(
        id=UUID(f"00000000-0000-4000-8000-00000000{hex_tail}"),
        content="x",
        created_at=_BASE + timedelta(seconds=seconds),
        **fields,
    )


def _page(docs: list[Document], *, next_after=None, exhausted=True) -> DocumentPage:
    return DocumentPage(docs, next_after=next_after, exhausted=exhausted)


class TestTotalOrder:
    def test_descending_passes(self) -> None:
        assert_total_order([_doc(2, "0001"), _doc(1, "0002"), _doc(0, "0003")])

    def test_ascending_fails(self) -> None:
        with pytest.raises(AssertionError, match="INV-E violated"):
            assert_total_order([_doc(0), _doc(1, "0002")])

    def test_a_repeat_fails_as_an_ordering_violation(self) -> None:
        """STRICT descent is why the same comparison catches an exactly-once break."""
        doc = _doc(1)
        with pytest.raises(AssertionError, match="INV-E violated"):
            assert_total_order([doc, doc])

    def test_a_created_at_tie_is_decided_by_id(self) -> None:
        """Equal timestamps must still descend — on ``id``, the keyset's second leg."""
        assert_total_order([_doc(1, "0002"), _doc(1, "0001")])
        with pytest.raises(AssertionError, match="INV-E violated"):
            assert_total_order([_doc(1, "0001"), _doc(1, "0002")])

    def test_naive_and_aware_timestamps_compare_without_raising(self) -> None:
        """The reason ``as_utc`` is in the key: the embedded tiers read back naive.

        Mixing the two shapes in one comparison raises ``TypeError`` rather than
        failing an assertion, which would surface as an error in an unrelated test
        rather than as an ordering failure here.
        """
        newer = _doc(1, "0001")
        older = _doc(0, "0002")
        older.created_at = older.created_at.replace(tzinfo=None)
        assert_total_order([newer, older])

    def test_a_string_id_compares_as_a_uuid_not_as_a_string(self) -> None:
        """The id half of the same normalization, and it changes the ANSWER.

        ``str(uuid)`` puts a dash at index 8, and ``-`` (0x2D) sorts below every hex
        digit — so two ids that differ only after the dash order differently as
        strings than as UUIDs. Comparing as UUIDs is what the stores do, so it is
        the correct semantics, not a convenience; a mixed ``str``/``UUID`` pair would
        also raise ``TypeError`` instead of failing an assertion.
        """
        newer, older = _doc(1, "0002"), _doc(1, "0001")
        # Off-type on purpose: ``Document.id`` is declared ``UUID``, and planting the
        # shape a store might actually hand back IS the subject. Same below. ``ty``
        # runs on ``src/`` only (see the ``lint`` target), so these do not gate CI;
        # were it ever pointed at ``tests/``, these three assignments are the
        # intended warnings, not oversights.
        older.id = str(older.id)
        assert_total_order([newer, older])

    def test_a_record_id_shaped_object_is_unwrapped(self) -> None:
        """SurrealDB hands back a ``RecordID`` carrying the uuid on ``.id``."""

        class _RecordID:
            def __init__(self, value: UUID) -> None:
                self.id = value

        newer, older = _doc(1, "0002"), _doc(1, "0001")
        newer.id = _RecordID(newer.id)
        assert_total_order([newer, older])

    def test_an_unnormalizable_id_fails_loudly_rather_than_being_coerced(self) -> None:
        """An id shape the helper cannot map must raise, never be coerced — a silent
        ``str(value)`` fallback would order rows by a rendering nothing stores."""
        doc = _doc(1)
        doc.id = 12345
        with pytest.raises(AssertionError, match="cannot be normalized"):
            assert_total_order([doc, _doc(0, "0002")])


class TestCompleteness:
    """The leg that catches a silently dropped row — the one no predicate can see."""

    def test_exact_match_passes(self) -> None:
        docs = [_doc(1, "0001"), _doc(0, "0002")]
        assert_walk_matches_expected(docs, [d.id for d in docs])

    def test_a_missing_document_fails_and_is_named(self) -> None:
        """The dangerous failure: the returned rows are all correct, one is absent."""
        returned = [_doc(1, "0001")]
        dropped = _doc(0, "0002")
        with pytest.raises(AssertionError, match="MISSING"):
            assert_walk_matches_expected(returned, [returned[0].id, dropped.id])

    def test_an_extra_document_fails_and_is_named(self) -> None:
        kept = _doc(1, "0001")
        extra = _doc(0, "0002")
        with pytest.raises(AssertionError, match="UNEXPECTED"):
            assert_walk_matches_expected([kept, extra], [kept.id])

    def test_expected_may_be_ids_or_documents_in_either_id_shape(self) -> None:
        docs = [_doc(1, "0001"), _doc(0, "0002")]
        assert_walk_matches_expected(docs, docs)  # documents
        assert_walk_matches_expected(docs, [str(d.id) for d in docs])  # dashed strings

    def test_soundness_passing_does_not_imply_completeness(self) -> None:
        """The two legs are independent, demonstrated rather than asserted in prose.

        The returned page satisfies the filter on every row, is ordered, and is
        missing a match. Soundness passes; completeness fails. This is exactly the
        shape a keyset bug produces, and the reason
        ``assert_documents_satisfy`` alone is not enough.
        """
        kept = _doc(1, "0001", source_type="report")
        dropped = _doc(0, "0002", source_type="report")
        ast = _ast({"source_type": "report"})

        assert_documents_satisfy([kept], ast)  # sound
        assert_total_order([kept])  # ordered
        with pytest.raises(AssertionError, match="MISSING"):
            assert_walk_matches_expected([kept], [kept.id, dropped.id])


class TestDocumentsSatisfy:
    def test_matching_rows_pass(self) -> None:
        assert_documents_satisfy([_doc(0, source_type="report")], _ast({"source_type": "report"}))

    def test_a_row_the_filter_excludes_fails(self) -> None:
        with pytest.raises(AssertionError, match="does not satisfy the full filter"):
            assert_documents_satisfy([_doc(0, source_type="library")], _ast({"source_type": "report"}))

    def test_a_metadata_path_is_evaluated_on_the_document_blob(self) -> None:
        """Metadata resolves through ``record.metadata`` regardless of field_mapping,
        which is why the identity context here agrees with every backend's own."""
        assert_documents_satisfy([_doc(0, metadata={"tier": "gold"})], _ast({"metadata.tier": "gold"}))
        with pytest.raises(AssertionError, match="does not satisfy the full filter"):
            assert_documents_satisfy([_doc(0, metadata={"tier": "silver"})], _ast({"metadata.tier": "gold"}))


class TestPageCompliance:
    def test_exhausted_page_without_a_cursor_passes(self) -> None:
        assert_page_compliant(_page([_doc(1, "0001"), _doc(0, "0002")]))

    def test_unexhausted_page_with_a_cursor_passes(self) -> None:
        cursor = DocumentCursor(created_at=_BASE, id=UUID(int=1))
        assert_page_compliant(_page([_doc(1)], next_after=cursor, exhausted=False))

    @pytest.mark.parametrize(
        ("next_after", "exhausted"),
        [
            (None, False),  # neither a resume position nor exhaustion — unactionable
            (DocumentCursor(created_at=_BASE, id=UUID(int=1)), True),  # both — a walk that never ends
        ],
    )
    def test_a_next_after_exhausted_mismatch_fails(self, next_after, exhausted) -> None:
        with pytest.raises(AssertionError, match="next_after/exhausted disagree"):
            assert_page_compliant(_page([_doc(1)], next_after=next_after, exhausted=exhausted))

    def test_a_filter_violating_row_fails(self) -> None:
        page = _page([_doc(0, source_type="library")])
        with pytest.raises(AssertionError, match="does not satisfy the full filter"):
            assert_page_compliant(page, _ast({"source_type": "report"}))

    def test_no_filter_skips_the_predicate_entirely(self) -> None:
        """``filter_ast=None`` must not silently compile a match-everything AST and
        claim the filter was checked — an unfiltered walk has no filter to check."""
        assert_page_compliant(_page([_doc(0, source_type="library")]))


class TestWalkCompliance:
    def test_a_clean_two_page_walk_returns_the_flat_rows_in_order(self) -> None:
        cursor = DocumentCursor(created_at=_BASE, id=UUID(int=1))
        first = _page([_doc(3, "0001"), _doc(2, "0002")], next_after=cursor, exhausted=False)
        second = _page([_doc(1, "0003"), _doc(0, "0004")])

        flat = assert_walk_compliant([first, second])

        assert [d.created_at for d in flat] == [_BASE + timedelta(seconds=s) for s in (3, 2, 1, 0)]

    def test_a_document_repeated_across_pages_fails(self) -> None:
        cursor = DocumentCursor(created_at=_BASE, id=UUID(int=1))
        repeated = _doc(2, "0002")
        first = _page([_doc(3, "0001"), repeated], next_after=cursor, exhausted=False)
        second = _page([repeated])

        with pytest.raises(AssertionError, match="exactly-once violated"):
            assert_walk_compliant([first, second])

    def test_an_order_that_resets_at_a_page_boundary_fails(self) -> None:
        """Each page descends on its own, and the walk still does not.

        This is the failure a per-page assertion cannot see, and the reason
        ``assert_walk_compliant`` re-checks the concatenation.
        """
        cursor = DocumentCursor(created_at=_BASE, id=UUID(int=1))
        first = _page([_doc(1, "0001"), _doc(0, "0002")], next_after=cursor, exhausted=False)
        second = _page([_doc(3, "0003"), _doc(2, "0004")])

        assert_page_compliant(first)
        assert_page_compliant(second)
        with pytest.raises(AssertionError, match="INV-E violated"):
            assert_walk_compliant([first, second])

    def test_expected_ids_threads_through_to_the_completeness_check(self) -> None:
        page = _page([_doc(1, "0001")])
        assert_walk_compliant([page], expected_ids=[page[0].id])
        with pytest.raises(AssertionError, match="MISSING"):
            assert_walk_compliant([page], expected_ids=[page[0].id, _doc(0, "0002").id])


class TestSplitHonestySpy:
    def test_flatten_walks_every_container_shape(self) -> None:
        """Positional tuples, bind dicts and executemany lists all reduce to values."""
        log = [
            ("SELECT 1", ("a", 2)),
            ("SELECT 2", {"f_0": "b", "lim": 10}),
            ("SELECT 3", [("c",), ("d",)]),
            ("SELECT 4", None),
        ]
        assert flatten_params(log) == {"a", "2", "b", "10", "c", "d"}

    def test_a_pushed_value_present_passes_and_absent_fails(self) -> None:
        present = [("SELECT * FROM documents WHERE ...", (PROBE_VALUE, 10))]
        assert_split_honest(present, pushed_values=[PROBE_VALUE])

        absent = [("SELECT * FROM documents WHERE ...", ("library", 10))]
        with pytest.raises(AssertionError, match="reported as pushed down"):
            assert_split_honest(absent, pushed_values=[PROBE_VALUE])

    def test_a_residual_value_absent_passes_and_present_fails(self) -> None:
        absent = [("SELECT * FROM documents WHERE ...", ("library", 10))]
        assert_split_honest(absent, residual_values=[PROBE_VALUE])

        present = [("SELECT * FROM documents WHERE ...", {"f_0": PROBE_VALUE})]
        with pytest.raises(AssertionError, match="reported as post-filtered"):
            assert_split_honest(present, residual_values=[PROBE_VALUE])

    def test_a_literal_inlined_into_the_sql_text_is_still_caught(self) -> None:
        """The tripwire for a compiler that stopped binding and started inlining.

        No compiler does this today (``_lance_fragment_to_text`` rewrites every
        ``?`` to a named bind, and the other three emit binds natively), which is
        exactly why it needs a test: were it to change, "the value is absent from
        the params" would become a false pass while SQL enforced the leaf anyway.
        """
        # A fixture string, never handed to a driver; inlining the literal IS the
        # condition under test, which is why S608 is suppressed rather than avoided.
        inlined = [(f"SELECT * FROM documents WHERE source_type = '{PROBE_VALUE}'", None)]  # noqa: S608
        assert_split_honest(inlined, pushed_values=[PROBE_VALUE])
        with pytest.raises(AssertionError, match="reported as post-filtered"):
            assert_split_honest(inlined, residual_values=[PROBE_VALUE])

    def test_an_empty_log_fails_instead_of_passing_vacuously(self) -> None:
        """The first failure the spy has to catch about ITSELF.

        A spy attached to the wrong seam captures nothing, and every
        ``residual_values`` assertion over an empty log is then trivially true — a
        green run indistinguishable from a real one.
        """
        with pytest.raises(AssertionError, match="captured no statements"):
            assert_split_honest([], residual_values=[PROBE_VALUE])

    def test_a_log_that_never_touched_the_table_fails(self) -> None:
        """The second, and the subtler one: a non-empty log of the WRONG statements.

        An instrumented seam that ran only a namespace lookup, a ``PRAGMA`` or a
        ``BEGIN`` clears the non-empty check while still having observed no scan, so
        the residual half would again pass for free.
        """
        wrong = [("PRAGMA foreign_keys=ON", None), ("SELECT id FROM memory_namespaces WHERE ...", ("ns",))]
        with pytest.raises(AssertionError, match="mentions 'document'"):
            assert_split_honest(wrong, residual_values=[PROBE_VALUE])

    def test_the_table_guard_accepts_both_table_names(self) -> None:
        """One default covers all four tiers: ``documents`` on three, ``document`` on
        SurrealDB — the latter being a substring of the former."""
        assert_split_honest([("SELECT * FROM documents WHERE ...", None)], residual_values=[PROBE_VALUE])
        assert_split_honest([("SELECT * FROM document WHERE ...", None)], residual_values=[PROBE_VALUE])
