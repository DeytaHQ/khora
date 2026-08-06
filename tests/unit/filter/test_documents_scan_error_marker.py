"""The SurrealDB document scan's ``CompileError`` discriminator vs. the real guard.

``@internal``. ``SurrealDBRelationalAdapter.scan_documents`` maps the compiler's
internal :class:`~khora.filter.context.CompileError` onto the public
:class:`~khora.filter.model.RecallFilterUnsupportedError` for exactly one cause —
``compile_surrealdb``'s injection guard rejecting a metadata path segment it
cannot render as a SurrealQL identifier. The guard has no error subclass of its
own, so the scan discriminates on a substring of the guard's message:
``_UNSAFE_METADATA_SEGMENT_MARKER``.

That constant lives in a storage module while the message it matches lives in a
compiler module the store does not own. Reword the guard and the mapping stops
firing: a hyphenated metadata key — legal JSON, common in the wild — starts
escaping as an internal fault instead of a structured rejection. That is the
*safe* direction to fail, but it should be caught here rather than discovered in
production.

**Why this is its own module rather than an assertion inside the scan's
integration tests.** The end-to-end mapping test
(``tests/integration/storage/backends/surrealdb/test_relational_scan_documents.py::
test_unsafe_metadata_segment_raises_the_public_error``) does drive the real
compiler and does go red on a reworded guard — verified by patching the marker to
a string the guard no longer emits. But that protection is **incidental**: that
test was written to pin the mapping, not the constant, and narrowing it to a stub
compiler would remove the protection with nothing to announce the loss. It also
fails for a confusing reason — "expected ``RecallFilterUnsupportedError``, got
``CompileError``" reads like a bug in the scan path rather than a reworded
message two modules away.

**Unit lane, and deliberately not behind ``pytest.importorskip("surrealdb")``.**
The whole path exercised here is pure compilation: no server, no adapter, no
``scan_documents``. Both imports resolve without the ``surrealdb`` extra — the
adapter module's only SurrealDB dependency is ``SurrealDBConnection``, which has
no module-level ``surrealdb`` import. Verified by blocking the package at the
import hook and re-running this file's imports and assertions. Putting it behind
an ``importorskip`` would skip it exactly in the degraded environment where a
silent mapping failure is least likely to be noticed.
"""

from __future__ import annotations

import pytest

from khora.filter import CompileError, RecallFilter, parse_to_ast
from khora.filter.compilers.surrealdb import compile_surrealdb
from khora.storage.backends.surrealdb.relational import (
    _UNSAFE_METADATA_SEGMENT_MARKER,
    _documents_compile_context,
)

pytestmark = pytest.mark.unit


def test_marker_still_matches_the_real_guard_message() -> None:
    """The discriminator matches what ``compile_surrealdb`` actually raises.

    Drives the real compiler with the real ``CompileContext`` the scan uses, so
    the message compared against is the one production would produce. Asserts the
    coupling directly instead of inferring it from an end-to-end mapping result.
    """
    ast = parse_to_ast(RecallFilter.model_validate({"metadata.foo-bar": {"$eq": "x"}}))

    with pytest.raises(CompileError) as excinfo:
        compile_surrealdb(ast, _documents_compile_context())

    assert _UNSAFE_METADATA_SEGMENT_MARKER in str(excinfo.value)


def test_the_marker_excludes_the_interpolated_segment() -> None:
    """Granularity: the marker is the invariant prefix, not the whole message.

    The guard's message interpolates the offending segment
    (``unsafe metadata path segment 'foo-bar' (not a SurrealQL identifier)``), so
    a marker that reached past the invariant prefix into the quoted segment would
    match only the one key it was written against and silently stop mapping every
    other hyphenated key. Pinned as its own assertion because widening the marker
    is a natural-looking "make the check stricter" edit whose damage is invisible:
    the mapping keeps working for the key in the tests and fails for every other.
    """
    ast = parse_to_ast(RecallFilter.model_validate({"metadata.due-date": {"$eq": "x"}}))

    with pytest.raises(CompileError) as excinfo:
        compile_surrealdb(ast, _documents_compile_context())

    # A different segment, and the same marker still matches.
    assert _UNSAFE_METADATA_SEGMENT_MARKER in str(excinfo.value)
    assert "due-date" in str(excinfo.value), "the guard should still report which segment it rejected"
    assert "due-date" not in _UNSAFE_METADATA_SEGMENT_MARKER
    assert "foo-bar" not in _UNSAFE_METADATA_SEGMENT_MARKER
