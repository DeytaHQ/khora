"""``SurrealDBConnection.query`` — the unexpected-driver-result-shape branch.

``query`` normalizes three shapes: a list of statement results (flattened), a
bare dict (wrapped), and anything else (dropped, returning ``[]``). The third is
a defensive fallback for a driver that returns something the SDK contract does
not describe, and returning ``[]`` is the right call — a normalizer is the wrong
place to raise.

**But it must not be SILENT, and that is what this module pins.** The keyset
scans built on this connection derive their only termination signal from the row
COUNT: ``exhausted = len(rows) < scan_limit``. A dropped result shape therefore
does not surface as an error or an empty page a caller can notice — it reads as
"the namespace ended here", and a walk stops early having returned a prefix of
the documents it was asked for. The caller cannot distinguish that from a
genuinely exhausted namespace, so the log is the only observability the failure
has (ADR-001 degradation convention).

loguru is not wired into ``caplog`` by default, so each test bridges the two for
the duration of the call.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytest.importorskip("surrealdb")

from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402

pytestmark = pytest.mark.unit


class _StubClient:
    """A driver stub returning one canned result for any statement."""

    def __init__(self, result: Any) -> None:
        self._result = result

    async def query(self, sql: str, bindings: dict[str, Any] | None = None) -> Any:
        return self._result


def _connection(result: Any) -> SurrealDBConnection:
    """A connected-looking connection whose driver returns ``result``.

    Constructed rather than connected: ``query`` is pure normalization over
    whatever the driver hands back, so standing up a real embedded instance would
    add a second thing that can fail without making the assertion stronger.
    """
    conn = SurrealDBConnection(mode="memory", namespace="ns", database="db")
    conn._client = _StubClient(result)  # noqa: SLF001
    conn._connected = True  # noqa: SLF001
    return conn


@pytest.fixture(name="loguru_to_caplog")
def _loguru_to_caplog(caplog: pytest.LogCaptureFixture):
    """Bridge loguru's WARNING records into ``caplog`` for the test's duration."""
    from loguru import logger as loguru_logger

    handler_id = loguru_logger.add(
        lambda msg: caplog.handler.emit(
            logging.LogRecord(
                name="khora",
                level=logging.WARNING,
                pathname="",
                lineno=0,
                msg=msg.record["message"],
                args=None,
                exc_info=None,
            )
        ),
        level="WARNING",
    )
    try:
        yield caplog
    finally:
        loguru_logger.remove(handler_id)


@pytest.mark.parametrize(
    ("label", "result"),
    [
        ("string", "unexpected"),
        ("int", 7),
        ("none", None),
    ],
)
async def test_an_unexpected_result_shape_returns_no_rows_and_warns(
    label: str, result: Any, loguru_to_caplog: pytest.LogCaptureFixture
) -> None:
    """Dropping the result is fine. Dropping it quietly is not.

    Both halves are asserted together because either alone would pass against a
    broken implementation: the ``[]`` alone is what the silent version already
    did, and a warning without the ``[]`` would mean the shape leaked onward to a
    caller expecting rows.

    The message has to carry enough to act on, so it names the type it actually
    got — "unexpected shape" with no type is a log line that sends whoever reads
    it straight back to a debugger.
    """
    conn = _connection(result)

    with loguru_to_caplog.at_level("WARNING"):
        rows = await conn.query("SELECT * FROM document", {})

    assert rows == []

    joined = "\n".join(rec.message for rec in loguru_to_caplog.records)
    assert "unexpected result shape" in joined
    assert type(result).__name__ in joined, f"{label}: the warning must name the type it got"
    # The reason this is worth logging at all — a row-count-driven caller reads
    # the empty list as exhaustion — belongs in the message, not only in a
    # comment that a reader parsing the log will never see.
    assert "exhaustion" in joined


@pytest.mark.parametrize(
    ("label", "result", "expected"),
    [
        ("list_of_dicts", [{"id": 1}, {"id": 2}], [{"id": 1}, {"id": 2}]),
        ("nested_statement_results", [[{"id": 1}], [{"id": 2}]], [{"id": 1}, {"id": 2}]),
        ("bare_dict", {"id": 1}, [{"id": 1}]),
        ("empty_list", [], []),
    ],
)
async def test_a_recognized_result_shape_is_normalized_without_warning(
    label: str, result: Any, expected: list[dict[str, Any]], loguru_to_caplog: pytest.LogCaptureFixture
) -> None:
    """The control, without which the test above proves nothing.

    A connection that warned on EVERY query would satisfy the assertions above
    perfectly. This pins the other side: each shape the contract does describe is
    normalized and passes silently — including the empty list, which is the case
    most easily confused with the dropped one and is the shape a legitimately
    exhausted scan actually returns.
    """
    conn = _connection(result)

    with loguru_to_caplog.at_level("WARNING"):
        rows = await conn.query("SELECT * FROM document", {})

    assert rows == expected
    assert not [rec for rec in loguru_to_caplog.records if "unexpected result shape" in rec.message], (
        f"{label}: a recognized shape must not warn"
    )
