"""Runtime safety guards for dream-phase apply mode (#667).

Architectural promises that #649 makes about the apply path live here as
runtime assertions:

- **No ``chunk_id`` mutation.** The chronicle ``chunks.id`` value is the
  back-pointer that ties temporal recall to the document that produced a
  chronicle event. Mutating it would silently break the temporal recall
  channel without surfacing a foreign-key error, since the dream-phase
  operates on a denormalized snapshot. Every :class:`UndoRecord`
  produced by an apply handler runs through
  :func:`_assert_no_chunk_id_mutation` before being committed.

Stability: **internal**.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from khora.dream.exceptions import DreamForbiddenOpError

if TYPE_CHECKING:
    from khora.dream.result import UndoRecord


def _assert_no_chunk_id_mutation(undo_record: UndoRecord) -> None:
    """Reject any apply handler that captured a ``chunk_id`` in its undo.

    Called by the orchestrator after every per-op apply handler returns.
    A top-level ``"chunk_id"`` key in ``undo_record.before`` is treated
    as evidence the handler touched a chronicle chunks row — that path
    is closed in the dream phase (#649 architectural promise).

    Args:
        undo_record: The handler's returned snapshot.

    Raises:
        DreamForbiddenOpError: ``before`` carries a top-level
            ``"chunk_id"`` key.
    """
    if "chunk_id" in undo_record.before:
        raise DreamForbiddenOpError(
            f"Apply handler for op_type={undo_record.op_type!r} "
            f"(op_id={undo_record.op_id}) captured a chunk_id in its undo "
            "snapshot. Mutating chronicle chunk_id is forbidden — the "
            "temporal-recall back-pointer must remain stable across "
            "dream runs."
        )


__all__ = ["_assert_no_chunk_id_mutation"]
