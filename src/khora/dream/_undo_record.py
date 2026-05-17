"""Backwards-compatible re-export of :class:`UndoRecord`.

The dataclass originally lived here as a transitional shim while #667
(the orchestrator apply path that owns the canonical type) was in
flight. Both PRs have since landed; the canonical home is
:mod:`khora.dream.result`. This module remains only so older imports
keep resolving.

New code should import from :mod:`khora.dream.result` directly.
"""

from __future__ import annotations

from khora.dream.result import UndoRecord

__all__ = ["UndoRecord"]
