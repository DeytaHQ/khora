"""Chronicle #855: reinforcement-on-recall must be off by default.

The reinforcement-on-recall feature changes the meaning of decay
(``max(source_timestamp, last_accessed_at)`` instead of just
``source_timestamp``) and adds an UPDATE per recall. Both are opt-in
behavior changes - so the public field default must stay False.
"""

from __future__ import annotations

from khora.config import KhoraConfig


def test_reinforcement_disabled_by_default() -> None:
    cfg = KhoraConfig()
    assert cfg.query.chronicle_enable_recall_reinforcement is False
