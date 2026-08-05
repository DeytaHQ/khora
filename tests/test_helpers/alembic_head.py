"""The head revision of the bundled Alembic chain, read from the sources.

Migration tests that run ``upgrade head`` and then assert what landed in
``khora_alembic_version`` used to spell the revision id out. Every new
migration therefore had to bump the same literal in six files, and those bumps
collided with any other migration PR in flight.

How much the resulting assertion is worth depends on where the compared value
came from, and the split is not by dialect:

* Where the test body has no upgrade of its own and the version table was
  written by the fixture's ``run_migrations()`` — only the hook-subscriptions
  case today — the comparison carries real signal. That call returns
  ``success=True, skipped=True`` without running anything when the database is
  ahead, and the fixture checks only ``success``, so the assertion is the one
  thing standing between a skipped migration and a green test.
* Everywhere else, on both dialects, a ``command.upgrade(cfg, "head")`` a few
  lines above wrote the value moments earlier and any real failure raises out
  of that call first. The comparison is close to decoration there — kept
  because it costs nothing, not because it is load-bearing.

This is only for *head* references. An explicit upgrade or downgrade target — a
migration's own revision, or its predecessor — stays a literal, because those
tests are about that specific revision and must not drift forward when the next
migration lands.
"""

from __future__ import annotations

from alembic.config import Config
from alembic.script import ScriptDirectory

from tests.test_helpers.schema_drift import MIGRATIONS_DIR

__all__ = ["current_head"]


def current_head() -> str:
    """Return the single head revision declared by the bundled migrations.

    Call this inside a test body, never at module scope: building the script
    directory imports every version module, and at module scope a broken chain
    becomes a collection error that takes unrelated tests down with it.

    Deliberately uncached — a memoized head would survive a migration being
    added mid-session and hand back a stale answer.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    head = ScriptDirectory.from_config(cfg).get_current_head()
    assert head is not None, f"no head revision found in {MIGRATIONS_DIR / 'versions'}"
    return head
