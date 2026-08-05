"""The head revision of the bundled Alembic chain, read from the sources.

Migration tests that run ``upgrade head`` and then assert what landed in
``khora_alembic_version`` used to spell the revision id out. Every new
migration therefore had to bump the same literal in six files, and those bumps
collided with any other migration PR in flight.

Every one of those assertions is decoration today. A ``command.upgrade(cfg,
"head")`` a few lines above wrote the value moments earlier and any real
failure raises out of that call first, so the comparison is kept because it
costs nothing, not because it is load-bearing. The hook-subscriptions case
reads like the exception — its version row is written by the fixture's
``run_migrations()``, which returns ``success=True, skipped=True`` without
running anything when the database is ahead, and the fixture checks only
``success`` — but that skip is unreachable as the fixture stands: it drops and
recreates ``public`` first, so the version table is empty, ahead-detection
cannot fire, and every other failure sets ``success=False``. The assertion
*would* become load-bearing if that reset ever stopped clearing the version
table.

This is only for *head* references. An explicit upgrade or downgrade target — a
migration's own revision, or its predecessor — stays a literal, because those
tests are about that specific revision and must not drift forward when the next
migration lands.
"""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

import khora.db.migrations

__all__ = ["current_head"]

#: Derive the migrations directory from the installed package — not relative to
#: this file — so the head is read from the same chain ``run_migrations()``
#: walks. The two are identical under an editable install and diverge under a
#: wheel, which is exactly the case ``test_versions_dir_is_fully_bundled``
#: exists to guard.
MIGRATIONS_DIR = Path(khora.db.migrations.__file__).parent


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
    if head is None:
        raise AssertionError(f"no head revision found in {MIGRATIONS_DIR / 'versions'}")
    return head
