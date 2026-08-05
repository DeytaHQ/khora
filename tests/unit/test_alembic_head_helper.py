"""Coverage for the shared migration-head helper.

``current_head()`` exists so migration tests stop spelling the chain head out
as a literal. The property that buys — adding a migration needs no test edit —
holds only while the lookup stays uncached, and nothing else in the suite would
notice if it stopped: a memoized head returns the stale-but-correct answer for
every test that ran before the new revision appeared, so the suite goes green
either way. That is what this module pins.

The chain is synthetic and the helper is pointed at it via monkeypatch, so
these tests never touch the real ``versions/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_helpers import alembic_head
from tests.test_helpers.alembic_head import current_head

pytestmark = pytest.mark.unit

_REVISION_TEMPLATE = '''\
"""synthetic revision"""

revision = "{revision}"
down_revision = {down_revision}


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
'''


def _write_revision(versions_dir: Path, revision: str, down_revision: str | None) -> None:
    down = "None" if down_revision is None else f'"{down_revision}"'
    (versions_dir / f"{revision}.py").write_text(_REVISION_TEMPLATE.format(revision=revision, down_revision=down))


@pytest.fixture
def synthetic_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``current_head()`` at a two-revision chain under ``tmp_path``."""
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    _write_revision(versions_dir, "001_first", None)
    _write_revision(versions_dir, "002_second", "001_first")
    monkeypatch.setattr(alembic_head, "MIGRATIONS_DIR", tmp_path)
    return versions_dir


def test_returns_the_chain_head(synthetic_chain: Path) -> None:
    assert current_head() == "002_second"


def test_tracks_a_revision_added_mid_session(synthetic_chain: Path) -> None:
    """A newly added revision must be visible to the very next call.

    This is the ticket's bought property in miniature. Memoizing the lookup —
    ``@lru_cache`` on ``current_head()`` — makes this test fail and is the one
    change that would silently reintroduce the per-migration test edit.
    """
    assert current_head() == "002_second"

    _write_revision(synthetic_chain, "003_third", "002_second")

    assert current_head() == "003_third", (
        "current_head() returned a stale head after a revision was added — the lookup must not be cached"
    )


def test_raises_when_the_chain_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "versions").mkdir()
    monkeypatch.setattr(alembic_head, "MIGRATIONS_DIR", tmp_path)

    with pytest.raises(AssertionError, match="no head revision found"):
        current_head()
