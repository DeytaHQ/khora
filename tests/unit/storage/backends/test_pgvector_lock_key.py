"""Invariants for the pgvector entity-upsert advisory-lock key derivation.

Regression test for #738: the previous implementation folded the 128-bit
``namespace_id`` UUID down to a single 32-bit ``int4`` advisory-lock key,
which produced birthday-paradox collisions at ~65K namespaces (probed
empirically at 120K in the issue's repro).

The replacement keeps Postgres's two-int advisory-lock form but populates
both 32-bit slots from the UUID, so the lock id is effectively 64-bit and
birthday-safe at billions of namespaces.
"""

from __future__ import annotations

from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import pytest

from khora.storage.backends.pgvector import _namespace_lock_keys

_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


@pytest.mark.unit
def test_no_collisions_at_one_million_random_uuids() -> None:
    """1M random ``uuid4`` namespaces must produce 1M distinct lock pairs.

    The previous 32-bit-fold implementation would produce ~117 collisions
    at this scale (1e6^2 / (2 * 2^32) ≈ 116.4); with the 64-bit pair the
    expectation is ~3e-8 collisions.
    """
    keys = {_namespace_lock_keys(uuid4()) for _ in range(1_000_000)}
    assert len(keys) == 1_000_000


@pytest.mark.unit
def test_keys_fit_postgres_int4_range() -> None:
    """Both keys must lie in signed-int32 range so asyncpg can bind them
    to ``pg_advisory_xact_lock(int4, int4)`` without overflow."""
    for _ in range(10_000):
        k1, k2 = _namespace_lock_keys(uuid4())
        assert _INT32_MIN <= k1 <= _INT32_MAX, f"k1 out of int4 range: {k1}"
        assert _INT32_MIN <= k2 <= _INT32_MAX, f"k2 out of int4 range: {k2}"
        assert isinstance(k1, int)
        assert isinstance(k2, int)


@pytest.mark.unit
def test_stable_across_calls() -> None:
    """Same UUID → same lock pair across calls (stability invariant)."""
    u = uuid4()
    assert _namespace_lock_keys(u) == _namespace_lock_keys(u)


@pytest.mark.unit
def test_uuid5_google_adk_pattern_has_high_entropy() -> None:
    """The Google ADK adapter derives namespace ids via
    ``uuid5(NAMESPACE_DNS, f"adk:{app_name}:{user_id}")``. With a fixed
    ``app_name`` and varying ``user_id``, both 32-bit halves of the key
    pair must remain well-distributed — catches regressions that narrow
    the hash to one half of the UUID.
    """
    keys = [_namespace_lock_keys(uuid5(NAMESPACE_DNS, f"adk:myapp:user_{i}")) for i in range(10_000)]
    distinct_k1 = len({k[0] for k in keys})
    distinct_k2 = len({k[1] for k in keys})
    # SHA-1-based UUID5 should give near-perfect distribution; 99% threshold
    # leaves headroom for legitimate UUID-encoding bit fixes (version, variant).
    assert distinct_k1 > 9_900, f"k1 distribution collapsed: {distinct_k1}/10000"
    assert distinct_k2 > 9_900, f"k2 distribution collapsed: {distinct_k2}/10000"


@pytest.mark.unit
def test_distinct_uuids_likely_distinct_keys() -> None:
    """Two distinct UUIDs almost-always produce distinct lock pairs.

    Stronger than a smoke test — sweeps 100K distinct UUIDs and asserts
    >= 99.99% distinct lock pairs (i.e. fewer than ~10 collisions in 100K).
    Expectation for a true 64-bit lock id is ~0 collisions; the loose
    threshold here lets the test survive future hash tweaks.
    """
    keys = [_namespace_lock_keys(uuid4()) for _ in range(100_000)]
    distinct = len(set(keys))
    assert distinct >= 99_990, f"only {distinct}/100000 distinct lock pairs"


@pytest.mark.unit
def test_known_uuid_lock_pair_pinned() -> None:
    """Pin one known UUID -> (k1, k2) pair to detect accidental algorithm
    drift in future refactors. Re-derive the expected value if the algorithm
    is intentionally changed; if it changes silently, this fails.
    """
    u = UUID("12345678-1234-5678-1234-567812345678")
    # hi64 = 0x1234567812345678; lo64 = 0x1234567812345678
    # fold_hi = (hi64 ^ (hi64 >> 32)) & 0xFFFFFFFF = 0x12345678 ^ 0x12345678 = 0
    # fold_lo = same shape = 0
    assert _namespace_lock_keys(u) == (0, 0)

    u2 = UUID("ffffffff-ffff-ffff-0000-000000000000")
    # hi64 = 0xFFFFFFFF_FFFFFFFF; (hi >> 32) = 0xFFFFFFFF; xor = 0 (low 32) -> 0
    # lo64 = 0; fold_lo = 0
    assert _namespace_lock_keys(u2) == (0, 0)

    u3 = UUID("12345678-9abc-def0-1234-56789abcdef0")
    # hi64 = 0x123456789abcdef0; hi >> 32 = 0x12345678; xor low32 = 0x88888888
    # int4 signed = -2004318072
    # lo64 = 0x123456789abcdef0; same -> -2004318072
    assert _namespace_lock_keys(u3) == (-2004318072, -2004318072)
