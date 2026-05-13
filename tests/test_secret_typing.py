"""ADR-084 footprint smoke test — requires deyta-core>=0.4.0 (DYT-3993)."""

import pytest

from khora.config.schema import KhoraConfig


def test_no_str_typed_secrets_in_khora_config() -> None:
    pytest.importorskip("deyta_core", reason="deyta-core not installed")
    from deyta_core.config import assert_no_str_typed_secrets

    assert_no_str_typed_secrets(KhoraConfig, mode="fail")
