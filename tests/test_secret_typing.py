"""ADR-084 footprint smoke test — requires deyta-core>=0.4.0 (DYT-3993)."""

import pytest

deyta_core = pytest.importorskip("deyta_core", reason="deyta-core not installed")
assert_no_str_typed_secrets = deyta_core.assert_no_str_typed_secrets

from khora.config.schema import KhoraConfig  # noqa: E402


def test_no_str_typed_secrets_in_khora_config() -> None:
    assert_no_str_typed_secrets(KhoraConfig, mode="warn")
