"""Config coverage for the Weaviate / Turbopuffer temporal-store backends.

These two backends gained operator-facing Pydantic config models
(``WeaviateConfig`` / ``TurbopufferConfig``) that hang off
``storage.weaviate`` / ``storage.turbopuffer``. This module locks in:

* env-var loading for every field, in BOTH the canonical single-underscore
  form (``KHORA_STORAGE_WEAVIATE_URL``) and the legacy double-underscore form
  (``KHORA_STORAGE__WEAVIATE__URL``),
* the conflict-detection validator fires for the new aliases when the two
  spellings disagree,
* defaults on a bare model, and
* ``extra="forbid"`` rejects unknown fields.

Secret-field typing + no-plaintext-leak for these same models lives in
``tests/unit/test_secret_typing.py`` (the secret-typing contract surface); this
module is the env-plumbing / shape surface.

Mirrors the structure of ``test_env_var_aliases.py``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from khora.config.schema import KhoraConfig, TurbopufferConfig, WeaviateConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# storage.weaviate — env loading (both alias forms)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var",
    [
        "KHORA_STORAGE_WEAVIATE_URL",
        "KHORA_STORAGE__WEAVIATE__URL",
    ],
)
def test_weaviate_url_both_forms(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    """Both alias spellings populate ``storage.weaviate.url`` (a SecretStr)."""
    monkeypatch.setenv(env_var, "http://localhost:8090")

    config = KhoraConfig()
    assert config.storage.weaviate is not None
    assert config.storage.weaviate.url.get_secret_value() == "http://localhost:8090"


@pytest.mark.parametrize(
    "env_var",
    [
        "KHORA_STORAGE_WEAVIATE_CLUSTER_URL",
        "KHORA_STORAGE__WEAVIATE__CLUSTER_URL",
    ],
)
def test_weaviate_cluster_url_both_forms(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    """Both alias spellings populate ``storage.weaviate.cluster_url``."""
    monkeypatch.setenv(env_var, "https://my-cluster.weaviate.network")

    config = KhoraConfig()
    assert config.storage.weaviate is not None
    assert config.storage.weaviate.cluster_url.get_secret_value() == "https://my-cluster.weaviate.network"


@pytest.mark.parametrize(
    "env_var",
    [
        "KHORA_STORAGE_WEAVIATE_API_KEY",
        "KHORA_STORAGE__WEAVIATE__API_KEY",
    ],
)
def test_weaviate_api_key_both_forms(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    """Both alias spellings populate ``storage.weaviate.api_key``."""
    monkeypatch.setenv("KHORA_STORAGE_WEAVIATE_CLUSTER_URL", "https://c.weaviate.network")
    monkeypatch.setenv(env_var, "wv-secret")

    config = KhoraConfig()
    assert config.storage.weaviate is not None
    assert config.storage.weaviate.api_key.get_secret_value() == "wv-secret"


@pytest.mark.parametrize(
    ("env_var", "field_name", "raw", "expected"),
    [
        ("KHORA_STORAGE_WEAVIATE_GRPC_PORT", "grpc_port", "50061", 50061),
        ("KHORA_STORAGE__WEAVIATE__GRPC_PORT", "grpc_port", "50061", 50061),
        ("KHORA_STORAGE_WEAVIATE_HTTP_SECURE", "http_secure", "true", True),
        ("KHORA_STORAGE__WEAVIATE__HTTP_SECURE", "http_secure", "true", True),
        ("KHORA_STORAGE_WEAVIATE_GRPC_SECURE", "grpc_secure", "true", True),
        ("KHORA_STORAGE_WEAVIATE_SKIP_INIT_CHECKS", "skip_init_checks", "true", True),
        ("KHORA_STORAGE__WEAVIATE__SKIP_INIT_CHECKS", "skip_init_checks", "true", True),
    ],
)
def test_weaviate_scalar_fields_both_forms(
    monkeypatch: pytest.MonkeyPatch, env_var: str, field_name: str, raw: str, expected: object
) -> None:
    """Scalar weaviate fields load from both alias spellings."""
    monkeypatch.setenv("KHORA_STORAGE_WEAVIATE_URL", "http://localhost:8090")
    monkeypatch.setenv(env_var, raw)

    config = KhoraConfig()
    assert config.storage.weaviate is not None
    assert getattr(config.storage.weaviate, field_name) == expected


def test_weaviate_alias_conflict_raises_when_values_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting both spellings of a weaviate field to different values must raise."""
    monkeypatch.setenv("KHORA_STORAGE_WEAVIATE_URL", "http://new:8090")
    monkeypatch.setenv("KHORA_STORAGE__WEAVIATE__URL", "http://old:8090")

    with pytest.raises(ValueError) as excinfo:
        KhoraConfig()
    message = str(excinfo.value)
    assert "KHORA_STORAGE_WEAVIATE_URL" in message
    assert "KHORA_STORAGE__WEAVIATE__URL" in message


# ---------------------------------------------------------------------------
# storage.turbopuffer — env loading (both alias forms)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var",
    [
        "KHORA_STORAGE_TURBOPUFFER_API_KEY",
        "KHORA_STORAGE__TURBOPUFFER__API_KEY",
    ],
)
def test_turbopuffer_api_key_both_forms(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    """Both alias spellings populate ``storage.turbopuffer.api_key`` (a SecretStr)."""
    monkeypatch.setenv(env_var, "tpuf-secret")

    config = KhoraConfig()
    assert config.storage.turbopuffer is not None
    assert config.storage.turbopuffer.api_key.get_secret_value() == "tpuf-secret"


@pytest.mark.parametrize(
    ("env_var", "field_name", "raw", "expected"),
    [
        ("KHORA_STORAGE_TURBOPUFFER_REGION", "region", "gcp-europe-west3", "gcp-europe-west3"),
        ("KHORA_STORAGE__TURBOPUFFER__REGION", "region", "gcp-europe-west3", "gcp-europe-west3"),
        ("KHORA_STORAGE_TURBOPUFFER_NAMESPACE_PREFIX", "namespace_prefix", "acme_", "acme_"),
        ("KHORA_STORAGE__TURBOPUFFER__NAMESPACE_PREFIX", "namespace_prefix", "acme_", "acme_"),
        ("KHORA_STORAGE_TURBOPUFFER_ANN_DISTANCE_THRESHOLD", "ann_distance_threshold", "0.35", 0.35),
        ("KHORA_STORAGE__TURBOPUFFER__ANN_DISTANCE_THRESHOLD", "ann_distance_threshold", "0.35", 0.35),
    ],
)
def test_turbopuffer_scalar_fields_both_forms(
    monkeypatch: pytest.MonkeyPatch, env_var: str, field_name: str, raw: str, expected: object
) -> None:
    """Scalar turbopuffer fields load from both alias spellings."""
    monkeypatch.setenv("KHORA_STORAGE_TURBOPUFFER_API_KEY", "tpuf-secret")
    monkeypatch.setenv(env_var, raw)

    config = KhoraConfig()
    assert config.storage.turbopuffer is not None
    assert getattr(config.storage.turbopuffer, field_name) == expected


def test_turbopuffer_base_url_both_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    """``base_url`` is a SecretStr; both alias spellings populate it."""
    monkeypatch.setenv("KHORA_STORAGE_TURBOPUFFER_API_KEY", "tpuf-secret")
    monkeypatch.setenv("KHORA_STORAGE__TURBOPUFFER__BASE_URL", "https://proxy.example.com")

    config = KhoraConfig()
    assert config.storage.turbopuffer is not None
    assert config.storage.turbopuffer.base_url.get_secret_value() == "https://proxy.example.com"


def test_turbopuffer_alias_conflict_raises_when_values_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting both spellings of a turbopuffer field to different values must raise."""
    monkeypatch.setenv("KHORA_STORAGE_TURBOPUFFER_REGION", "gcp-us-central1")
    monkeypatch.setenv("KHORA_STORAGE__TURBOPUFFER__REGION", "gcp-europe-west3")

    with pytest.raises(ValueError) as excinfo:
        KhoraConfig()
    message = str(excinfo.value)
    assert "KHORA_STORAGE_TURBOPUFFER_REGION" in message
    assert "KHORA_STORAGE__TURBOPUFFER__REGION" in message


# ---------------------------------------------------------------------------
# Defaults on a bare model
# ---------------------------------------------------------------------------


def test_weaviate_defaults() -> None:
    """A bare ``WeaviateConfig()`` carries the documented defaults."""
    cfg = WeaviateConfig()
    assert cfg.url is None
    assert cfg.cluster_url is None
    assert cfg.api_key is None
    assert cfg.grpc_port == 50051
    assert cfg.http_secure is False
    assert cfg.grpc_secure is False
    assert cfg.skip_init_checks is False


def test_turbopuffer_defaults() -> None:
    """A bare ``TurbopufferConfig()`` does not raise and carries documented defaults.

    ``api_key`` defaults to ``None`` so the model is constructible without a
    secret; the factory enforces presence at use time (covered in the dispatch
    tests).
    """
    cfg = TurbopufferConfig()
    assert cfg.api_key is None
    assert cfg.region == "gcp-us-central1"
    assert cfg.base_url is None
    assert cfg.namespace_prefix == "khora_"
    assert cfg.ann_distance_threshold is None


# ---------------------------------------------------------------------------
# extra="forbid" — unknown fields rejected
# ---------------------------------------------------------------------------


def test_weaviate_rejects_unknown_field() -> None:
    """``extra="forbid"`` surfaces a typo'd field instead of silently dropping it."""
    with pytest.raises(ValidationError):
        WeaviateConfig(bogus_field="x")


def test_turbopuffer_rejects_unknown_field() -> None:
    """``extra="forbid"`` surfaces a typo'd field instead of silently dropping it."""
    with pytest.raises(ValidationError):
        TurbopufferConfig(bogus_field="x")
