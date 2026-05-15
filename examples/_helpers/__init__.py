"""Shared helpers for the khora integration examples.

These helpers exist so adapter examples can run without external
services or API keys:

- ``mock_llm.install_mock_llm`` patches ``litellm.acompletion`` /
  ``litellm.aembedding`` with deterministic stubs.
- ``khora_fixtures.embedded_khora`` yields a ``Khora`` instance bound
  to an in-memory ``sqlite_lance`` backend (no Postgres, no Neo4j).

Not part of the khora public API. Intended for use by files under
``examples/integrations/<framework>/example.py``.
"""

from examples._helpers.khora_fixtures import embedded_khora
from examples._helpers.mock_llm import MockLLM, install_mock_llm

__all__ = ["MockLLM", "install_mock_llm", "embedded_khora"]
