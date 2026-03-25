"""Unit tests for the ontology CLI: sources, sampling, inference, output, flow."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from khora.cli.ontology.inference.domain import DomainResult
from khora.cli.ontology.inference.prompts import (
    DOMAIN_DETECTION_SYSTEM,
    DOMAIN_DETECTION_USER,
    ENTITY_INFERENCE_SYSTEM,
    ENTITY_INFERENCE_USER,
    PROMPT_GENERATION_SYSTEM,
    PROMPT_GENERATION_USER,
    RELATIONSHIP_INFERENCE_SYSTEM,
    RELATIONSHIP_INFERENCE_USER,
    RULE_INFERENCE_SYSTEM,
    RULE_INFERENCE_USER,
)
from khora.cli.ontology.llm import (
    BudgetExhaustedError,
    LLMResponseError,
    OntologyLLM,
)
from khora.cli.ontology.output.serializer import serialize_ontology
from khora.cli.ontology.output.validator import validate_ontology
from khora.cli.ontology.sampling.sampler import DataSampler
from khora.cli.ontology.session import OntologySession
from khora.cli.ontology.sources.base import SampleChunk, SourceSummary
from khora.cli.ontology.sources.detection import detect_source
from khora.cli.ontology.sources.local import (
    LocalDirectorySource,
    LocalFileSource,
    _extract_csv_text,
    _extract_json_text,
    _extract_jsonl_text,
)
from khora.extraction.skills.base import (
    ConfidenceConfig,
    EntityTypeConfig,
    ExpertiseConfig,
    RelationshipTypeConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_YAML = textwrap.dedent("""\
    name: test_ontology
    version: "1.0.0"
    description: "Test ontology"
    entity_types:
      - name: PERSON
        description: "A person"
        attributes:
          required: [name]
          optional: [email]
        identifiers: [name, email]
      - name: ORGANIZATION
        description: "An org"
        attributes:
          required: [name]
        identifiers: [name]
    relationship_types:
      - name: WORKS_FOR
        description: "Employment"
        source_types: [PERSON]
        target_types: [ORGANIZATION]
        bidirectional: false
    correlation_rules:
      - name: email_match
        match_fields: [email]
        entity_types: [PERSON]
        confidence: 0.9
    inference_rules: []
    confidence:
      min_entity: 0.5
      min_relationship: 0.3
      min_inferred: 0.2
""")


def _make_expertise() -> ExpertiseConfig:
    """Build a minimal valid ExpertiseConfig for tests."""
    return ExpertiseConfig(
        name="test",
        version="1.0.0",
        description="Test ontology",
        system_prompt="You are a test extractor.",
        entity_types=[
            EntityTypeConfig(name="PERSON", description="A person", identifiers=["name"]),
            EntityTypeConfig(name="ORGANIZATION", description="An org", identifiers=["name"]),
        ],
        relationship_types=[
            RelationshipTypeConfig(
                name="WORKS_FOR",
                description="Employment",
                source_types=["PERSON"],
                target_types=["ORGANIZATION"],
            ),
        ],
        confidence=ConfidenceConfig(min_entity=0.5, min_relationship=0.3),
    )


# ===========================================================================
# Sources
# ===========================================================================


@pytest.mark.unit
class TestLocalFileSource:
    def test_scan_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Hello world\nLine two\nLine three")
        src = LocalFileSource(f)
        summary = src.scan()
        assert summary.source_type == "file"
        assert summary.file_count == 1
        assert summary.total_bytes > 0
        assert ".txt" in summary.extensions

    def test_sample_small_file(self, tmp_path: Path) -> None:
        f = tmp_path / "small.txt"
        f.write_text("Short content")
        src = LocalFileSource(f)
        chunks = src.sample(budget_chars=10_000)
        assert len(chunks) == 1
        assert "Short content" in chunks[0].content

    def test_sample_large_file_splits(self, tmp_path: Path) -> None:
        f = tmp_path / "large.txt"
        # Use lines so readline() in the sampler works correctly
        f.write_text(("Line of text number N\n") * 5_000)
        src = LocalFileSource(f)
        chunks = src.sample(budget_chars=3_000)
        # Should have head + middle + tail
        assert len(chunks) >= 2
        total = sum(c.char_count for c in chunks)
        assert total <= 3_000

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            LocalFileSource(tmp_path / "nope.txt")


@pytest.mark.unit
class TestLocalDirectorySource:
    def test_scan_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content a")
        (tmp_path / "b.json").write_text('{"key": "val"}')
        (tmp_path / "c.bin").write_bytes(b"\x00\x01\x02")  # unsupported

        src = LocalDirectorySource(tmp_path)
        summary = src.scan()
        assert summary.source_type == "directory"
        assert summary.file_count == 2  # .bin excluded
        assert ".txt" in summary.extensions
        assert ".json" in summary.extensions

    def test_sample_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("Text content here")
        (tmp_path / "b.csv").write_text("col1,col2\nval1,val2")

        src = LocalDirectorySource(tmp_path)
        src.scan()
        chunks = src.sample(budget_chars=5_000)
        assert len(chunks) >= 1
        assert sum(c.char_count for c in chunks) > 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        src = LocalDirectorySource(tmp_path)
        summary = src.scan()
        assert summary.file_count == 0
        chunks = src.sample(budget_chars=5_000)
        assert chunks == []


@pytest.mark.unit
class TestSourceDetection:
    def test_detect_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        src = detect_source(str(f))
        assert isinstance(src, LocalFileSource)

    def test_detect_directory(self, tmp_path: Path) -> None:
        src = detect_source(str(tmp_path))
        assert isinstance(src, LocalDirectorySource)

    def test_detect_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            detect_source("/nonexistent/path.txt")


# ===========================================================================
# Text extraction
# ===========================================================================


@pytest.mark.unit
class TestTextExtraction:
    def test_json_extraction(self) -> None:
        raw = json.dumps({"name": "Alice", "age": 30, "tags": ["a", "b"]})
        result = _extract_json_text(raw)
        assert "name: Alice" in result
        assert "age: 30" in result

    def test_jsonl_extraction(self) -> None:
        raw = '{"name": "Alice"}\n{"name": "Bob"}'
        result = _extract_jsonl_text(raw)
        assert "Alice" in result
        assert "Bob" in result

    def test_csv_extraction(self) -> None:
        raw = "name,email\nAlice,alice@example.com\nBob,bob@example.com"
        result = _extract_csv_text(raw)
        assert "Alice" in result
        assert "alice@example.com" in result


# ===========================================================================
# Sampling
# ===========================================================================


@pytest.mark.unit
class TestDataSampler:
    def test_add_source_and_sample(self, tmp_path: Path) -> None:
        (tmp_path / "test.txt").write_text("Hello world " * 100)
        src = LocalFileSource(tmp_path / "test.txt")

        sampler = DataSampler()
        summary = sampler.add_source(src)
        assert summary.source_type == "file"

        samples = sampler.sample_all(budget_chars=500)
        assert len(samples) >= 1
        assert sampler.total_chars > 0

    def test_format_for_llm(self, tmp_path: Path) -> None:
        (tmp_path / "test.txt").write_text("Sample data for LLM")
        src = LocalFileSource(tmp_path / "test.txt")

        sampler = DataSampler()
        sampler.add_source(src)
        sampler.sample_all(budget_chars=1_000)

        formatted = sampler.format_samples_for_llm(500)
        assert "---" in formatted
        assert "source:" in formatted

    def test_empty_sampler(self) -> None:
        sampler = DataSampler()
        assert sampler.sample_all() == []
        assert sampler.total_chars == 0
        assert sampler.format_samples_for_llm() == ""


# ===========================================================================
# LLM wrapper
# ===========================================================================


@pytest.mark.unit
class TestOntologyLLM:
    def test_budget_tracking(self) -> None:
        llm = OntologyLLM(model="gpt-4o", budget_usd=1.0)
        assert llm.budget_remaining == 1.0
        summary = llm.usage_summary
        assert summary["calls"] == 0
        assert summary["cost_usd"] == 0.0

    def test_parse_json_direct(self) -> None:
        llm = OntologyLLM()
        result = llm._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_markdown_block(self) -> None:
        llm = OntologyLLM()
        raw = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = llm._parse_json(raw)
        assert result == {"key": "value"}

    def test_parse_json_with_prefix(self) -> None:
        llm = OntologyLLM()
        raw = 'Sure! Here is the JSON: {"key": "value"}'
        result = llm._parse_json(raw)
        assert result == {"key": "value"}

    def test_parse_json_invalid_raises(self) -> None:
        llm = OntologyLLM()
        with pytest.raises(LLMResponseError):
            llm._parse_json("This is not JSON at all")

    def test_budget_exhausted_error(self) -> None:
        err = BudgetExhaustedError(used=1.5, budget=1.0)
        assert "1.5" in str(err)
        assert "1.0" in str(err)


# ===========================================================================
# Prompt templates
# ===========================================================================


@pytest.mark.unit
class TestPromptTemplates:
    def test_domain_detection_formats(self) -> None:
        user = DOMAIN_DETECTION_USER.format(source_count=3, total_chars=30_000, samples="test data here")
        assert "3 source(s)" in user
        assert "30000" in user
        assert "test data here" in user
        assert len(DOMAIN_DETECTION_SYSTEM) > 100

    def test_entity_inference_formats(self) -> None:
        system = ENTITY_INFERENCE_SYSTEM.format(domain="Legal", min_types=5, max_types=18)
        assert "Legal" in system
        assert "5" in system
        user = ENTITY_INFERENCE_USER.format(domain="Legal", key_concepts="contracts, parties", samples="text")
        assert "Legal" in user
        assert "contracts" in user

    def test_relationship_inference_formats(self) -> None:
        system = RELATIONSHIP_INFERENCE_SYSTEM.format(
            domain="Finance",
            entity_type_names="PERSON, COMPANY",
            min_rels=3,
            max_rels=10,
        )
        assert "Finance" in system
        assert "PERSON, COMPANY" in system
        user = RELATIONSHIP_INFERENCE_USER.format(
            domain="Finance",
            entity_type_names="PERSON, COMPANY",
            samples="data",
        )
        assert "PERSON, COMPANY" in user

    def test_rule_inference_formats(self) -> None:
        system = RULE_INFERENCE_SYSTEM.format(domain="Healthcare")
        assert "Healthcare" in system
        user = RULE_INFERENCE_USER.format(
            domain="Healthcare",
            entity_type_names="PATIENT, DOCTOR",
            relationship_type_names="TREATS, DIAGNOSED_WITH",
        )
        assert "PATIENT" in user
        assert "TREATS" in user

    def test_prompt_generation_formats(self) -> None:
        system = PROMPT_GENERATION_SYSTEM.format(domain="E-commerce")
        assert "E-commerce" in system
        user = PROMPT_GENERATION_USER.format(
            domain="E-commerce",
            entity_type_names="PRODUCT, CUSTOMER",
            relationship_type_names="PURCHASED, REVIEWED",
            key_concepts="shopping, cart, checkout",
        )
        assert "PRODUCT" in user
        assert "shopping" in user


# ===========================================================================
# Domain result
# ===========================================================================


@pytest.mark.unit
class TestDomainResult:
    def test_from_dict(self) -> None:
        data = {
            "primary_domain": "Legal",
            "secondary_domains": ["Finance"],
            "languages": ["English", "German"],
            "data_structure": "semi-structured",
            "ontology_scope": "single",
            "estimated_entity_types": 12,
            "key_concepts": ["contract", "party", "clause"],
        }
        result = DomainResult.from_dict(data)
        assert result.primary_domain == "Legal"
        assert result.languages == ["English", "German"]
        assert len(result.key_concepts) == 3

    def test_defaults(self) -> None:
        result = DomainResult.from_dict({})
        assert result.primary_domain == "General"
        assert result.ontology_scope == "single"


# ===========================================================================
# Output: validation
# ===========================================================================


@pytest.mark.unit
class TestValidateOntology:
    def test_valid_config(self) -> None:
        config = _make_expertise()
        result = validate_ontology(config)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_relationship_reference(self) -> None:
        config = ExpertiseConfig(
            name="bad",
            entity_types=[EntityTypeConfig(name="PERSON")],
            relationship_types=[
                RelationshipTypeConfig(
                    name="WORKS_FOR",
                    source_types=["PERSON"],
                    target_types=["NONEXISTENT"],
                )
            ],
        )
        result = validate_ontology(config)
        assert not result.is_valid
        assert any("NONEXISTENT" in e for e in result.errors)

    def test_duplicate_entity_names(self) -> None:
        config = ExpertiseConfig(
            name="dupes",
            entity_types=[
                EntityTypeConfig(name="PERSON"),
                EntityTypeConfig(name="PERSON"),
            ],
        )
        result = validate_ontology(config)
        assert not result.is_valid
        assert any("Duplicate" in e for e in result.errors)

    def test_wildcard_types_pass(self) -> None:
        config = ExpertiseConfig(
            name="wild",
            entity_types=[EntityTypeConfig(name="THING")],
            relationship_types=[RelationshipTypeConfig(name="RELATES_TO", source_types=["*"], target_types=["*"])],
        )
        result = validate_ontology(config)
        assert result.is_valid

    def test_missing_prompt_warning(self) -> None:
        config = ExpertiseConfig(name="noprompt", system_prompt=None)
        result = validate_ontology(config)
        assert any("system prompt" in w for w in result.warnings)


# ===========================================================================
# Output: serialization
# ===========================================================================


@pytest.mark.unit
class TestSerializeOntology:
    def test_roundtrip(self) -> None:
        config = _make_expertise()
        yaml_str = serialize_ontology(config)

        # Parse back
        data = yaml.safe_load(yaml_str)
        restored = ExpertiseConfig.from_dict(data)

        assert restored.name == config.name
        assert len(restored.entity_types) == len(config.entity_types)
        assert len(restored.relationship_types) == len(config.relationship_types)

    def test_clean_output_no_none(self) -> None:
        config = ExpertiseConfig(name="minimal")
        yaml_str = serialize_ontology(config)
        assert "null" not in yaml_str.lower()
        assert "None" not in yaml_str


# ===========================================================================
# Session
# ===========================================================================


@pytest.mark.unit
class TestOntologySession:
    def test_save_and_load(self, tmp_path: Path) -> None:
        session = OntologySession(
            phase="domain_detected",
            sources=["./data/"],
            model="gpt-4o",
            budget_usd=2.0,
            domain={"primary_domain": "Legal"},
        )
        path = tmp_path / "session.json"
        session.save(path)

        loaded = OntologySession.load(path)
        assert loaded.phase == "domain_detected"
        assert loaded.sources == ["./data/"]
        assert loaded.domain == {"primary_domain": "Legal"}

    def test_get_domain_result(self) -> None:
        session = OntologySession(domain={"primary_domain": "Finance", "key_concepts": ["stock", "bond"]})
        dr = session.get_domain_result()
        assert dr is not None
        assert dr.primary_domain == "Finance"

    def test_get_domain_result_none(self) -> None:
        session = OntologySession()
        assert session.get_domain_result() is None

    def test_get_expertise_config(self) -> None:
        config = _make_expertise()
        session = OntologySession(draft=config.to_dict())
        restored = session.get_expertise_config()
        assert restored is not None
        assert restored.name == "test"


# ===========================================================================
# CLI integration (Click CliRunner)
# ===========================================================================


@pytest.mark.unit
class TestOntologyCLI:
    def test_ontology_group_help(self) -> None:
        from click.testing import CliRunner

        from khora.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["ontology", "--help"])
        assert result.exit_code == 0
        assert "construct" in result.output
        assert "validate" in result.output
        assert "preview" in result.output

    def test_validate_builtin_general(self) -> None:
        from click.testing import CliRunner

        from khora.cli import cli

        yaml_path = "src/khora/extraction/skills/builtin/general.yaml"
        if not Path(yaml_path).exists():
            pytest.skip("general.yaml not found")

        runner = CliRunner()
        result = runner.invoke(cli, ["ontology", "validate", yaml_path])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_validate_invalid_yaml(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from khora.cli import cli

        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "name: bad\nrelationship_types:\n  - name: R\n    source_types: [NOPE]\n    target_types: [NOPE]"
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["ontology", "validate", str(bad)])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_preview_builtin(self) -> None:
        from click.testing import CliRunner

        from khora.cli import cli

        yaml_path = "src/khora/extraction/skills/builtin/general.yaml"
        if not Path(yaml_path).exists():
            pytest.skip("general.yaml not found")

        runner = CliRunner()
        result = runner.invoke(cli, ["ontology", "preview", yaml_path])
        assert result.exit_code == 0
        assert "PERSON" in result.output
        assert "WORKS_FOR" in result.output

    def test_construct_help(self) -> None:
        from click.testing import CliRunner

        from khora.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["ontology", "construct", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output
        assert "--model" in result.output
        assert "--budget" in result.output


# ===========================================================================
# SourceSummary
# ===========================================================================


@pytest.mark.unit
class TestSourceSummary:
    def test_size_human_bytes(self) -> None:
        s = SourceSummary(source_id="x", source_type="file", path=Path("."), total_bytes=500)
        assert s.size_human == "500 B"

    def test_size_human_kb(self) -> None:
        s = SourceSummary(source_id="x", source_type="file", path=Path("."), total_bytes=2048)
        assert "KB" in s.size_human

    def test_size_human_mb(self) -> None:
        s = SourceSummary(source_id="x", source_type="file", path=Path("."), total_bytes=5_000_000)
        assert "MB" in s.size_human


# ===========================================================================
# SampleChunk
# ===========================================================================


@pytest.mark.unit
class TestSampleChunk:
    def test_char_count(self) -> None:
        chunk = SampleChunk(source_id="test", content="Hello world")
        assert chunk.char_count == 11

    def test_metadata_default(self) -> None:
        chunk = SampleChunk(source_id="test", content="data")
        assert chunk.metadata == {}
