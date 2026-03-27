"""Ontology construction flow orchestrator.

Coordinates the phases: source scanning → sampling → domain detection →
entity/relationship/rule inference → output.  Manages session state and
provides the interactive TUI experience.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path

import click
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from khora.extraction.skills.base import (
    ConfidenceConfig,
    ExpansionConfig,
    ExpertiseConfig,
)

from .inference.domain import DomainDetector, DomainResult
from .inference.entity_inferrer import EntityInferrer
from .inference.prompt_generator import PromptGenerator
from .inference.relationship_inferrer import RelationshipInferrer
from .inference.rule_inferrer import RuleInferrer
from .llm import BudgetExhaustedError, OntologyLLM
from .output.serializer import serialize_ontology, write_ontology
from .output.validator import validate_ontology
from .sampling.sampler import DataSampler
from .session import OntologySession
from .sources.detection import detect_source
from .tui.console import console
from .tui.panels import (
    render_entity_types_table,
    render_ontology_tree,
    render_relationship_types_table,
    render_source_summary_table,
)


class OntologyConstructFlow:
    """Orchestrates the full ontology construction pipeline."""

    def __init__(
        self,
        *,
        sources: tuple[str, ...],
        output: str,
        model: str,
        budget: float,
        extends_skill: str | None,
        non_interactive: bool,
        resume: str | None,
    ) -> None:
        self._output = Path(output)
        self._non_interactive = non_interactive
        self._extends_skill = extends_skill

        # Initialize or resume session
        if resume:
            self._session = OntologySession.load(Path(resume))
            console.print(f"[dim]Resumed session from {resume} (phase: {self._session.phase})[/]")
        else:
            self._session = OntologySession(
                sources=list(sources),
                model=model,
                budget_usd=budget,
                output_path=output,
                extends_skill=extends_skill,
            )

        self._llm = OntologyLLM(model=self._session.model, budget_usd=self._session.budget_usd)
        self._sampler = DataSampler()

    async def run(self) -> None:
        """Execute the full construction pipeline."""
        try:
            await self._run_phases()
        except BudgetExhaustedError as e:
            console.print(f"\n[yellow]Budget exhausted: {e}[/]")
            if self._session.draft:
                console.print("[dim]Writing current draft to output...[/]")
                self._write_output()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
            if not self._non_interactive and Confirm.ask("Save session for later resume?", default=True):
                self._session.save()
                console.print(f"[dim]Session saved to {OntologySession.SESSION_FILE}[/]")
            raise SystemExit(130)

    async def _run_phases(self) -> None:
        """Run each phase in sequence."""
        # Phase 1: Source scanning
        if self._session.phase in ("init",):
            await self._phase_sources()
            self._session.phase = "sampled"

        # Phase 2: Domain detection
        if self._session.phase in ("sampled",):
            await self._phase_domain()
            self._session.phase = "domain_detected"

        # Phase 3: Ontology inference
        if self._session.phase in ("domain_detected",):
            await self._phase_inference()
            self._session.phase = "inferred"

        # Phase 4: Output
        await self._phase_output()
        self._session.phase = "complete"

    # ------------------------------------------------------------------
    # Phase 1: Source scanning & sampling
    # ------------------------------------------------------------------

    async def _phase_sources(self) -> None:
        console.print(Rule("[bold magenta]Phase 1: Source Scanning[/]"))

        sources = self._session.sources
        if not sources and not self._non_interactive:
            sources = self._prompt_for_sources()
            self._session.sources = sources

        if not sources:
            console.print("[red]No data sources provided. Use --source or enter interactively.[/]")
            raise SystemExit(1)

        # Scan sources
        summaries = []
        for raw in sources:
            try:
                src = detect_source(raw)
                summary = self._sampler.add_source(src)
                summaries.append(
                    {
                        "source": summary.source_id,
                        "type": summary.source_type,
                        "files": str(summary.file_count),
                        "size": summary.size_human,
                    }
                )
            except (FileNotFoundError, NotADirectoryError, ValueError) as e:
                console.print(f"[yellow]Skipping source: {e}[/]")

        if not self._sampler.sources:
            console.print("[red]No valid sources found.[/]")
            raise SystemExit(1)

        console.print(render_source_summary_table(summaries))

        # Sample
        with console.status("[bold cyan]Sampling data...[/]"):
            self._sampler.sample_all(budget_chars=30_000)

        console.print(
            f"[green]Sampled {self._sampler.total_chars:,} characters "
            f"from {self._sampler.source_count} source(s).[/]\n"
        )

    def _prompt_for_sources(self) -> list[str]:
        """Interactively prompt user for data source paths.

        Offers internet discovery if API keys are available and the user
        has no local sources to provide.
        """
        import os

        has_discovery = bool(os.environ.get("PERPLEXITY_API_KEY") or os.environ.get("FIRECRAWL_API_KEY"))

        if has_discovery:
            console.print(
                "[dim]Enter data source paths (files or directories), "
                "or type [bold cyan]discover[/bold cyan] to search the internet for sources. "
                "Empty line to finish.[/]"
            )
        else:
            console.print("[dim]Enter data source paths (files or directories). Empty line to finish.[/]")

        sources: list[str] = []
        while True:
            raw = Prompt.ask("  Source path", default="")
            if not raw:
                break
            if raw.strip().lower() == "discover" and has_discovery:
                discovered = self._run_discovery()
                sources.extend(discovered)
                if discovered:
                    console.print(f"[green]Added {len(discovered)} discovered source(s).[/]")
                break
            sources.append(raw)
        return sources

    def _run_discovery(self) -> list[str]:
        """Launch the interactive discovery agent and return local paths."""
        from .discover import run_discovery_session

        return asyncio.run(run_discovery_session())

    # ------------------------------------------------------------------
    # Phase 2: Domain detection
    # ------------------------------------------------------------------

    async def _phase_domain(self) -> None:
        console.print(Rule("[bold magenta]Phase 2: Domain Detection[/]"))

        formatted = self._sampler.format_samples_for_llm(30_000)

        with console.status("[bold cyan]Analyzing domain...[/]"):
            detector = DomainDetector(self._llm)
            domain = await detector.detect(
                formatted_samples=formatted,
                source_count=self._sampler.source_count,
                total_chars=self._sampler.total_chars,
            )

        self._session.domain = asdict(domain)
        self._print_domain_result(domain)
        self._update_usage()

        if not self._non_interactive:
            if not Confirm.ask("Proceed with these settings?", default=True):
                console.print("[dim]Aborting.[/]")
                raise SystemExit(0)

    def _print_domain_result(self, domain: DomainResult) -> None:
        """Display domain detection results."""
        table = Table(title="Domain Analysis", show_lines=True)
        table.add_column("Property", style="bold", width=25)
        table.add_column("Value", width=55)

        table.add_row("Primary domain", f"[bright_blue]{domain.primary_domain}[/]")
        if domain.secondary_domains:
            table.add_row("Secondary domains", ", ".join(domain.secondary_domains))
        table.add_row("Languages", ", ".join(domain.languages))
        table.add_row("Data structure", domain.data_structure)
        table.add_row("Ontology scope", domain.ontology_scope)
        table.add_row("Est. entity types", str(domain.estimated_entity_types))
        table.add_row("Est. relationship types", str(domain.estimated_relationship_types))
        if domain.key_concepts:
            table.add_row("Key concepts", ", ".join(domain.key_concepts[:10]))
        if domain.scope_reasoning:
            table.add_row("Reasoning", domain.scope_reasoning)

        console.print(table)
        console.print()

    # ------------------------------------------------------------------
    # Phase 3: Ontology inference
    # ------------------------------------------------------------------

    async def _phase_inference(self) -> None:
        console.print(Rule("[bold magenta]Phase 3: Ontology Inference[/]"))

        domain = self._session.get_domain_result()
        if domain is None:
            console.print("[red]Domain detection result missing. Cannot proceed.[/]")
            raise SystemExit(1)

        formatted = self._sampler.format_samples_for_llm(25_000)

        # 3a: Entity types
        with console.status("[bold cyan]Inferring entity types...[/]"):
            entity_inferrer = EntityInferrer(self._llm)
            entity_types = await entity_inferrer.infer(domain, formatted)
        self._update_usage()

        console.print(render_entity_types_table(entity_types))
        console.print()

        # 3b: Relationship types
        with console.status("[bold cyan]Inferring relationship types...[/]"):
            rel_inferrer = RelationshipInferrer(self._llm)
            rel_types = await rel_inferrer.infer(domain, entity_types, formatted)
        self._update_usage()

        console.print(render_relationship_types_table(rel_types))
        console.print()

        # 3c: Rules
        with console.status("[bold cyan]Generating rules...[/]"):
            rule_inferrer = RuleInferrer(self._llm)
            corr_rules, inf_rules = await rule_inferrer.infer(domain, entity_types, rel_types)
        self._update_usage()

        console.print(f"[green]Generated {len(corr_rules)} correlation rules, {len(inf_rules)} inference rules.[/]")

        # 3d: System prompt
        with console.status("[bold cyan]Generating system prompt...[/]"):
            prompt_gen = PromptGenerator(self._llm)
            system_prompt = await prompt_gen.generate(domain, entity_types, rel_types)
        self._update_usage()

        # Assemble ExpertiseConfig
        name = domain.primary_domain.lower().replace(" ", "_")
        config = ExpertiseConfig(
            name=name,
            version="1.0.0",
            description=f"Ontology for {domain.primary_domain} domain (auto-generated)",
            system_prompt=system_prompt,
            entity_types=entity_types,
            relationship_types=rel_types,
            correlation_rules=corr_rules,
            inference_rules=inf_rules,
            confidence=ConfidenceConfig(min_entity=0.5, min_relationship=0.25, min_inferred=0.3),
            expansion=ExpansionConfig(enabled=True, depth=2, inference_mode="smart"),
        )

        self._session.draft = config.to_dict()
        console.print()
        console.print(render_ontology_tree(config))

    # ------------------------------------------------------------------
    # Phase 4: Output
    # ------------------------------------------------------------------

    async def _phase_output(self) -> None:
        console.print(Rule("[bold magenta]Phase 4: Output[/]"))

        config = self._session.get_expertise_config()
        if config is None:
            console.print("[red]No ontology draft available.[/]")
            raise SystemExit(1)

        # Validate
        vr = validate_ontology(config)
        if vr.errors:
            console.print("[yellow]Validation errors:[/]")
            for err in vr.errors:
                console.print(f"  [red]- {err}[/]")
        if vr.warnings:
            console.print("[dim]Warnings:[/]")
            for warn in vr.warnings:
                console.print(f"  [dim]- {warn}[/]")
        if vr.is_valid:
            console.print("[green]Ontology is valid.[/]")
        console.print()

        # Preview YAML
        yaml_str = serialize_ontology(config)
        console.print(
            Panel(Syntax(yaml_str, "yaml", line_numbers=True), title="Generated Ontology", border_style="cyan")
        )

        # Offer $EDITOR
        if not self._non_interactive:
            if Confirm.ask("Open in $EDITOR for manual edits?", default=False):
                edited = click.edit(yaml_str, extension=".yaml")
                if edited and edited != yaml_str:
                    import yaml

                    try:
                        data = yaml.safe_load(edited)
                        config = ExpertiseConfig.from_dict(data)
                        self._session.draft = config.to_dict()
                        console.print("[green]Edits applied.[/]")
                    except Exception as e:
                        console.print(f"[yellow]Could not parse edits: {e}. Using original.[/]")

        self._write_output()

    def _write_output(self) -> None:
        """Write the current draft to disk."""
        config = self._session.get_expertise_config()
        if config is None:
            return

        write_ontology(config, self._output)
        console.print(f"\n[bold green]Ontology written to {self._output}[/]")
        self._print_usage_summary()

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def _update_usage(self) -> None:
        """Sync LLM usage into session."""
        summary = self._llm.usage_summary
        self._session.tokens_used = summary["total_tokens"]
        self._session.cost_usd = summary["cost_usd"]

    def _print_usage_summary(self) -> None:
        """Print final token/cost summary."""
        summary = self._llm.usage_summary
        console.print(
            f"[dim]LLM usage: {summary['calls']} calls, "
            f"{summary['total_tokens']:,} tokens, "
            f"${summary['cost_usd']:.4f}[/]"
        )


def run_construct(
    source: tuple[str, ...],
    output: str,
    model: str,
    budget: float,
    extends_skill: str | None,
    non_interactive: bool,
    resume: str | None,
) -> None:
    """Entry point called by the Click command."""
    flow = OntologyConstructFlow(
        sources=source,
        output=output,
        model=model,
        budget=budget,
        extends_skill=extends_skill,
        non_interactive=non_interactive,
        resume=resume,
    )
    asyncio.run(flow.run())
