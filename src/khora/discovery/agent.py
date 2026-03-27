"""Discovery agent — state-machine-based orchestrator.

Ties together the DiscoveryUI, DiscoveryPlanner, PerplexityClient,
and FirecrawlClient into a coherent interactive loop that walks
through the AgentPhase transitions.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from .clients.firecrawl import FirecrawlClient
from .clients.perplexity import PerplexityClient
from .planner import DiscoveryPlanner
from .state import (
    AgentPhase,
    DiscoveredSource,
    FetchAttempt,
    FetchMethod,
    FetchResult,
    SessionState,
)
from .ui import DiscoveryUI


class DiscoveryAgent:
    """State-machine-based interactive agent for datasource discovery.

    Each phase is handled by a dedicated ``_handle_<phase>`` method that
    returns the next phase.  The LLM is used *within* phases (via the
    planner) but does not control phase transitions — keeping the flow
    predictable and testable.

    Usage::

        agent = DiscoveryAgent(ui=ui, output_dir=Path("./data"))
        state = await agent.run()
        paths = [f.local_path for f in state.successful_fetches]
    """

    def __init__(
        self,
        *,
        ui: DiscoveryUI,
        output_dir: Path,
        state: SessionState | None = None,
        planner: DiscoveryPlanner | None = None,
        perplexity: PerplexityClient | None = None,
        firecrawl: FirecrawlClient | None = None,
    ) -> None:
        self._ui = ui
        self._output_dir = output_dir
        self._state = state or SessionState(output_dir=str(output_dir))
        self._planner = planner or DiscoveryPlanner()

        # Detect available API keys
        self._has_perplexity = perplexity is not None or bool(os.environ.get("PERPLEXITY_API_KEY"))
        self._has_firecrawl = firecrawl is not None or bool(os.environ.get("FIRECRAWL_API_KEY"))

        # Store injected clients (or None to create on demand)
        self._perplexity = perplexity
        self._firecrawl = firecrawl

    @property
    def state(self) -> SessionState:
        return self._state

    async def run(self) -> SessionState:
        """Execute the discovery loop until DONE or interrupted."""
        handlers = {
            AgentPhase.GATHER_INTENT: self._handle_gather_intent,
            AgentPhase.SEARCH: self._handle_search,
            AgentPhase.PRESENT_RESULTS: self._handle_present_results,
            AgentPhase.SELECT_SOURCES: self._handle_select_sources,
            AgentPhase.FETCH: self._handle_fetch,
            AgentPhase.REVIEW: self._handle_review,
        }

        while self._state.phase != AgentPhase.DONE:
            handler = handlers.get(self._state.phase)
            if handler is None:
                break

            next_phase = await handler()
            self._state.phase = next_phase
            self._state.iteration += 1

            if self._state.iteration >= self._state.max_iterations:
                self._ui.show_max_iterations()
                break

        return self._state

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    async def _handle_gather_intent(self) -> AgentPhase:
        intent = await self._ui.prompt_intent()
        if not intent:
            return AgentPhase.DONE

        self._state.user_intent = intent
        self._state.conversation_history.append({"role": "user", "content": intent})

        # Use planner to formulate queries
        plan = await self._planner.formulate_queries(
            intent,
            previous_queries=self._state.search_queries or None,
        )
        self._state.search_queries = plan.search_queries
        self._state.total_cost_usd = self._planner.cost_usd

        logger.debug(f"Query plan: {plan.domain} — {len(plan.search_queries)} queries")
        return AgentPhase.SEARCH

    async def _handle_search(self) -> AgentPhase:
        if not self._has_perplexity:
            self._ui.show_no_perplexity()
            url = await self._ui.prompt_url()
            if url:
                self._state.discovered.append(DiscoveredSource(url=url, title=url, discovered_via="user"))
                return AgentPhase.PRESENT_RESULTS
            return AgentPhase.GATHER_INTENT

        queries = self._state.search_queries or [self._state.user_intent]
        all_citations: list[str] = []
        last_answer = ""

        for query in queries:
            with self._ui.show_searching(query):
                try:
                    if self._perplexity:
                        client = self._perplexity
                        response = await client.search(query)
                    else:
                        async with PerplexityClient() as client:
                            response = await client.search(query)

                    all_citations.extend(response.citations)
                    if response.answer:
                        last_answer = response.answer
                except Exception as e:
                    self._ui.show_search_failed(str(e))
                    logger.warning(f"Search failed for query '{query}': {e}")

        if not all_citations:
            if last_answer:
                self._ui.show_search_results(last_answer, 0)
            self._ui.show_no_results()
            return AgentPhase.GATHER_INTENT

        # Deduplicate citations
        seen: set[str] = set()
        unique_citations = []
        for c in all_citations:
            if c not in seen:
                seen.add(c)
                unique_citations.append(c)

        # Show raw answer
        self._ui.show_search_results(last_answer, len(unique_citations))

        # Use planner to classify and rank
        self._state.discovered = await self._planner.classify_sources(
            self._state.user_intent,
            unique_citations,
        )
        self._state.total_cost_usd = self._planner.cost_usd

        return AgentPhase.PRESENT_RESULTS

    async def _handle_present_results(self) -> AgentPhase:
        if not self._state.discovered:
            return AgentPhase.GATHER_INTENT

        self._ui.show_sources(self._state.discovered)
        return AgentPhase.SELECT_SOURCES

    async def _handle_select_sources(self) -> AgentPhase:
        indices = await self._ui.prompt_source_selection(len(self._state.discovered))
        if indices is None:
            return AgentPhase.GATHER_INTENT

        self._state.selected_indices = indices
        return AgentPhase.FETCH

    async def _handle_fetch(self) -> AgentPhase:
        selected = self._state.selected_sources
        if not selected:
            self._ui.show_no_selection()
            return AgentPhase.SELECT_SOURCES

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._state.fetched.clear()

        for src in selected:
            self._ui.show_fetch_start(src.url)
            strategy = self._planner.plan_fetch_strategy(src, has_firecrawl=self._has_firecrawl)

            if strategy.method == "firecrawl_scrape" and self._has_firecrawl:
                await self._fetch_with_firecrawl(src)
            elif strategy.method == "direct_download":
                await self._fetch_direct(src)
            elif strategy.method == "generated_script":
                await self._fetch_with_script(src)
            else:
                await self._fetch_direct(src)

        return AgentPhase.REVIEW

    async def _handle_review(self) -> AgentPhase:
        self._ui.show_review_summary(self._state.fetched)

        # Check for index pages and offer deep crawl
        index_sources = await self._check_for_index_pages()
        if index_sources:
            # Add new sources from index pages and go back to fetch
            self._state.discovered.extend(index_sources)
            new_indices = list(range(len(self._state.discovered) - len(index_sources), len(self._state.discovered)))
            self._state.selected_indices = new_indices
            return AgentPhase.FETCH

        # Run validation on successful fetches
        successful = self._state.successful_fetches
        if successful:
            from .validation import validate_batch

            paths = [Path(f.local_path) for f in successful if f.local_path and Path(f.local_path).exists()]
            if paths:
                val_results = validate_batch(paths, query=self._state.user_intent)

                # Enrich with LLM summaries for the top results (limit to 5 to save cost)
                enriched: list[dict] = []
                for vr in val_results:
                    entry = vr.to_dict()
                    if vr.decision in ("accept", "review") and len(enriched) < 5:
                        try:
                            content = Path(vr.path).read_text(encoding="utf-8", errors="replace")
                            summary = await self._planner.summarize_content(content, Path(vr.path).name)
                            entry["content_summary"] = summary
                        except Exception:
                            entry["content_summary"] = ""
                    enriched.append(entry)

                self._ui.show_validation_results(enriched)

        if self._planner.cost_usd > 0:
            self._ui.show_cost(self._planner.cost_usd)

        choice = await self._ui.prompt_review_action()
        if choice == "accept":
            return AgentPhase.DONE
        elif choice == "retry":
            self._state.fetched.clear()
            return AgentPhase.FETCH
        elif choice == "search":
            return AgentPhase.GATHER_INTENT
        else:
            return AgentPhase.DONE

    # ------------------------------------------------------------------
    # Index page detection
    # ------------------------------------------------------------------

    async def _check_for_index_pages(self) -> list[DiscoveredSource]:
        """Check successful fetches for index pages and offer deep crawl.

        Returns new DiscoveredSource objects for document links if the
        user chooses to download them, or an empty list otherwise.
        """
        from .validation import ContentClass, classify_content

        new_sources: list[DiscoveredSource] = []

        for fetch in self._state.successful_fetches:
            if not fetch.local_path or not Path(fetch.local_path).exists():
                continue

            try:
                content = Path(fetch.local_path).read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            classification = classify_content(content, fetch.source.url)
            if classification.content_class != ContentClass.INDEX:
                continue

            doc_count = len(classification.document_links)
            sub_count = len(classification.subpage_links)

            if doc_count == 0 and sub_count == 0:
                continue

            self._ui.show_index_detected(fetch.source.title, doc_count, sub_count)

            action = await self._ui.prompt_index_action(doc_count)

            if action == "skip":
                continue

            links = classification.document_links
            if action == "pick":
                start, end = await self._ui.prompt_pick_range(len(links))
                links = links[start:end]

            # Create DiscoveredSource for each selected document link
            for title, url in links:
                ext = Path(url.split("?")[0]).suffix.lower()
                from .state import SourceType

                source_type = SourceType.PDF if ext == ".pdf" else SourceType.CSV if ext == ".csv" else SourceType.OTHER
                new_sources.append(
                    DiscoveredSource(
                        url=url,
                        title=title or Path(url).name,
                        source_type=source_type,
                        access_method="direct_download",
                        discovered_via="index_extraction",
                        discovery_query=self._state.user_intent,
                        relevance_score=fetch.source.relevance_score,
                    )
                )

            self._ui.show_info(f"  [{len(new_sources)} document(s) queued for download]")

        return new_sources

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------

    async def _fetch_with_firecrawl(self, src: DiscoveredSource) -> None:
        """Fetch a source using Firecrawl scrape."""
        try:
            with self._ui.show_fetching(src.url):
                if self._firecrawl:
                    result = await self._firecrawl.scrape(src.url)
                else:
                    async with FirecrawlClient() as fc:
                        result = await fc.scrape(src.url)

            if result.markdown:
                out_file = self._save_content(src, result.markdown, ".md")
                self._state.fetched.append(
                    FetchResult(
                        source=src,
                        local_path=str(out_file),
                        content_type="text/markdown",
                        size_bytes=len(result.markdown.encode()),
                        success=True,
                        attempts=[
                            FetchAttempt(
                                method=FetchMethod.FIRECRAWL_SCRAPE,
                                success=True,
                                bytes_fetched=len(result.markdown.encode()),
                            )
                        ],
                    )
                )
                self._ui.show_fetch_saved(out_file.name, len(result.markdown))
            else:
                self._ui.show_fetch_empty(src.url)
                self._state.fetched.append(FetchResult(source=src, local_path="", error="no content"))
        except Exception as e:
            self._ui.show_fetch_failed(str(e))
            self._state.fetched.append(FetchResult(source=src, local_path="", error=str(e)))

    async def _fetch_direct(self, src: DiscoveredSource) -> None:
        """Fetch a source via direct HTTP download."""
        import httpx

        try:
            with self._ui.show_fetching(src.url):
                async with httpx.AsyncClient(timeout=30.0) as http:
                    resp = await http.get(src.url, follow_redirects=True)
                    resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            ext = ".html" if "html" in content_type else ".txt"
            out_file = self._save_content(src, resp.text, ext)
            self._state.fetched.append(
                FetchResult(
                    source=src,
                    local_path=str(out_file),
                    content_type=content_type,
                    size_bytes=len(resp.content),
                    success=True,
                    attempts=[
                        FetchAttempt(
                            method=FetchMethod.DIRECT_DOWNLOAD,
                            success=True,
                            bytes_fetched=len(resp.content),
                        )
                    ],
                )
            )
            self._ui.show_fetch_saved(out_file.name, len(resp.content), unit="bytes")
        except Exception as e:
            self._ui.show_fetch_failed(str(e))
            self._state.fetched.append(FetchResult(source=src, local_path="", error=str(e)))

    async def _fetch_with_script(self, src: DiscoveredSource) -> None:
        """Generate a fetch script via LLM, validate, confirm, and execute."""
        from .codegen import execute_script, extract_urls, render_template, validate_script

        try:
            # Generate script via planner
            self._ui.show_info("[dim]Generating fetch script...[/]")
            raw_body = await self._planner.generate_fetch_script(src, str(self._output_dir))

            script = render_template(title=src.title, url=src.url, fetch_body=raw_body)

            # AST validation
            violations = validate_script(script)
            if violations:
                msg = "Script failed validation:\n" + "\n".join(f"  {v}" for v in violations)
                self._ui.show_fetch_failed(msg)
                self._state.fetched.append(FetchResult(source=src, local_path="", error=msg))
                return

            # Show script and URLs to user for confirmation
            urls = extract_urls(script)
            self._ui.show_info(f"\n[bold]Generated script for: {src.title}[/]")
            if urls:
                self._ui.show_info(f"[dim]Script will contact: {', '.join(urls)}[/]")
            self._ui.show_data_preview("fetch_script.py", script, max_chars=2000)

            from rich.prompt import Confirm

            approved = Confirm.ask("Execute this script?", default=False)
            if not approved:
                self._ui.show_info("[dim]Script execution skipped by user.[/]")
                return

            # Execute in sandbox
            with self._ui.show_fetching(src.url):
                result = await execute_script(
                    script,
                    self._output_dir,
                    timeout=120,
                )

            if result.success and result.files_created:
                for f in result.files_created:
                    self._ui.show_fetch_saved(f, 0, unit="file(s)")
                self._state.fetched.append(
                    FetchResult(
                        source=src,
                        local_path=result.files_created[0],
                        content_type="application/octet-stream",
                        size_bytes=sum(
                            Path(self._output_dir / f).stat().st_size
                            for f in result.files_created
                            if (self._output_dir / f).exists()
                        ),
                        success=True,
                        attempts=[
                            FetchAttempt(
                                method=FetchMethod.GENERATED_SCRIPT,
                                success=True,
                            )
                        ],
                    )
                )
            else:
                error = result.stderr or result.summary.get("error", "Script produced no output")
                self._ui.show_fetch_failed(error[:200])
                self._state.fetched.append(FetchResult(source=src, local_path="", error=error[:500]))

        except Exception as e:
            self._ui.show_fetch_failed(str(e))
            self._state.fetched.append(FetchResult(source=src, local_path="", error=str(e)))

    def _save_content(self, src: DiscoveredSource, content: str, ext: str) -> Path:
        """Write content to a file in the output directory."""
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in src.title[:50])
        if not safe_name:
            safe_name = "source"
        out_file = self._output_dir / f"{safe_name}{ext}"
        out_file.write_text(content, encoding="utf-8")
        return out_file
