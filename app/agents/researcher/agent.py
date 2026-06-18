"""Production Researcher agent node for the LangGraph workflow."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import SecretStr

from app.agents.researcher.models import (
    ResearchAgentConfig,
    SearchClient,
    SearchContext,
    TavilySearchRequest,
)
from app.agents.researcher.query_generator import ResearchQueryGenerator
from app.agents.researcher.scoring import deduplicate_and_rank
from app.agents.researcher.tavily_client import TavilySearchClient, TavilySearchError
from app.workflows.business_research.state import (
    AgentName,
    EvidenceItem,
    ResearchState,
    Source,
    SourceType,
    WorkflowError,
    utc_now,
)
import logging

from app.services.llm_service import LLMService
from app.services.memory_service import ResearchMemoryService
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


class ResearchAgent:
    """LangGraph-compatible Researcher node.

    Responsibilities:
    - Generate search queries.
    - Search the web using Tavily.
    - Deduplicate sources.
    - Score source relevance and credibility.
    - Return structured Source and EvidenceItem updates.
    """

    def __init__(
        self,
        *,
        config: ResearchAgentConfig,
        search_client: SearchClient | None = None,
        query_generator: ResearchQueryGenerator | None = None,
        llm_service: LLMService | None = None,
        memory_service: ResearchMemoryService | None = None,
    ) -> None:
        self.config = config
        self.search_client = search_client or TavilySearchClient(config)
        self.query_generator = query_generator or ResearchQueryGenerator(
            llm=llm_service.primary_llm if llm_service else None,
            max_queries=config.max_queries,
        )
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.reranker = memory_service.reranker if memory_service else None

    @classmethod
    def from_env(
        cls,
        *,
        query_generator: ResearchQueryGenerator | None = None,
        search_client: SearchClient | None = None,
        llm_service: LLMService | None = None,
        memory_service: ResearchMemoryService | None = None,
        **overrides: Any,
    ) -> "ResearchAgent":
        api_key = overrides.pop("tavily_api_key", None) or os.getenv("TAVILY_API_KEY")
        config = ResearchAgentConfig(
            tavily_api_key=SecretStr(api_key) if api_key else None,
            **overrides,
        )
        return cls(
            config=config,
            search_client=search_client,
            query_generator=query_generator,
            llm_service=llm_service,
            memory_service=memory_service,
        )

    @traceable(name="ResearcherAgent", run_type="chain")
    async def __call__(self, state: ResearchState) -> ResearchState:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({
                "topic": state.get("query"),
                "report_id": state.get("session_id"),
            })
        context = self._build_context(state)

        try:
            query_plan = await self.query_generator.generate(context)
            responses = await asyncio.gather(
                *[self._run_query(candidate.query) for candidate in query_plan.queries],
                return_exceptions=True,
            )
        except Exception as exc:
            return self._error_update(f"research query generation failed: {exc}")

        query_results = []
        errors: list[WorkflowError] = []

        for candidate, response in zip(query_plan.queries, responses, strict=False):
            if isinstance(response, Exception):
                errors.append(
                    WorkflowError(
                        agent=AgentName.RESEARCHER,
                        message=f"Tavily search failed for '{candidate.query}': {response}",
                        recoverable=True,
                    )
                )
                continue
            for result in response.results:
                query_results.append((candidate, result))

        ranked_results = deduplicate_and_rank(
            query_results,
            min_relevance_score=self.config.min_relevance_score,
            max_sources=self.config.max_sources,
            reranker=self.reranker if self.config.use_reranker else None,
        )

        sources: list[Source] = []
        evidence: list[EvidenceItem] = []

        for index, ranked in enumerate(ranked_results, start=1):
            source = Source(
                source_type=SourceType.WEB,
                title=ranked.result.title,
                url=ranked.result.url,
                credibility_score=ranked.credibility_score,
                metadata={
                    "query": ranked.query,
                    "task_id": ranked.task_id,
                    "rank": index,
                    "tavily_score": ranked.result.score,
                    "relevance_score": ranked.relevance_score,
                    "published_date": ranked.result.published_date,
                    **ranked.result.metadata,
                },
            )
            sources.append(source)

            evidence_text = ranked.result.raw_content or ranked.result.content
            if evidence_text:
                evidence.append(
                    EvidenceItem(
                        source_id=source.id,
                        task_id=ranked.task_id,
                        text=evidence_text,
                        relevance_score=ranked.relevance_score,
                        citation_label=f"[S{index}]",
                    )
                )

        if not sources:
            errors.append(
                WorkflowError(
                    agent=AgentName.RESEARCHER,
                    message="research completed but no sources met the relevance threshold",
                    recoverable=True,
                )
            )

        if self.memory_service is not None and evidence:
            try:
                logger.info("ResearchAgent ingesting %d evidence items into FAISS memory...", len(evidence))
                await self.memory_service.ingest_evidence(evidence)
            except Exception as e:
                logger.error("ResearchAgent failed to ingest evidence into FAISS memory: %s", e)

        # Log final source count to LangSmith run tree metadata
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({"source_count": len(sources)})

        return {
            "sources": sources,
            "evidence": evidence,
            "errors": errors,
            "updated_at": utc_now(),
        }

    async def _run_query(self, query: str):
        request = TavilySearchRequest(
            query=query,
            search_depth=self.config.search_depth,
            topic=self.config.topic,
            max_results=self.config.results_per_query,
            include_answer=self.config.include_answer,
            include_raw_content=self.config.include_raw_content,
            include_usage=self.config.include_usage,
        )
        try:
            return await self.search_client.search(request)
        except TavilySearchError:
            raise

    def _build_context(self, state: ResearchState) -> SearchContext:
        plan = state.get("plan")
        research_tasks = state.get("research_tasks") or list(plan.tasks if plan else [])
        previous_source_urls = tuple(
            str(source.url)
            for source in state.get("sources", [])
            if source.url is not None
        )

        return SearchContext(
            session_id=state["session_id"],
            query=state["query"],
            business_context=state.get("business_context"),
            tasks=tuple(research_tasks),
            retry_reason=state.get("retry_reason"),
            previous_source_urls=previous_source_urls,
        )

    def _error_update(self, message: str) -> ResearchState:
        return {
            "errors": [
                WorkflowError(
                    agent=AgentName.RESEARCHER,
                    message=message,
                    recoverable=True,
                )
            ],
            "updated_at": utc_now(),
        }
