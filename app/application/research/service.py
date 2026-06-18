"""Async research orchestration service."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import SecretStr

from app.agents.analyst import AnalystAgent
from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearchAgent, ResearchAgentConfig
from app.agents.writer import WriterAgent
from app.services.llm_service import LLMService
from app.services.memory_service import ResearchMemoryService
from app.api.v1.schemas import (
    ResearchDetailResponse,
    ResearchJobStatus,
    ResearchRequest,
    ResearchStatusResponse,
)
from app.application.research.models import ResearchSessionRecord
from app.application.research.store import InMemoryResearchStore
from app.core.config import Settings
from app.core.errors import ReportNotReadyError, ResearchCapacityError
from app.workflows.business_research import BusinessResearchAgents, build_business_research_graph
from app.workflows.business_research.state import FinalReport, ResearchState, WorkflowStatus

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        llm_service: LLMService,
        memory_service: ResearchMemoryService,
        store: InMemoryResearchStore | None = None,
    ) -> None:
        self.settings = settings
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.store = store or InMemoryResearchStore()
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_research_jobs)

    async def create_research(self, request: ResearchRequest) -> ResearchSessionRecord:
        if self._semaphore.locked():
            raise ResearchCapacityError("Research capacity is currently exhausted")

        now = utc_now()
        record = ResearchSessionRecord(
            id=str(uuid4()),
            query=request.query,
            business_context=request.business_context,
            user_id=request.user_id,
            max_iterations=request.max_iterations,
            created_at=now,
            updated_at=now,
        )
        logger.info("research.create id=%s query=%r", record.id, record.query)
        return await self.store.create(record)

    async def get_research(self, research_id: str) -> ResearchSessionRecord:
        return await self.store.get(research_id)

    async def get_completed_report(self, research_id: str) -> ResearchSessionRecord:
        record = await self.store.get(research_id)
        if record.status is not ResearchJobStatus.COMPLETED or record.report is None:
            raise ReportNotReadyError(f"Report for research session '{research_id}' is not ready")
        return record

    async def run_research(self, research_id: str) -> None:
        async with self._semaphore:
            await self._mark_running(research_id)
            try:
                record = await self.store.get(research_id)
                state = await self._execute_workflow(record)
                await self._mark_completed(research_id, state)
                logger.info("research.completed id=%s", research_id)
            except Exception as exc:
                logger.exception("research.failed id=%s error=%s", research_id, exc)
                await self._mark_failed(research_id, str(exc))

    async def _execute_workflow(self, record: ResearchSessionRecord) -> ResearchState:
        if not self.settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required to run web research")

        agents = BusinessResearchAgents(
            planner=PlannerAgent(llm_service=self.llm_service),
            researcher=ResearchAgent(
                config=ResearchAgentConfig(
                    tavily_api_key=SecretStr(self.settings.tavily_api_key),
                    max_queries=6,
                    results_per_query=5,
                    max_sources=15,
                    use_reranker=self.settings.reranker_model is not None,
                    reranker_model=self.settings.reranker_model,
                    rerank_top_k=self.settings.rerank_top_k,
                ),
                llm_service=self.llm_service,
                memory_service=self.memory_service,
            ),
            analyst=AnalystAgent(llm_service=self.llm_service, memory_service=self.memory_service),
            critic=CriticAgent(llm_service=self.llm_service),
            writer=WriterAgent(llm_service=self.llm_service),
        )
        graph = build_business_research_graph(agents)
        initial_state: ResearchState = {
            "session_id": record.id,
            "user_id": record.user_id,
            "query": record.query,
            "business_context": record.business_context,
            "retry_count": 0,
            "iteration_count": 1,
            "max_iterations": record.max_iterations,
            "status": WorkflowStatus.CREATED,
            "created_at": record.created_at,
            "updated_at": utc_now(),
        }
        return await graph.ainvoke(
            initial_state,
            config={
                "configurable": {"thread_id": record.id},
                "run_name": "business-research-workflow",
                "tags": [
                    "research",
                    self.settings.environment,
                ],
                "metadata": {
                    "research_id": record.id,
                    "report_id": record.id,
                    "user_id": record.user_id,
                    "query": record.query,
                    "topic": record.query,
                },
            },
        )

    async def _mark_running(self, research_id: str) -> None:
        now = utc_now()

        def update(record: ResearchSessionRecord) -> ResearchSessionRecord:
            return record.model_copy(
                update={
                    "status": ResearchJobStatus.RUNNING,
                    "updated_at": now,
                    "error": None,
                }
            )

        await self.store.update(research_id, update)

    async def _mark_completed(self, research_id: str, state: ResearchState) -> None:
        now = utc_now()
        report = state.get("final_report")
        if not isinstance(report, FinalReport):
            raise ValueError("workflow completed without a final report")

        def update(record: ResearchSessionRecord) -> ResearchSessionRecord:
            return record.model_copy(
                update={
                    "status": ResearchJobStatus.COMPLETED,
                    "updated_at": now,
                    "completed_at": now,
                    "plan": state.get("plan"),
                    "critic_review": state.get("critic_review"),
                    "report": report,
                    "raw_state": self._safe_state_summary(state),
                }
            )

        await self.store.update(research_id, update)

    async def _mark_failed(self, research_id: str, message: str) -> None:
        now = utc_now()

        def update(record: ResearchSessionRecord) -> ResearchSessionRecord:
            return record.model_copy(
                update={
                    "status": ResearchJobStatus.FAILED,
                    "updated_at": now,
                    "completed_at": now,
                    "error": message,
                }
            )

        await self.store.update(research_id, update)

    def to_status_response(self, record: ResearchSessionRecord) -> ResearchStatusResponse:
        return ResearchStatusResponse(
            id=record.id,
            status=record.status,
            query=record.query,
            created_at=record.created_at,
            updated_at=record.updated_at,
            completed_at=record.completed_at,
            error=record.error,
            confidence_score=record.report.confidence_score if record.report else None,
        )

    def to_detail_response(self, record: ResearchSessionRecord) -> ResearchDetailResponse:
        status = self.to_status_response(record)
        return ResearchDetailResponse(
            **status.model_dump(),
            business_context=record.business_context,
            user_id=record.user_id,
            plan=record.plan,
            critic_review=record.critic_review,
            report_available=record.report is not None,
        )

    def _safe_state_summary(self, state: ResearchState) -> dict:
        return {
            "status": state.get("status"),
            "retry_count": state.get("retry_count"),
            "iteration_count": state.get("iteration_count"),
            "source_count": len(state.get("sources", [])),
            "evidence_count": len(state.get("evidence", [])),
        }
