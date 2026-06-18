"""Research API endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from sse_starlette.sse import EventSourceResponse

from app.api.v1.schemas import (
    ResearchCreateResponse,
    ResearchDetailResponse,
    ResearchJobStatus,
    ResearchReportResponse,
    ResearchRequest,
    ResearchStatusResponse,
    ReviewRequest,
    ReviewDetailsResponse,
)
from app.application.research.service import ResearchService
from app.dependencies import get_research_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["research"])


@router.post(
    "/research",
    response_model=ResearchCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    service: ResearchService = Depends(get_research_service),
) -> ResearchCreateResponse:
    session = await service.create_research(request)
    background_tasks.add_task(service.run_research, session.id)
    return ResearchCreateResponse(
        id=session.id,
        status=session.status,
        query=session.query,
        created_at=session.created_at,
        status_url=f"/research/{session.id}/status",
        report_url=f"/research/{session.id}/report",
    )


@router.get("/research/{research_id}", response_model=ResearchDetailResponse)
async def get_research(
    research_id: str,
    service: ResearchService = Depends(get_research_service),
) -> ResearchDetailResponse:
    session = await service.get_research(research_id)
    return service.to_detail_response(session)


@router.get("/research/{research_id}/status", response_model=ResearchStatusResponse)
async def get_research_status(
    research_id: str,
    service: ResearchService = Depends(get_research_service),
) -> ResearchStatusResponse:
    session = await service.get_research(research_id)
    return service.to_status_response(session)


@router.get("/research/{research_id}/report", response_model=ResearchReportResponse)
async def get_research_report(
    research_id: str,
    service: ResearchService = Depends(get_research_service),
) -> ResearchReportResponse:
    session = await service.get_completed_report(research_id)
    return ResearchReportResponse(
        id=session.id,
        status=session.status,
        report=session.report,
    )


@router.get("/research/{research_id}/stream")
async def stream_research_progress(
    research_id: str,
    service: ResearchService = Depends(get_research_service),
) -> EventSourceResponse:
    """SSE endpoint streaming real-time status of the research workflow."""

    async def event_generator():
        while True:
            try:
                record = await service.get_research(research_id)
                current_status = record.status.value
                current_iteration = 1
                if record.raw_state:
                    current_iteration = record.raw_state.get("iteration_count", 1)

                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "id": research_id,
                        "status": current_status,
                        "iteration": current_iteration,
                        "error": record.error,
                    }),
                }

                if record.status in {ResearchJobStatus.COMPLETED, ResearchJobStatus.FAILED}:
                    break
            except Exception as e:
                logger.error("SSE stream failed for job %s: %s", research_id, e)
                yield {
                    "event": "error",
                    "data": str(e),
                }
                break

            await asyncio.sleep(1.0)

    return EventSourceResponse(event_generator())


@router.get("/models/status")
async def get_models_status(request: Request) -> dict[str, Any]:
    """Health check endpoint for local Ollama models."""
    llm_service = request.app.state.llm_service
    health = await llm_service.check_health()
    return health


@router.post("/review/{report_id}", status_code=status.HTTP_202_ACCEPTED)
async def submit_review(
    report_id: str,
    review: ReviewRequest,
    background_tasks: BackgroundTasks,
    service: ResearchService = Depends(get_research_service),
) -> dict[str, str]:
    """Submit a reviewer decision and resume the research workflow."""
    background_tasks.add_task(
        service.resume_research,
        report_id,
        review.approval_status,
        review.reviewer_comments,
    )
    return {"message": "Review submitted successfully and workflow execution resumed."}


@router.get("/review/{report_id}", response_model=ReviewDetailsResponse)
async def get_review_details(
    report_id: str,
    service: ResearchService = Depends(get_research_service),
) -> ReviewDetailsResponse:
    """Retrieve report draft, critic review, and metadata for review screen."""
    from typing import Any
    session = await service.get_research(report_id)

    from app.workflows.business_research import BusinessResearchAgents, build_business_research_graph
    from app.agents.planner import PlannerAgent
    from app.agents.researcher import ResearchAgent, ResearchAgentConfig
    from app.agents.analyst import AnalystAgent
    from app.agents.critic import CriticAgent
    from app.agents.writer import WriterAgent
    from pydantic import SecretStr

    agents = BusinessResearchAgents(
        planner=PlannerAgent(llm_service=service.llm_service),
        researcher=ResearchAgent(
            config=ResearchAgentConfig(
                tavily_api_key=SecretStr(service.settings.tavily_api_key or "mock"),
                max_queries=1,
            ),
            llm_service=service.llm_service,
            memory_service=service.memory_service,
        ),
        analyst=AnalystAgent(llm_service=service.llm_service, memory_service=service.memory_service),
        critic=CriticAgent(llm_service=service.llm_service),
        writer=WriterAgent(llm_service=service.llm_service),
    )
    graph = build_business_research_graph(agents)
    config = {"configurable": {"thread_id": report_id}}
    state_snapshot = await graph.aget_state(config)
    current_state = state_snapshot.values if state_snapshot else {}

    analysis = current_state.get("analysis")
    report_draft = {}
    confidence_score = 0.0
    if analysis:
        report_draft = {
            "summary": analysis.summary,
            "findings": [f.model_dump() for f in analysis.findings] if analysis.findings else [],
            "metrics": [m.model_dump() for m in analysis.metrics] if analysis.metrics else [],
            "trends": [t.model_dump() for t in analysis.trends] if analysis.trends else [],
        }
        confidence_score = analysis.confidence_score

    critic_review = current_state.get("critic_review")
    critic_score = critic_review.confidence_score if critic_review else None
    source_count = len(current_state.get("sources", []))
    review_history = current_state.get("review_history", [])

    return ReviewDetailsResponse(
        report_id=report_id,
        status=session.status,
        query=session.query,
        report_draft=report_draft or None,
        critic_score=critic_score,
        source_count=source_count,
        confidence_score=confidence_score,
        reviewer_comments=current_state.get("reviewer_comments"),
        review_history=review_history,
    )
