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
