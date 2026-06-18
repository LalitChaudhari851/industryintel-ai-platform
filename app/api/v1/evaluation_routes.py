"""FastAPI routes for retrieving research evaluation metrics and performance trends."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["evaluation"])


def get_evaluation_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> EvaluationService:
    """Dependency injector to construct the EvaluationService."""
    research_service = request.app.state.research_service
    return EvaluationService(research_service.store, settings)


@router.get("/evaluation/metrics")
async def get_evaluation_metrics(
    service: EvaluationService = Depends(get_evaluation_service),
) -> Dict[str, Any]:
    """Retrieve aggregated research quality, agent, report, and LangSmith evaluation metrics."""
    logger.info("Fetching aggregated evaluation metrics")
    return await service.get_aggregated_metrics()


@router.get("/evaluation/reports")
async def get_evaluation_reports(
    service: EvaluationService = Depends(get_evaluation_service),
) -> List[Dict[str, Any]]:
    """Retrieve detailed telemetry records for all completed reports."""
    logger.info("Fetching report-level evaluation list")
    return await service.get_reports()


@router.get("/evaluation/trends")
async def get_evaluation_trends(
    service: EvaluationService = Depends(get_evaluation_service),
) -> List[Dict[str, Any]]:
    """Retrieve chronological trend data for quality, confidence, latency, and source count."""
    logger.info("Fetching chronological evaluation trends")
    return await service.get_trends()
