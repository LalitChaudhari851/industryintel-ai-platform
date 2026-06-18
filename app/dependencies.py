"""FastAPI dependency providers."""

from __future__ import annotations

from fastapi import Request

from app.application.research.service import ResearchService


def get_research_service(request: Request) -> ResearchService:
    return request.app.state.research_service
