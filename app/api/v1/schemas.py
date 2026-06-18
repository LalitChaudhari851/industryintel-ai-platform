"""Pydantic request and response schemas for research APIs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from app.workflows.business_research.state import CriticReview, FinalReport, ResearchPlan


class ResearchJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PENDING_REVIEW = "pending_review"
    COMPLETED = "completed"
    FAILED = "failed"



class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=2000)
    business_context: str | None = Field(default=None, max_length=4000)
    user_id: str | None = Field(default=None, max_length=128)
    max_iterations: int = Field(default=3, ge=1, le=3)


class ResearchCreateResponse(BaseModel):
    id: str
    status: ResearchJobStatus
    query: str
    created_at: datetime
    status_url: str
    report_url: str


class ResearchStatusResponse(BaseModel):
    id: str
    status: ResearchJobStatus
    query: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    confidence_score: float | None = None


class ResearchDetailResponse(ResearchStatusResponse):
    business_context: str | None = None
    user_id: str | None = None
    plan: ResearchPlan | None = None
    critic_review: CriticReview | None = None
    report_available: bool = False


class ResearchReportResponse(BaseModel):
    id: str
    status: ResearchJobStatus
    report: FinalReport


class ReviewRequest(BaseModel):
    approval_status: str  # approved, rejected, request_more_research
    reviewer_comments: str | None = None


class ReviewDetailsResponse(BaseModel):
    report_id: str
    status: ResearchJobStatus
    query: str
    report_draft: Dict[str, Any] | None = None
    critic_score: float | None = None
    source_count: int = 0
    confidence_score: float | None = None
    reviewer_comments: str | None = None
    review_history: List[Dict[str, Any]] = Field(default_factory=list)
