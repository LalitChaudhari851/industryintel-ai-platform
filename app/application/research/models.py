"""Application-level research session models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.schemas import ResearchJobStatus
from app.workflows.business_research.state import CriticReview, FinalReport, ResearchPlan


class ResearchSessionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    business_context: str | None = None
    user_id: str | None = None
    max_iterations: int = 3
    status: ResearchJobStatus = ResearchJobStatus.QUEUED
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    plan: ResearchPlan | None = None
    critic_review: CriticReview | None = None
    report: FinalReport | None = None
    raw_state: dict = Field(default_factory=dict)
