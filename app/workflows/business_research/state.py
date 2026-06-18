"""Typed state and Pydantic v2 models for the business research workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal, NotRequired, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def append_items[T](left: list[T] | None, right: list[T] | None) -> list[T]:
    """LangGraph reducer that appends list updates without mutating prior state."""
    return [*(left or []), *(right or [])]


class AgentName(StrEnum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CRITIC = "critic"
    WRITER = "writer"


class WorkflowStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    CRITIQUING = "critiquing"
    WRITING = "writing"
    PENDING_REVIEW = "pending_review"
    COMPLETED = "completed"
    FAILED = "failed"



class TaskType(StrEnum):
    MARKET = "market"
    COMPANY = "company"
    COMPETITOR = "competitor"
    FINANCIAL = "financial"
    RISK = "risk"
    TREND = "trend"
    GENERAL = "general"


class SourceType(StrEnum):
    WEB = "web"
    INTERNAL_KNOWLEDGE = "internal_knowledge"
    DOCUMENT = "document"


class QualityDecision(StrEnum):
    PASS = "pass"
    RETRY_RESEARCH = "retry_research"
    WRITE_WITH_LIMITATIONS = "write_with_limitations"
    FAIL = "fail"


class RetryReason(StrEnum):
    INSUFFICIENT_SOURCES = "insufficient_sources"
    WEAK_CITATIONS = "weak_citations"
    STALE_OR_LOW_QUALITY_SOURCES = "stale_or_low_quality_sources"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    ANALYSIS_DOES_NOT_ANSWER_QUERY = "analysis_does_not_answer_query"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ResearchTask(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    task_type: TaskType = TaskType.GENERAL
    objective: str = Field(min_length=1)
    search_queries: tuple[str, ...] = Field(default_factory=tuple)
    priority: int = Field(default=3, ge=1, le=5)
    required_source_count: int = Field(default=2, ge=1, le=10)


class ResearchPlan(StrictModel):
    objective: str = Field(min_length=1)
    tasks: tuple[ResearchTask, ...] = Field(default_factory=tuple)
    expected_output: str = Field(min_length=1)
    success_criteria: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("tasks")
    @classmethod
    def require_tasks(cls, tasks: tuple[ResearchTask, ...]) -> tuple[ResearchTask, ...]:
        if not tasks:
            raise ValueError("research plan must include at least one task")
        return tasks


class Source(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: SourceType
    title: str = Field(min_length=1)
    url: HttpUrl | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str = Field(min_length=1)
    task_id: str | None = None
    text: str = Field(min_length=1)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    citation_label: str | None = None


class AnalysisFinding(StrictModel):
    claim: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    implication: str | None = None


class ExtractedMetric(StrictModel):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: str | None = None
    context: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedTrend(StrictModel):
    topic: str = Field(min_length=1)
    direction: Literal["increasing", "decreasing", "stable", "mixed", "unknown"]
    description: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalysisResult(StrictModel):
    summary: str = Field(min_length=1)
    findings: tuple[AnalysisFinding, ...] = Field(default_factory=tuple)
    metrics: tuple[ExtractedMetric, ...] = Field(default_factory=tuple)
    trends: tuple[ExtractedTrend, ...] = Field(default_factory=tuple)
    risks: tuple[str, ...] = Field(default_factory=tuple)
    opportunities: tuple[str, ...] = Field(default_factory=tuple)
    recommendations: tuple[str, ...] = Field(default_factory=tuple)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)


class EvidenceAssessment(StrictModel):
    claim: str = Field(min_length=1)
    supported: bool
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    support_score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)


class Contradiction(StrictModel):
    description: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    severity: Literal["low", "medium", "high"] = "medium"


class CriticFinding(StrictModel):
    issue: str = Field(min_length=1)
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    retry_reason: RetryReason | None = None
    related_claim: str | None = None


class CriticReview(StrictModel):
    decision: QualityDecision
    confidence_score: float = Field(ge=0.0, le=1.0)
    findings: tuple[CriticFinding, ...] = Field(default_factory=tuple)
    evidence_assessments: tuple[EvidenceAssessment, ...] = Field(default_factory=tuple)
    contradictions: tuple[Contradiction, ...] = Field(default_factory=tuple)
    missing_evidence: tuple[str, ...] = Field(default_factory=tuple)
    notes: str | None = None


class Citation(StrictModel):
    claim: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    evidence_id: str | None = None
    supporting_text: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class FinalReport(StrictModel):
    title: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    sections: dict[str, str] = Field(default_factory=dict)
    citations: tuple[Citation, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    confidence_score: float = Field(ge=0.0, le=1.0)
    word_count: int = Field(default=0, ge=0)


class WorkflowError(StrictModel):
    agent: AgentName | None = None
    message: str = Field(min_length=1)
    recoverable: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class ResearchState(TypedDict, total=False):
    """Shared LangGraph state.

    Agent nodes should return partial updates for these keys only.
    """

    session_id: str
    user_id: NotRequired[str | None]
    query: str
    business_context: NotRequired[str | None]
    status: WorkflowStatus
    plan: NotRequired[ResearchPlan | None]
    research_tasks: NotRequired[list[ResearchTask]]
    sources: Annotated[list[Source], append_items]
    evidence: Annotated[list[EvidenceItem], append_items]
    analysis: NotRequired[AnalysisResult | None]
    critic_review: NotRequired[CriticReview | None]
    final_report: NotRequired[FinalReport | None]
    iteration_count: int
    retry_count: int
    max_iterations: int
    retry_reason: NotRequired[RetryReason | None]
    memory_context: NotRequired[list[str]]
    errors: Annotated[list[WorkflowError], append_items]
    approval_status: NotRequired[str | None]
    reviewer_comments: NotRequired[str | None]
    review_timestamp: NotRequired[str | None]
    review_pending_at: NotRequired[str | None]
    review_history: NotRequired[list[dict[str, Any]]]
    created_at: datetime
    updated_at: datetime
