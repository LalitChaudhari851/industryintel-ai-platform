"""Tests for the Evaluation Service."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from app.api.v1.schemas import ResearchJobStatus
from app.application.research.models import ResearchSessionRecord
from app.application.research.store import InMemoryResearchStore
from app.core.config import get_settings
from app.services.evaluation_service import EvaluationService
from app.workflows.business_research.state import Citation, FinalReport, CriticReview, QualityDecision


@pytest.fixture
def store() -> InMemoryResearchStore:
    return InMemoryResearchStore()


@pytest.fixture
def settings() -> Any:
    return get_settings()


@pytest.fixture
def service(store: InMemoryResearchStore, settings: Any) -> EvaluationService:
    return EvaluationService(store, settings)


@pytest.mark.asyncio
async def test_evaluation_service_empty_store_fallback(service: EvaluationService) -> None:
    """Verify that EvaluationService defaults to simulated evaluation data when store is empty."""
    metrics = await service.get_aggregated_metrics()
    assert isinstance(metrics, dict)
    assert "research_quality" in metrics
    assert "agent_metrics" in metrics
    assert "report_metrics" in metrics
    assert metrics["research_quality"]["source_count"] == 8.4

    reports = await service.get_reports()
    assert len(reports) == 8
    assert reports[0]["query"] == "Quantum Computing Commercialization"

    trends = await service.get_trends()
    assert len(trends) == 10
    assert "quality_score" in trends[0]


@pytest.mark.asyncio
async def test_evaluation_service_calculated_metrics(service: EvaluationService, store: InMemoryResearchStore) -> None:
    """Verify that EvaluationService computes metrics dynamically when at least 3 completed records exist."""
    # Create 3 completed sessions
    for idx in range(3):
        report = FinalReport(
            title=f"Briefing {idx}",
            executive_summary="Summary text...",
            sections={"Key Findings": "Findings detail..."},
            citations=(Citation(claim="claim", source_id=f"src-{idx}", confidence=0.85),),
            limitations=(),
            confidence_score=0.9,
            word_count=50,
        )
        critic = CriticReview(
            decision=QualityDecision.PASS,
            confidence_score=0.92,
            findings=(),
            evidence_assessments=(),
            contradictions=(),
            missing_evidence=(),
        )
        record = ResearchSessionRecord(
            id=f"session-{idx}",
            query=f"Query {idx}",
            status=ResearchJobStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            report=report,
            critic_review=critic,
            raw_state={
                "sources": [
                    {"title": "Source 1", "url": f"https://domain{idx}.com/path", "credibility_score": 0.9},
                    {"title": "Source 2", "url": f"https://domain{idx}.com/other", "credibility_score": 0.8},
                ],
                "iteration_count": 1,
                "review_history": [{"status": "approved"}]
            },
        )
        await store.create(record)

    metrics = await service.get_aggregated_metrics()
    assert metrics["research_quality"]["source_count"] == 2.0
    # Source diversity is unique domains per run: domain{idx}.com (1 unique domain)
    assert metrics["research_quality"]["source_diversity"] == 1.0
    assert metrics["report_metrics"]["quality_score"] == 92.0
    assert metrics["report_metrics"]["confidence_score"] == 90.0

    reports = await service.get_reports()
    assert len(reports) == 3
    assert reports[0]["query"] == "Query 0"
    assert reports[0]["source_diversity"] == 1

    trends = await service.get_trends()
    assert len(trends) == 3
    assert trends[0]["quality_score"] == 92.0
