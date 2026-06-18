"""Tests for the PDF export API endpoint."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.v1.schemas import ResearchJobStatus
from app.application.research.models import ResearchSessionRecord
from app.dependencies import get_research_service
from app.main import app
from app.workflows.business_research.state import FinalReport


@pytest.fixture(autouse=True)
def mock_embeddings_and_reranker():
    """Mock SentenceTransformer and BGEReranker to avoid downloading weights or running heavy local models during API tests."""
    with patch("app.services.memory_service.SentenceTransformer") as mock_st, \
         patch("app.agents.researcher.reranker.BGEReranker") as mock_rr, \
         patch("app.services.memory_service.FAISS") as mock_faiss:
        yield mock_st, mock_rr, mock_faiss


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache to prevent tracing env overrides from leaking."""
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_export_pdf_endpoint_success() -> None:
    """Test GET /research/{research_id}/pdf endpoint returns a streaming PDF file."""
    mock_service = MagicMock()
    app.dependency_overrides[get_research_service] = lambda: mock_service

    # Mock DB record and report
    report = FinalReport(
        title="Dynamic Market Research Briefing",
        executive_summary="This is a summary of the strategic intelligence briefing.",
        sections={
            "Market Dynamics": "High market growth observed in electric drivetrains.",
        },
        citations=(),
        limitations=(),
        confidence_score=0.9,
        word_count=50,
    )
    record = ResearchSessionRecord(
        id="session-123",
        query="Test query",
        status=ResearchJobStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        report=report,
        raw_state={"sources": []},
    )
    mock_service.get_completed_report = AsyncMock(return_value=record)

    try:
        with TestClient(app) as client:
            response = client.get("/research/session-123/pdf")
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert "attachment" in response.headers["content-disposition"]
            assert response.content.startswith(b"%PDF")
    finally:
        app.dependency_overrides.clear()
