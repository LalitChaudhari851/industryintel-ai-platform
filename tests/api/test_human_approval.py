"""Tests for human approval workflow endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.v1.schemas import ResearchJobStatus
from app.application.research.models import ResearchSessionRecord
from app.dependencies import get_research_service
from app.main import app


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


def test_submit_review_success() -> None:
    """Test POST /review/{report_id} endpoint successfully submits review."""
    mock_service = MagicMock()
    mock_service.resume_research = AsyncMock()
    
    # Use FastAPI dependency overrides
    app.dependency_overrides[get_research_service] = lambda: mock_service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/review/session-123",
                json={"approval_status": "approved", "reviewer_comments": "Looks good!"}
            )
            assert response.status_code == 202
            data = response.json()
            assert "Review submitted successfully" in data["message"]
            mock_service.resume_research.assert_called_once_with(
                "session-123", "approved", "Looks good!"
            )
    finally:
        app.dependency_overrides.clear()


@patch("app.workflows.business_research.build_business_research_graph")
def test_get_review_details(mock_build_graph: MagicMock) -> None:
    """Test GET /review/{report_id} successfully retrieves review metadata and draft."""
    mock_service = MagicMock()

    # Mock DB record
    record = ResearchSessionRecord(
        id="session-123",
        query="Test query",
        status=ResearchJobStatus.PENDING_REVIEW,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    mock_service.get_research = AsyncMock(return_value=record)
    app.dependency_overrides[get_research_service] = lambda: mock_service

    # Mock Graph and state snapshot
    mock_graph = MagicMock()
    mock_build_graph.return_value = mock_graph

    mock_snapshot = MagicMock()
    mock_snapshot.values = {
        "query": "Test query",
        "sources": [MagicMock(), MagicMock()],
        "analysis": MagicMock(
            summary="Draft summary test",
            findings=[],
            metrics=[],
            trends=[]
        ),
        "critic_review": MagicMock(confidence_score=0.85),
        "reviewer_comments": "Some comment",
        "review_history": [{"status": "rejected", "comments": "too brief"}]
    }
    mock_graph.aget_state = AsyncMock(return_value=mock_snapshot)

    try:
        with TestClient(app) as client:
            response = client.get("/review/session-123")
            assert response.status_code == 200
            data = response.json()
            assert data["report_id"] == "session-123"
            assert data["status"] == "pending_review"
            assert data["critic_score"] == 0.85
            assert data["source_count"] == 2
            assert data["report_draft"]["summary"] == "Draft summary test"
            assert len(data["review_history"]) == 1
            assert data["review_history"][0]["status"] == "rejected"
    finally:
        app.dependency_overrides.clear()
