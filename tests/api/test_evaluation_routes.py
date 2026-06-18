"""Tests for the evaluation API routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.v1.evaluation_routes import get_evaluation_service
from app.main import app


@pytest.fixture(autouse=True)
def mock_embeddings_and_reranker():
    """Mock heavy local model initializations to speed up test startup."""
    with patch("app.services.memory_service.SentenceTransformer") as mock_st, \
         patch("app.agents.researcher.reranker.BGEReranker") as mock_rr, \
         patch("app.services.memory_service.FAISS") as mock_faiss:
        yield mock_st, mock_rr, mock_faiss


def test_get_evaluation_metrics_success() -> None:
    """Verify GET /evaluation/metrics returns status 200 and metric contents."""
    mock_service = MagicMock()
    mock_service.get_aggregated_metrics = AsyncMock(return_value={
        "research_quality": {"source_count": 5.0},
        "agent_metrics": {"agent_success_rate": 95.0},
        "report_metrics": {"quality_score": 88.0},
        "langsmith_metrics": {"configured": False}
    })
    
    app.dependency_overrides[get_evaluation_service] = lambda: mock_service

    try:
        with TestClient(app) as client:
            response = client.get("/evaluation/metrics")
            assert response.status_code == 200
            data = response.json()
            assert "research_quality" in data
            assert data["research_quality"]["source_count"] == 5.0
            assert data["agent_metrics"]["agent_success_rate"] == 95.0
    finally:
        app.dependency_overrides.clear()


def test_get_evaluation_reports_success() -> None:
    """Verify GET /evaluation/reports returns list of report metrics."""
    mock_service = MagicMock()
    mock_service.get_reports = AsyncMock(return_value=[
        {
            "id": "session-abc",
            "query": "Test Topic",
            "status": "completed",
            "quality_score": 90.0,
            "duration": 22.5
        }
    ])
    
    app.dependency_overrides[get_evaluation_service] = lambda: mock_service

    try:
        with TestClient(app) as client:
            response = client.get("/evaluation/reports")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["id"] == "session-abc"
            assert data[0]["query"] == "Test Topic"
    finally:
        app.dependency_overrides.clear()


def test_get_evaluation_trends_success() -> None:
    """Verify GET /evaluation/trends returns trend arrays."""
    mock_service = MagicMock()
    mock_service.get_trends = AsyncMock(return_value=[
        {
            "timestamp": "2026-06-18T12:00:00",
            "quality_score": 85.0,
            "latency": 24.5
        }
    ])
    
    app.dependency_overrides[get_evaluation_service] = lambda: mock_service

    try:
        with TestClient(app) as client:
            response = client.get("/evaluation/trends")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["quality_score"] == 85.0
    finally:
        app.dependency_overrides.clear()
