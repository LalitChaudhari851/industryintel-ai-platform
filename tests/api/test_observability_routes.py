"""Tests for the observability routes."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
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
    """Clear lru_cache for get_settings before and after every test to ensure env changes are picked up."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_observability_stats_disabled() -> None:
    """Test that when LangSmith tracing is disabled, it returns unconfigured response."""
    with patch.dict("os.environ", {"LANGSMITH_TRACING": "false", "LANGSMITH_API_KEY": ""}):
        with TestClient(app) as client:
            response = client.get("/observability/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["configured"] is False
            assert "disabled" in data["message"]
            assert data["average_latencies"] == {}
            assert data["agent_success_rates"] == {}


@patch("langsmith.Client")
def test_observability_stats_enabled_success(mock_client_class: MagicMock) -> None:
    """Test that when LangSmith tracing is enabled, it returns fetched runs data."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    start_time = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 6, 18, 12, 0, 5, tzinfo=timezone.utc)

    # We need to simulate runs representing agents and workflow parent run
    mock_agent_run = MagicMock()
    mock_agent_run.name = "PlannerAgent"
    mock_agent_run.start_time = start_time
    mock_agent_run.end_time = end_time
    mock_agent_run.error = None

    mock_workflow_run = MagicMock()
    mock_workflow_run.name = "business-research-workflow"
    mock_workflow_run.start_time = start_time
    mock_workflow_run.end_time = end_time
    mock_workflow_run.error = None
    mock_workflow_run.id = "run-id-123"
    mock_workflow_run.metadata = {
        "quality_score": 0.85,
        "source_count": 5,
        "retry_count": 1,
        "topic": "Electric Vehicles",
        "report_id": "report-123",
    }

    mock_client.list_runs.return_value = [mock_agent_run, mock_workflow_run]

    with patch.dict("os.environ", {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "lsk-testkey"}):
        with TestClient(app) as client:
            response = client.get("/observability/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["configured"] is True
            assert data["average_latencies"]["PlannerAgent"] == 5.0
            assert data["agent_success_rates"]["PlannerAgent"] == 100.0
            assert data["retry_frequency"]["1"] == 1
            assert len(data["quality_trends"]) == 1
            assert data["quality_trends"][0]["topic"] == "Electric Vehicles"
            assert data["quality_trends"][0]["quality_score"] == 0.85


@patch("langsmith.Client")
def test_observability_stats_client_exception(mock_client_class: MagicMock) -> None:
    """Test that when client raises exception, it returns configured: False."""
    mock_client_class.side_effect = Exception("LangSmith connection timeout")

    with patch.dict("os.environ", {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "lsk-testkey"}):
        with TestClient(app) as client:
            response = client.get("/observability/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["configured"] is False
            assert "LangSmith connection timeout" in data["message"]
