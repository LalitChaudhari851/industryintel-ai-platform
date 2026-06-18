"""Shared pytest fixtures for testing the AI Industry Intelligence Platform."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.services.llm_service import LLMService
from app.services.memory_service import ResearchMemoryService


@pytest.fixture
def mock_settings() -> Settings:
    """Mock Settings fixture."""
    return Settings(
        tavily_api_key="tvly-mock-key",
        ollama_base_url="http://localhost:11434",
        primary_model="qwen3:8b",
        fallback_model="llama3.1:8b",
        embedding_model="BAAI/bge-base-en-v1.5",
        reranker_model="BAAI/bge-reranker-base",
        faiss_index_path="./data/test_faiss_index",
    )


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Mock LLMService fixture."""
    mock = MagicMock(spec=LLMService)
    mock.check_health.return_value = {
        "primary": True,
        "fallback": True,
        "ollama_connected": True,
        "primary_model": "qwen3:8b",
        "fallback_model": "llama3.1:8b",
    }
    return mock


@pytest.fixture
def mock_memory_service() -> MagicMock:
    """Mock ResearchMemoryService fixture."""
    mock = MagicMock(spec=ResearchMemoryService)
    return mock
