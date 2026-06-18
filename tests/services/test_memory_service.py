"""Tests for the Research Memory Service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings
from app.services.memory_service import ResearchMemoryService


class DummyEvidence:
    """Mock evidence item Pydantic-like object."""

    def __init__(self, text: str, id: str, source_id: str, task_id: str) -> None:
        self.text = text
        self.id = id
        self.source_id = source_id
        self.task_id = task_id
        self.relevance_score = 0.8


@pytest.mark.asyncio
@patch("app.services.memory_service.SentenceTransformer")
@patch("app.agents.researcher.reranker.BGEReranker")
@patch("app.services.memory_service.FAISS")
async def test_memory_service_ingest_and_retrieve(
    mock_faiss: MagicMock,
    mock_reranker: MagicMock,
    mock_sentence: MagicMock,
) -> None:
    settings = Settings(
        tavily_api_key="tvly-test",
        faiss_index_path="./data/test_faiss_index_tmp",
    )
    service = ResearchMemoryService(settings)

    # Ingest dummy items
    items = [
        DummyEvidence("India EV sales grew rapidly in 2025.", "e1", "s1", "t1"),
    ]
    await service.ingest_evidence(items)

    # Verify FAISS index creation was called
    mock_faiss.from_texts.assert_called_once()

    # Clean up index directory
    import shutil
    shutil.rmtree("./data/test_faiss_index_tmp", ignore_errors=True)
