"""Tests for the LLM Service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.services.llm_service import LLMService


class MockOutputSchema(BaseModel):
    name: str
    value: int


@pytest.mark.asyncio
@patch("app.services.llm_service.ChatOllama")
async def test_llm_service_initialization(mock_chat_ollama: MagicMock) -> None:
    settings = Settings(
        tavily_api_key="tvly-test",
        ollama_base_url="http://localhost:11434",
        primary_model="qwen3:8b",
        fallback_model="llama3.1:8b",
    )
    service = LLMService(settings)
    assert service.primary_model == "qwen3:8b"
    assert service.fallback_model == "llama3.1:8b"
    assert mock_chat_ollama.call_count == 2


@pytest.mark.asyncio
@patch("app.services.llm_service.ChatOllama")
async def test_llm_service_generate_text_primary(mock_chat_ollama: MagicMock) -> None:
    settings = Settings(tavily_api_key="tvly-test")
    mock_primary = MagicMock()
    mock_fallback = MagicMock()
    mock_chat_ollama.side_effect = [mock_primary, mock_fallback]

    service = LLMService(settings)
    assert service.primary_llm is mock_primary
    assert service.fallback_llm is mock_fallback

    # Mock primary LLM ainvoke success
    mock_response = MagicMock()
    mock_response.content = "Primary Response"
    mock_primary.ainvoke = AsyncMock(return_value=mock_response)

    result = await service.generate_text("Hello")
    assert result == "Primary Response"
    mock_primary.ainvoke.assert_called_once()
    assert not mock_fallback.ainvoke.called


@pytest.mark.asyncio
@patch("app.services.llm_service.ChatOllama")
async def test_llm_service_generate_text_fallback(mock_chat_ollama: MagicMock) -> None:
    settings = Settings(tavily_api_key="tvly-test")
    mock_primary = MagicMock()
    mock_fallback = MagicMock()
    mock_chat_ollama.side_effect = [mock_primary, mock_fallback]

    service = LLMService(settings)
    assert service.primary_llm is mock_primary
    assert service.fallback_llm is mock_fallback

    # Mock primary failure, fallback success
    mock_primary.ainvoke = AsyncMock(side_effect=Exception("Primary Down"))
    mock_response = MagicMock()
    mock_response.content = "Fallback Response"
    mock_fallback.ainvoke = AsyncMock(return_value=mock_response)

    result = await service.generate_text("Hello")
    assert result == "Fallback Response"
    mock_primary.ainvoke.assert_called_once()
    mock_fallback.ainvoke.assert_called_once()
