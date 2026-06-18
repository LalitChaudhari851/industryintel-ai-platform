"""Base Agent class for Ollama-powered agents in the platform."""

from __future__ import annotations

import logging
from typing import Type, TypeVar

from pydantic import BaseModel

from app.services.llm_service import LLMService
from app.workflows.business_research.state import AgentName

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    """Base Agent class providing common LLM operations with fallback support."""

    def __init__(self, llm_service: LLMService, name: AgentName) -> None:
        self.llm = llm_service
        self.name = name

    async def invoke_llm_structured(self, prompt: str, schema: Type[T], system_prompt: str | None = None) -> T:
        """Invoke LLM with a schema to get structured output, falling back if necessary."""
        try:
            return await self.llm.generate_structured(prompt, schema, system_prompt=system_prompt)
        except Exception as e:
            logger.warning(
                "Agent %s failed structured output generation on primary model: %s. Trying fallback.",
                self.name,
                e,
            )
            return await self.llm.generate_structured(
                prompt, schema, system_prompt=system_prompt, use_fallback=True
            )

    async def invoke_llm_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """Invoke LLM to get raw text output, falling back if necessary."""
        try:
            return await self.llm.generate_text(prompt, system_prompt=system_prompt)
        except Exception as e:
            logger.warning(
                "Agent %s failed text generation on primary model: %s. Trying fallback.",
                self.name,
                e,
            )
            return await self.llm.generate_text(prompt, system_prompt=system_prompt, use_fallback=True)
