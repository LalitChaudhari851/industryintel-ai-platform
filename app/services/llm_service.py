"""LLM Service wrapper using Ollama ChatOllama and automatic fallback."""

from __future__ import annotations

import logging
from typing import Type, TypeVar

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.core.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Ollama LLM Service with primary/fallback routing and structured output parsing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.primary_model = settings.primary_model
        self.fallback_model = settings.fallback_model
        self.base_url = settings.ollama_base_url

        self.primary_llm = ChatOllama(
            base_url=self.base_url,
            model=self.primary_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )
        self.fallback_llm = ChatOllama(
            base_url=self.base_url,
            model=self.fallback_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )

    async def generate_text(self, prompt: str, system_prompt: str | None = None, use_fallback: bool = False) -> str:
        """Generate plain text from LLM with automatic fallback."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        if use_fallback:
            logger.info("Forced use of fallback LLM model: %s", self.fallback_model)
            response = await self.fallback_llm.ainvoke(messages)
            return str(response.content)

        try:
            logger.info("Calling primary LLM for text: %s", self.primary_model)
            response = await self.primary_llm.ainvoke(messages)
            return str(response.content)
        except Exception as e:
            logger.warning("Primary LLM text generation failed: %s. Falling back to %s", e, self.fallback_model)
            try:
                response = await self.fallback_llm.ainvoke(messages)
                return str(response.content)
            except Exception as fe:
                logger.error("Fallback LLM text generation failed: %s", fe)
                raise fe

    async def generate_structured(
        self, prompt: str, schema: Type[T], system_prompt: str | None = None, use_fallback: bool = False
    ) -> T:
        """Generate structured output validated by a Pydantic model with automatic fallback."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        if use_fallback:
            logger.info("Forced use of fallback structured LLM model: %s", self.fallback_model)
            structured_llm = self.fallback_llm.with_structured_output(schema)
            return await structured_llm.ainvoke(messages)

        try:
            logger.info("Calling primary LLM for structured: %s -> %s", self.primary_model, schema.__name__)
            structured_llm = self.primary_llm.with_structured_output(schema)
            return await structured_llm.ainvoke(messages)
        except Exception as e:
            logger.warning("Primary LLM structured output failed: %s. Falling back to %s", e, self.fallback_model)
            try:
                structured_llm = self.fallback_llm.with_structured_output(schema)
                return await structured_llm.ainvoke(messages)
            except Exception as fe:
                logger.error("Fallback LLM structured output failed: %s", fe)
                raise fe

    async def check_health(self) -> dict[str, bool | str]:
        """Check status of Ollama server and configured models."""
        url = f"{self.base_url.rstrip('/')}/api/tags"
        health = {
            "primary": False,
            "fallback": False,
            "ollama_connected": False,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    health["ollama_connected"] = True
                    models = [m["name"] for m in res.json().get("models", [])]
                    # Check tags matching model names
                    health["primary"] = any(self.primary_model in m or m in self.primary_model for m in models)
                    health["fallback"] = any(self.fallback_model in m or m in self.fallback_model for m in models)
        except Exception as e:
            logger.warning("Ollama health check connection failed: %s", e)
        return health
