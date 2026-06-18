"""LangChain retriever tool wrapping the FAISS Memory Service."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.services.memory_service import ResearchMemoryService

logger = logging.getLogger(__name__)


def create_faiss_retriever_tool(memory_service: ResearchMemoryService) -> BaseTool:
    """Creates a LangChain tool for querying the research memory index."""

    @tool("search_memory")
    async def search_memory(query: str) -> str:
        """Query the internal memory store to find past research details, claims, metrics, and evidence.

        Use this tool to retrieve relevant prior context before analyzing or reviewing.
        """
        try:
            results = await memory_service.retrieve_relevant(query)
            if not results:
                return "No relevant past evidence found in memory."

            formatted = []
            for i, item in enumerate(results, 1):
                text = item.get("text", "")
                score = item.get("rerank_score", item.get("score", 0.0))
                meta = item.get("metadata", {})
                source_id = meta.get("source_id", "unknown")
                formatted.append(
                    f"Result {i} (Relevance Score: {score:.3f}, Source ID: {source_id}):\n{text}"
                )
            return "\n\n".join(formatted)
        except Exception as e:
            logger.error("Error retrieving from memory tool: %s", e)
            return f"Error retrieving from memory: {e}"

    return search_memory
