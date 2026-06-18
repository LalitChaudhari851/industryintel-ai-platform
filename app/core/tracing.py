"""LangSmith tracing configuration."""

from __future__ import annotations

import logging
import os

from app.core.config import Settings

logger = logging.getLogger(__name__)


def configure_langsmith(settings: Settings) -> None:
    """Configure LangSmith through environment variables used by LangChain.

    LangSmith automatically traces LangChain/LangGraph runs when tracing is
    enabled and the relevant environment variables are present.
    """
    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info("langsmith.tracing disabled")
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id

    logger.info(
        "langsmith.tracing enabled project=%s endpoint=%s",
        settings.langsmith_project,
        settings.langsmith_endpoint or "default",
    )
