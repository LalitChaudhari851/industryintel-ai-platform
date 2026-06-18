"""Configuration models for the Planner Agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlannerAgentConfig(BaseModel):
    """Configuration settings for the Planner agent node."""

    max_tasks: int = Field(default=5, ge=1, le=10)
    use_llm: bool = Field(default=True)
