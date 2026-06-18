"""Pydantic models for the Analyst agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class AnalystAgentConfig(StrictModel):
    max_findings: int = Field(default=8, ge=1, le=25)
    max_metrics: int = Field(default=12, ge=1, le=50)
    max_trends: int = Field(default=8, ge=1, le=25)
    min_sentence_chars: int = Field(default=35, ge=10, le=300)
    min_finding_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class AnalystLLMInput(StrictModel):
    query: str
    evidence: tuple[dict[str, Any], ...]
    max_findings: int
    max_metrics: int
    max_trends: int
