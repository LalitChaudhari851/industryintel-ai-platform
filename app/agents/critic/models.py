"""Pydantic models for the Critic agent."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class CriticAgentConfig(StrictModel):
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_sources: int = Field(default=2, ge=1, le=20)
    min_supported_claim_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    min_token_overlap: float = Field(default=0.18, ge=0.0, le=1.0)
