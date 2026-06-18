"""Pydantic models for the Writer agent."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class WriterAgentConfig(StrictModel):
    min_words: int = Field(default=1000, ge=500, le=3000)
    max_words: int = Field(default=1500, ge=800, le=4000)
    max_findings: int = Field(default=8, ge=1, le=20)
    max_metrics: int = Field(default=10, ge=1, le=30)
    max_trends: int = Field(default=6, ge=1, le=20)
    max_references: int = Field(default=12, ge=1, le=50)

    @model_validator(mode="after")
    def validate_word_window(self) -> "WriterAgentConfig":
        if self.min_words > self.max_words:
            raise ValueError("min_words must be less than or equal to max_words")
        return self
