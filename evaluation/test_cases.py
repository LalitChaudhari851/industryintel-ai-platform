"""Golden test cases for platform evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenTestCase(BaseModel):
    id: str
    query: str
    business_context: str | None = None
    expected_topics: tuple[str, ...] = Field(default_factory=tuple)


TEST_CASES = [
    GoldenTestCase(
        id="TC-001",
        query="Compare OpenAI vs Anthropic competitive strategy in 2026",
        business_context="Evaluating model costs, developer API lock-in, and raw performance.",
        expected_topics=("OpenAI", "Anthropic", "API", "pricing", "pricing comparison"),
    ),
    GoldenTestCase(
        id="TC-002",
        query="Analyze NVIDIA Blackwell GPU supply chain bottlenecks",
        business_context="Focus on TSMC CoWoS capacity, advanced packaging constraints, and HBM3e yields.",
        expected_topics=("NVIDIA", "Blackwell", "TSMC", "CoWoS", "packaging", "HBM3e"),
    ),
    GoldenTestCase(
        id="TC-003",
        query="Evaluate the impact of EU AI Act on open-source LLMs",
        business_context="Focus on systemic risk compliance, copyright disclosures, and fine-tuning liabilities.",
        expected_topics=("EU AI Act", "open-source", "LLM", "compliance", "copyright", "systemic risk"),
    ),
]
