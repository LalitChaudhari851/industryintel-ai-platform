"""LLM-as-judge accuracy evaluator using local Ollama model."""

from __future__ import annotations

import logging
from typing import Any, Dict

from pydantic import BaseModel, Field

from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class AccuracyRating(BaseModel):
    """Pydantic schema for the LLM judge output."""

    accuracy_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Factual accuracy score between 0.0 and 1.0 (where 1.0 is perfectly accurate with no unsupported claims).",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Factual claims in the report that are not supported by the citations or evidence.",
    )
    rationale: str = Field(
        min_length=10,
        description="Detailed explanation of the rating and findings.",
    )


JUDGE_SYSTEM_PROMPT = """You are an independent quality auditor. Your task is to evaluate a generated business intelligence report against the citations and grounding evidence to judge its factual accuracy and identify any hallucinated or unsupported claims.

You must rate accuracy on a scale from 0.0 (completely hallucinated/unsupported) to 1.0 (perfectly accurate and grounded).
Provide a list of any claims made in the report that have no supporting details in the citations or evidence, and write a thorough rationale.
"""

JUDGE_USER_TEMPLATE = """Report Title:
"{title}"

Executive Summary:
{executive_summary}

Report Content:
{content_dump}

Citations Grounding:
{citations}

Critically assess this report and verify if it matches the grounding citations.
"""


class LLMAccuracyJudge:
    """Judge that uses Qwen3 via LLMService to assess factual grounding accuracy."""

    def __init__(self, llm_service: LLMService) -> None:
        self.llm = llm_service

    async def evaluate_report(self, report: Dict[str, Any]) -> AccuracyRating:
        """Run LLM accuracy analysis on a generated report."""
        title = report.get("title", "")
        summary = report.get("executive_summary", "")
        sections = report.get("sections", {})
        citations = report.get("citations", [])

        # Build content dump string
        content_dump = "\n\n".join(f"## {t}\n{c}" for t, c in sections.items())

        # Build citations printout
        citations_dump = "\n".join(
            f"- Claim: {c.get('claim', '')} | Supporting Evidence: \"{c.get('supporting_text', '')}\""
            for c in citations
        )

        prompt = JUDGE_USER_TEMPLATE.format(
            title=title,
            executive_summary=summary,
            content_dump=content_dump,
            citations=citations_dump,
        )

        try:
            logger.info("LLMAccuracyJudge starting report audit...")
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema=AccuracyRating,
                system_prompt=JUDGE_SYSTEM_PROMPT,
            )
            logger.info("LLMAccuracyJudge completed audit with score: %.2f", result.accuracy_score)
            return result
        except Exception as e:
            logger.error("LLMAccuracyJudge failed to audit report: %s. Returning default score.", e)
            return AccuracyRating(
                accuracy_score=0.75,
                unsupported_claims=["Error: judge failed to complete evaluation run"],
                rationale=f"Auditing encountered exception: {e}",
            )
