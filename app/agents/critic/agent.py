"""LLM-backed Critic agent with deterministic verification fallback."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agents.base import BaseAgent
from app.agents.critic.evaluation import (
    assess_findings,
    critic_findings,
    decision_for_quality,
    detect_contradictions,
    missing_evidence_items,
    quality_score,
)
from app.agents.critic.models import CriticAgentConfig
from app.agents.critic.prompts import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE
from app.services.llm_service import LLMService
from app.workflows.business_research.state import (
    AgentName,
    CriticReview,
    QualityDecision,
    ResearchState,
    RetryReason,
    WorkflowError,
    utc_now,
)

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """LangGraph-compatible Critic node combining LLM critique and deterministic checks."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        config: CriticAgentConfig | None = None,
    ) -> None:
        self.config = config or CriticAgentConfig()
        self.use_llm = llm_service is not None

        if llm_service is not None:
            super().__init__(llm_service, AgentName.CRITIC)
        else:
            self.llm = None  # type: ignore
            self.name = AgentName.CRITIC

    @traceable(name="CriticAgent", run_type="chain")
    async def __call__(self, state: ResearchState) -> ResearchState:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({
                "topic": state.get("query"),
                "report_id": state.get("session_id"),
            })
        analysis = state.get("analysis")
        evidence_items = list(state.get("evidence", []))
        sources = list(state.get("sources", []))

        if analysis is None:
            return {
                "critic_review": CriticReview(
                    decision=QualityDecision.RETRY_RESEARCH,
                    confidence_score=0.0,
                    findings=(
                        {
                            "issue": "Analysis result is missing.",
                            "severity": "critical",
                            "retry_reason": RetryReason.ANALYSIS_DOES_NOT_ANSWER_QUERY,
                        },
                    ),
                    missing_evidence=("analysis result",),
                    notes="Critic cannot verify absent analysis.",
                ),
                "errors": [
                    WorkflowError(
                        agent=AgentName.CRITIC,
                        message="analysis result is missing",
                        recoverable=True,
                    )
                ],
                "updated_at": utc_now(),
            }

        llm_result = await self._critique_with_llm(state, analysis, evidence_items)
        review = llm_result or self._critique_deterministically(analysis, evidence_items, sources)

        # Enforce retry budget limit and fallback logic in route decision if retry is requested
        iteration = state.get("iteration_count", 1)
        max_iterations = state.get("max_iterations", 3)

        if review.decision == QualityDecision.RETRY_RESEARCH and iteration >= max_iterations:
            logger.info(
                "CriticAgent downgrading retry decision to write_with_limitations due to iteration budget limits."
            )
            review = CriticReview(
                decision=QualityDecision.WRITE_WITH_LIMITATIONS,
                confidence_score=review.confidence_score,
                findings=review.findings,
                evidence_assessments=review.evidence_assessments,
                contradictions=review.contradictions,
                missing_evidence=review.missing_evidence,
                notes=f"Budget exceeded. Downgraded retry. Original notes: {review.notes}",
            )

        # Log final quality score and retry details to LangSmith metadata
        run_tree = get_current_run_tree()
        if run_tree and review:
            run_tree.add_metadata({
                "quality_score": float(review.confidence_score),
                "retry_count": int(state.get("retry_count", 0)),
                "iteration_count": int(iteration),
            })

        return {
            "critic_review": review,
            "updated_at": utc_now(),
        }

    async def _critique_with_llm(
        self,
        state: ResearchState,
        analysis: Any,
        evidence_items: list[Any],
    ) -> CriticReview | None:
        if not self.use_llm:
            return None

        # Build payload of evidence for critic context
        evidence_payload = [
            {
                "id": item.id,
                "text": item.text[:1000],  # Truncate to save context
                "source_id": item.source_id,
            }
            for item in evidence_items
        ]

        prompt = CRITIC_USER_TEMPLATE.format(
            query=state["query"],
            analysis=analysis.model_dump_json(indent=2),
            evidence=json.dumps(evidence_payload, ensure_ascii=True),
        )

        try:
            logger.info("CriticAgent performing structured review with LLM...")
            review = await self.invoke_llm_structured(
                prompt, CriticReview, system_prompt=CRITIC_SYSTEM_PROMPT
            )
            logger.info(
                "CriticAgent completed structured review with LLM. Decision: %s, Score: %.2f",
                review.decision,
                review.confidence_score,
            )
            return review
        except Exception as e:
            logger.error("CriticAgent LLM critique failed: %s. Falling back to deterministic review.", e)
            return None

    def _critique_deterministically(
        self,
        analysis: Any,
        evidence_items: list[Any],
        sources: list[Any],
    ) -> CriticReview:
        logger.info("CriticAgent running deterministic verification fallback...")
        evidence_by_id_map = {item.id: item for item in evidence_items}
        assessments = assess_findings(
            analysis,
            evidence_by_id_map,
            config=self.config,
        )
        contradictions = detect_contradictions(analysis, evidence_items)
        quality = quality_score(
            analysis=analysis,
            evidence_assessments=assessments,
            contradictions=contradictions,
            sources=sources,
            evidence_items=evidence_items,
            config=self.config,
        )
        findings = critic_findings(
            quality=quality,
            evidence_assessments=assessments,
            contradictions=contradictions,
            source_count=len({item.source_id for item in evidence_items}),
            config=self.config,
        )

        return CriticReview(
            decision=decision_for_quality(quality, self.config),
            confidence_score=quality,
            findings=findings,
            evidence_assessments=assessments,
            contradictions=contradictions,
            missing_evidence=missing_evidence_items(assessments),
            notes=(
                "Quality threshold passed."
                if quality >= self.config.quality_threshold
                else "Quality below threshold; retry research."
            ),
        )
