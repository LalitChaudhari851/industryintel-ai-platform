"""LLM-backed report Writer agent with deterministic fallback composer."""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.agents.writer.composer import (
    approved_evidence_ids,
    approved_findings,
    approved_metrics,
    approved_trends,
    build_citations,
    compose_sections,
)
from app.agents.writer.models import WriterAgentConfig
from app.agents.writer.prompts import WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE
from app.services.llm_service import LLMService
from app.workflows.business_research.state import (
    AgentName,
    FinalReport,
    ResearchState,
    WorkflowError,
    utc_now,
)

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


# Temporary helper class to parse the LLM structured output schema
from pydantic import BaseModel, Field

class LLMReportOutput(BaseModel):
    title: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    sections: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class WriterAgent(BaseAgent):
    """LangGraph-compatible Writer node combining LLM generation and deterministic fallback composition."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        config: WriterAgentConfig | None = None,
    ) -> None:
        self.config = config or WriterAgentConfig()
        self.use_llm = llm_service is not None

        if llm_service is not None:
            super().__init__(llm_service, AgentName.WRITER)
        else:
            self.llm = None  # type: ignore
            self.name = AgentName.WRITER

    @traceable(name="WriterAgent", run_type="chain")
    async def __call__(self, state: ResearchState) -> ResearchState:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({
                "topic": state.get("query"),
                "report_id": state.get("session_id"),
            })
        analysis = state.get("analysis")
        if analysis is None:
            return {
                "errors": [
                    WorkflowError(
                        agent=AgentName.WRITER,
                        message="analysis result is required before writing a report",
                        recoverable=False,
                    )
                ],
                "updated_at": utc_now(),
            }

        critic_review = state.get("critic_review")
        sources = list(state.get("sources", []))
        evidence_by_id = {item.id: item for item in state.get("evidence", [])}

        # Select approved analytical details
        findings = approved_findings(analysis, critic_review, config=self.config)
        evidence_ids = approved_evidence_ids(findings)
        metrics = approved_metrics(analysis, evidence_ids, config=self.config)
        trends = approved_trends(analysis, evidence_ids, config=self.config)
        citations = build_citations(findings, metrics, trends, evidence_by_id)

        confidence = critic_review.confidence_score if critic_review else analysis.confidence_score

        # Attempt LLM generation if enabled
        llm_report = None
        if self.use_llm:
            llm_report = await self._generate_with_llm(
                state, findings, metrics, trends, citations, critic_review, confidence
            )

        if llm_report is not None:
            report = llm_report
        else:
            report = self._compose_deterministically(
                state, analysis, critic_review, findings, metrics, trends, citations, sources, evidence_by_id, confidence
            )

        # Log final quality score and source counts to LangSmith metadata
        run_tree = get_current_run_tree()
        if run_tree and report:
            run_tree.add_metadata({
                "quality_score": float(report.confidence_score),
                "source_count": len(sources),
            })

        return {
            "final_report": report,
            "updated_at": utc_now(),
        }

    async def _generate_with_llm(
        self,
        state: ResearchState,
        findings: list[Any],
        metrics: list[Any],
        trends: list[Any],
        citations: tuple[Any, ...],
        critic_review: Any,
        confidence: float,
    ) -> FinalReport | None:
        try:
            # Build citation-source mapping for LLM reference
            citations_mapping = [
                {
                    "label": cit.citation_label if hasattr(cit, "citation_label") else f"[S{i+1}]",
                    "claim": cit.claim,
                    "source_id": cit.source_id,
                }
                for i, cit in enumerate(citations)
            ]

            prompt = WRITER_USER_TEMPLATE.format(
                query=state["query"],
                findings=json_dumps_pydantic(findings),
                metrics=json_dumps_pydantic(metrics),
                trends=json_dumps_pydantic(trends),
                citations_mapping=json_dumps_pydantic(citations_mapping),
                critic_notes=critic_review.notes if critic_review else "None",
            )

            logger.info("WriterAgent generating report with LLM...")
            parsed_output = await self.invoke_llm_structured(
                prompt, LLMReportOutput, system_prompt=WRITER_SYSTEM_PROMPT
            )

            # Compute word count of LLM output
            total_words = len(parsed_output.executive_summary.split()) + sum(
                len(text.split()) for text in parsed_output.sections.values()
            )

            logger.info("WriterAgent successfully generated report with LLM. Word count: %d", total_words)

            return FinalReport(
                title=parsed_output.title,
                executive_summary=parsed_output.executive_summary,
                sections=parsed_output.sections,
                citations=citations,
                limitations=tuple(parsed_output.limitations),
                confidence_score=confidence,
                word_count=total_words,
            )

        except Exception as e:
            logger.error("WriterAgent LLM generation failed: %s. Falling back to deterministic composer.", e)
            return None

    def _compose_deterministically(
        self,
        state: ResearchState,
        analysis: Any,
        critic_review: Any,
        findings: list[Any],
        metrics: list[Any],
        trends: list[Any],
        citations: tuple[Any, ...],
        sources: list[Any],
        evidence_by_id: dict[str, Any],
        confidence: float,
    ) -> FinalReport:
        logger.info("WriterAgent running deterministic report composer fallback...")
        executive_summary, sections, limitations, total_words = compose_sections(
            query=state["query"],
            analysis=analysis,
            critic_review=critic_review,
            findings=findings,
            metrics=metrics,
            trends=trends,
            citations=citations,
            sources=sources,
            evidence_by_id=evidence_by_id,
            config=self.config,
        )

        return FinalReport(
            title=f"Business Research Report: {state['query']}",
            executive_summary=executive_summary,
            sections=sections,
            citations=citations,
            limitations=limitations,
            confidence_score=confidence,
            word_count=total_words,
        )


def json_dumps_pydantic(obj: Any) -> str:
    """Helper to dump list of pydantic models or primitives into JSON."""
    try:
        if isinstance(obj, list):
            return json.dumps([item.model_dump() if hasattr(item, "model_dump") else item for item in obj], indent=2)
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)
