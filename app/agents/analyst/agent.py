"""LLM-backed business analyst agent with deterministic fallback."""

from __future__ import annotations

import json
import logging
from typing import Any, List

from app.agents.analyst.extraction import (
    confidence_score,
    evidence_by_id,
    extract_findings,
    extract_metrics,
    extract_opportunities,
    extract_risks,
    extract_trends,
    recommendations_from_findings,
    source_by_id,
    summarize,
)
from app.agents.analyst.models import AnalystAgentConfig
from app.agents.analyst.prompts import ANALYST_SYSTEM_PROMPT, ANALYST_USER_TEMPLATE
from app.agents.base import BaseAgent
from app.services.llm_service import LLMService
from app.services.memory_service import ResearchMemoryService
from app.workflows.business_research.state import (
    AgentName,
    AnalysisResult,
    EvidenceItem,
    ResearchState,
    WorkflowError,
    utc_now,
)

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """LangGraph-compatible Analyst node combining LLM synthesis and deterministic extraction fallback."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        config: AnalystAgentConfig | None = None,
        memory_service: ResearchMemoryService | None = None,
    ) -> None:
        self.config = config or AnalystAgentConfig()
        self.memory_service = memory_service
        self.use_llm = llm_service is not None

        if llm_service is not None:
            super().__init__(llm_service, AgentName.ANALYST)
        else:
            self.llm = None  # type: ignore
            self.name = AgentName.ANALYST

    @traceable(name="AnalystAgent", run_type="chain")
    async def __call__(self, state: ResearchState) -> ResearchState:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({
                "topic": state.get("query"),
                "report_id": state.get("session_id"),
            })
        evidence_items = list(state.get("evidence", []))
        sources = source_by_id(state.get("sources", []))

        # Retrieve relevant past memory context before analysis if memory_service is available
        memory_context: List[str] = []
        if self.memory_service is not None:
            try:
                logger.info("AnalystAgent retrieving past research context from FAISS...")
                results = await self.memory_service.retrieve_relevant(state["query"])
                memory_context = [item["text"] for item in results]
            except Exception as e:
                logger.error("AnalystAgent failed to retrieve context from FAISS: %s", e)

        llm_result = await self._analyze_with_llm(state, evidence_items, memory_context)
        analysis = llm_result or self._analyze_deterministically(state, evidence_items, sources)

        update_data: ResearchState = {
            "analysis": analysis,
            "updated_at": utc_now(),
        }
        if memory_context:
            update_data["memory_context"] = memory_context

        # Log quality score to LangSmith metadata
        run_tree = get_current_run_tree()
        if run_tree and analysis:
            run_tree.add_metadata({"quality_score": float(analysis.confidence_score)})

        return update_data

    async def _analyze_with_llm(
        self,
        state: ResearchState,
        evidence_items: list[EvidenceItem],
        memory_context: List[str],
    ) -> AnalysisResult | None:
        if not self.use_llm:
            return None

        # Build raw evidence payload with IDs
        evidence_payload = [
            {
                "id": item.id,
                "source_id": item.source_id,
                "task_id": item.task_id,
                "relevance_score": item.relevance_score,
                "text": item.text[:1500],  # Truncate per item to manage context window
            }
            for item in evidence_items
        ]

        prompt = ANALYST_USER_TEMPLATE.format(
            query=state["query"],
            evidence=json.dumps(evidence_payload, ensure_ascii=True),
            memory_context=json.dumps(memory_context, ensure_ascii=True) if memory_context else "None",
        )

        try:
            logger.info("AnalystAgent performing structured analysis with LLM...")
            analysis = await self.invoke_llm_structured(
                prompt, AnalysisResult, system_prompt=ANALYST_SYSTEM_PROMPT
            )
            logger.info(
                "AnalystAgent successfully completed structured analysis with LLM (confidence: %.2f)",
                analysis.confidence_score,
            )
            return analysis
        except Exception as e:
            logger.error("AnalystAgent LLM analysis failed: %s. Falling back to deterministic extraction.", e)
            return None

    def _analyze_deterministically(
        self,
        state: ResearchState,
        evidence_items: list[EvidenceItem],
        sources: dict[str, Any],
    ) -> AnalysisResult:
        logger.info("AnalystAgent running deterministic extraction fallback...")
        findings = extract_findings(
            evidence_items,
            sources,
            max_findings=self.config.max_findings,
            min_sentence_chars=self.config.min_sentence_chars,
            min_confidence=self.config.min_finding_confidence,
        )
        metrics = extract_metrics(
            evidence_items,
            sources,
            max_metrics=self.config.max_metrics,
        )
        trends = extract_trends(
            evidence_items,
            sources,
            max_trends=self.config.max_trends,
        )
        confidence = confidence_score(findings, metrics, trends, evidence_items, sources)

        if not evidence_by_id(evidence_items):
            return AnalysisResult(
                summary=f"{state['query']}: no evidence was available for analysis.",
                confidence_score=0.0,
            )

        return AnalysisResult(
            summary=summarize(state["query"], findings, metrics),
            findings=findings,
            metrics=metrics,
            trends=trends,
            risks=extract_risks(evidence_items),
            opportunities=extract_opportunities(evidence_items),
            recommendations=recommendations_from_findings(findings),
            confidence_score=confidence,
        )
