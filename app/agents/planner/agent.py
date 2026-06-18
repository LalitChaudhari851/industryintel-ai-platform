"""LLM-backed research planner with deterministic fallback."""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.agents.planner.models import PlannerAgentConfig
from app.agents.planner.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from app.services.llm_service import LLMService
from app.workflows.business_research.state import (
    AgentName,
    ResearchPlan,
    ResearchState,
    ResearchTask,
    TaskType,
    utc_now,
)

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Generates structured research plans from user queries using LLM or deterministic rules."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        config: PlannerAgentConfig | None = None,
    ) -> None:
        self.config = config or PlannerAgentConfig()
        self.use_llm = llm_service is not None and self.config.use_llm

        if llm_service is not None:
            super().__init__(llm_service, AgentName.PLANNER)
        else:
            self.llm = None  # type: ignore
            self.name = AgentName.PLANNER

    @traceable(name="PlannerAgent", run_type="chain")
    async def __call__(self, state: ResearchState) -> ResearchState:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({
                "topic": state.get("query"),
                "report_id": state.get("session_id"),
            })
        query = state["query"]
        context = state.get("business_context") or "None provided"

        if not self.use_llm:
            logger.info("PlannerAgent running in deterministic mode.")
            return self._run_deterministic(query)

        prompt = PLANNER_USER_TEMPLATE.format(query=query, context=context)
        try:
            logger.info("PlannerAgent generating research plan using LLM...")
            plan = await self.invoke_llm_structured(
                prompt, ResearchPlan, system_prompt=PLANNER_SYSTEM_PROMPT
            )
            # Enforce max tasks constraint from config
            tasks = list(plan.tasks[: self.config.max_tasks])
            logger.info("PlannerAgent dynamically created %d research tasks", len(tasks))

            return {
                "plan": ResearchPlan(
                    objective=plan.objective,
                    tasks=tuple(tasks),
                    expected_output=plan.expected_output,
                    success_criteria=plan.success_criteria,
                ),
                "research_tasks": tasks,
                "updated_at": utc_now(),
            }
        except Exception as e:
            logger.error(
                "PlannerAgent failed to generate plan using LLM: %s. Falling back to deterministic plan.",
                e,
            )
            return self._run_deterministic(query)

    def _run_deterministic(self, query: str) -> ResearchState:
        """Fallback deterministic planner logic."""
        tasks = (
            ResearchTask(
                task_type=TaskType.MARKET,
                objective=f"Identify market context, size, growth, and current dynamics for: {query}",
                search_queries=(
                    f"{query} market size growth latest report",
                    f"{query} industry trends official data",
                ),
                priority=5,
                required_source_count=3,
            ),
            ResearchTask(
                task_type=TaskType.COMPETITOR,
                objective=f"Identify major companies, competitors, and ecosystem participants for: {query}",
                search_queries=(
                    f"{query} leading companies competitors market share",
                    f"{query} competitive landscape analysis",
                ),
                priority=4,
                required_source_count=2,
            ),
            ResearchTask(
                task_type=TaskType.RISK,
                objective=f"Identify material risks, constraints, and uncertainties for: {query}",
                search_queries=(
                    f"{query} risks challenges regulation supply chain",
                    f"{query} barriers constraints market adoption",
                ),
                priority=4,
                required_source_count=2,
            ),
        )
        plan = ResearchPlan(
            objective=query,
            tasks=tasks,
            expected_output="Evidence-backed professional business research report",
            success_criteria=(
                "At least two credible sources are used.",
                "Key findings are supported by evidence.",
                "The final report includes risks, opportunities, recommendations, and references.",
            ),
        )
        return {
            "plan": plan,
            "research_tasks": list(tasks),
            "updated_at": utc_now(),
        }
