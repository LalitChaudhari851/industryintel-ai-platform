from __future__ import annotations

import json
import re
from collections.abc import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from app.agents.researcher.models import (
    SearchContext,
    SearchQueryCandidate,
    SearchQueryPlan,
)
from app.agents.researcher.prompts import QUERY_GENERATION_TEMPLATE


class ResearchQueryGenerator:
    """Generate Tavily queries with LangChain, falling back to deterministic rules."""

    def __init__(self, *, llm: BaseLanguageModel | None = None, max_queries: int = 6) -> None:
        self.llm = llm
        self.max_queries = max_queries
        self.prompt = (
            PromptTemplate.from_template(QUERY_GENERATION_TEMPLATE)
            if PromptTemplate is not None
            else None
        )

    async def generate(self, context: SearchContext) -> SearchQueryPlan:
        llm_plan = await self._generate_with_llm(context)
        fallback_plan = self._fallback_queries(context)

        merged: list[SearchQueryCandidate] = []
        seen: set[str] = set()

        for candidate in [*llm_plan.queries, *fallback_plan.queries]:
            normalized = candidate.query.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(candidate)
            if len(merged) >= self.max_queries:
                break

        return SearchQueryPlan(queries=tuple(merged))

    async def _generate_with_llm(self, context: SearchContext) -> SearchQueryPlan:
        if self.llm is None or self.prompt is None:
            return SearchQueryPlan()

        chain = self.prompt | self.llm
        response = await chain.ainvoke(
            {
                "query": context.query,
                "business_context": context.business_context or "None",
                "tasks": self._format_tasks(context),
                "retry_reason": context.retry_reason or "None",
                "max_queries": self.max_queries,
            }
        )

        text = getattr(response, "content", str(response))
        return self._parse_llm_json(text)

    def _parse_llm_json(self, text: str) -> SearchQueryPlan:
        json_text = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", json_text, flags=re.DOTALL)
        if fenced:
            json_text = fenced.group(1).strip()

        try:
            return SearchQueryPlan.model_validate(json.loads(json_text))
        except (json.JSONDecodeError, ValueError):
            return SearchQueryPlan()

    def _fallback_queries(self, context: SearchContext) -> SearchQueryPlan:
        candidates: list[SearchQueryCandidate] = [
            SearchQueryCandidate(
                query=context.query,
                rationale="Use the original user request as the broad research query.",
            )
        ]

        for task in context.tasks:
            task_queries = self._task_queries(context.query, task.id, task.objective, task.search_queries)
            candidates.extend(task_queries)

        if context.retry_reason is not None:
            candidates.append(
                SearchQueryCandidate(
                    query=f"{context.query} evidence data sources official report",
                    rationale=f"Retry query to address {context.retry_reason.value}.",
                )
            )

        return SearchQueryPlan(queries=tuple(candidates[: self.max_queries]))

    def _task_queries(
        self,
        root_query: str,
        task_id: str,
        objective: str,
        explicit_queries: Sequence[str],
    ) -> list[SearchQueryCandidate]:
        candidates = [
            SearchQueryCandidate(
                query=query,
                rationale="Planner-provided search query.",
                task_id=task_id,
            )
            for query in explicit_queries
        ]
        candidates.append(
            SearchQueryCandidate(
                query=f"{root_query} {objective}",
                rationale="Combine the root request with the task objective.",
                task_id=task_id,
            )
        )
        candidates.append(
            SearchQueryCandidate(
                query=f"{objective} market analysis latest data official sources",
                rationale="Seek current and credible business evidence for this task.",
                task_id=task_id,
            )
        )
        return candidates

    def _format_tasks(self, context: SearchContext) -> str:
        if not context.tasks:
            return "None"
        return "\n".join(
            f"- id={task.id}; type={task.task_type}; objective={task.objective}"
            for task in context.tasks
        )
