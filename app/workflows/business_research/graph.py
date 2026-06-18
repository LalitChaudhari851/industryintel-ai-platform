"""LangGraph definition for the AI business research analyst workflow."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.workflows.business_research.routing import (
    initialize_defaults,
    mark_failure,
    mark_research_retry,
    route_after_critic,
)
from app.workflows.business_research.state import ResearchState, WorkflowStatus, utc_now

AgentNode: TypeAlias = Callable[[ResearchState], ResearchState | Awaitable[ResearchState]]


@dataclass(frozen=True, slots=True)
class BusinessResearchAgents:
    """Agent node callables supplied by the application layer.

    This module intentionally does not implement agent behavior. Each callable
    receives ResearchState and returns a partial ResearchState update.
    """

    planner: AgentNode
    researcher: AgentNode
    analyst: AgentNode
    critic: AgentNode
    writer: AgentNode


def _status_update(status: WorkflowStatus) -> ResearchState:
    return {"status": status, "updated_at": utc_now()}


def build_business_research_graph(
    agents: BusinessResearchAgents,
    *,
    checkpointer: MemorySaver | None = None,
):
    """Build and compile the production workflow with MemorySaver checkpointing."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("initialize", initialize_defaults)
    workflow.add_node("planner", agents.planner)
    workflow.add_node("researcher", agents.researcher)
    workflow.add_node("analyst", agents.analyst)
    workflow.add_node("critic", agents.critic)
    workflow.add_node("writer", agents.writer)
    workflow.add_node("prepare_research_retry", mark_research_retry)
    workflow.add_node("fail", mark_failure)
    workflow.add_node("mark_planning", lambda _: _status_update(WorkflowStatus.PLANNING))
    workflow.add_node("mark_researching", lambda _: _status_update(WorkflowStatus.RESEARCHING))
    workflow.add_node("mark_analyzing", lambda _: _status_update(WorkflowStatus.ANALYZING))
    workflow.add_node("mark_critiquing", lambda _: _status_update(WorkflowStatus.CRITIQUING))
    workflow.add_node("mark_writing", lambda _: _status_update(WorkflowStatus.WRITING))
    workflow.add_node("mark_completed", lambda _: _status_update(WorkflowStatus.COMPLETED))

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "mark_planning")
    workflow.add_edge("mark_planning", "planner")
    workflow.add_edge("planner", "mark_researching")
    workflow.add_edge("mark_researching", "researcher")
    workflow.add_edge("researcher", "mark_analyzing")
    workflow.add_edge("mark_analyzing", "analyst")
    workflow.add_edge("analyst", "mark_critiquing")
    workflow.add_edge("mark_critiquing", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "prepare_research_retry",
            "writer": "mark_writing",
            "fail": "fail",
        },
    )

    workflow.add_edge("prepare_research_retry", "researcher")
    workflow.add_edge("mark_writing", "writer")
    workflow.add_edge("writer", "mark_completed")
    workflow.add_edge("mark_completed", END)
    workflow.add_edge("fail", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())
