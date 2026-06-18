"""Business research LangGraph workflow package.

Heavy LangGraph imports are loaded lazily so state models can be reused by
agents and tests without requiring graph runtime dependencies at import time.
"""

from app.workflows.business_research.state import ResearchState

__all__ = [
    "BusinessResearchAgents",
    "ResearchState",
    "build_business_research_graph",
]


def __getattr__(name: str):
    if name in {"BusinessResearchAgents", "build_business_research_graph"}:
        from app.workflows.business_research.graph import (
            BusinessResearchAgents,
            build_business_research_graph,
        )

        return {
            "BusinessResearchAgents": BusinessResearchAgents,
            "build_business_research_graph": build_business_research_graph,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
