from __future__ import annotations

import pytest

from app.agents.researcher.agent import ResearchAgent
from app.agents.researcher.models import (
    ResearchAgentConfig,
    TavilyResult,
    TavilySearchRequest,
    TavilySearchResponse,
)
from app.workflows.business_research.state import ResearchPlan, ResearchTask


class FakeSearchClient:
    def __init__(self) -> None:
        self.requests: list[TavilySearchRequest] = []

    async def search(self, request: TavilySearchRequest) -> TavilySearchResponse:
        self.requests.append(request)
        return TavilySearchResponse(
            query=request.query,
            results=(
                TavilyResult(
                    title="Official EV market report",
                    url="https://example.com/report?utm_source=test",
                    content="India EV market sales growth official data battery charging policy",
                    score=0.92,
                ),
                TavilyResult(
                    title="Duplicate EV market report",
                    url="https://www.example.com/report",
                    content="India EV market official report duplicate",
                    score=0.81,
                ),
                TavilyResult(
                    title="Unrelated sports article",
                    url="https://sports.example.com/story",
                    content="Cricket match report",
                    score=0.1,
                ),
            ),
        )


@pytest.mark.asyncio
async def test_research_agent_returns_deduplicated_structured_sources() -> None:
    fake_client = FakeSearchClient()
    agent = ResearchAgent(
        config=ResearchAgentConfig(
            max_queries=2,
            max_sources=5,
            min_relevance_score=0.3,
        ),
        search_client=fake_client,
    )
    task = ResearchTask(
        objective="Find official India EV market data",
        search_queries=("India EV market official report",),
    )
    state = {
        "session_id": "session-1",
        "query": "Analyze India's EV market",
        "plan": ResearchPlan(
            objective="Analyze India's EV market",
            tasks=(task,),
            expected_output="Market research summary",
        ),
    }

    update = await agent(state)

    assert len(fake_client.requests) == 2
    assert len(update["sources"]) == 1
    assert len(update["evidence"]) == 1
    assert update["sources"][0].title == "Official EV market report"
    assert update["sources"][0].metadata["relevance_score"] > 0.3
    assert update["evidence"][0].source_id == update["sources"][0].id
    assert update["errors"] == []
