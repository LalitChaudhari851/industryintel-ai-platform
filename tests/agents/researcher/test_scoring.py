from __future__ import annotations

from app.agents.researcher.models import SearchQueryCandidate, TavilyResult
from app.agents.researcher.scoring import deduplicate_and_rank, normalize_url


def test_normalize_url_removes_tracking_and_www() -> None:
    assert (
        normalize_url("https://www.example.com/report/?utm_source=x&b=2&a=1")
        == "https://example.com/report?a=1&b=2"
    )


def test_deduplicate_and_rank_keeps_best_result() -> None:
    query = SearchQueryCandidate(query="AI chip market NVIDIA AMD", rationale="test")
    lower = TavilyResult(
        title="NVIDIA AMD AI chips",
        url="https://www.example.com/ai-chips?utm_campaign=x",
        content="NVIDIA AMD AI chip market share",
        score=0.5,
    )
    higher = TavilyResult(
        title="NVIDIA AMD AI chips market report",
        url="https://example.com/ai-chips",
        content="NVIDIA AMD AI chip market share data accelerator demand",
        score=0.9,
    )

    ranked = deduplicate_and_rank(
        [(query, lower), (query, higher)],
        min_relevance_score=0.0,
        max_sources=5,
    )

    assert len(ranked) == 1
    assert ranked[0].result.title == "NVIDIA AMD AI chips market report"
