from __future__ import annotations

import pytest

from app.agents.analyst.agent import AnalystAgent
from app.agents.critic.agent import CriticAgent
from app.workflows.business_research.state import (
    EvidenceItem,
    QualityDecision,
    Source,
    SourceType,
)


@pytest.mark.asyncio
async def test_critic_passes_high_quality_supported_analysis() -> None:
    source_1 = Source(
        source_type=SourceType.WEB,
        title="Official EV market report",
        url="https://example.com/report-1",
        credibility_score=0.92,
    )
    source_2 = Source(
        source_type=SourceType.WEB,
        title="Industry EV market report",
        url="https://example.com/report-2",
        credibility_score=0.86,
    )
    evidence = [
        EvidenceItem(
            id="evidence-1",
            source_id=source_1.id,
            text="India EV market sales grew 42% in 2025 as demand increased and charging investment expanded.",
            relevance_score=0.93,
        ),
        EvidenceItem(
            id="evidence-2",
            source_id=source_2.id,
            text="Battery costs declined 18% while market adoption growth supported stronger EV demand.",
            relevance_score=0.88,
        ),
    ]
    state = {
        "session_id": "session-1",
        "query": "Analyze India EV market",
        "sources": [source_1, source_2],
        "evidence": evidence,
    }
    analysis_update = await AnalystAgent()(state)

    critic_update = await CriticAgent()({**state, **analysis_update})

    review = critic_update["critic_review"]
    assert review.decision is QualityDecision.PASS
    assert review.confidence_score >= 0.7
    assert review.evidence_assessments


@pytest.mark.asyncio
async def test_critic_triggers_retry_when_quality_below_threshold() -> None:
    analysis_update = await AnalystAgent()(
        {
            "session_id": "session-1",
            "query": "Analyze India EV market",
            "sources": [],
            "evidence": [],
        }
    )

    critic_update = await CriticAgent()(
        {
            "session_id": "session-1",
            "query": "Analyze India EV market",
            "sources": [],
            "evidence": [],
            **analysis_update,
        }
    )

    review = critic_update["critic_review"]
    assert review.decision is QualityDecision.RETRY_RESEARCH
    assert review.confidence_score < 0.7
