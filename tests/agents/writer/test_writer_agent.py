from __future__ import annotations

import pytest

from app.agents.analyst.agent import AnalystAgent
from app.agents.critic.agent import CriticAgent
from app.agents.writer.agent import WriterAgent
from app.workflows.business_research.state import EvidenceItem, Source, SourceType


@pytest.mark.asyncio
async def test_writer_generates_professional_structured_report() -> None:
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
            id="e1",
            source_id=source_1.id,
            text=(
                "India EV market sales grew 42% in 2025 as demand increased and charging investment expanded. "
                "Fleet electrification growth created a major opportunity for battery suppliers and charging operators."
            ),
            relevance_score=0.93,
        ),
        EvidenceItem(
            id="e2",
            source_id=source_2.id,
            text=(
                "Battery costs declined 18% while market adoption growth supported stronger EV demand. "
                "However, charging infrastructure shortage remained a key risk for adoption outside major cities."
            ),
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
    writer_update = await WriterAgent()({**state, **analysis_update, **critic_update})

    report = writer_update["final_report"]
    assert 1000 <= report.word_count <= 1500
    assert set(report.sections) == {
        "Executive Summary",
        "Key Findings",
        "Risks",
        "Opportunities",
        "Recommendations",
        "References",
    }
    assert report.executive_summary
    assert report.citations
    assert report.confidence_score >= 0.7
