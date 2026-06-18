from __future__ import annotations

import pytest

from app.agents.analyst.agent import AnalystAgent
from app.workflows.business_research.state import EvidenceItem, Source, SourceType


@pytest.mark.asyncio
async def test_analyst_extracts_structured_findings_metrics_and_trends() -> None:
    source = Source(
        source_type=SourceType.WEB,
        title="India EV market report",
        url="https://example.com/ev-report",
        credibility_score=0.9,
    )
    evidence = EvidenceItem(
        id="evidence-1",
        source_id=source.id,
        text=(
            "India EV market sales grew 42% in 2025 as charging investment expanded. "
            "Battery costs declined 18% and demand growth created a major opportunity."
        ),
        relevance_score=0.92,
    )

    update = await AnalystAgent()(
        {
            "session_id": "session-1",
            "query": "Analyze India EV market",
            "sources": [source],
            "evidence": [evidence],
        }
    )

    analysis = update["analysis"]
    assert analysis.findings
    assert analysis.metrics
    assert analysis.trends
    assert analysis.confidence_score >= 0.7
    assert analysis.metrics[0].evidence_ids == ("evidence-1",)
