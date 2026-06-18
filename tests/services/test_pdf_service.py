"""Tests for the PDF Export Service."""

from __future__ import annotations

import io
from datetime import datetime, timezone

from app.application.research.models import ResearchSessionRecord
from app.services.pdf_service import PDFExportService
from app.workflows.business_research.state import Citation, FinalReport


def test_pdf_export_service_generates_pdf() -> None:
    """Test that PDFExportService compiles a report into a valid PDF bytes buffer."""
    report = FinalReport(
        title="Dynamic Market Research Briefing",
        executive_summary="This is a summary of the strategic intelligence briefing.",
        sections={
            "Market Dynamics": "High market growth observed in electric drivetrains.",
            "Risks": "High battery raw material supply volatility.",
            "Recommendations": "Diversify sourcing of battery suppliers.",
        },
        citations=(
            Citation(claim="High market growth", source_id="ev-report-1", confidence=0.9),
        ),
        limitations=("Limited sample size.",),
        confidence_score=0.88,
        word_count=100,
    )
    
    session = ResearchSessionRecord(
        id="test-session-id",
        query="Electric Vehicles Market Growth",
        status="completed",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        report=report,
        raw_state={
            "sources": [{"title": "EV Report", "url": "https://example.com", "credibility_score": 0.9}]
        },
    )
    
    pdf_service = PDFExportService()
    pdf_buffer = pdf_service.generate_report_pdf(session)
    
    assert isinstance(pdf_buffer, io.BytesIO)
    pdf_bytes = pdf_buffer.getvalue()
    assert len(pdf_bytes) > 0
    # Verify PDF signature/header
    assert pdf_bytes.startswith(b"%PDF")
