"""PDF generation service for exporting executive intelligence briefings."""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.application.research.models import ResearchSessionRecord


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to calculate total page count and render running headers and footers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        # Ignore cover page (page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))  # Slate Gray

        # Header
        self.drawString(54, 750, "AI Industry Intelligence Platform — Executive Briefing")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))  # Light Gray border
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)

        # Footer
        self.line(54, 60, 558, 60)
        self.drawString(54, 48, "CONFIDENTIAL — FOR INTERNAL USE ONLY")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 48, page_str)
        self.restoreState()


class PDFExportService:
    """Service to compile executive briefings into professional publication-quality PDFs."""

    def __init__(self) -> None:
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        # Text adjustments
        self.styles["Normal"].textColor = colors.HexColor("#1e293b")  # Dark Slate
        self.styles["Normal"].fontSize = 10
        self.styles["Normal"].leading = 14

        # Title / Cover Page Style
        self.cover_title_style = ParagraphStyle(
            "CoverTitle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=32,
            textColor=colors.HexColor("#1e3a8a"),  # Navy Blue
            spaceAfter=10,
        )

        self.cover_subtitle_style = ParagraphStyle(
            "CoverSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#475569"),  # Cool Gray
            spaceAfter=30,
        )

        # Section Headings (H1)
        self.h1_style = ParagraphStyle(
            "SectionH1",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1e3a8a"),
            spaceBefore=18,
            spaceAfter=8,
            keepWithNext=True,
        )

        # Section Subheadings (H2)
        self.h2_style = ParagraphStyle(
            "SectionH2",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0f766e"),  # Teal accent
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        )

        # Body text
        self.body_style = ParagraphStyle(
            "ReportBody",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14.5,
            spaceAfter=8,
        )

        # Meta info
        self.meta_style = ParagraphStyle(
            "MetaLabel",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=colors.HexColor("#1e3a8a"),
        )
        self.meta_val_style = ParagraphStyle(
            "MetaValue",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
        )

    def _markdown_to_html(self, text: str) -> str:
        """Sanitize and format simple markdown tags for ReportLab Paragraph parser."""
        if not text:
            return ""
        # Escape HTML entities
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Bold Markdown **text** -> <b>text</b>
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        # Italic Markdown *text* -> <i>text</i>
        text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
        # Bullet list marker conversion
        text = re.sub(r"^\s*-\s+(.*?)$", r"&bull; \1", text, flags=re.MULTILINE)
        # Linebreaks to HTML breaks
        text = text.replace("\n", "<br/>")
        return text

    def _get_section(self, sections: Dict[str, str], keys: List[str], fallback: str = "") -> str:
        """Retrieve matching section key case-insensitively with list of fallback alternatives."""
        for key in keys:
            if key in sections:
                return sections[key]
            for k, v in sections.items():
                if k.lower() == key.lower():
                    return v
        return fallback

    def generate_report_pdf(self, session: ResearchSessionRecord) -> io.BytesIO:
        """Create a styled PDF binary buffer for the completed session briefing."""
        report = session.report
        if report is None:
            raise ValueError("Cannot export report to PDF: final report is missing in session record.")

        buffer = io.BytesIO()
        # Document template with standard letter margins (0.75in / 54pt)
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=72,
            bottomMargin=72,
        )

        story = []

        # --- COVER PAGE ---
        story.append(Spacer(1, 40))
        # Top banner bar
        banner_data = [[""]]
        banner_table = Table(banner_data, colWidths=[504], rowHeights=[8])
        banner_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e3a8a")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("AI INDUSTRY INTELLIGENCE PLATFORM", self.cover_subtitle_style))
        story.append(Paragraph(report.title, self.cover_title_style))
        story.append(Paragraph("EXECUTIVE BRIEFING & STRATEGIC INSIGHTS", self.cover_subtitle_style))
        story.append(Spacer(1, 50))

        # Metadata table block
        q_score = session.critic_review.confidence_score if session.critic_review else report.confidence_score
        c_score = report.confidence_score
        source_count = len(report.citations) if report.citations else len(session.raw_state.get("sources", []))
        
        meta_rows = [
            [
                Paragraph("Report ID:", self.meta_style),
                Paragraph(session.id, self.meta_val_style),
                Paragraph("Target Query:", self.meta_style),
                Paragraph(session.query, self.meta_val_style),
            ],
            [
                Paragraph("Quality Score:", self.meta_style),
                Paragraph(f"{q_score * 100:.1f}%", self.meta_val_style),
                Paragraph("Confidence Score:", self.meta_style),
                Paragraph(f"{c_score * 100:.1f}%", self.meta_val_style),
            ],
            [
                Paragraph("Sources Analyzed:", self.meta_style),
                Paragraph(str(source_count), self.meta_val_style),
                Paragraph("Created Date:", self.meta_style),
                Paragraph(session.completed_at.strftime("%Y-%m-%d") if session.completed_at else datetime.now(timezone.utc).strftime("%Y-%m-%d"), self.meta_val_style),
            ],
        ]
        
        meta_table = Table(meta_rows, colWidths=[110, 142, 110, 142])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),  # light slate gray
            ("PADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(meta_table)
        
        # Cover footer
        story.append(Spacer(1, 120))
        story.append(Paragraph("<b>Author</b>: Autonomous Intelligence Swarm Engine<br/>"
                               "<b>Status</b>: Verified & Approved Briefing", self.meta_val_style))
        story.append(PageBreak())

        # --- CONTENT PAGES ---

        # 2. Executive Summary
        story.append(Paragraph("1. Executive Summary", self.h1_style))
        summary_html = self._markdown_to_html(report.executive_summary)
        story.append(Paragraph(summary_html, self.body_style))

        # 3. Key Findings
        story.append(Paragraph("2. Key Findings", self.h1_style))
        findings_content = self._get_section(report.sections, ["Key Findings", "Market Dynamics", "Findings"])
        if findings_content:
            story.append(Paragraph(self._markdown_to_html(findings_content), self.body_style))
        else:
            story.append(Paragraph("No specific findings section details compiled.", self.body_style))

        # 4. Strategic Analysis
        story.append(Paragraph("3. Strategic Analysis", self.h1_style))
        analysis_content = self._get_section(report.sections, ["Strategic Analysis", "Competitive Landscape", "Market Dynamics"])
        if analysis_content:
            story.append(Paragraph(self._markdown_to_html(analysis_content), self.body_style))
        else:
            story.append(Paragraph("No strategic analysis section compiled.", self.body_style))

        # 5. Risks
        story.append(Paragraph("4. Risks & Limitations", self.h1_style))
        risks_content = self._get_section(report.sections, ["Risks", "Risks & Regulatory Constraints"])
        if risks_content:
            story.append(Paragraph(self._markdown_to_html(risks_content), self.body_style))
        else:
            story.append(Paragraph("No risks assessment compiled.", self.body_style))

        # 6. Opportunities
        story.append(Paragraph("5. Opportunities", self.h1_style))
        opps_content = self._get_section(report.sections, ["Opportunities", "Opportunities & Strategic Alternatives"])
        if opps_content:
            story.append(Paragraph(self._markdown_to_html(opps_content), self.body_style))
        else:
            story.append(Paragraph("No strategic opportunities compiled.", self.body_style))

        # 7. Recommendations
        story.append(Paragraph("6. Recommendations", self.h1_style))
        recs_content = self._get_section(report.sections, ["Recommendations", "Strategic Recommendations"])
        if recs_content:
            story.append(Paragraph(self._markdown_to_html(recs_content), self.body_style))
        else:
            story.append(Paragraph("No actionable recommendations compiled.", self.body_style))

        # 8. References
        story.append(Paragraph("7. References", self.h1_style))
        if report.citations:
            story.append(Paragraph("The following list matches key claims in the briefing back to source documents:", self.body_style))
            ref_rows = [
                [
                    Paragraph("<b>Ref Code</b>", self.meta_style),
                    Paragraph("<b>Attributed Claim</b>", self.meta_style),
                    Paragraph("<b>Domain Source</b>", self.meta_style),
                ]
            ]
            for idx, cit in enumerate(report.citations):
                ref_rows.append([
                    Paragraph(f"Ref #{idx + 1}", self.meta_val_style),
                    Paragraph(cit.claim, self.meta_val_style),
                    Paragraph(cit.source_id or "Web Query", self.meta_val_style),
                ])
            
            ref_table = Table(ref_rows, colWidths=[70, 314, 120])
            ref_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(ref_table)
        else:
            story.append(Paragraph("No source citations logged in the executive briefing.", self.body_style))

        # Build document with running footers/headers
        doc.build(story, canvasmaker=NumberedCanvas)
        buffer.seek(0)
        return buffer
