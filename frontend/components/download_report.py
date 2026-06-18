"""Download Report component for Streamlit."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def generate_markdown_content(report: Dict[str, Any], sources: List[Dict[str, Any]]) -> str:
    """Builds a beautifully formatted Markdown string of the intelligence report."""
    md = []
    md.append(f"# {report.get('title', 'Business Intelligence Report')}")
    md.append(f"\n**Swarm Confidence Score:** {report.get('confidence_score', 0.0) * 100:.1f}%")
    md.append(f"**Report Word Count:** {report.get('word_count', 0)}")
    md.append("\n---\n")

    md.append("## Executive Summary")
    md.append(f"\n{report.get('executive_summary', '')}")

    sections = report.get("sections", {})
    for title, content in sections.items():
        md.append(f"\n## {title}")
        md.append(f"\n{content}")

    limitations = report.get("limitations", [])
    if limitations:
        md.append("\n## Research Limitations")
        for lim in limitations:
            md.append(f"- {lim}")

    if sources:
        md.append("\n---\n")
        md.append("## Sources & References")
        for i, src in enumerate(sources, 1):
            md.append(
                f"- **[S{i}]** [{src.get('title')}]({src.get('url')}) "
                f"(Credibility Score: {src.get('credibility_score', 0.5) * 100:.0f}%)"
            )

    return "\n".join(md)


def render_download_buttons(
    report: Dict[str, Any],
    sources: List[Dict[str, Any]],
    api_client: Any = None,
    research_id: str = None,
) -> None:
    """Renders download buttons allowing the user to export the report to markdown and professional PDF."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📥 Export Intelligence Report")

    try:
        markdown_text = generate_markdown_content(report, sources)
        
        # Download Markdown Button
        st.download_button(
            label="Download Markdown (.md)",
            data=markdown_text,
            file_name="intelligence_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        # Download PDF Button
        if api_client and research_id:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            pdf_bytes = loop.run_until_complete(api_client.get_report_pdf(research_id))

            st.download_button(
                label="Download Executive PDF (.pdf)",
                data=pdf_bytes,
                file_name=f"Executive_Briefing_Report_{research_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Failed to generate download file: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
