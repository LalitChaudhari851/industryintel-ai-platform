"""Report Viewer component for Streamlit."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_report_viewer(report: Dict[str, Any]) -> None:
    """Renders the generated executive intelligence report, sections, and metadata."""
    if not report:
        st.info("No report data available.")
        return

    # Extract metadata
    title = report.get("title", "Business Intelligence Report")
    exec_summary = report.get("executive_summary", "")
    sections = report.get("sections", {})
    confidence = report.get("confidence_score", 0.0)
    word_count = report.get("word_count", 0)

    # Header with title
    st.markdown(f'<h1 style="font-size: 2.2rem; margin-top: 10px;">📄 {title}</h1>', unsafe_allow_html=True)

    # Metadata metrics block
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Confidence Score</span>
                <h2 style="color: #60a5fa; margin: 4px 0 0 0;">{confidence * 100:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Report Word Count</span>
                <h2 style="color: #a78bfa; margin: 4px 0 0 0;">{word_count:,} words</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")  # space

    # Executive Summary Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3>🎯 Executive Summary</h3>", unsafe_allow_html=True)
    st.markdown(exec_summary)
    st.markdown("</div>", unsafe_allow_html=True)

    # Report Sections Rendered in Accordions/Tabs
    st.markdown("### 📘 Detailed Analysis Sections")
    
    if sections:
        for sec_title, sec_content in sections.items():
            with st.expander(f"🔹 {sec_title}", expanded=True):
                st.markdown(sec_content)
    else:
        st.warning("No sections found in report.")
