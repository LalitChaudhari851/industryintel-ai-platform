"""Citation and Source Viewer component for Streamlit."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_citation_viewer(citations: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> None:
    """Renders the sources used during research and detailed citation mappings."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📚 Research Sources & Citations")

    tab1, tab2 = st.tabs(["🌐 Sources Visited", "🔗 Citation Grounding"])

    with tab1:
        if not sources:
            st.info("No sources recorded for this research session.")
        else:
            st.markdown("Below is the list of web sources evaluated and utilized by the research swarm:")
            for i, src in enumerate(sources, 1):
                title = src.get("title", f"Source {i}")
                url = src.get("url", "#")
                credibility = src.get("credibility_score", 0.5)

                st.markdown(
                    f"**[S{i}] [{title}]({url})**  \n"
                    f"*Credibility Score: {credibility * 100:.0f}%*"
                )
                st.write("")

    with tab2:
        if not citations:
            st.info("No citation details mapped for this report.")
        else:
            st.markdown("Mapping claims generated in the report to verified evidence chunks:")
            for i, cit in enumerate(citations, 1):
                claim = cit.get("claim", "")
                label = cit.get("citation_label", f"[S{i}]")
                support_text = cit.get("supporting_text", "")
                
                # Check confidence score
                conf = cit.get("confidence", 0.5)

                with st.expander(f"📍 Claim {i}: {claim[:80]}... ({label})"):
                    st.markdown(f"**Full Claim:** {claim}")
                    st.markdown(f"**Source Attribution:** {label}")
                    st.markdown(f"**Verified Evidence:** *\"{support_text}\"*")
                    st.markdown(f"**Grounding Confidence:** {conf * 100:.0f}%")

    st.markdown("</div>", unsafe_allow_html=True)
