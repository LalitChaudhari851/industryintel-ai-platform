"""Progress Tracker component for Streamlit."""

from __future__ import annotations

import streamlit as st


def render_progress_tracker(status: str, iteration: int) -> None:
    """Renders a dynamic visual step tracker indicating which agent is active."""
    stages = [
        {"key": "planning", "label": "Planner", "desc": "Deconstructing query..."},
        {"key": "researching", "label": "Researcher", "desc": "Running Tavily search..."},
        {"key": "analyzing", "label": "Analyst", "desc": "Synthesizing evidence..."},
        {"key": "critiquing", "label": "Critic", "desc": f"Checking facts (Loop {iteration})..."},
        {"key": "writing", "label": "Writer", "desc": "Composing final report..."},
    ]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Header with running status indicator
    if status in {"queued", "created"}:
        st.markdown(
            '<h3>Workflow Status: <span class="status-pill status-queued">Queued</span></h3>',
            unsafe_allow_html=True,
        )
    elif status in {"completed"}:
        st.markdown(
            '<h3>Workflow Status: <span class="status-pill status-completed">Completed</span></h3>',
            unsafe_allow_html=True,
        )
    elif status in {"failed"}:
        st.markdown(
            '<h3>Workflow Status: <span class="status-pill status-failed">Failed</span></h3>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<h3>Workflow Status: <span class="status-pill status-running">Running</span></h3>',
            unsafe_allow_html=True,
        )

    # Active index lookup
    active_idx = -1
    for i, s in enumerate(stages):
        if s["key"] == status:
            active_idx = i
            break

    # Timeline UI
    cols = st.columns(len(stages))
    for idx, s in enumerate(stages):
        with cols[idx]:
            if status == "completed":
                st.markdown(f"🟢 **{s['label']}**")
                st.caption("Finished")
            elif status == "failed":
                if idx <= active_idx:
                    st.markdown(f"🔴 **{s['label']}**")
                    st.caption("Halted")
                else:
                    st.markdown(f"⚪ {s['label']}")
                    st.caption("Cancelled")
            else:
                if idx < active_idx:
                    st.markdown(f"🟢 **{s['label']}**")
                    st.caption("Done")
                elif idx == active_idx:
                    st.markdown(f"🔵 **{s['label']}**")
                    st.caption(s["desc"])
                else:
                    st.markdown(f"⚪ {s['label']}")
                    st.caption("Pending")

    st.markdown("</div>", unsafe_allow_html=True)
