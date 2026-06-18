"""Agent Workflow Visualization component for Streamlit."""

from __future__ import annotations

import streamlit as st


def render_agent_workflow(status: str) -> None:
    """Renders a visual node map of the multi-agent swarm highlighting the active agent."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🗺️ Active Swarm Node Map")

    # Determine node highlights
    nodes = {
        "planning": "",
        "researching": "",
        "analyzing": "",
        "critiquing": "",
        "writing": "",
    }

    if status in nodes:
        nodes[status] = "active-node"

    # CSS + HTML for the interactive swarm nodes
    html_code = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
        <div class="swarm-node {nodes['planning']}">
            <div class="node-title">Planner Agent</div>
            <div class="node-subtitle">Query Deconstruction</div>
        </div>
        <div class="swarm-arrow">➜</div>
        <div class="swarm-node {nodes['researching']}">
            <div class="node-title">Researcher Agent</div>
            <div class="node-subtitle">Tavily + BGE Embed</div>
        </div>
        <div class="swarm-arrow">➜</div>
        <div class="swarm-node {nodes['analyzing']}">
            <div class="node-title">Analyst Agent</div>
            <div class="node-subtitle">Insights Synthesis</div>
        </div>
        <div class="swarm-arrow">➜</div>
        <div class="swarm-node {nodes['critiquing']}">
            <div class="node-title">Critic Agent</div>
            <div class="node-subtitle">Fact & Source Check</div>
        </div>
        <div class="swarm-arrow">➜</div>
        <div class="swarm-node {nodes['writing']}">
            <div class="node-title">Writer Agent</div>
            <div class="node-subtitle">Prose Composition</div>
        </div>
    </div>

    <style>
    .swarm-node {{
        flex: 1;
        min-width: 140px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .active-node {{
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%) !important;
        border: 1.5px solid #8b5cf6 !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
        transform: scale(1.03);
    }}
    .node-title {{
        font-weight: 700;
        font-size: 0.9rem;
        color: #ffffff;
        margin-bottom: 4px;
    }}
    .node-subtitle {{
        font-size: 0.75rem;
        color: #94a3b8;
    }}
    .swarm-arrow {{
        color: #475569;
        font-size: 1.2rem;
        user-select: none;
    }}
    @media (max-width: 768px) {{
        .swarm-arrow {{
            display: none;
        }}
    }}
    </style>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
