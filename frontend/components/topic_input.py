"""Topic Input component for Streamlit."""

from __future__ import annotations

import asyncio
from typing import Any, Tuple

import streamlit as st

from frontend.utils.api_client import APIClient


@st.cache_data(ttl=15)
def get_models_health(base_url: str) -> dict[str, Any]:
    """Cache models health check to prevent laggy UI renders."""
    client = APIClient(base_url)
    try:
        # Run async in synchronous cached function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(client.check_models())
    except Exception:
        return {"ollama_connected": False}


def render_topic_input(api_client: APIClient) -> Tuple[bool, str, str, int]:
    """Renders the research initiation form and models health status."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔍 Launch Autonomous Research")

    # Render model checks
    health = get_models_health(api_client.base_url)
    if health.get("ollama_connected"):
        p_ok = "🟢 Online" if health.get("primary") else "🔴 Missing"
        f_ok = "🟢 Online" if health.get("fallback") else "🔴 Missing"
        st.info(
            f"⚡ **Local Ollama Services Connected** | "
            f"Primary Model ({health.get('primary_model')}): {p_ok} | "
            f"Fallback ({health.get('fallback_model')}): {f_ok}"
        )
    else:
        st.warning(
            "⚠️ **Local Ollama Services Offline** | "
            "Please ensure Ollama is running at http://localhost:11434 and models are pulled."
        )

    with st.form("research_form"):
        query = st.text_input(
            "Research Query",
            placeholder="e.g., Compare OpenAI vs Anthropic competitive strategy in 2026",
            help="Enter the main topic, industry, competitor, or technology you want the agent swarm to research.",
        )
        business_context = st.text_area(
            "Additional Business Context (Optional)",
            placeholder="e.g., We are a enterprise analytics vendor evaluating API reliability, cost, and rate limits...",
            help="Provide any context or specific constraints to guide the dynamic planning agent.",
        )

        col1, col2 = st.columns(2)
        with col1:
            max_iterations = st.slider(
                "Max Research Iterations (Critique Loops)",
                min_value=1,
                max_value=3,
                value=3,
                help="Max loops the Critic agent can trigger to self-correct findings before generating report.",
            )
        with col2:
            st.write("")  # spacing
            st.write("")  # spacing
            submit = st.form_submit_button("Launch Agent Swarm", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return submit, query.strip(), business_context.strip(), max_iterations
