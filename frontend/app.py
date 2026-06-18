"""Main Entrypoint for the Streamlit Front-End App."""

from __future__ import annotations

import asyncio
import logging
import time

import streamlit as st

from frontend.components.agent_workflow import render_agent_workflow
from frontend.components.citation_viewer import render_citation_viewer
from frontend.components.download_report import render_download_buttons
from frontend.components.progress_tracker import render_progress_tracker
from frontend.components.report_viewer import render_report_viewer
from frontend.components.topic_input import render_topic_input
from frontend.components.observability_dashboard import render_observability_dashboard
from frontend.utils.api_client import APIClient

# Page configuration - Expand sidebar for page navigation
st.set_page_config(
    page_title="AI Industry Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

# Initialize API client
api_client = APIClient("http://localhost:8000")


def load_css() -> None:
    """Load custom CSS themes."""
    try:
        with open("frontend/styles/theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        # Fallback if file not found
        pass


def main() -> None:
    load_css()

    # Sidebar Navigation Controls
    st.sidebar.markdown(
        '<h2 style="font-family: \'Outfit\'; margin-bottom: 0px;">🧠 Control Panel</h2>',
        unsafe_allow_html=True,
    )
    st.sidebar.write("Navigate between platform workspaces:")
    
    app_mode = st.sidebar.radio(
        "Page Selector",
        ["🔍 Research Workspace", "📊 Observability Dashboard"],
        label_visibility="collapsed"
    )

    # 1. Observability Dashboard Page
    if app_mode == "📊 Observability Dashboard":
        st.markdown(
            '<h1>📊 <span class="gradient-text">Observability Hub</span></h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 25px;'>"
            "Trace execution metrics, latency breakdowns, and report quality trends compiled from LangSmith runs.</p>",
            unsafe_allow_html=True,
        )

        with st.spinner("Fetching run metrics from LangSmith via backend..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                stats = loop.run_until_complete(api_client.get_observability_stats())
                render_observability_dashboard(stats)
            except Exception as e:
                st.error(f"Failed to load observability metrics: {e}")

    # 2. Research Workspace Page (Default)
    else:
        # Beautiful Header
        st.markdown(
            '<h1>🧠 <span class="gradient-text">AI Industry Intelligence Platform</span></h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 25px;'>"
            "Autonomous multi-agent intelligence swarm researching markets, companies, competitors, and trends locally.</p>",
            unsafe_allow_html=True,
        )

        # Initialize session state variables
        if "research_id" not in st.session_state:
            st.session_state.research_id = None
        if "job_status" not in st.session_state:
            st.session_state.job_status = None
        if "error_msg" not in st.session_state:
            st.session_state.error_msg = None

        # Step 1: Input form (if no active research job)
        if st.session_state.research_id is None:
            submit, query, context, max_iter = render_topic_input(api_client)
            if submit:
                if not query:
                    st.error("Please enter a research query.")
                    return

                with st.spinner("Initializing autonomous research swarm..."):
                    try:
                        # Run request synchronously within Streamlit
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(
                            api_client.create_research(
                                query=query, business_context=context, max_iterations=max_iter
                            )
                        )
                        st.session_state.research_id = result.get("id")
                        st.session_state.job_status = result.get("status")
                        st.session_state.error_msg = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to initialize research job: {e}")
                        logger.error("Failed to initialize research: %s", e)

        # Step 2: Running/Progress state
        elif st.session_state.job_status in {"queued", "running"}:
            # Fetch current status
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                detail = loop.run_until_complete(api_client.get_detail(st.session_state.research_id))
                st.session_state.job_status = detail.get("status")
                st.session_state.error_msg = detail.get("error")

                # Extract active agent state from the workflow record
                agent_status = "planning"
                iteration_count = 1
                if detail.get("plan"):
                    agent_status = "researching"

                raw_state = detail.get("raw_state")
                if raw_state:
                    agent_status = raw_state.get("status", "planning")
                    iteration_count = raw_state.get("iteration_count", 1)

                # Renders components
                render_progress_tracker(agent_status, iteration_count)
                st.write("")
                render_agent_workflow(agent_status)

                # Polling delay + rerun to keep updates live
                time.sleep(2)
                st.rerun()

            except Exception as e:
                st.error(f"Error checking workflow progress: {e}")
                time.sleep(3)
                st.rerun()

        # Step 3: Finished / Report state
        elif st.session_state.job_status == "completed":
            try:
                # Fetch report and details
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                report_res = loop.run_until_complete(api_client.get_report(st.session_state.research_id))
                detail_res = loop.run_until_complete(api_client.get_detail(st.session_state.research_id))
                
                report = report_res.get("report", {})
                citations = report.get("citations", [])
                
                # Reconstruct source list
                sources_list = []
                seen_source_ids = set()
                for cit in citations:
                    s_id = cit.get("source_id")
                    if s_id and s_id not in seen_source_ids:
                        seen_source_ids.add(s_id)
                        sources_list.append({
                            "id": s_id,
                            "title": cit.get("claim", "Source Title"),
                            "url": "https://www.google.com/search?q=" + cit.get("claim", ""),
                            "credibility_score": cit.get("confidence", 0.8)
                        })

                col_left, col_right = st.columns([3, 1])

                with col_left:
                    render_report_viewer(report)

                with col_right:
                    render_download_buttons(report, sources_list)
                    st.write("")
                    render_citation_viewer(citations, sources_list)
                    
                    st.write("")
                    if st.button("Start New Research", use_container_width=True):
                        st.session_state.research_id = None
                        st.session_state.job_status = None
                        st.session_state.error_msg = None
                        st.rerun()

            except Exception as e:
                st.error(f"Failed to display report: {e}")

        # Step 4: Failed State
        elif st.session_state.job_status == "failed":
            st.error(f"❌ Swarm Execution Failed: {st.session_state.error_msg or 'Unknown Error'}")
            
            if st.button("Try Another Research Query", use_container_width=True):
                st.session_state.research_id = None
                st.session_state.job_status = None
                st.session_state.error_msg = None
                st.rerun()


if __name__ == "__main__":
    main()
