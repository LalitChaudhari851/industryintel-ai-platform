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
from frontend.components.evaluation_dashboard import render_evaluation_dashboard
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
        ["🔍 Research Workspace", "📈 Evaluation Dashboard", "📊 Observability Dashboard"],
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

    # 2. Evaluation Dashboard Page
    elif app_mode == "📈 Evaluation Dashboard":
        st.markdown(
            '<h1>📈 <span class="gradient-text">Evaluation Dashboard</span></h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 25px;'>"
            "Analyze system-wide metrics across research quality, agent precision, briefing scores, and execution trends.</p>",
            unsafe_allow_html=True,
        )

        with st.spinner("Fetching evaluation records & trends..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                metrics = loop.run_until_complete(api_client.get_evaluation_metrics())
                reports = loop.run_until_complete(api_client.get_evaluation_reports())
                trends = loop.run_until_complete(api_client.get_evaluation_trends())
                render_evaluation_dashboard(metrics, reports, trends)
            except Exception as e:
                st.error(f"Failed to load evaluation dashboard: {e}")

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

        # Step 2.5: Pending Review State
        elif st.session_state.job_status == "pending_review":
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("👥 Human-in-the-Loop Approval Required")

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                review_data = loop.run_until_complete(api_client.get_review(st.session_state.research_id))

                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    st.metric("Critic Quality Score", f"{review_data.get('critic_score', 0.0) * 100:.1f}%" if review_data.get('critic_score') is not None else "N/A")
                with col_c2:
                    st.metric("Analyst Confidence Score", f"{review_data.get('confidence_score', 0.0) * 100:.1f}%" if review_data.get('confidence_score') is not None else "N/A")
                with col_c3:
                    st.metric("Sources Gathered", review_data.get("source_count", 0))

                st.write("")

                # Show report draft (Findings and Metrics)
                draft = review_data.get("report_draft")
                if draft:
                    st.markdown("### 📝 Draft Findings & Summary")
                    st.write(draft.get("summary", "No summary provided."))

                    st.markdown("#### Key Findings")
                    for finding in draft.get("findings", []):
                        st.markdown(f"- **{finding.get('claim')}** (Confidence: {finding.get('confidence', 0.0) * 100:.0f}%)")
                        if finding.get("implication"):
                            st.caption(f"  *Implication: {finding.get('implication')}*")

                    st.markdown("#### Extracted Metrics")
                    for metric in draft.get("metrics", []):
                        st.markdown(f"- **{metric.get('name')}**: {metric.get('value')} {metric.get('unit') or ''}  \n  *{metric.get('context')}*")
                else:
                    st.info("No draft content compiled yet.")

                st.write("")
                st.markdown("---")
                st.markdown("### ✍️ Reviewer Feedback")
                comments = st.text_area("Provide feedback / comments (optional for approval, required for rejection/research requests)", key="reviewer_comments")

                # Review Buttons
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    if st.button("✅ Approve Draft", use_container_width=True):
                        with st.spinner("Submitting approval..."):
                            loop.run_until_complete(api_client.submit_review(st.session_state.research_id, "approved", comments))
                            st.session_state.job_status = "running"
                            st.success("Draft approved! Resuming workflow...")
                            time.sleep(1.5)
                            st.rerun()
                with col_b2:
                    if st.button("❌ Reject (Send to Analyst)", use_container_width=True):
                        if not comments:
                            st.error("Please enter comments explaining the rejection reasons.")
                        else:
                            with st.spinner("Submitting rejection..."):
                                loop.run_until_complete(api_client.submit_review(st.session_state.research_id, "rejected", comments))
                                st.session_state.job_status = "running"
                                st.warning("Draft rejected. Returning to Analyst agent...")
                                time.sleep(1.5)
                                st.rerun()
                with col_b3:
                    if st.button("🔄 Request More Research (Send to Researcher)", use_container_width=True):
                        if not comments:
                            st.error("Please enter comments explaining the research gap.")
                        else:
                            with st.spinner("Submitting research request..."):
                                loop.run_until_complete(api_client.submit_review(st.session_state.research_id, "request_more_research", comments))
                                st.session_state.job_status = "running"
                                st.info("Request submitted. Returning to Researcher agent...")
                                time.sleep(1.5)
                                st.rerun()

                # Show history if present
                history = review_data.get("review_history", [])
                if history:
                    st.markdown("---")
                    st.markdown("### 🕒 Review History")
                    for idx, record in enumerate(reversed(history)):
                        status_label = record.get('status', 'unknown').upper()
                        st.markdown(
                            f"**Review #{len(history) - idx}** — *{status_label}*  \n"
                            f"Comments: {record.get('comments') or 'None'}  \n"
                            f"Time: {record.get('timestamp')[:16].replace('T', ' ')}"
                        )

            except Exception as e:
                st.error(f"Error loading review interface: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

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
                    render_download_buttons(
                        report,
                        sources_list,
                        api_client=api_client,
                        research_id=st.session_state.research_id,
                    )
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
