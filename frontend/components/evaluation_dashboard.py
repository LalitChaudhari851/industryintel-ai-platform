"""Evaluation Dashboard component for displaying research quality, agent, report, and trace metrics."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Any, Dict, List


def render_evaluation_dashboard(metrics: Dict[str, Any], reports: List[Dict[str, Any]], trends: List[Dict[str, Any]]) -> None:
    """Renders the comprehensive research evaluation dashboard."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Research Swarm Evaluation Dashboard")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 0.95rem; margin-top: -10px; margin-bottom: 20px;'>"
        "Continuous evaluation scores across Research Quality, Agent Execution efficiency, Report Quality, and LangSmith metadata.</p>",
        unsafe_allow_html=True,
    )

    # Extract metrics categories
    research_quality = metrics.get("research_quality", {})
    agent_metrics = metrics.get("agent_metrics", {})
    report_metrics = metrics.get("report_metrics", {})
    ls_metrics = metrics.get("langsmith_metrics", {})

    # 1. KPI Sections Tabs
    tab_quality, tab_report, tab_agents, tab_traces = st.tabs([
        "🔍 Research Quality", 
        "📝 Report Quality", 
        "🤖 Agent Telemetry", 
        "🛠️ Trace Observability"
    ])

    with tab_quality:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Source Count", f"{research_quality.get('source_count', 0.0)}")
        with col2:
            st.metric("Source Diversity (Domains)", f"{research_quality.get('source_diversity', 0.0)}")
        with col3:
            st.metric("Citation Coverage", f"{research_quality.get('citation_coverage', 0.0)}%")
        with col4:
            st.metric("Retrieval Precision", f"{research_quality.get('retrieval_precision', 0.0)}%")
        with col5:
            st.metric("Retrieval Recall", f"{research_quality.get('retrieval_recall', 0.0)}%")

    with tab_report:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Critic Quality Score", f"{report_metrics.get('quality_score', 0.0)}%")
        with col2:
            st.metric("Confidence Score", f"{report_metrics.get('confidence_score', 0.0)}%")
        with col3:
            st.metric("Grounding Score", f"{report_metrics.get('grounding_score', 0.0)}%")
        with col4:
            st.metric("Human Approval Rate", f"{report_metrics.get('approval_rate', 0.0)}%")
        with col5:
            st.metric("Human Rejection Rate", f"{report_metrics.get('rejection_rate', 0.0)}%")

    with tab_agents:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Agent Success Rate", f"{agent_metrics.get('agent_success_rate', 0.0)}%")
        with col2:
            st.metric("Agent Failure Rate", f"{agent_metrics.get('agent_failure_rate', 0.0)}%")
        with col3:
            st.metric("Avg Retry Loops", f"{agent_metrics.get('retry_frequency', 0.0)}")
        with col4:
            st.metric("Avg Execution Time", f"{agent_metrics.get('average_execution_time', 0.0)}s")

    with tab_traces:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("LangSmith Active Traces", f"{ls_metrics.get('trace_count', 0)}")
        with col2:
            st.metric("System Error Rate", f"{ls_metrics.get('error_rate', 0.0)}%")
        with col3:
            st.metric("Tracing Configuration Status", "Active" if ls_metrics.get("configured", False) else "Disabled (Local Fallback)")

    st.write("")
    st.markdown("---")

    # 2. Performance Comparison & Latencies Grid
    col_chart_left, col_chart_right = st.columns(2)

    with col_chart_left:
        st.markdown("### 🤖 Agent Efficiency & Success Rates")
        agents_data = agent_metrics.get("agents", {})
        if agents_data:
            df_agents = pd.DataFrame([
                {
                    "Agent": name.replace("Agent", ""),
                    "Avg Latency (s)": details.get("avg_latency"),
                    "Success Rate (%)": details.get("success_rate")
                }
                for name, details in agents_data.items()
            ])
            # Display bar chart of latencies
            st.bar_chart(df_agents.set_index("Agent")["Avg Latency (s)"])
        else:
            st.info("No detailed agent analytics logged yet.")

    with col_chart_right:
        st.markdown("### 🔄 Agent Workflow Success Rates")
        if agents_data:
            st.bar_chart(df_agents.set_index("Agent")["Success Rate (%)"])
        else:
            st.info("No detailed agent success rates available.")

    st.markdown("---")

    # 3. Quality & Latency Trends Over Time
    st.markdown("### 📈 Historical Evaluation Timeline")
    if trends:
        df_trends = pd.DataFrame(trends)
        
        # Chronological sorting and formatting
        df_trends["timestamp"] = pd.to_datetime(df_trends["timestamp"])
        df_trends.sort_values(by="timestamp", inplace=True)
        df_trends["date_label"] = df_trends["timestamp"].dt.strftime("%m-%d %H:%M")
        
        col_trend_l, col_trend_r = st.columns(2)
        
        with col_trend_l:
            st.markdown("**Evaluated Quality & Grounding Score Trends (%)**")
            chart_df = df_trends[["date_label", "quality_score", "grounding_score", "confidence_score"]].copy()
            chart_df.rename(columns={
                "quality_score": "Critic Quality",
                "grounding_score": "Source Grounding",
                "confidence_score": "Writer Confidence"
            }, inplace=True)
            st.line_chart(chart_df.set_index("date_label"))

        with col_trend_r:
            st.markdown("**Workflow Latency Timeline (seconds)**")
            st.area_chart(df_trends.set_index("date_label")["latency"])
    else:
        st.info("No trend points recorded. Completed runs will populate time-series metrics.")

    st.markdown("---")

    # 4. Report Analytics Directory Table
    st.markdown("### 📋 Executive Briefing Telemetry Directory")
    if reports:
        df_reports = pd.DataFrame(reports)
        df_reports.rename(columns={
            "query": "Topic Query",
            "quality_score": "Quality Score (%)",
            "confidence_score": "Confidence Score (%)",
            "grounding_score": "Grounding Score (%)",
            "source_count": "Sources",
            "citation_count": "Citations",
            "source_diversity": "Domain Diversity",
            "duration": "Latency (s)",
            "approval_status": "Human Decision"
        }, inplace=True)

        cols_to_show = [
            "Topic Query", 
            "Quality Score (%)", 
            "Confidence Score (%)", 
            "Grounding Score (%)", 
            "Sources", 
            "Citations",
            "Domain Diversity",
            "Latency (s)",
            "Human Decision"
        ]
        
        st.dataframe(
            df_reports[cols_to_show],
            use_container_width=True
        )
    else:
        st.info("No completed reports cataloged in the briefing system.")

    st.markdown("</div>", unsafe_allow_html=True)
