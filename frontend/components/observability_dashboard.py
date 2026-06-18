"""Observability Dashboard component for displaying LangSmith statistics."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Any, Dict


def render_observability_dashboard(stats: Dict[str, Any]) -> None:
    """Renders the observability stats fetched from the backend."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Swarm Observability Dashboard (LangSmith)")

    # 1. Check if configured
    if not stats.get("configured", False):
        st.warning(
            "⚠️ **LangSmith Observability is Not Active**  \n"
            "Tracing is disabled or credentials are not configured in your environment.  \n\n"
            "To enable production-grade tracing, set the following environment variables in your `.env` file:  \n"
            "```bash\n"
            "LANGSMITH_TRACING=true\n"
            "LANGSMITH_API_KEY=lsv2-your-api-key\n"
            "LANGSMITH_PROJECT=ai-industry-intelligence\n"
            "```\n"
            "Then restart the application stack."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Extract metrics
    average_latencies = stats.get("average_latencies", {})
    agent_success_rates = stats.get("agent_success_rates", {})
    retry_frequency = stats.get("retry_frequency", {})
    quality_trends = stats.get("quality_trends", [])

    # High-level Composite Metrics
    col1, col2, col3 = st.columns(3)
    
    total_runs = len(quality_trends)
    avg_latency = 0.0
    if average_latencies:
        avg_latency = round(sum(average_latencies.values()) / len(average_latencies), 1)

    completed_runs = sum(1 for r in quality_trends if r.get("status") == "COMPLETED")
    overall_success = (completed_runs / total_runs * 100) if total_runs > 0 else 100.0

    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Total Runs Tracked</span>
                <h2 style="color: #60a5fa; margin: 4px 0 0 0;">{total_runs}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Avg Agent Latency</span>
                <h2 style="color: #a78bfa; margin: 4px 0 0 0;">{avg_latency}s</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Overall Swarm Success</span>
                <h2 style="color: #34d399; margin: 4px 0 0 0;">{overall_success:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")  # spacing
    st.markdown("---")
    
    # 2. Agent Latency & Performance Breakdown
    st.markdown("### 🤖 Agent Performance Metrics")
    
    # Render table breakdown
    agent_data = []
    for name in sorted(average_latencies.keys()):
        agent_data.append({
            "Agent Node": name.replace("Agent", ""),
            "Average Latency (s)": average_latencies[name],
            "Success Rate (%)": f"{agent_success_rates[name]}%"
        })
    st.table(pd.DataFrame(agent_data))

    # Bar chart for latencies
    if average_latencies:
        st.markdown("**Latency Comparison (seconds)**")
        chart_data = pd.DataFrame({
            "Agent": [name.replace("Agent", "") for name in average_latencies.keys()],
            "Latency": list(average_latencies.values())
        })
        st.bar_chart(chart_data.set_index("Agent"))

    st.markdown("---")

    # 3. Quality trends & Retries Grid
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📈 Quality Score Trend")
        if not quality_trends:
            st.info("No quality data recorded yet.")
        else:
            scores = [r.get("quality_score") for r in quality_trends if r.get("quality_score") is not None]
            topics = [r.get("topic")[:20] + "..." for r in quality_trends if r.get("quality_score") is not None]
            if scores:
                trend_df = pd.DataFrame({
                    "Query Topic": topics,
                    "Quality Score": scores
                })
                st.line_chart(trend_df.set_index("Query Topic"))
            else:
                st.info("No report confidence scores extracted from LangSmith history.")

    with col_right:
        st.markdown("### 🔄 Retry Frequency Distribution")
        if not retry_frequency:
            st.info("No retry frequency data recorded yet.")
        else:
            retries = [f"{k} Retries" for k in sorted(retry_frequency.keys())]
            counts = [retry_frequency[k] for k in sorted(retry_frequency.keys())]
            retry_df = pd.DataFrame({
                "Iteration Loops": retries,
                "Frequency Count": counts
            })
            st.bar_chart(retry_df.set_index("Iteration Loops"))

    st.markdown("---")

    # 4. Human-in-the-Loop Metrics
    st.markdown("### 👥 Human-in-the-Loop Review Metrics")
    approval_metrics = stats.get("approval_metrics", {})
    if approval_metrics and approval_metrics.get("total_decisions", 0) > 0:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Reviews", approval_metrics.get("total_decisions", 0))
        with col_m2:
            st.metric("Avg Approval Latency", f"{approval_metrics.get('average_approval_latency', 0.0)}s")
        with col_m3:
            st.metric("Rejection Rate", f"{approval_metrics.get('rejection_rate', 0.0)}%")
        with col_m4:
            st.metric("Research Request Rate", f"{approval_metrics.get('research_request_rate', 0.0)}%")
            
        # Also render a small breakdown
        st.markdown("**Review Decision Breakdown**")
        breakdown_data = pd.DataFrame({
            "Decision": ["Approvals", "Rejections", "Research Requests"],
            "Count": [
                approval_metrics.get("approvals", 0),
                approval_metrics.get("rejections", 0),
                approval_metrics.get("research_requests", 0)
            ]
        })
        st.bar_chart(breakdown_data.set_index("Decision"))
    else:
        st.info("No Human-in-the-Loop review metrics recorded yet.")

    st.markdown("---")

    # 5. Raw Runs Logs
    st.markdown("### 📋 Historical Run Log (Last 100 parent runs)")
    if quality_trends:
        log_df = pd.DataFrame(quality_trends)
        log_df.rename(columns={
            "topic": "Topic Query",
            "quality_score": "Quality Score",
            "source_count": "Sources Checked",
            "retry_count": "Retries",
            "status": "Final Status",
            "timestamp": "Execution Time"
        }, inplace=True)
        # Format columns for display
        if "Quality Score" in log_df.columns:
            log_df["Quality Score"] = log_df["Quality Score"].map(lambda val: f"{val * 100:.1f}%" if pd.notnull(val) else "N/A")
        if "Execution Time" in log_df.columns:
            log_df["Execution Time"] = log_df["Execution Time"].map(lambda val: val[:16].replace("T", " ") if pd.notnull(val) else "N/A")
            
        st.dataframe(
            log_df[["Topic Query", "Quality Score", "Sources Checked", "Retries", "Final Status", "Execution Time"]],
            use_container_width=True
        )
    else:
        st.info("No run log records retrieved from LangSmith.")

    st.markdown("</div>", unsafe_allow_html=True)
