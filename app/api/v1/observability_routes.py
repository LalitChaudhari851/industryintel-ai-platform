"""FastAPI routes for fetching LangSmith observability statistics."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["observability"])


@router.get("/observability/stats")
async def get_observability_stats(
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Fetch aggregated observability runs and custom metadata from LangSmith."""
    api_key = os.getenv("LANGSMITH_API_KEY") or settings.langsmith_api_key
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() in {"true", "1", "yes"}

    if not api_key or not tracing_enabled:
        return {
            "configured": False,
            "message": "LangSmith tracing is disabled or API key is not configured.",
            "average_latencies": {},
            "agent_success_rates": {},
            "retry_frequency": {},
            "quality_trends": [],
        }

    try:
        from langsmith import Client

        client = Client(
            api_url=os.getenv("LANGSMITH_ENDPOINT") or settings.langsmith_endpoint,
            api_key=api_key,
        )

        logger.info("Fetching runs for project: %s", settings.langsmith_project)
        # Fetch the last 100 runs in the project
        runs = list(
            client.list_runs(
                project_name=settings.langsmith_project,
                limit=100,
            )
        )

        agent_names = {"PlannerAgent", "ResearcherAgent", "AnalystAgent", "CriticAgent", "WriterAgent"}
        
        agent_latencies: Dict[str, List[float]] = {name: [] for name in agent_names}
        agent_success: Dict[str, int] = {name: 0 for name in agent_names}
        agent_failures: Dict[str, int] = {name: 0 for name in agent_names}
        
        retry_frequency: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        quality_trends: List[Dict[str, Any]] = []

        for run in runs:
            # 1. Capture Agent level metrics
            if run.name in agent_names:
                if run.start_time and run.end_time:
                    latency = (run.end_time - run.start_time).total_seconds()
                    agent_latencies[run.name].append(latency)

                if run.error:
                    agent_failures[run.name] += 1
                else:
                    agent_success[run.name] += 1

            # 2. Capture Parent Workflow custom metadata
            if run.name == "business-research-workflow":
                meta = run.metadata or {}
                q_score = meta.get("quality_score")
                s_count = meta.get("source_count")
                retries = meta.get("retry_count", 0)

                # Track retry frequencies
                try:
                    r_val = int(retries)
                    retry_frequency[r_val] = retry_frequency.get(r_val, 0) + 1
                except (ValueError, TypeError):
                    pass

                quality_trends.append(
                    {
                        "report_id": meta.get("report_id", str(run.id)),
                        "topic": meta.get("topic", meta.get("query", "Unknown Topic")),
                        "quality_score": float(q_score) if q_score is not None else None,
                        "source_count": int(s_count) if s_count is not None else None,
                        "retry_count": int(retries),
                        "timestamp": run.start_time.isoformat() if run.start_time else None,
                        "status": "FAILED" if run.error else "COMPLETED",
                    }
                )

        # Compute averages
        average_latencies = {}
        agent_success_rates = {}

        for name in agent_names:
            lats = agent_latencies[name]
            average_latencies[name] = round(sum(lats) / len(lats), 2) if lats else 0.0

            total = agent_success[name] + agent_failures[name]
            agent_success_rates[name] = (
                round((agent_success[name] / total) * 100, 1) if total > 0 else 100.0
            )

        # Sort quality trends chronologically
        quality_trends.reverse()

        return {
            "configured": True,
            "average_latencies": average_latencies,
            "agent_success_rates": agent_success_rates,
            "retry_frequency": retry_frequency,
            "quality_trends": quality_trends,
        }

    except Exception as e:
        logger.error("Failed to query LangSmith run metrics: %s", e)
        return {
            "configured": False,
            "message": f"Connection to LangSmith failed: {e}",
            "average_latencies": {},
            "agent_success_rates": {},
            "retry_frequency": {},
            "quality_trends": [],
        }
