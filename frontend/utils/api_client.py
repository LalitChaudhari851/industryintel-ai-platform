"""API Client for interacting with the FastAPI backend."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class APIClient:
    """Async API client for communicating with the backend FastAPI service."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    async def create_research(
        self, query: str, business_context: Optional[str] = None, max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Submit a new research job to the backend."""
        payload = {
            "query": query,
            "business_context": business_context,
            "max_iterations": max_iterations,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(f"{self.base_url}/research", json=payload)
            res.raise_for_status()
            return res.json()

    async def get_status(self, research_id: str) -> Dict[str, Any]:
        """Get the high-level status of a research job."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/research/{research_id}/status")
            res.raise_for_status()
            return res.json()

    async def get_report(self, research_id: str) -> Dict[str, Any]:
        """Retrieve the completed report of a research job."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/research/{research_id}/report")
            res.raise_for_status()
            return res.json()

    async def get_detail(self, research_id: str) -> Dict[str, Any]:
        """Retrieve full details of a research job, including plan and review."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/research/{research_id}")
            res.raise_for_status()
            return res.json()

    async def check_models(self) -> Dict[str, Any]:
        """Check the connection and loading status of Ollama models on the backend."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/models/status")
            res.raise_for_status()
            return res.json()

    async def get_observability_stats(self) -> Dict[str, Any]:
        """Retrieve LangSmith observability metrics from FastAPI backend."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{self.base_url}/observability/stats")
            res.raise_for_status()
            return res.json()

    async def get_review(self, report_id: str) -> Dict[str, Any]:
        """Retrieve review details and report draft for pending review."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/review/{report_id}")
            res.raise_for_status()
            return res.json()

    async def submit_review(self, report_id: str, approval_status: str, comments: Optional[str] = None) -> Dict[str, Any]:
        """Submit review decision to backend FastAPI service."""
        payload = {
            "approval_status": approval_status,
            "reviewer_comments": comments,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(f"{self.base_url}/review/{report_id}", json=payload)
            res.raise_for_status()
            return res.json()

    async def get_report_pdf(self, research_id: str) -> bytes:
        """Download executive briefing PDF from backend FastAPI service."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.get(f"{self.base_url}/research/{research_id}/pdf")
            res.raise_for_status()
            return res.content

    async def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Fetch aggregated evaluation metrics from backend."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/evaluation/metrics")
            res.raise_for_status()
            return res.json()

    async def get_evaluation_reports(self) -> List[Dict[str, Any]]:
        """Fetch individual report evaluation metrics from backend."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/evaluation/reports")
            res.raise_for_status()
            return res.json()

    async def get_evaluation_trends(self) -> List[Dict[str, Any]]:
        """Fetch chronological evaluation trends from backend."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{self.base_url}/evaluation/trends")
            res.raise_for_status()
            return res.json()
