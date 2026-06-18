"""Prompts for the Researcher agent."""

from __future__ import annotations

QUERY_GENERATION_TEMPLATE = """You are a business research analyst.
Generate targeted web search queries for the research request.

Original query:
{query}

Business context:
{business_context}

Research tasks:
{tasks}

Retry reason:
{retry_reason}

Return strict JSON only with this shape:
{{
  "queries": [
    {{"query": "search query", "rationale": "why this query helps", "task_id": "task id or null"}}
  ]
}}

Rules:
- Prefer official sources, financial filings, credible market data, and recent reports.
- Include specific company, geography, date, market, or competitor terms when available.
- Avoid duplicates.
- Return at most {max_queries} queries.
"""
