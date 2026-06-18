"""Tavily Search API client."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.agents.researcher.models import (
    ResearchAgentConfig,
    TavilyResult,
    TavilySearchRequest,
    TavilySearchResponse,
)


class TavilySearchError(RuntimeError):
    """Raised when Tavily search fails."""


class TavilySearchClient:
    endpoint = "https://api.tavily.com/search"

    def __init__(self, config: ResearchAgentConfig) -> None:
        if config.tavily_api_key is None:
            raise ValueError("TAVILY_API_KEY is required for TavilySearchClient")
        self.config = config
        self.api_key = config.tavily_api_key.get_secret_value()

    async def search(self, request: TavilySearchRequest) -> TavilySearchResponse:
        return await asyncio.to_thread(self._search_sync, request)

    def _search_sync(self, request: TavilySearchRequest) -> TavilySearchResponse:
        body = request.model_dump(mode="json", exclude_none=True)
        http_request = Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
            },
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TavilySearchError(f"Tavily HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise TavilySearchError(f"Tavily request failed: {exc}") from exc

        return self._parse_response(payload, request.query)

    def _parse_response(self, payload: dict[str, Any], query: str) -> TavilySearchResponse:
        results = []
        for item in payload.get("results", []):
            metadata = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "title",
                    "url",
                    "content",
                    "score",
                    "raw_content",
                    "published_date",
                }
            }
            results.append(
                TavilyResult(
                    title=item.get("title") or "Untitled source",
                    url=item["url"],
                    content=item.get("content") or "",
                    score=float(item.get("score") or 0.0),
                    raw_content=item.get("raw_content"),
                    published_date=item.get("published_date"),
                    metadata=metadata,
                )
            )

        return TavilySearchResponse(
            query=payload.get("query") or query,
            answer=payload.get("answer"),
            results=tuple(results),
            response_time=payload.get("response_time"),
            request_id=payload.get("request_id"),
            usage=payload.get("usage") or {},
        )
