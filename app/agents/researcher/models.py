"""Pydantic models for the Researcher agent."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, field_validator

from app.workflows.business_research.state import ResearchTask, RetryReason


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ResearchAgentConfig(StrictModel):
    tavily_api_key: SecretStr | None = None
    max_queries: int = Field(default=6, ge=1, le=20)
    results_per_query: int = Field(default=5, ge=1, le=20)
    max_sources: int = Field(default=15, ge=1, le=50)
    min_relevance_score: float = Field(default=0.35, ge=0.0, le=1.0)
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic"
    topic: Literal["general", "news", "finance"] = "general"
    include_raw_content: bool | Literal["markdown", "text"] = False
    include_answer: bool | Literal["basic", "advanced"] = False
    include_usage: bool = True
    timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)
    user_agent: str = "autonomous-research-platform/1.0"
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 5


class SearchContext(StrictModel):
    session_id: str
    query: str = Field(min_length=1)
    business_context: str | None = None
    tasks: tuple[ResearchTask, ...] = Field(default_factory=tuple)
    retry_reason: RetryReason | None = None
    previous_source_urls: tuple[str, ...] = Field(default_factory=tuple)


class SearchQueryCandidate(StrictModel):
    query: str = Field(min_length=3)
    rationale: str = Field(min_length=1)
    task_id: str | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, query: str) -> str:
        return " ".join(query.split())


class SearchQueryPlan(StrictModel):
    queries: tuple[SearchQueryCandidate, ...] = Field(default_factory=tuple)


class TavilySearchRequest(StrictModel):
    query: str
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic"
    topic: Literal["general", "news", "finance"] = "general"
    max_results: int = Field(default=5, ge=1, le=20)
    include_answer: bool | Literal["basic", "advanced"] = False
    include_raw_content: bool | Literal["markdown", "text"] = False
    include_usage: bool = True


class TavilyResult(StrictModel):
    title: str = Field(min_length=1)
    url: HttpUrl
    content: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_content: str | None = None
    published_date: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TavilySearchResponse(StrictModel):
    query: str
    answer: str | None = None
    results: tuple[TavilyResult, ...] = Field(default_factory=tuple)
    response_time: str | None = None
    request_id: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class RankedSearchResult(StrictModel):
    query: str
    result: TavilyResult
    task_id: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    credibility_score: float = Field(ge=0.0, le=1.0)


class SearchClient(Protocol):
    async def search(self, request: TavilySearchRequest) -> TavilySearchResponse:
        """Run a Tavily-compatible web search request."""
