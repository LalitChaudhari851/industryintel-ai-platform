"""Application configuration."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "AI Industry Intelligence Platform"
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    tavily_api_key: str | None = None
    max_concurrent_research_jobs: int = Field(default=8, ge=1, le=128)

    # Ollama LLM
    ollama_base_url: str = "http://localhost:11434"
    primary_model: str = "qwen3:8b"
    fallback_model: str = "llama3.1:8b"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=256, le=32768)
    llm_timeout: int = Field(default=120, ge=10, le=600)

    # Embedding & Reranker
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    faiss_index_path: str = "./data/faiss_index"
    faiss_top_k: int = Field(default=10, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-industry-intelligence"
    langsmith_endpoint: str | None = None
    langsmith_workspace_id: str | None = None

    @property
    def is_development(self) -> bool:
        return self.environment.lower() in {"dev", "development", "local"}


@lru_cache
def get_settings() -> Settings:
    return Settings(
        environment=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_concurrent_research_jobs=int(os.getenv("MAX_CONCURRENT_RESEARCH_JOBS", "8")),
        # Ollama
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        primary_model=os.getenv("PRIMARY_MODEL", "qwen3:8b"),
        fallback_model=os.getenv("FALLBACK_MODEL", "llama3.1:8b"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        llm_timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        # Embedding & Reranker
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"),
        faiss_top_k=int(os.getenv("FAISS_TOP_K", "10")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
        # LangSmith
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower()
        in {"1", "true", "yes", "on"},
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "ai-industry-intelligence"),
        langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT"),
        langsmith_workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID"),
    )
