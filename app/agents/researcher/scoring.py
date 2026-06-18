"""Source normalization, deduplication, and relevance scoring."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.agents.researcher.models import RankedSearchResult, SearchQueryCandidate, TavilyResult

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9$%.-]*")
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "igshid"}

HIGH_CREDIBILITY_SUFFIXES = (
    ".gov",
    ".edu",
)
HIGH_CREDIBILITY_HOST_PARTS = (
    "sec.gov",
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "rbi.org.in",
    "commerce.gov",
    "investor.",
    "ir.",
)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key not in TRACKING_QUERY_KEYS
        and not any(key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    normalized_query = urlencode(sorted(query_items))
    path = parsed.path.rstrip("/") or "/"

    return urlunparse((scheme, netloc, path, "", normalized_query, ""))


def lexical_relevance(query: str, title: str, content: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    document_tokens = tokenize(f"{title} {content}")
    if not document_tokens:
        return 0.0

    overlap = len(query_tokens & document_tokens)
    coverage = overlap / len(query_tokens)
    density = overlap / math.sqrt(max(len(document_tokens), 1))
    return min(1.0, (coverage * 0.75) + (density * 0.25))


def credibility_score(url: str) -> float:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    score = 0.55
    if any(host.endswith(suffix) for suffix in HIGH_CREDIBILITY_SUFFIXES):
        score += 0.25
    if any(part in host for part in HIGH_CREDIBILITY_HOST_PARTS):
        score += 0.25
    if any(part in host for part in ("wikipedia.org", "medium.com", "substack.com")):
        score -= 0.1
    return max(0.0, min(1.0, score))


def blended_relevance(query: str, result: TavilyResult) -> float:
    lexical = lexical_relevance(query, result.title, result.content or result.raw_content or "")
    tavily_score = result.score or 0.0
    return max(0.0, min(1.0, (tavily_score * 0.65) + (lexical * 0.35)))


def deduplicate_and_rank(
    query_results: Iterable[tuple[SearchQueryCandidate, TavilyResult]],
    *,
    min_relevance_score: float,
    max_sources: int,
    reranker: Any | None = None,
) -> list[RankedSearchResult]:
    import logging
    logger = logging.getLogger(__name__)

    by_url: dict[str, RankedSearchResult] = {}
    candidates_to_rerank = []

    for query_candidate, result in query_results:
        normalized = normalize_url(str(result.url))
        relevance = blended_relevance(query_candidate.query, result)
        
        candidates_to_rerank.append({
            "query": query_candidate.query,
            "title": result.title,
            "text": result.content or result.raw_content or "",
            "result": result,
            "query_candidate": query_candidate,
            "normalized_url": normalized,
            "relevance": relevance,
        })

    if reranker is not None and candidates_to_rerank:
        pairs = [[item["query"], item["text"]] for item in candidates_to_rerank]
        try:
            logger.info("Reranking %d candidates with BGE Reranker...", len(candidates_to_rerank))
            scores = reranker.model.predict(pairs)
            for item, score in zip(candidates_to_rerank, scores):
                # Sigmoid normalization of the logit score
                sigmoid_score = 1.0 / (1.0 + math.exp(-float(score)))
                # Blended relevance and rerank score combination
                item["relevance"] = (sigmoid_score * 0.70) + (item["relevance"] * 0.30)
        except Exception as e:
            logger.error("Reranker prediction failed in deduplicate_and_rank: %s", e)

    for item in candidates_to_rerank:
        relevance = item["relevance"]
        if relevance < min_relevance_score:
            continue

        ranked = RankedSearchResult(
            query=item["query"],
            result=item["result"],
            task_id=item["query_candidate"].task_id,
            relevance_score=relevance,
            credibility_score=credibility_score(item["normalized_url"]),
        )

        current = by_url.get(item["normalized_url"])
        if current is None or ranked.relevance_score > current.relevance_score:
            by_url[item["normalized_url"]] = ranked

    return sorted(
        by_url.values(),
        key=lambda item: (item.relevance_score, item.credibility_score),
        reverse=True,
    )[:max_sources]
