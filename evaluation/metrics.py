"""Quality metrics calculator for research reports."""

from __future__ import annotations

from urllib.parse import urlparse
from typing import Any, Dict, List


def calculate_source_diversity(sources: List[Dict[str, Any]]) -> float:
    """Calculates domain diversity: unique domains divided by total sources.

    Returns a score between 0.0 and 1.0.
    """
    if not sources:
        return 0.0

    domains = set()
    for src in sources:
        url = src.get("url")
        if url:
            try:
                domain = urlparse(str(url)).netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
                domains.add(domain)
            except Exception:
                pass

    return len(domains) / len(sources)


def calculate_citation_coverage(citations: List[Dict[str, Any]], sections: Dict[str, str]) -> float:
    """Calculates citation density: ratio of sentences with citations to total sentences in sections."""
    if not sections:
        return 0.0

    total_sentences = 0
    cited_sentences = 0

    for content in sections.values():
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        total_sentences += len(sentences)
        for s in sentences:
            if any(marker in s for marker in ["[S", "[s"]):
                cited_sentences += 1

    if total_sentences == 0:
        return 1.0

    return min(1.0, cited_sentences / total_sentences)


def calculate_composite_score(
    diversity: float,
    coverage: float,
    confidence: float,
    accuracy: float,
) -> float:
    """Weighted composite score for research quality."""
    weights = {
        "diversity": 0.20,
        "coverage": 0.25,
        "confidence": 0.25,
        "accuracy": 0.30,
    }
    return (
        (diversity * weights["diversity"])
        + (coverage * weights["coverage"])
        + (confidence * weights["confidence"])
        + (accuracy * weights["accuracy"])
    )
