"""Deterministic structured extraction utilities for the Analyst agent."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from app.workflows.business_research.state import (
    AnalysisFinding,
    EvidenceItem,
    ExtractedMetric,
    ExtractedTrend,
    Source,
)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
METRIC_RE = re.compile(
    r"(?P<value>(?:[$₹€£]\s*)?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage points|bps|x|times|million|billion|trillion|crore|lakh|bn|mn|m|k)?)",
    flags=re.IGNORECASE,
)
TREND_PATTERNS = {
    "increasing": re.compile(
        r"\b(grow|growing|growth|increase|increasing|rise|rising|accelerat|expand|surge|gain)\b",
        flags=re.IGNORECASE,
    ),
    "decreasing": re.compile(
        r"\b(decline|declining|decrease|decreasing|fall|falling|drop|shrinking|slowdown|contract)\b",
        flags=re.IGNORECASE,
    ),
    "stable": re.compile(r"\b(stable|flat|steady|unchanged|plateau)\b", flags=re.IGNORECASE),
}
RISK_RE = re.compile(r"\b(risk|challenge|threat|headwind|constraint|shortage|regulatory|margin pressure)\b", re.IGNORECASE)
OPPORTUNITY_RE = re.compile(r"\b(opportunity|tailwind|demand|growth|expansion|adoption|investment)\b", re.IGNORECASE)
BUSINESS_SIGNAL_RE = re.compile(
    r"\b(market|revenue|sales|growth|share|margin|profit|cost|demand|supply|customer|competitor|investment|forecast|adoption|valuation|capacity|pricing)\b",
    flags=re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]


def evidence_by_id(evidence: Iterable[EvidenceItem]) -> dict[str, EvidenceItem]:
    return {item.id: item for item in evidence}


def source_by_id(sources: Iterable[Source]) -> dict[str, Source]:
    return {source.id: source for source in sources}


def source_credibility(source_id: str, sources: dict[str, Source]) -> float:
    source = sources.get(source_id)
    return source.credibility_score if source else 0.5


def sentence_confidence(sentence: str, evidence: EvidenceItem, sources: dict[str, Source]) -> float:
    signal_bonus = 0.15 if BUSINESS_SIGNAL_RE.search(sentence) else 0.0
    score = (evidence.relevance_score * 0.55) + (source_credibility(evidence.source_id, sources) * 0.30) + signal_bonus
    return max(0.0, min(1.0, score))


def extract_findings(
    evidence_items: list[EvidenceItem],
    sources: dict[str, Source],
    *,
    max_findings: int,
    min_sentence_chars: int,
    min_confidence: float,
) -> tuple[AnalysisFinding, ...]:
    candidates: list[AnalysisFinding] = []

    for evidence in evidence_items:
        for sentence in split_sentences(evidence.text):
            if len(sentence) < min_sentence_chars or not BUSINESS_SIGNAL_RE.search(sentence):
                continue
            confidence = sentence_confidence(sentence, evidence, sources)
            if confidence < min_confidence:
                continue
            candidates.append(
                AnalysisFinding(
                    claim=sentence,
                    evidence_ids=(evidence.id,),
                    confidence=confidence,
                    implication=_infer_implication(sentence),
                )
            )

    candidates.sort(key=lambda finding: finding.confidence, reverse=True)
    return tuple(_dedupe_findings(candidates)[:max_findings])


def extract_metrics(
    evidence_items: list[EvidenceItem],
    sources: dict[str, Source],
    *,
    max_metrics: int,
) -> tuple[ExtractedMetric, ...]:
    metrics: list[ExtractedMetric] = []

    for evidence in evidence_items:
        for sentence in split_sentences(evidence.text):
            for match in METRIC_RE.finditer(sentence):
                value = " ".join(match.group("value").split())
                if not any(char.isdigit() for char in value):
                    continue
                name = _metric_name(sentence, value)
                metrics.append(
                    ExtractedMetric(
                        name=name,
                        value=value,
                        unit=_metric_unit(value),
                        context=sentence,
                        evidence_ids=(evidence.id,),
                        confidence=sentence_confidence(sentence, evidence, sources),
                    )
                )

    metrics.sort(key=lambda metric: metric.confidence, reverse=True)
    return tuple(_dedupe_metrics(metrics)[:max_metrics])


def extract_trends(
    evidence_items: list[EvidenceItem],
    sources: dict[str, Source],
    *,
    max_trends: int,
) -> tuple[ExtractedTrend, ...]:
    trends: list[ExtractedTrend] = []

    for evidence in evidence_items:
        for sentence in split_sentences(evidence.text):
            direction = _trend_direction(sentence)
            if direction == "unknown":
                continue
            trends.append(
                ExtractedTrend(
                    topic=_trend_topic(sentence),
                    direction=direction,
                    description=sentence,
                    evidence_ids=(evidence.id,),
                    confidence=sentence_confidence(sentence, evidence, sources),
                )
            )

    trends.sort(key=lambda trend: trend.confidence, reverse=True)
    return tuple(_dedupe_trends(trends)[:max_trends])


def extract_risks(evidence_items: list[EvidenceItem], *, limit: int = 5) -> tuple[str, ...]:
    risks = []
    for evidence in evidence_items:
        for sentence in split_sentences(evidence.text):
            if RISK_RE.search(sentence):
                risks.append(sentence)
    return tuple(_dedupe_text(risks)[:limit])


def extract_opportunities(evidence_items: list[EvidenceItem], *, limit: int = 5) -> tuple[str, ...]:
    opportunities = []
    for evidence in evidence_items:
        for sentence in split_sentences(evidence.text):
            if OPPORTUNITY_RE.search(sentence):
                opportunities.append(sentence)
    return tuple(_dedupe_text(opportunities)[:limit])


def confidence_score(
    findings: tuple[AnalysisFinding, ...],
    metrics: tuple[ExtractedMetric, ...],
    trends: tuple[ExtractedTrend, ...],
    evidence_items: list[EvidenceItem],
    sources: dict[str, Source],
) -> float:
    if not evidence_items:
        return 0.0

    avg_evidence = sum(item.relevance_score for item in evidence_items) / len(evidence_items)
    avg_source = sum(source_credibility(item.source_id, sources) for item in evidence_items) / len(evidence_items)
    finding_score = _average([finding.confidence for finding in findings])
    metric_bonus = min(len(metrics) / 6, 1.0) * 0.10
    trend_bonus = min(len(trends) / 4, 1.0) * 0.10

    score = (avg_evidence * 0.35) + (avg_source * 0.25) + (finding_score * 0.30) + metric_bonus + trend_bonus
    return round(max(0.0, min(1.0, score)), 3)


def summarize(query: str, findings: tuple[AnalysisFinding, ...], metrics: tuple[ExtractedMetric, ...]) -> str:
    if findings:
        return f"{query}: {findings[0].claim}"
    if metrics:
        return f"{query}: key quantified evidence includes {metrics[0].value} for {metrics[0].name}."
    return f"{query}: insufficient evidence was available for a high-confidence analysis."


def recommendations_from_findings(findings: tuple[AnalysisFinding, ...], *, limit: int = 4) -> tuple[str, ...]:
    recommendations = []
    for finding in findings[:limit]:
        recommendations.append(f"Use this evidence in the final report: {finding.claim}")
    return tuple(recommendations)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _infer_implication(sentence: str) -> str | None:
    if RISK_RE.search(sentence):
        return "This may create execution or market risk."
    if OPPORTUNITY_RE.search(sentence):
        return "This may indicate a market opportunity or positive demand signal."
    return None


def _metric_name(sentence: str, value: str) -> str:
    before_value = sentence.split(value, 1)[0]
    tokens = re.findall(r"[A-Za-z][A-Za-z-]+", before_value)[-5:]
    name = " ".join(tokens).strip()
    return name or "reported metric"


def _metric_unit(value: str) -> str | None:
    match = re.search(r"(%|percent|percentage points|bps|x|times|million|billion|trillion|crore|lakh|bn|mn|m|k)$", value, re.I)
    return match.group(1) if match else None


def _trend_direction(sentence: str) -> str:
    matches = [direction for direction, pattern in TREND_PATTERNS.items() if pattern.search(sentence)]
    if len(set(matches)) > 1:
        return "mixed"
    return matches[0] if matches else "unknown"


def _trend_topic(sentence: str) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z-]+", sentence)
    stopwords = {"the", "and", "for", "with", "from", "that", "this", "into", "were", "was", "are"}
    common = Counter(token.lower() for token in tokens if token.lower() not in stopwords)
    if not common:
        return "business trend"
    return " ".join(token for token, _ in common.most_common(3))


def _dedupe_findings(findings: list[AnalysisFinding]) -> list[AnalysisFinding]:
    seen: set[str] = set()
    unique = []
    for finding in findings:
        key = _dedupe_key(finding.claim)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def _dedupe_metrics(metrics: list[ExtractedMetric]) -> list[ExtractedMetric]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for metric in metrics:
        key = (_dedupe_key(metric.name), metric.value.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(metric)
    return unique


def _dedupe_trends(trends: list[ExtractedTrend]) -> list[ExtractedTrend]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for trend in trends:
        key = (_dedupe_key(trend.topic), trend.direction)
        if key in seen:
            continue
        seen.add(key)
        unique.append(trend)
    return unique


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for item in items:
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _dedupe_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()[:120]
