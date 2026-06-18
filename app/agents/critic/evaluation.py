"""Structured evidence verification for the Critic agent."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.agents.critic.models import CriticAgentConfig
from app.workflows.business_research.state import (
    AnalysisFinding,
    AnalysisResult,
    Contradiction,
    CriticFinding,
    EvidenceAssessment,
    EvidenceItem,
    QualityDecision,
    RetryReason,
    Source,
)

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9$%.-]*")
INCREASE_RE = re.compile(r"\b(grow|growth|increase|increasing|rise|rising|surge|expand|gain)\b", re.I)
DECREASE_RE = re.compile(r"\b(decline|decrease|decreasing|fall|falling|drop|shrink|contract)\b", re.I)


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}


def overlap_score(claim: str, evidence_text: str) -> float:
    claim_tokens = token_set(claim)
    if not claim_tokens:
        return 0.0
    evidence_tokens = token_set(evidence_text)
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def assess_findings(
    analysis: AnalysisResult,
    evidence_by_id: dict[str, EvidenceItem],
    *,
    config: CriticAgentConfig,
) -> tuple[EvidenceAssessment, ...]:
    assessments = []
    for finding in analysis.findings:
        supporting_items = [evidence_by_id[item_id] for item_id in finding.evidence_ids if item_id in evidence_by_id]
        support_score = _support_score(finding, supporting_items)
        assessments.append(
            EvidenceAssessment(
                claim=finding.claim,
                supported=support_score >= config.min_token_overlap,
                evidence_ids=tuple(item.id for item in supporting_items),
                support_score=round(support_score, 3),
                explanation=(
                    "Claim has lexical support in cited evidence."
                    if support_score >= config.min_token_overlap
                    else "Claim has weak or missing lexical support in cited evidence."
                ),
            )
        )
    return tuple(assessments)


def detect_contradictions(
    analysis: AnalysisResult,
    evidence_items: Iterable[EvidenceItem],
) -> tuple[Contradiction, ...]:
    contradictions: list[Contradiction] = []

    trend_directions: dict[str, set[str]] = {}
    trend_evidence: dict[str, set[str]] = {}
    for trend in analysis.trends:
        key = _topic_key(trend.topic)
        trend_directions.setdefault(key, set()).add(trend.direction)
        trend_evidence.setdefault(key, set()).update(trend.evidence_ids)

    for topic, directions in trend_directions.items():
        if "increasing" in directions and "decreasing" in directions:
            contradictions.append(
                Contradiction(
                    description=f"Trend direction conflict detected for {topic}.",
                    evidence_ids=tuple(sorted(trend_evidence.get(topic, set()))),
                    severity="high",
                )
            )

    evidence_texts = list(evidence_items)
    for left_index, left in enumerate(evidence_texts):
        for right in evidence_texts[left_index + 1 :]:
            shared_tokens = token_set(left.text) & token_set(right.text)
            if len(shared_tokens) < 4:
                continue
            left_inc = bool(INCREASE_RE.search(left.text))
            left_dec = bool(DECREASE_RE.search(left.text))
            right_inc = bool(INCREASE_RE.search(right.text))
            right_dec = bool(DECREASE_RE.search(right.text))
            if (left_inc and right_dec) or (left_dec and right_inc):
                contradictions.append(
                    Contradiction(
                        description="Evidence contains opposing directional signals for overlapping topics.",
                        evidence_ids=(left.id, right.id),
                        severity="medium",
                    )
                )

    return tuple(contradictions[:5])


def quality_score(
    *,
    analysis: AnalysisResult,
    evidence_assessments: tuple[EvidenceAssessment, ...],
    contradictions: tuple[Contradiction, ...],
    sources: list[Source],
    evidence_items: list[EvidenceItem],
    config: CriticAgentConfig,
) -> float:
    source_coverage = min(len({item.source_id for item in evidence_items}) / config.min_sources, 1.0)
    support_ratio = (
        sum(1 for assessment in evidence_assessments if assessment.supported) / len(evidence_assessments)
        if evidence_assessments
        else 0.0
    )
    avg_support = (
        sum(assessment.support_score for assessment in evidence_assessments) / len(evidence_assessments)
        if evidence_assessments
        else 0.0
    )
    avg_source_credibility = (
        sum(source.credibility_score for source in sources) / len(sources)
        if sources
        else 0.0
    )
    contradiction_penalty = min(
        sum(0.18 if item.severity == "high" else 0.10 for item in contradictions),
        0.35,
    )

    score = (
        (analysis.confidence_score * 0.30)
        + (source_coverage * 0.20)
        + (support_ratio * 0.20)
        + (avg_support * 0.15)
        + (avg_source_credibility * 0.15)
        - contradiction_penalty
    )
    return round(max(0.0, min(1.0, score)), 3)


def critic_findings(
    *,
    quality: float,
    evidence_assessments: tuple[EvidenceAssessment, ...],
    contradictions: tuple[Contradiction, ...],
    source_count: int,
    config: CriticAgentConfig,
) -> tuple[CriticFinding, ...]:
    findings: list[CriticFinding] = []

    if source_count < config.min_sources:
        findings.append(
            CriticFinding(
                issue=f"Only {source_count} unique source(s) available; at least {config.min_sources} required.",
                severity="high",
                retry_reason=RetryReason.INSUFFICIENT_SOURCES,
            )
        )

    unsupported = [assessment for assessment in evidence_assessments if not assessment.supported]
    supported_ratio = (
        (len(evidence_assessments) - len(unsupported)) / len(evidence_assessments)
        if evidence_assessments
        else 0.0
    )
    if supported_ratio < config.min_supported_claim_ratio:
        findings.append(
            CriticFinding(
                issue="Too many analytical claims have weak evidence support.",
                severity="high",
                retry_reason=RetryReason.UNSUPPORTED_CLAIMS,
            )
        )

    if contradictions:
        findings.append(
            CriticFinding(
                issue="Contradictory evidence or trend direction was detected.",
                severity="high",
                retry_reason=RetryReason.CONFLICTING_EVIDENCE,
            )
        )

    if quality < config.quality_threshold and not findings:
        findings.append(
            CriticFinding(
                issue=f"Overall research quality score {quality:.2f} is below threshold {config.quality_threshold:.2f}.",
                severity="medium",
                retry_reason=RetryReason.WEAK_CITATIONS,
            )
        )

    return tuple(findings)


def decision_for_quality(quality: float, config: CriticAgentConfig) -> QualityDecision:
    return QualityDecision.PASS if quality >= config.quality_threshold else QualityDecision.RETRY_RESEARCH


def missing_evidence_items(evidence_assessments: tuple[EvidenceAssessment, ...]) -> tuple[str, ...]:
    return tuple(
        assessment.claim
        for assessment in evidence_assessments
        if not assessment.supported
    )


def _support_score(finding: AnalysisFinding, evidence_items: list[EvidenceItem]) -> float:
    if not evidence_items:
        return 0.0
    scores = [overlap_score(finding.claim, item.text) * item.relevance_score for item in evidence_items]
    return max(scores) if scores else 0.0


def _topic_key(topic: str) -> str:
    return re.sub(r"\W+", " ", topic.lower()).strip()
