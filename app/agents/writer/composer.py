"""Evidence-bound report composition utilities."""

from __future__ import annotations

import re

from app.agents.writer.models import WriterAgentConfig
from app.workflows.business_research.state import (
    AnalysisFinding,
    AnalysisResult,
    Citation,
    CriticReview,
    EvidenceItem,
    ExtractedMetric,
    ExtractedTrend,
    Source,
)

WORD_RE = re.compile(r"\b[\w$%.-]+\b")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def approved_findings(
    analysis: AnalysisResult,
    critic_review: CriticReview | None,
    *,
    config: WriterAgentConfig,
) -> tuple[AnalysisFinding, ...]:
    if critic_review and critic_review.evidence_assessments:
        supported_claims = {
            assessment.claim
            for assessment in critic_review.evidence_assessments
            if assessment.supported
        }
        findings = tuple(finding for finding in analysis.findings if finding.claim in supported_claims)
    else:
        findings = tuple(finding for finding in analysis.findings if finding.confidence >= 0.45)

    if not findings:
        findings = tuple(sorted(analysis.findings, key=lambda item: item.confidence, reverse=True)[:2])
    return findings[: config.max_findings]


def approved_metrics(
    analysis: AnalysisResult,
    approved_evidence_ids: set[str],
    *,
    config: WriterAgentConfig,
) -> tuple[ExtractedMetric, ...]:
    if approved_evidence_ids:
        metrics = tuple(
            metric
            for metric in analysis.metrics
            if approved_evidence_ids.intersection(metric.evidence_ids)
        )
    else:
        metrics = analysis.metrics
    return tuple(sorted(metrics, key=lambda item: item.confidence, reverse=True)[: config.max_metrics])


def approved_trends(
    analysis: AnalysisResult,
    approved_evidence_ids: set[str],
    *,
    config: WriterAgentConfig,
) -> tuple[ExtractedTrend, ...]:
    if approved_evidence_ids:
        trends = tuple(
            trend
            for trend in analysis.trends
            if approved_evidence_ids.intersection(trend.evidence_ids)
        )
    else:
        trends = analysis.trends
    return tuple(sorted(trends, key=lambda item: item.confidence, reverse=True)[: config.max_trends])


def approved_evidence_ids(findings: tuple[AnalysisFinding, ...]) -> set[str]:
    return {evidence_id for finding in findings for evidence_id in finding.evidence_ids}


def build_citations(
    findings: tuple[AnalysisFinding, ...],
    metrics: tuple[ExtractedMetric, ...],
    trends: tuple[ExtractedTrend, ...],
    evidence_by_id: dict[str, EvidenceItem],
) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    seen: set[tuple[str, str | None]] = set()

    for finding in findings:
        citations.extend(
            _citations_for_claim(
                claim=finding.claim,
                evidence_ids=finding.evidence_ids,
                confidence=finding.confidence,
                evidence_by_id=evidence_by_id,
            )
        )
    for metric in metrics:
        citations.extend(
            _citations_for_claim(
                claim=f"{metric.name}: {metric.value}",
                evidence_ids=metric.evidence_ids,
                confidence=metric.confidence,
                evidence_by_id=evidence_by_id,
            )
        )
    for trend in trends:
        citations.extend(
            _citations_for_claim(
                claim=f"{trend.topic}: {trend.description}",
                evidence_ids=trend.evidence_ids,
                confidence=trend.confidence,
                evidence_by_id=evidence_by_id,
            )
        )

    unique: list[Citation] = []
    for citation in citations:
        key = (citation.claim, citation.evidence_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return tuple(unique)


def compose_sections(
    *,
    query: str,
    analysis: AnalysisResult,
    critic_review: CriticReview | None,
    findings: tuple[AnalysisFinding, ...],
    metrics: tuple[ExtractedMetric, ...],
    trends: tuple[ExtractedTrend, ...],
    citations: tuple[Citation, ...],
    sources: list[Source],
    evidence_by_id: dict[str, EvidenceItem],
    config: WriterAgentConfig,
) -> tuple[str, dict[str, str], tuple[str, ...], int]:
    citation_lookup = _citation_lookup(citations)
    references = _references(sources, citations, config=config)
    limitations = _limitations(critic_review, findings, citations)

    executive_summary = _executive_summary(query, analysis, findings, metrics, trends, critic_review)
    sections = {
        "Executive Summary": executive_summary,
        "Key Findings": _key_findings(findings, metrics, trends, citation_lookup),
        "Risks": _risks(analysis, critic_review),
        "Opportunities": _opportunities(analysis, findings, trends),
        "Recommendations": _recommendations(analysis, findings, metrics, trends),
        "References": references,
    }

    sections = _expand_to_minimum_words(
        query=query,
        sections=sections,
        findings=findings,
        metrics=metrics,
        trends=trends,
        citation_lookup=citation_lookup,
        min_words=config.min_words,
    )
    sections = _trim_to_max_words(sections, config.max_words)
    total_words = word_count(_render_sections(sections))
    return sections["Executive Summary"], sections, tuple(limitations), total_words


def _citations_for_claim(
    *,
    claim: str,
    evidence_ids: tuple[str, ...],
    confidence: float,
    evidence_by_id: dict[str, EvidenceItem],
) -> list[Citation]:
    citations = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue
        citations.append(
            Citation(
                claim=claim,
                source_id=evidence.source_id,
                evidence_id=evidence.id,
                supporting_text=_supporting_text(evidence.text),
                confidence=confidence,
            )
        )
    return citations


def _supporting_text(text: str, *, max_chars: int = 350) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= max_chars else f"{clean[: max_chars - 3].rstrip()}..."


def _citation_lookup(citations: tuple[Citation, ...]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for index, citation in enumerate(citations, start=1):
        labels[f"{citation.claim}|{citation.evidence_id}"] = f"[{index}]"
    return labels


def _labels_for_claim(claim: str, evidence_ids: tuple[str, ...], citation_lookup: dict[str, str]) -> str:
    labels = [
        citation_lookup[key]
        for evidence_id in evidence_ids
        if (key := f"{claim}|{evidence_id}") in citation_lookup
    ]
    return " ".join(labels)


def _executive_summary(
    query: str,
    analysis: AnalysisResult,
    findings: tuple[AnalysisFinding, ...],
    metrics: tuple[ExtractedMetric, ...],
    trends: tuple[ExtractedTrend, ...],
    critic_review: CriticReview | None,
) -> str:
    confidence = critic_review.confidence_score if critic_review else analysis.confidence_score
    primary_finding = findings[0].claim if findings else analysis.summary
    metric_text = _metric_sentence(metrics)
    trend_text = _trend_sentence(trends)
    return (
        f"This report evaluates {query} using the evidence that passed the research and review workflow. "
        f"The core conclusion is that {primary_finding} {metric_text} {trend_text} "
        f"The overall confidence score is {confidence:.2f}, which reflects the strength of the cited evidence, "
        "the number of supported claims, the credibility of the sources, and the absence or presence of contradictions. "
        "The analysis is designed for executive use: it separates evidence-backed findings from interpretation, "
        "surfaces the main risks and opportunities, and converts the research into practical recommendations."
    )


def _key_findings(
    findings: tuple[AnalysisFinding, ...],
    metrics: tuple[ExtractedMetric, ...],
    trends: tuple[ExtractedTrend, ...],
    citation_lookup: dict[str, str],
) -> str:
    paragraphs = []
    for index, finding in enumerate(findings, start=1):
        labels = _labels_for_claim(finding.claim, finding.evidence_ids, citation_lookup)
        implication = f" The business implication is: {finding.implication}" if finding.implication else ""
        paragraphs.append(
            f"{index}. {finding.claim} {labels} Confidence for this finding is {finding.confidence:.2f}.{implication}"
        )

    if metrics:
        metric_lines = []
        for metric in metrics:
            labels = _labels_for_claim(f"{metric.name}: {metric.value}", metric.evidence_ids, citation_lookup)
            metric_lines.append(
                f"{metric.name} was reported as {metric.value}. Context: {metric.context} {labels}"
            )
        paragraphs.append("The most relevant quantitative metrics are: " + " ".join(metric_lines))

    if trends:
        trend_lines = []
        for trend in trends:
            labels = _labels_for_claim(f"{trend.topic}: {trend.description}", trend.evidence_ids, citation_lookup)
            trend_lines.append(
                f"{trend.topic} shows a {trend.direction} direction: {trend.description} {labels}"
            )
        paragraphs.append("The trend evidence indicates the following directional signals: " + " ".join(trend_lines))

    return "\n\n".join(paragraphs) if paragraphs else "No approved findings were available."


def _risks(analysis: AnalysisResult, critic_review: CriticReview | None) -> str:
    risks = list(analysis.risks)
    if critic_review:
        risks.extend(finding.issue for finding in critic_review.findings)
        risks.extend(contradiction.description for contradiction in critic_review.contradictions)

    if not risks:
        return (
            "The reviewed evidence does not surface a specific material risk. "
            "The main residual risk is evidence incompleteness: decision-makers should treat the report as a structured "
            "view of available sources, not as a substitute for primary diligence."
        )

    return " ".join(f"{index}. {risk}" for index, risk in enumerate(_unique(risks), start=1))


def _opportunities(
    analysis: AnalysisResult,
    findings: tuple[AnalysisFinding, ...],
    trends: tuple[ExtractedTrend, ...],
) -> str:
    opportunities = list(analysis.opportunities)
    opportunities.extend(
        finding.claim
        for finding in findings
        if finding.implication and "opportunity" in finding.implication.lower()
    )
    opportunities.extend(
        trend.description
        for trend in trends
        if trend.direction in {"increasing", "stable"}
    )

    if not opportunities:
        return (
            "The evidence does not justify a strong opportunity claim. "
            "A prudent next step is to gather additional primary or official sources before committing resources."
        )
    return " ".join(f"{index}. {item}" for index, item in enumerate(_unique(opportunities), start=1))


def _recommendations(
    analysis: AnalysisResult,
    findings: tuple[AnalysisFinding, ...],
    metrics: tuple[ExtractedMetric, ...],
    trends: tuple[ExtractedTrend, ...],
) -> str:
    recommendations = list(analysis.recommendations)
    if findings:
        recommendations.append("Prioritize decisions around the highest-confidence findings and preserve their evidence trail in any executive communication.")
    if metrics:
        recommendations.append("Use the extracted metrics as the first quantitative baseline, but validate them against primary filings, official releases, or paid market datasets before financial modeling.")
    if trends:
        recommendations.append("Monitor the identified trend directions over time because trend reversals would materially change strategic interpretation.")
    recommendations.append("Commission additional research for claims with lower confidence or limited source coverage before using them in investment, partnership, or market-entry decisions.")
    return " ".join(f"{index}. {item}" for index, item in enumerate(_unique(recommendations), start=1))


def _references(
    sources: list[Source],
    citations: tuple[Citation, ...],
    *,
    config: WriterAgentConfig,
) -> str:
    cited_source_ids = {citation.source_id for citation in citations}
    cited_sources = [source for source in sources if source.id in cited_source_ids][: config.max_references]
    if not cited_sources:
        return "No references were available."

    rows = []
    for index, source in enumerate(cited_sources, start=1):
        url = f" {source.url}" if source.url else ""
        publisher = f", {source.publisher}" if source.publisher else ""
        rows.append(f"[{index}] {source.title}{publisher}.{url}")
    return "\n".join(rows)


def _limitations(
    critic_review: CriticReview | None,
    findings: tuple[AnalysisFinding, ...],
    citations: tuple[Citation, ...],
) -> list[str]:
    limitations = []
    if not findings:
        limitations.append("No approved findings were available.")
    if not citations:
        limitations.append("No citations could be generated from the available evidence.")
    if critic_review and critic_review.missing_evidence:
        limitations.append("Some claims had missing or weak evidence support.")
    if critic_review and critic_review.contradictions:
        limitations.append("Potential contradictions were detected and should be reviewed before high-stakes use.")
    return limitations


def _expand_to_minimum_words(
    *,
    query: str,
    sections: dict[str, str],
    findings: tuple[AnalysisFinding, ...],
    metrics: tuple[ExtractedMetric, ...],
    trends: tuple[ExtractedTrend, ...],
    citation_lookup: dict[str, str],
    min_words: int,
) -> dict[str, str]:
    expanded = dict(sections)
    expansion_paragraphs = _expansion_paragraphs(query, findings, metrics, trends, citation_lookup)
    index = 0
    while word_count(_render_sections(expanded)) < min_words and index < len(expansion_paragraphs):
        expanded["Key Findings"] = f"{expanded['Key Findings']}\n\n{expansion_paragraphs[index]}"
        index += 1
    return expanded


def _expansion_paragraphs(
    query: str,
    findings: tuple[AnalysisFinding, ...],
    metrics: tuple[ExtractedMetric, ...],
    trends: tuple[ExtractedTrend, ...],
    citation_lookup: dict[str, str],
) -> list[str]:
    primary = findings[0] if findings else None
    finding_text = primary.claim if primary else "the approved evidence base is limited"
    finding_label = _labels_for_claim(primary.claim, primary.evidence_ids, citation_lookup) if primary else ""
    metric_text = _metric_sentence(metrics)
    trend_text = _trend_sentence(trends)

    return [
        (
            f"For {query}, the most important interpretation is not simply the presence of individual data points, "
            f"but the consistency of the approved evidence around the central finding: {finding_text} {finding_label} "
            "Executives should treat this as a decision input rather than a standalone forecast. The finding is useful "
            "because it gives the organization a concrete basis for prioritization, while still requiring follow-up "
            "validation before financial commitments, public claims, or board-level decisions."
        ),
        (
            f"The quantitative layer of the analysis should be read with discipline. {metric_text} Metrics extracted "
            "from evidence are valuable because they make the report operational: they can be tracked, challenged, and "
            "compared over time. At the same time, the report should avoid overfitting strategy to one number. A better "
            "operating approach is to use these metrics as a baseline, then refresh them as more current sources become "
            "available."
        ),
        (
            f"The trend layer adds strategic context. {trend_text} Trend evidence helps separate temporary data points "
            "from broader movement in the market. When a trend is increasing, leadership should ask whether the company "
            "has the capabilities, partnerships, distribution, capital, and timing to benefit from it. When a trend is "
            "mixed or uncertain, the right response is staged investment and tighter monitoring."
        ),
        (
            "From a risk-management perspective, the report should be used as a structured research artifact with an "
            "explicit evidence trail. The strongest sections are those connected directly to citations; weaker areas "
            "should become research tasks rather than assumptions. This discipline reduces hallucination risk, makes "
            "the analysis auditable, and allows future agents or human analysts to update the conclusion when better "
            "sources arrive."
        ),
        (
            "For execution, the recommended next move is to convert the findings into a short decision memo: what the "
            "organization believes, what evidence supports it, what would change the conclusion, and what action is "
            "justified now. That memo should preserve the citations and confidence levels from this report. Doing so "
            "keeps the business discussion grounded in verified evidence rather than narrative momentum."
        ),
        (
            "The final takeaway is that this report provides a professional evidence-backed view, not a prediction with "
            "false precision. It is strongest when used to frame choices, identify diligence gaps, and focus follow-up "
            "research. The cited findings, metrics, and trends should become the foundation for further expert review, "
            "financial analysis, customer discovery, or competitive monitoring."
        ),
    ]


def _trim_to_max_words(sections: dict[str, str], max_words: int) -> dict[str, str]:
    rendered = _render_sections(sections)
    if word_count(rendered) <= max_words:
        return sections

    trimmed = dict(sections)
    overflow = word_count(rendered) - max_words
    key = "Key Findings"
    words = trimmed[key].split()
    if len(words) > overflow + 50:
        trimmed[key] = " ".join(words[: len(words) - overflow])
    return trimmed


def _render_sections(sections: dict[str, str]) -> str:
    return "\n\n".join(f"{name}\n{body}" for name, body in sections.items())


def _metric_sentence(metrics: tuple[ExtractedMetric, ...]) -> str:
    if not metrics:
        return "The evidence did not provide a robust set of quantitative metrics."
    top = metrics[:3]
    metric_text = "; ".join(f"{metric.name} at {metric.value}" for metric in top)
    return f"Key metrics include {metric_text}."


def _trend_sentence(trends: tuple[ExtractedTrend, ...]) -> str:
    if not trends:
        return "No clear directional trend was approved beyond the core findings."
    top = trends[:3]
    trend_text = "; ".join(f"{trend.topic} is {trend.direction}" for trend in top)
    return f"Approved trend signals show that {trend_text}."


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for item in items:
        key = re.sub(r"\W+", " ", item.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
