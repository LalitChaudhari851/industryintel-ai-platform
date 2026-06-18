"""Routing functions for the business research LangGraph workflow."""

from __future__ import annotations

from typing import Literal

from app.workflows.business_research.state import (
    QualityDecision,
    ResearchState,
    RetryReason,
    WorkflowError,
    WorkflowStatus,
    utc_now,
)

CriticRoute = Literal["researcher", "writer", "fail"]


def initialize_defaults(state: ResearchState) -> ResearchState:
    """Normalize required workflow defaults before the first agent runs."""
    now = utc_now()
    return {
        **state,
        "status": state.get("status", WorkflowStatus.CREATED),
        "iteration_count": state.get("iteration_count", 1),
        "retry_count": state.get("retry_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "sources": state.get("sources", []),
        "evidence": state.get("evidence", []),
        "errors": state.get("errors", []),
        "created_at": state.get("created_at", now),
        "updated_at": now,
    }


def route_after_critic(state: ResearchState) -> CriticRoute:
    """Route Critic output to Researcher retry, Writer, or terminal failure."""
    review = state.get("critic_review")
    iteration_count = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)

    if review is None:
        return "fail"

    if review.decision is QualityDecision.PASS:
        return "writer"

    if review.decision is QualityDecision.FAIL:
        return "fail"

    if (
        review.decision is QualityDecision.RETRY_RESEARCH
        and iteration_count < max_iterations
    ):
        return "researcher"

    return "writer"


def mark_research_retry(state: ResearchState) -> ResearchState:
    """Increment retry count and capture the strongest Critic retry reason."""
    review = state.get("critic_review")
    retry_reason = RetryReason.INSUFFICIENT_SOURCES

    if review:
        critical_or_high = [
            finding
            for finding in review.findings
            if finding.retry_reason is not None and finding.severity in {"critical", "high"}
        ]
        retryable = critical_or_high or [
            finding for finding in review.findings if finding.retry_reason is not None
        ]
        if retryable:
            retry_reason = retryable[0].retry_reason or retry_reason

    return {
        "iteration_count": state.get("iteration_count", 1) + 1,
        "retry_count": state.get("retry_count", 0) + 1,
        "retry_reason": retry_reason,
        "status": WorkflowStatus.RESEARCHING,
        "updated_at": utc_now(),
    }


def mark_failure(state: ResearchState) -> ResearchState:
    """Produce a terminal failure state when routing cannot continue safely."""
    message = "workflow failed before a final report could be produced"
    if state.get("critic_review") is None:
        message = "critic did not produce a review"

    return {
        "status": WorkflowStatus.FAILED,
        "updated_at": utc_now(),
        "errors": [
            WorkflowError(
                message=message,
                recoverable=False,
            )
        ],
    }


def human_approval_node(state: ResearchState) -> ResearchState:
    """Execute human approval node step to record review history and update retry counts."""
    status = state.get("approval_status")
    comments = state.get("reviewer_comments")
    history = list(state.get("review_history") or [])

    new_record = {
        "status": status,
        "comments": comments,
        "timestamp": utc_now().isoformat(),
        "retry_count": state.get("retry_count", 0),
    }
    history.append(new_record)

    update: ResearchState = {
        "review_history": history,
        "updated_at": utc_now(),
    }

    if status in {"rejected", "request_more_research"}:
        update["retry_count"] = state.get("retry_count", 0) + 1
        update["iteration_count"] = state.get("iteration_count", 1) + 1

    return update


def route_after_approval(state: ResearchState) -> Literal["writer", "analyst", "researcher", "fail"]:
    """Route decision after human review."""
    status = state.get("approval_status")
    retry_count = state.get("retry_count", 0)

    if status == "approved":
        return "writer"

    if retry_count >= 3:
        return "fail"

    if status == "rejected":
        return "analyst"

    if status == "request_more_research":
        return "researcher"

    return "fail"
