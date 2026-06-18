"""Application exceptions and HTTP error helpers."""

from __future__ import annotations


class AppError(Exception):
    status_code = 500
    error_code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ResearchNotFoundError(AppError):
    status_code = 404
    error_code = "research_not_found"


class ReportNotReadyError(AppError):
    status_code = 409
    error_code = "report_not_ready"


class ResearchCapacityError(AppError):
    status_code = 429
    error_code = "research_capacity_exceeded"
