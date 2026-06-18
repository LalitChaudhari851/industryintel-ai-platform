"""FastAPI entrypoint for the AI business research backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1.routes import router as research_router
from app.api.v1.observability_routes import router as observability_router
from app.application.research.service import ResearchService
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.core.tracing import configure_langsmith

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_langsmith(settings)
    app.state.settings = settings

    from app.services.llm_service import LLMService
    from app.services.memory_service import ResearchMemoryService

    llm_service = LLMService(settings)
    memory_service = ResearchMemoryService(settings)

    app.state.llm_service = llm_service
    app.state.memory_service = memory_service

    app.state.research_service = ResearchService(
        settings=settings,
        llm_service=llm_service,
        memory_service=memory_service,
    )
    logger.info("app.started name=%s environment=%s", settings.app_name, settings.environment)
    yield
    logger.info("app.stopped")


app = FastAPI(
    title="Autonomous Research & Intelligence Platform",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    logger.warning("app_error code=%s message=%s", exc.error_code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.error_code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("validation_error errors=%s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "message": "Request validation failed", "details": exc.errors()}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error error=%s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Internal server error"}},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(research_router)
app.include_router(observability_router)
