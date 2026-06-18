"""In-memory research session store.

The interface is intentionally async so it can be replaced by PostgreSQL or
Redis-backed storage without changing the FastAPI route layer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.application.research.models import ResearchSessionRecord
from app.core.errors import ResearchNotFoundError


class InMemoryResearchStore:
    def __init__(self) -> None:
        self._records: dict[str, ResearchSessionRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, record: ResearchSessionRecord) -> ResearchSessionRecord:
        async with self._lock:
            self._records[record.id] = record
            return record

    async def get(self, research_id: str) -> ResearchSessionRecord:
        async with self._lock:
            record = self._records.get(research_id)
            if record is None:
                raise ResearchNotFoundError(f"Research session '{research_id}' was not found")
            return record

    async def update(
        self,
        research_id: str,
        updater: Callable[[ResearchSessionRecord], ResearchSessionRecord],
    ) -> ResearchSessionRecord:
        async with self._lock:
            record = self._records.get(research_id)
            if record is None:
                raise ResearchNotFoundError(f"Research session '{research_id}' was not found")
            updated = updater(record)
            self._records[research_id] = updated
            return updated

    async def list_all(self) -> list[ResearchSessionRecord]:
        async with self._lock:
            return list(self._records.values())
