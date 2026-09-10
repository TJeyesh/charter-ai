"""
Charter-AI — FastAPI Dependency Injection.

Provides database sessions, model instances, and service objects
to route handlers via FastAPI's Depends() system.
"""

from typing import Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from src.utils.logging import get_logger

logger = get_logger(__name__)


async def get_session() -> AsyncGenerator[Optional[AsyncSession], None]:
    """
    Yield an async database session for request-scoped use.
    If database drivers or connection are unavailable (e.g. SIH demo mode),
    yields None gracefully so route handlers can fall back to the processed dataset.
    """
    try:
        from src.data.db import get_db_session
        async for session in get_db_session():
            yield session
            return
    except Exception as e:
        logger.debug("Database session unavailable (%s); falling back to mock/processed repository.", e)
        yield None
