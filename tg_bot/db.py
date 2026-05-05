from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from tg_bot.models import Base

logger = logging.getLogger(__name__)


def build_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, echo=False, pool_size=5, max_overflow=10)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def reset_db(engine: AsyncEngine) -> None:
    """Drop all tables and recreate them from scratch."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("All tables dropped.")
    await init_db(engine)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified.")

    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT create_hypertable('bars_5m', 'ts', if_not_exists => TRUE);")
            )
        logger.info("TimescaleDB hypertable created for bars_5m.")
    except Exception:
        logger.debug("TimescaleDB hypertable skipped (extension may not be available).")
