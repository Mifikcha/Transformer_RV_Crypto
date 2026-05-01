"""Utilities to bootstrap bar history from Bybit into PostgreSQL.

Reuses the same ingestion logic as Telegram bot workers in `view`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


# Ensure project root is importable when running from dashboard/ directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from view.bybit_client import BybitClient
from view.config import Settings
from view.db import build_engine, build_session_factory, init_db
from view.ingestion_worker import bootstrap_history, heal_gap, ingest_latest_bar


async def _pull_history_async(days: int) -> tuple[bool, str]:
    settings = Settings()
    engine = build_engine(settings.database_url)
    session_factory = build_session_factory(engine)

    try:
        await init_db(engine)
        bybit = BybitClient(settings.bybit_base_url, settings.symbol)

        async with session_factory() as session:
            await bootstrap_history(bybit, session, days=days)

        # After bootstrap fetch latest closed bars and heal any internal gaps.
        async with session_factory() as session:
            await ingest_latest_bar(bybit, session)
            for _ in range(5):
                healed = await heal_gap(bybit, session)
                if healed == 0:
                    break

        return True, f"История подтянута успешно (days={days})."
    except Exception as exc:  # pragma: no cover
        return False, f"Ошибка bootstrap: {exc}"
    finally:
        await engine.dispose()


def pull_history(days: int) -> tuple[bool, str]:
    """Sync wrapper for Streamlit button callbacks."""
    return asyncio.run(_pull_history_async(days=days))

