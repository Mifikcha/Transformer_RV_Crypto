"""Streamlit-friendly wrapper for batch prediction backfill.

Mirrors the design of ``history_loader.py``: re-uses the same async
worker that the Telegram bot calls in its scheduled cycle, so the
dashboard and the bot stay in lockstep on inference behaviour.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

# Ensure project root is importable when running from dashboard/ directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from tg_bot.config import Settings
from tg_bot.db import build_engine, build_session_factory
from tg_bot.inference import RVInference
from tg_bot.prediction_worker import backfill_predictions


async def _async_backfill(
    progress_cb: Callable[[int, int], None] | None,
) -> tuple[bool, str, dict]:
    settings = Settings()
    engine = build_engine(settings.database_url)
    session_factory = build_session_factory(engine)

    try:
        # Loading model weights is the slowest part of the call (~1–2 s).
        # We do it OUTSIDE the session so the DB connection is free during it.
        inference = RVInference(settings.model_paths_list, settings.features_path)

        async with session_factory() as session:
            stats = await backfill_predictions(
                session,
                inference,
                settings,
                progress_cb=progress_cb,
            )

        msg = (
            f"Бэкфилл завершён: записано {stats.get('filled', 0)} predictions, "
            f"досчитано {stats.get('actuals_filled', 0)} rv_actual; "
            f"degraded={stats.get('degraded', 0)}, "
            f"пропущено по короткому хвосту={stats.get('skipped_short_tail', 0)}."
        )
        return True, msg, stats
    except Exception as exc:  # pragma: no cover — surfaced to the user via Streamlit
        return False, f"Ошибка backfill: {exc}", {}
    finally:
        await engine.dispose()


def run_backfill(progress_cb: Callable[[int, int], None] | None = None) -> tuple[bool, str, dict]:
    """Sync wrapper for Streamlit button callbacks."""
    return asyncio.run(_async_backfill(progress_cb))
