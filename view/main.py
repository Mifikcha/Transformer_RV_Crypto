"""
Entry point: bootstraps DB, starts scheduler (ingestion + prediction + notification)
and launches the Telegram bot.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select, func
import numpy as np

from view.bot import create_dispatcher
from view.bybit_client import BybitClient
from view.config import Settings
from view.db import build_engine, build_session_factory, init_db, reset_db
from view.inference import RVInference
from view.ingestion_worker import bootstrap_history, heal_gap, heal_funding_oi, ingest_latest_bar
from view.models import Bar5m
from view.notification_worker import notify_cycle
from view.prediction_worker import run_prediction, backfill_actual_rv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RV Bot")
    parser.add_argument(
        "--reset-db", action="store_true",
        help="Drop all tables and bootstrap from scratch before starting.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = Settings()

    # Database
    engine = build_engine(settings.database_url)
    if args.reset_db:
        logger.info("--reset-db: dropping all tables and starting fresh.")
        await reset_db(engine)
    else:
        await init_db(engine)
    session_factory = build_session_factory(engine)

    # Clients & ML
    bybit = BybitClient(settings.bybit_base_url, settings.symbol)
    rv_inference = RVInference(settings.model_paths_list, settings.features_path)
    bot = Bot(token=settings.telegram_bot_token)

    # Bootstrap if needed (always runs after --reset-db since tables are empty)
    async with session_factory() as session:
        count = await session.scalar(select(func.count(Bar5m.ts)))
        if (count or 0) < settings.min_bars_for_inference:
            logger.info("Bootstrap: bars_5m has %d rows, need %d.", count or 0, settings.min_bars_for_inference)
            await bootstrap_history(bybit, session, days=25)

    # Ingest latest bars first, so heal_gap can see the full picture.
    async with session_factory() as session:
        await ingest_latest_bar(bybit, session)

    # Now heal any internal gaps (between old data and freshly ingested data).
    for _attempt in range(10):
        async with session_factory() as session:
            healed = await heal_gap(bybit, session)
            if healed == 0:
                break

    # Correct funding_rate / open_interest that bootstrap wrote as a
    # single snapshot value for all bars.
    async with session_factory() as session:
        await heal_funding_oi(bybit, session)

    # Recompute rv_actual with gap-aware forward-GK method.
    async with session_factory() as session:
        from view.models import RvActual as _RvA
        deleted = await session.execute(_RvA.__table__.delete())
        await session.commit()
        logger.info("Cleared rv_actual table for recomputation (%s rows).", deleted.rowcount)
        await backfill_actual_rv(session)

    # ---- scheduled jobs ----
    # Ingest → predict → notify run sequentially in one job to avoid race conditions:
    # prediction needs the bars that ingestion just fetched.
    async def job_cycle() -> None:
        # 1. Ingest latest bars
        async with session_factory() as session:
            try:
                await ingest_latest_bar(bybit, session)
            except Exception:
                logger.exception("Ingestion step failed")

        # 2. Run prediction (including backfill of actual RV)
        async with session_factory() as session:
            try:
                await run_prediction(session, rv_inference, settings)
            except Exception:
                logger.exception("Prediction step failed")

        # 3. Send notifications
        async with session_factory() as session:
            try:
                await notify_cycle(bot, session, settings)
            except Exception:
                logger.exception("Notification step failed")

    async def job_healthcheck() -> None:
        async with session_factory() as session:
            try:
                await _send_healthcheck(bot, session, settings)
            except Exception:
                logger.exception("Healthcheck job failed")

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(job_cycle, "interval", seconds=settings.fetch_interval_seconds)
    scheduler.add_job(job_healthcheck, "interval", hours=1)
    scheduler.start()
    logger.info(
        "Scheduler started: cycle=%ds (ingest→predict→notify), healthcheck=1h",
        settings.fetch_interval_seconds,
    )

    # Telegram bot (blocks event loop)
    dp = create_dispatcher(settings, session_factory)
    logger.info("Starting Telegram bot polling...")
    await dp.start_polling(bot)


async def _send_healthcheck(bot: Bot, session, settings: Settings) -> None:
    """Hourly health status + personal full report."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import func as sqlfunc, text

    now = datetime.now(timezone.utc)

    max_bar_ts = await session.scalar(select(sqlfunc.max(Bar5m.ts)))
    bar_count = await session.scalar(select(sqlfunc.count(Bar5m.ts)))

    from view.models import Prediction
    max_pred_ts = await session.scalar(select(sqlfunc.max(Prediction.ts)))

    bar_lag_min = (now - max_bar_ts).total_seconds() / 60 if max_bar_ts else 999
    status = "\u2705" if bar_lag_min < 15 else "\u26a0\ufe0f DATA LAG"

    ts_str = now.strftime("%Y-%m-%d %H:%M")
    bar_ts_str = max_bar_ts.strftime("%H:%M") if max_bar_ts else "?"
    pred_ts_str = max_pred_ts.strftime("%H:%M") if max_pred_ts else "?"

    health_text = (
        f"{status} RV Bot alive \u00b7 {ts_str} UTC\n"
        f"  Последний бар:      {bar_ts_str} UTC ({bar_lag_min:.0f} мин назад)\n"
        f"  Последний прогноз:  {pred_ts_str} UTC\n"
        f"  Баров в БД:         {bar_count:,}"
    )

    # Group/channel posting disabled by request: send only direct reports below.

    # ---------- Personal full report (accuracy + rv + debug + data) ----------
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 3:
            return 0.0
        c = np.corrcoef(y_true, y_pred)
        return float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0

    def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_pred - y_true))

    # /rv block
    from view.models import Prediction, RvActual
    pred_row = await session.execute(
        select(Prediction).order_by(Prediction.created_at.desc()).limit(1)
    )
    pred = pred_row.scalar_one_or_none()
    rv_block = "RV: нет прогнозов."
    if pred is not None:
        actual = await session.scalar(select(RvActual.rv_3bar).where(RvActual.ts == pred.ts))
        rv_block = (
            f"RV\n"
            f"ts: {pred.ts}\n"
            f"rv_3bar: {float(pred.rv_3bar or 0):.6f}\n"
            f"rv_12bar: {float(pred.rv_12bar or 0):.6f}\n"
            f"actual_3bar: {float(actual):.6f}" if actual is not None else
            f"RV\n"
            f"ts: {pred.ts}\n"
            f"rv_3bar: {float(pred.rv_3bar or 0):.6f}\n"
            f"rv_12bar: {float(pred.rv_12bar or 0):.6f}\n"
            f"actual_3bar: n/a"
        )

    # /accuracy block
    acc_stmt = text(
        """
        SELECT p.rv_3bar, p.rv_12bar, a.rv_3bar, a.rv_12bar
        FROM predictions p
        JOIN rv_actual a ON a.ts = p.ts
        WHERE p.rv_3bar IS NOT NULL
          AND p.rv_12bar IS NOT NULL
          AND a.rv_3bar IS NOT NULL
          AND a.rv_12bar IS NOT NULL
          AND COALESCE(p.degraded, FALSE) = FALSE
        ORDER BY p.ts DESC
        LIMIT 1000
        """
    )
    acc_rows = (await session.execute(acc_stmt)).fetchall()
    if len(acc_rows) >= 5:
        p3 = np.array([r[0] for r in acc_rows], dtype=float)
        p12 = np.array([r[1] for r in acc_rows], dtype=float)
        a3 = np.array([r[2] for r in acc_rows], dtype=float)
        a12 = np.array([r[3] for r in acc_rows], dtype=float)
        acc_block = (
            f"Accuracy (N={len(acc_rows)})\n"
            f"R2: {_r2(a3, p3):.4f} / {_r2(a12, p12):.4f}\n"
            f"r: {_pearson(a3, p3):.4f} / {_pearson(a12, p12):.4f}\n"
            f"MAE: {_mae(a3, p3):.6f} / {_mae(a12, p12):.6f}\n"
            f"Bias: {_bias(a3, p3):+.6f} / {_bias(a12, p12):+.6f}"
        )
    else:
        acc_block = f"Accuracy: недостаточно данных (N={len(acc_rows)})."

    # /debug block
    total_pred = await session.scalar(select(sqlfunc.count(Prediction.ts)))
    total_actual = await session.scalar(select(sqlfunc.count(RvActual.ts)))
    min_pred_ts = await session.scalar(select(sqlfunc.min(Prediction.ts)))
    max_pred_ts = await session.scalar(select(sqlfunc.max(Prediction.ts)))
    min_actual_ts = await session.scalar(select(sqlfunc.min(RvActual.ts)))
    max_actual_ts = await session.scalar(select(sqlfunc.max(RvActual.ts)))
    debug_block = (
        "Debug\n"
        f"predictions: {int(total_pred or 0)} ({min_pred_ts} .. {max_pred_ts})\n"
        f"rv_actual: {int(total_actual or 0)} ({min_actual_ts} .. {max_actual_ts})\n"
        f"pending: {int(total_pred or 0) - int(total_actual or 0)}"
    )

    # /data block
    min_bar_ts = await session.scalar(select(sqlfunc.min(Bar5m.ts)))
    data_block = (
        "Data\n"
        f"last_bar: {max_bar_ts}\n"
        f"lag_min: {bar_lag_min:.1f}\n"
        f"first_bar: {min_bar_ts}\n"
        f"bars_total: {int(bar_count or 0)}"
    )

    report_text = (
        f"{status} Hourly Personal Report ({ts_str} UTC)\n\n"
        f"{acc_block}\n\n"
        f"{rv_block}\n\n"
        f"{debug_block}\n\n"
        f"{data_block}\n\n"
        f"Uptime: {_format_uptime()}"
    )

    recipients = settings.alert_chat_ids_list or sorted(settings.allowed_chat_ids_set)
    for chat_id in recipients:
        try:
            await bot.send_message(chat_id=chat_id, text=report_text)
        except Exception as e:
            logger.error("Failed to send personal hourly report to %s: %s", chat_id, e)


def _format_uptime() -> str:
    delta = datetime.now(timezone.utc) - PROCESS_STARTED_AT_UTC
    total_sec = int(delta.total_seconds())
    days, rem = divmod(total_sec, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}д {hours}ч {minutes}м"
    return f"{hours}ч {minutes}м"


if __name__ == "__main__":
    asyncio.run(main())
