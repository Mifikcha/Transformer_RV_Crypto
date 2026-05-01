"""
Notification worker: send prediction alerts to Telegram.

- Regular updates to the team channel.
- Spike alerts to individual chats when RV exceeds threshold.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from aiogram import Bot
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from view.config import Settings
from view.models import Prediction, NotificationLog

logger = logging.getLogger(__name__)


def get_regime_label(rv_3bar: float, median_24h: float) -> tuple[str, str]:
    if median_24h <= 0:
        return ("\U0001f7e1", "недоступно")
    ratio = rv_3bar / median_24h
    if ratio < 0.7:
        return ("\U0001f7e2", "низкая")
    if ratio < 1.3:
        return ("\U0001f7e1", "умеренная")
    if ratio < 2.0:
        return ("\U0001f7e0", "высокая")
    return ("\U0001f534", "экстремальная")


def format_regular_message(
    pred: Prediction,
    rv_actual: float | None,
    median_24h: float,
    symbol: str = "BTCUSDT",
) -> str:
    emoji, label = get_regime_label(pred.rv_3bar or 0, median_24h)
    ts_str = pred.ts.strftime("%Y-%m-%d %H:%M") if pred.ts else "?"
    actual_str = f"{rv_actual:.6f}" if rv_actual is not None else "n/a"

    sep = "\u2501" * 20
    return (
        f"\U0001f4ca RV Forecast \u00b7 {symbol} \u00b7 {ts_str} UTC\n"
        f"{sep}\n"
        f"Прогноз 15min:  {pred.rv_3bar:.6f}  {emoji}\n"
        f"Прогноз 1h:     {pred.rv_12bar:.6f}\n"
        f"Текущая RV:     {actual_str}\n"
        f"\n"
        f"Режим: {label}\n"
        f"Модель: {pred.model_ver or '?'}"
    )


def format_spike_message(
    pred: Prediction,
    median_24h: float,
    symbol: str = "BTCUSDT",
) -> str:
    pct = ((pred.rv_3bar / median_24h) - 1) * 100 if median_24h > 0 else 0
    ts_str = pred.ts.strftime("%Y-%m-%d %H:%M") if pred.ts else "?"

    return (
        f"\u26a0\ufe0f VOLATILITY SPIKE EXPECTED \u00b7 {symbol}\n"
        f"\n"
        f"Прогноз RV(15min): {pred.rv_3bar:.6f}  (+{pct:.0f}% vs норма)\n"
        f"Прогноз RV(1h):    {pred.rv_12bar:.6f}\n"
        f"\n"
        f"Время: {ts_str} UTC"
    )


async def _get_median_24h(session: AsyncSession) -> float:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    stmt = select(func.percentile_cont(0.5).within_group(Prediction.rv_3bar)).where(
        Prediction.ts >= cutoff
    )
    try:
        val = await session.scalar(stmt)
    except Exception:
        val = None
    return float(val) if val else 0.0


async def _get_actual_rv(session: AsyncSession, ts: datetime) -> float | None:
    from view.models import RvActual
    stmt = select(RvActual.rv_3bar).where(RvActual.ts == ts)
    return await session.scalar(stmt)


async def notify_cycle(
    bot: Bot,
    session: AsyncSession,
    settings: Settings,
) -> None:
    """Find un-notified predictions, send direct notifications only, log delivery."""
    already_sent = select(NotificationLog.prediction_id)
    stmt = (
        select(Prediction)
        .where(Prediction.id.notin_(already_sent))
        .order_by(Prediction.created_at.asc())
    )
    result = await session.execute(stmt)
    pending_preds = result.scalars().all()

    if not pending_preds:
        return

    median_24h = await _get_median_24h(session)
    recipients = settings.alert_chat_ids_list or sorted(settings.allowed_chat_ids_set)

    for pred in pending_preds:
        rv_actual = await _get_actual_rv(session, pred.ts) if pred.ts else None

        # Regular message ONLY to direct chats (no group/channel messages).
        msg = format_regular_message(pred, rv_actual, median_24h, settings.symbol)
        for chat_id in recipients:
            try:
                await bot.send_message(chat_id=chat_id, text=msg)
            except Exception as e:
                logger.error("Failed to send regular DM to %s: %s", chat_id, e)

        # Spike alert to direct chats
        if pred.rv_3bar and median_24h > 0 and pred.rv_3bar > median_24h * settings.rv_spike_multiplier:
            spike_msg = format_spike_message(pred, median_24h, settings.symbol)
            for chat_id in recipients:
                try:
                    await bot.send_message(chat_id=chat_id, text=spike_msg)
                except Exception as e:
                    logger.error("Failed to send spike alert to %s: %s", chat_id, e)

        alert_type = "spike" if (
            pred.rv_3bar and median_24h > 0 and pred.rv_3bar > median_24h * settings.rv_spike_multiplier
        ) else "regular"

        log_entry = NotificationLog(
            id=uuid.uuid4(),
            prediction_id=pred.id,
            alert_type=alert_type,
        )
        session.add(log_entry)

    await session.commit()
    logger.info("Notified %d predictions.", len(pending_preds))
