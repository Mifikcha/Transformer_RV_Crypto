"""
Telegram bot: /rv, /history, /accuracy, /regime, /model, /data commands.

Requires aiogram >=3.4.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, TelegramObject
from aiogram import BaseMiddleware
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tg_bot.config import Settings
from tg_bot.models import Bar5m, Prediction, RvActual
from tg_bot.notification_worker import get_regime_label

logger = logging.getLogger(__name__)

router = Router()
MSK = timezone(timedelta(hours=3))
BOT_STARTED_AT_UTC = datetime.now(timezone.utc)
_FOLD_TRAIN_R2: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (0.0, 0.0),
    2: (0.0, 0.0),
    3: (0.0, 0.0),
    4: (0.7283, 0.5984),
}


def _format_uptime() -> str:
    delta = datetime.now(timezone.utc) - BOT_STARTED_AT_UTC
    total_sec = int(delta.total_seconds())
    days, rem = divmod(total_sec, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}д {hours}ч {minutes}м"
    return f"{hours}ч {minutes}м"


def _resolve_train_reference(settings: Settings) -> tuple[float, float, str]:
    paths = settings.model_paths_list
    fold_vals: list[tuple[float, float]] = []
    fold_names: list[str] = []
    for p in paths:
        stem = Path(p).stem
        fold = None
        if "fold_rv_" in stem:
            try:
                fold = int(stem.split("fold_rv_")[-1].split("_")[0])
            except ValueError:
                fold = None
        if fold is not None and fold in _FOLD_TRAIN_R2:
            fold_vals.append(_FOLD_TRAIN_R2[fold])
            fold_names.append(str(fold))
    if fold_vals:
        t3 = float(np.mean([x[0] for x in fold_vals]))
        t12 = float(np.mean([x[1] for x in fold_vals]))
        return t3, t12, ",".join(fold_names)
    return 0.6936, 0.5528, "legacy"


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

class AuthMiddleware(BaseMiddleware):
    def __init__(self, allowed_ids: set[int]) -> None:
        self.allowed_ids = allowed_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        message: Message | None = data.get("event_update", {}).get("message") if isinstance(data.get("event_update"), dict) else None
        if message is None and isinstance(event, Message):
            message = event
        if message and message.from_user:
            if self.allowed_ids and message.from_user.id not in self.allowed_ids:
                await message.answer("\U0001f6ab Нет доступа")
                return
        return await handler(event, data)


# ---------------------------------------------------------------------------
# /rv -- latest prediction + regime
# ---------------------------------------------------------------------------

@router.message(Command("rv"))
async def cmd_rv(message: Message, session: AsyncSession, settings: Settings) -> None:
    stmt = select(Prediction).order_by(Prediction.created_at.desc()).limit(1)
    result = await session.execute(stmt)
    pred = result.scalar_one_or_none()
    if pred is None:
        await message.answer("Нет прогнозов пока.")
        return

    # median 24h
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    median_val = await session.scalar(
        select(func.percentile_cont(0.5).within_group(Prediction.rv_3bar)).where(Prediction.ts >= cutoff)
    ) or 0

    # actual RV
    actual = await session.scalar(select(RvActual.rv_3bar).where(RvActual.ts == pred.ts))

    emoji, label = get_regime_label(pred.rv_3bar or 0, float(median_val))
    ts_str = pred.ts.strftime("%Y-%m-%d %H:%M") if pred.ts else "?"
    actual_str = f"{actual:.6f}" if actual else "n/a"

    sep = "\u2501" * 20
    text = (
        f"\U0001f4ca RV Прогноз \u00b7 {ts_str} UTC\n"
        f"{sep}\n"
        f"RV 15min:  {pred.rv_3bar:.6f}  {emoji}\n"
        f"RV 1h:     {pred.rv_12bar:.6f}\n"
        f"Текущая:   {actual_str}\n"
        f"Режим: {label}\n"
        f"Модель: {pred.model_ver or '?'}\n"
        f"Uptime: {_format_uptime()}"
    )
    await message.answer(text)


# ---------------------------------------------------------------------------
# /history -- predictions for last N hours
# ---------------------------------------------------------------------------

@router.message(Command("history"))
async def cmd_history(message: Message, session: AsyncSession, **kwargs: Any) -> None:
    # parse hours argument
    hours = 24
    parts = (message.text or "").split()
    if len(parts) > 1:
        arg = parts[1].lower().replace("h", "")
        try:
            hours = int(arg)
        except ValueError:
            pass

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    stmt = (
        select(Prediction)
        .where(Prediction.ts >= cutoff)
        .order_by(Prediction.ts.desc())
        .limit(50)
    )
    result = await session.execute(stmt)
    preds = result.scalars().all()

    if not preds:
        await message.answer(f"Нет прогнозов за последние {hours}ч.")
        return

    lines = [f"\U0001f4cb История за {hours}ч ({len(preds)} записей)\n"]
    for p in preds[:20]:
        ts_str = p.ts.strftime("%H:%M") if p.ts else "?"
        actual = await session.scalar(select(RvActual.rv_3bar).where(RvActual.ts == p.ts))
        actual_str = f"{actual:.6f}" if actual else "---"
        lines.append(f"{ts_str}  pred={p.rv_3bar:.6f}  fact={actual_str}")

    await message.answer("\n".join(lines))


# ---------------------------------------------------------------------------
# /accuracy -- rolling R2 and MAE vs train best
# ---------------------------------------------------------------------------

@router.message(Command("accuracy"))
async def cmd_accuracy(message: Message, session: AsyncSession, settings: Settings, **kwargs: Any) -> None:
    # Parse optional hours argument: /accuracy 4h → last 4 hours only
    hours = 0  # 0 = all data
    parts = (message.text or "").split()
    if len(parts) > 1:
        arg = parts[1].lower().replace("h", "").replace("ч", "")
        try:
            hours = int(arg)
        except ValueError:
            pass

    time_filter = ""
    if hours > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S+00")
        time_filter = f"AND p.ts >= '{cutoff_str}'"

    stmt = text(f"""
        SELECT p.rv_3bar  AS pred_3,
               p.rv_12bar AS pred_12,
               a.rv_3bar  AS actual_3,
               a.rv_12bar AS actual_12,
               p.ts
        FROM predictions p
        JOIN rv_actual a ON a.ts = p.ts
        WHERE p.rv_3bar IS NOT NULL
          AND a.rv_3bar IS NOT NULL
          AND COALESCE(p.degraded, FALSE) = FALSE
          {time_filter}
        ORDER BY p.ts DESC
        LIMIT 1000
    """)
    result = await session.execute(stmt)
    rows = result.fetchall()

    if len(rows) < 5:
        await message.answer(f"Недостаточно данных для accuracy ({len(rows)} пар, нужно >= 5).")
        return

    pred_3 = np.array([r[0] for r in rows])
    pred_12 = np.array([r[1] for r in rows])
    actual_3 = np.array([r[2] for r in rows])
    actual_12 = np.array([r[3] for r in rows])

    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 3:
            return 0.0
        c = np.corrcoef(y_true, y_pred)
        return float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0

    def _log_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-10
        return _r2(np.log(np.maximum(y_true, eps)), np.log(np.maximum(y_pred, eps)))

    def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_pred - y_true))

    n = len(rows)
    r2_3 = _r2(actual_3, pred_3)
    r2_12 = _r2(actual_12, pred_12)
    mae_3 = _mae(actual_3, pred_3)
    mae_12 = _mae(actual_12, pred_12)
    corr_3 = _pearson(actual_3, pred_3)
    corr_12 = _pearson(actual_12, pred_12)
    logr2_3 = _log_r2(actual_3, pred_3)
    logr2_12 = _log_r2(actual_12, pred_12)
    bias_3 = _bias(actual_3, pred_3)
    bias_12 = _bias(actual_12, pred_12)

    train_r2_3, train_r2_12, train_fold_label = _resolve_train_reference(settings)
    reliability = "\u26a0\ufe0f (N<50)" if n < 50 else "\u2705"
    period = f"за {hours}ч" if hours > 0 else "все"

    lbl_r2 = "R²"
    lbl_logr2 = "log R²"
    lbl_corr = "Pearson r"
    lbl_mae = "MAE"
    lbl_bias = "Bias"
    hdr_empty = ""
    hdr_15 = "15min"
    hdr_1h = "1h"

    stmt_30d = text(
        """
        SELECT p.rv_3bar, p.rv_12bar, a.rv_3bar, a.rv_12bar
        FROM predictions p
        JOIN rv_actual a ON a.ts = p.ts
        WHERE p.ts >= NOW() - INTERVAL '30 days'
          AND COALESCE(p.degraded, FALSE) = FALSE
          AND p.rv_3bar IS NOT NULL
          AND p.rv_12bar IS NOT NULL
          AND a.rv_3bar IS NOT NULL
          AND a.rv_12bar IS NOT NULL
        """
    )
    rows_30d = (await session.execute(stmt_30d)).fetchall()
    if rows_30d:
        p3_30 = np.array([r[0] for r in rows_30d], dtype=float)
        p12_30 = np.array([r[1] for r in rows_30d], dtype=float)
        a3_30 = np.array([r[2] for r in rows_30d], dtype=float)
        a12_30 = np.array([r[3] for r in rows_30d], dtype=float)
        corr3_30 = _pearson(a3_30, p3_30)
        corr12_30 = _pearson(a12_30, p12_30)
        bias3_30 = _bias(a3_30, p3_30)
        bias12_30 = _bias(a12_30, p12_30)
    else:
        corr3_30 = corr12_30 = 0.0
        bias3_30 = bias12_30 = 0.0

    lines_out = [
        f"\U0001f4c8 Accuracy \u00b7 {n} пар ({period}) {reliability}",
        "",
        f"{hdr_empty:>12s}{hdr_15:>10s}{hdr_1h:>10s}",
    ]
    if n >= 100:
        lines_out.append(f"{lbl_r2:>12s}{r2_3:>10.4f}{r2_12:>10.4f}")
        lines_out.append(f"{lbl_logr2:>12s}{logr2_3:>10.4f}{logr2_12:>10.4f}")
    else:
        lines_out.append(f"{lbl_r2:>12s}{'hidden':>10s}{'hidden':>10s}")
        lines_out.append(f"{lbl_logr2:>12s}{'hidden':>10s}{'hidden':>10s}")
    lines_out.extend(
        [
            f"{lbl_corr:>12s}{corr_3:>10.4f}{corr_12:>10.4f}",
            f"{lbl_mae:>12s}{mae_3:>10.6f}{mae_12:>10.6f}",
            f"{lbl_bias:>12s}{bias_3:>+10.6f}{bias_12:>+10.6f}",
            "",
            f"{'Pearson 30d':>12s}{corr3_30:>10.4f}{corr12_30:>10.4f}",
            f"{'Bias 30d':>12s}{bias3_30:>+10.6f}{bias12_30:>+10.6f}",
            "",
            f"Train fold: {train_fold_label}",
            f"Train R\u00b2: {train_r2_3:.4f} / {train_r2_12:.4f}",
            f"Drift R\u00b2: {r2_3 - train_r2_3:+.4f} / {r2_12 - train_r2_12:+.4f}",
            f"Uptime: {_format_uptime()}",
        ]
    )
    text_out = "\n".join(lines_out)

    if r2_3 < train_r2_3 - 0.1 or r2_12 < train_r2_12 - 0.1:
        text_out += "  \u26a0\ufe0f"

    if n <= 20:
        lines = ["\n\n\U0001f50d Последние пары (pred | actual):"]
        for r in rows[:8]:
            ts_str = r[4].strftime("%H:%M") if r[4] else "?"
            lines.append(
                f"  {ts_str}  3b: {r[0]:.5f}|{r[2]:.5f}  12b: {r[1]:.5f}|{r[3]:.5f}"
            )
        text_out += "\n".join(lines)

    await message.answer(text_out)


# ---------------------------------------------------------------------------
# /regime -- current regime + 7-day history
# ---------------------------------------------------------------------------

@router.message(Command("regime"))
async def cmd_regime(message: Message, session: AsyncSession, **kwargs: Any) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    stmt = (
        select(Prediction.ts, Prediction.rv_3bar)
        .where(Prediction.ts >= cutoff)
        .order_by(Prediction.ts.desc())
        .limit(2016)  # 7 days * 288
    )
    result = await session.execute(stmt)
    rows = result.fetchall()

    if not rows:
        await message.answer("Нет данных за 7 дней.")
        return

    all_rv = [r[1] for r in rows if r[1] is not None]
    if not all_rv:
        await message.answer("Нет RV-данных.")
        return

    median_rv = float(np.median(all_rv))
    latest_rv = all_rv[0]
    emoji, label = get_regime_label(latest_rv, median_rv)

    regime_counts = {"низкая": 0, "умеренная": 0, "высокая": 0, "экстремальная": 0}
    for rv in all_rv:
        _, l = get_regime_label(rv, median_rv)
        if l in regime_counts:
            regime_counts[l] += 1
    total = sum(regime_counts.values()) or 1

    lines = [
        f"{emoji} Текущий режим: {label}",
        f"RV: {latest_rv:.6f}  |  Медиана 7д: {median_rv:.6f}",
        "",
        "Распределение за 7 дней:",
    ]
    for regime, cnt in regime_counts.items():
        pct = cnt / total * 100
        lines.append(f"  {regime}: {pct:.0f}%")

    await message.answer("\n".join(lines))


# ---------------------------------------------------------------------------
# /model -- model info
# ---------------------------------------------------------------------------

@router.message(Command("model"))
async def cmd_model(message: Message, session: AsyncSession, **kwargs: Any) -> None:
    stmt = select(Prediction.model_ver, Prediction.created_at).order_by(Prediction.created_at.desc()).limit(1)
    result = await session.execute(stmt)
    row = result.first()
    if row is None:
        await message.answer("Нет прогнозов.")
        return

    ver, created = row
    ts_str = created.strftime("%Y-%m-%d %H:%M:%S") if created else "?"
    await message.answer(
        f"\U0001f916 Модель\n"
        f"Версия: {ver or '?'}\n"
        f"Последний прогноз: {ts_str} UTC"
    )


# ---------------------------------------------------------------------------
# /data -- data status
# ---------------------------------------------------------------------------

@router.message(Command("data"))
async def cmd_data(message: Message, session: AsyncSession, **kwargs: Any) -> None:
    max_ts = await session.scalar(select(func.max(Bar5m.ts)))
    total = await session.scalar(select(func.count(Bar5m.ts)))
    min_ts = await session.scalar(select(func.min(Bar5m.ts)))

    if max_ts is None:
        await message.answer("Нет данных в bars_5m.")
        return

    now = datetime.now(timezone.utc)
    lag = now - max_ts
    lag_min = lag.total_seconds() / 60
    max_ts_msk = max_ts.astimezone(MSK)
    min_ts_msk = min_ts.astimezone(MSK) if min_ts else None

    # Data quality over last 24h: gap count and max gap in minutes.
    cutoff_24h = now - timedelta(hours=24)
    ts_rows = await session.execute(
        select(Bar5m.ts).where(Bar5m.ts >= cutoff_24h).order_by(Bar5m.ts.asc())
    )
    ts_list = [row[0] for row in ts_rows.fetchall()]

    gap_count = 0
    max_gap_min = 0.0
    for i in range(1, len(ts_list)):
        gap_min = (ts_list[i] - ts_list[i - 1]).total_seconds() / 60.0
        if gap_min > 5.5:
            gap_count += 1
            if gap_min > max_gap_min:
                max_gap_min = gap_min

    status = "\u2705" if lag_min < 15 else "\u26a0\ufe0f"
    text_out = (
        f"\U0001f4be Данные {status}\n"
        f"Последний бар: {max_ts_msk.strftime('%Y-%m-%d %H:%M')} МСК\n"
        f"Задержка: {lag_min:.0f} мин\n"
        f"Первый бар: {min_ts_msk.strftime('%Y-%m-%d %H:%M') if min_ts_msk else '?'} МСК\n"
        f"Всего баров: {total:,}\n"
        f"Дыры 24ч: {gap_count}\n"
        f"Макс. gap 24ч: {max_gap_min:.0f} мин"
    )
    await message.answer(text_out)


# ---------------------------------------------------------------------------
# /debug -- diagnostic dump for accuracy troubleshooting
# ---------------------------------------------------------------------------

@router.message(Command("debug"))
async def cmd_debug(message: Message, session: AsyncSession, **kwargs: Any) -> None:
    total_pred = await session.scalar(select(func.count(Prediction.ts)))
    total_actual = await session.scalar(select(func.count(RvActual.ts)))
    min_pred_ts = await session.scalar(select(func.min(Prediction.ts)))
    max_pred_ts = await session.scalar(select(func.max(Prediction.ts)))
    min_actual_ts = await session.scalar(select(func.min(RvActual.ts)))
    max_actual_ts = await session.scalar(select(func.max(RvActual.ts)))

    def _msk(ts):
        if ts is None:
            return "—"
        return ts.astimezone(MSK).strftime("%m-%d %H:%M")

    lines = [
        "\U0001f527 Debug info",
        f"Predictions: {total_pred}  ({_msk(min_pred_ts)} .. {_msk(max_pred_ts)})",
        f"RvActual:    {total_actual}  ({_msk(min_actual_ts)} .. {_msk(max_actual_ts)})",
        f"Pending (no actual): {(total_pred or 0) - (total_actual or 0)}",
        "",
    ]

    # Show last 10 joined pairs
    stmt = text("""
        SELECT p.ts, p.rv_3bar, p.rv_12bar,
               a.rv_3bar AS a3, a.rv_12bar AS a12
        FROM predictions p
        LEFT JOIN rv_actual a ON a.ts = p.ts
        ORDER BY p.ts DESC
        LIMIT 10
    """)
    result = await session.execute(stmt)
    rows = result.fetchall()

    lines.append("Последние 10 прогнозов:")
    for r in rows:
        ts_msk = r[0].astimezone(MSK).strftime("%m-%d %H:%M") if r[0] else "?"
        p3 = f"{r[1]:.5f}" if r[1] is not None else "None"
        p12 = f"{r[2]:.5f}" if r[2] is not None else "None"
        a3 = f"{r[3]:.5f}" if r[3] is not None else "—"
        a12 = f"{r[4]:.5f}" if r[4] is not None else "—"
        has_actual = "\u2705" if r[3] is not None else "\u23f3"
        lines.append(f"  {has_actual} {ts_msk}  p={p3}/{p12}  a={a3}/{a12}")

    # Show worst 5 pairs by absolute error (1h)
    stmt2 = text("""
        SELECT p.ts, p.rv_12bar, a.rv_12bar,
               ABS(p.rv_12bar - a.rv_12bar) AS err
        FROM predictions p
        JOIN rv_actual a ON a.ts = p.ts
        WHERE p.rv_12bar IS NOT NULL AND a.rv_12bar IS NOT NULL
        ORDER BY err DESC
        LIMIT 5
    """)
    result2 = await session.execute(stmt2)
    worst = result2.fetchall()

    if worst:
        lines.append("")
        lines.append("Worst 5 пар по ошибке 1h:")
        for r in worst:
            ts_msk = r[0].astimezone(MSK).strftime("%m-%d %H:%M") if r[0] else "?"
            lines.append(
                f"  {ts_msk}  pred={r[1]:.5f}  actual={r[2]:.5f}  err={r[3]:.5f}"
            )

    await message.answer("\n".join(lines))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_dispatcher(settings: Settings, session_factory: async_sessionmaker) -> Dispatcher:
    dp = Dispatcher()

    if settings.allowed_chat_ids_set:
        router.message.middleware(AuthMiddleware(settings.allowed_chat_ids_set))

    # Inject session and settings into every handler via middleware
    class DbMiddleware(BaseMiddleware):
        async def __call__(
            self,
            handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
            event: TelegramObject,
            data: dict[str, Any],
        ) -> Any:
            async with session_factory() as session:
                data["session"] = session
                data["settings"] = settings
                return await handler(event, data)

    router.message.middleware(DbMiddleware())
    dp.include_router(router)
    return dp
