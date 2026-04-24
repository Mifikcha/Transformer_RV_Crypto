"""
Prediction worker: load bars from DB, compute features, run inference, persist results.

Actual (forward) RV is backfilled once the required future bars have arrived,
using the same Garman-Klass estimator as the training targets.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from view.config import Settings
from view.feature_engine import FeatureEngine, MONTH_BARS
from view.inference import RVInference
from view.models import Bar5m, Prediction, RvActual

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bar loading
# ---------------------------------------------------------------------------

async def _load_bars(session: AsyncSession, limit: int) -> list[dict]:
    stmt = select(Bar5m).order_by(Bar5m.ts.desc()).limit(limit)
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    rows.reverse()
    bars: list[dict] = []
    for r in rows:
        bars.append({
            "ts": r.ts,
            "open_perp": r.open_perp,
            "high_perp": r.high_perp,
            "low_perp": r.low_perp,
            "close_perp": r.close_perp,
            "volume_perp": r.volume_perp,
            "turnover_perp": r.turnover_perp,
            "volume_spot": r.volume_spot,
            "funding_rate": r.funding_rate,
            "open_interest": r.open_interest,
        })
    return bars


# ---------------------------------------------------------------------------
# GK-based forward RV (matches training targets from scripts/add_rv_targets.py)
# ---------------------------------------------------------------------------

def _gk_point(o: float, h: float, l: float, c: float) -> float:
    """Single-bar Garman-Klass point volatility."""
    ratio_hl = max(h / l, 1.0001)
    ratio_co = max(min(c / o, 1e6), 1e-6)
    ln_hl = np.log(ratio_hl)
    ln_co = np.log(ratio_co)
    inner = 0.5 * ln_hl ** 2 - (2 * np.log(2) - 1) * ln_co ** 2
    return float(np.sqrt(max(0.0, inner)))


async def _forward_bars(session: AsyncSession, ts, horizon: int) -> list:
    """Load `horizon` bars starting from `ts` (inclusive), sorted ASC.
    Training target rv_3bar_fwd[T] = GK-RV over bars [T, T+1, T+2],
    so the current bar is included."""
    stmt = (
        select(Bar5m)
        .where(Bar5m.ts >= ts)
        .order_by(Bar5m.ts.asc())
        .limit(horizon)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _forward_rv_gk(bars: list) -> float:
    """GK-based realized vol over a list of Bar5m rows."""
    gk_sq_sum = 0.0
    for b in bars:
        gk_sq_sum += _gk_point(b.open_perp, b.high_perp, b.low_perp, b.close_perp) ** 2
    return float(np.sqrt(gk_sq_sum))


MAX_GAP_SECONDS = 330  # 5min + 30s tolerance


def _trim_to_continuous_tail(bars: list[dict]) -> list[dict]:
    """Return the longest gap-free suffix of bars.

    Walk backwards from the end; the first gap encountered marks the cut point.
    Everything before that gap is discarded.
    """
    if len(bars) <= 1:
        return bars
    for i in range(len(bars) - 1, 0, -1):
        delta = (bars[i]["ts"] - bars[i - 1]["ts"]).total_seconds()
        if delta > MAX_GAP_SECONDS:
            return bars[i:]
    return bars


def _bars_are_consecutive(bars: list) -> bool:
    """Check that every pair of adjacent bars is <= MAX_GAP_SECONDS apart."""
    for i in range(1, len(bars)):
        delta = (bars[i].ts - bars[i - 1].ts).total_seconds()
        if delta > MAX_GAP_SECONDS:
            return False
    return True


async def _load_log_bias_stats(session: AsyncSession) -> tuple[float | None, float | None, float | None, float | None, int]:
    stmt = text(
        """
        SELECT
          AVG(LN(NULLIF(a.rv_3bar,0))  - LN(NULLIF(p.rv_3bar,0)))   AS lb3,
          AVG(LN(NULLIF(a.rv_12bar,0)) - LN(NULLIF(p.rv_12bar,0)))  AS lb12,
          STDDEV(LN(NULLIF(a.rv_3bar,0))  - LN(NULLIF(p.rv_3bar,0)))  AS ls3,
          STDDEV(LN(NULLIF(a.rv_12bar,0)) - LN(NULLIF(p.rv_12bar,0))) AS ls12,
          COUNT(*) AS n_pairs
        FROM predictions p
        JOIN rv_actual a ON a.ts = p.ts
        WHERE p.rv_3bar IS NOT NULL
          AND p.rv_12bar IS NOT NULL
          AND a.rv_3bar IS NOT NULL
          AND a.rv_12bar IS NOT NULL
          AND COALESCE(p.degraded, FALSE) = FALSE
        """
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return None, None, None, None, 0
    return row[0], row[1], row[2], row[3], int(row[4] or 0)


async def backfill_actual_rv(session: AsyncSession) -> int:
    """
    For predictions that are old enough, compute forward GK RV from real bars.
    Skips predictions whose forward window crosses a data gap.
    Returns the number of newly filled rv_actual rows.
    """
    already_filled = select(RvActual.ts)
    stmt = (
        select(Prediction)
        .where(Prediction.ts.notin_(already_filled))
        .order_by(Prediction.ts.asc())
    )
    result = await session.execute(stmt)
    pending = result.scalars().all()

    filled = 0
    skipped_gap = 0
    for pred in pending:
        fwd_12 = await _forward_bars(session, pred.ts, 12)
        if len(fwd_12) < 12:
            continue

        if not _bars_are_consecutive(fwd_12):
            skipped_gap += 1
            logger.debug("Skipping rv_actual for ts=%s: gap in forward bars.", pred.ts)
            continue

        fwd_3 = fwd_12[:3]
        rv_3_actual = _forward_rv_gk(fwd_3)
        rv_12_actual = _forward_rv_gk(fwd_12)

        actual = RvActual(ts=pred.ts, rv_3bar=rv_3_actual, rv_12bar=rv_12_actual)
        await session.merge(actual)
        filled += 1

    if filled or skipped_gap:
        await session.commit()
        logger.info(
            "Backfill rv_actual: filled=%d, skipped_gap=%d", filled, skipped_gap
        )
    return filled


# ---------------------------------------------------------------------------
# Main prediction cycle
# ---------------------------------------------------------------------------

async def run_prediction(
    session: AsyncSession,
    inference: RVInference,
    settings: Settings,
) -> None:
    # --- backfill actual RV for past predictions first ---
    await backfill_actual_rv(session)
    lb3, lb12, ls3, ls12, bias_n = await _load_log_bias_stats(session)
    if bias_n >= settings.min_pairs_for_bias_calibration:
        inference.update_log_bias(lb3=lb3, lb12=lb12, ls3=ls3, ls12=ls12)
        logger.info(
            "Bias calibration active: n=%d lb3=%.4f lb12=%.4f",
            bias_n, float(lb3 or 0), float(lb12 or 0),
        )

    # --- load bars & compute features ---
    bars = await _load_bars(session, settings.bar_buffer_size)
    if not bars:
        logger.warning("No bars in DB, skipping prediction.")
        return

    bars = _trim_to_continuous_tail(bars)
    logger.debug("Continuous tail: %d bars (%s .. %s)", len(bars), bars[0]["ts"], bars[-1]["ts"])

    if len(bars) < settings.min_bars_for_inference:
        logger.warning(
            "Continuous tail too short for inference: %d bars (need %d). "
            "Waiting for more data to arrive.",
            len(bars), settings.min_bars_for_inference,
        )
        return

    engine = FeatureEngine(buffer_size=settings.bar_buffer_size, min_bars=settings.min_bars_for_inference)
    for b in bars:
        engine.add_bar(b)

    window = engine.get_window(seq_len=240, feature_cols=inference.feature_columns)
    if window is None:
        logger.warning("Not enough data for inference window (have %d bars).", len(bars))
        return

    har = engine.compute_har_context()
    if har is None:
        logger.warning(
            "HAR context unavailable (%d bars, need >= %d for weekly). "
            "Skipping prediction — zeros would corrupt the model output.",
            len(bars), 720,
        )
        return
    logger.info("HAR=%s tail=%d window=%d", np.array2string(har, precision=6), len(bars), len(window))

    degraded = False
    if len(bars) < MONTH_BARS:
        degraded = True
    for i, val in enumerate(har):
        if i < len(inference.har_bounds):
            lo, hi = inference.har_bounds[i]
            if val < lo or val > hi:
                degraded = True
                break

    result = inference.predict(window, har)
    last_bar = bars[-1]
    last_ts = last_bar["ts"]

    existing = await session.scalar(select(Prediction.id).where(Prediction.ts == last_ts).limit(1))
    if existing is not None:
        logger.debug("Prediction for ts=%s already exists, skipping.", last_ts)
        return

    pred = Prediction(
        id=uuid.uuid4(),
        ts=last_ts,
        rv_3bar=result.get("rv_3bar"),
        rv_12bar=result.get("rv_12bar"),
        model_ver=inference.model_ver,
        degraded=degraded,
    )
    session.add(pred)
    await session.commit()

    logger.info(
        "Prediction ts=%s rv_3bar=%.6f rv_12bar=%.6f degraded=%s",
        last_ts,
        result.get("rv_3bar", 0),
        result.get("rv_12bar", 0),
        degraded,
    )
