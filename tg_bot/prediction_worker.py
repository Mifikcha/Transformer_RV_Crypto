"""
Prediction worker: load bars from DB, compute features, run inference, persist results.

Actual (forward) RV is backfilled once the required future bars have arrived,
using the same Garman-Klass estimator as the training targets.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Awaitable, Callable

import numpy as np
import pandas as pd
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from view.config import Settings
from view.feature_engine import (
    FEATURE_COLS,
    FeatureEngine,
    HAR_RV_BASE,
    MONTH_BARS,
    WEEK_BARS,
)
from view.inference import RVInference
from view.models import Bar5m, Prediction, RvActual

logger = logging.getLogger(__name__)

# Length of the inference input window (must match training; same as the live cycle).
_SEQ_LEN = 240


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


# ---------------------------------------------------------------------------
# Batch backfill of predictions over a historical range
# ---------------------------------------------------------------------------

async def _load_bars_range(
    session: AsyncSession,
    start_ts: datetime | None,
    end_ts: datetime | None,
) -> list[dict]:
    """Load all bars in [start_ts, end_ts] (inclusive), sorted ASC."""
    stmt = select(Bar5m)
    if start_ts is not None:
        stmt = stmt.where(Bar5m.ts >= start_ts)
    if end_ts is not None:
        stmt = stmt.where(Bar5m.ts <= end_ts)
    stmt = stmt.order_by(Bar5m.ts.asc())
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
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


def _continuous_tail_lengths(bars: list[dict]) -> list[int]:
    """For every bar i, the length of the gap-free run ending at i (1-indexed count).

    Equivalent to running ``_trim_to_continuous_tail`` for the prefix ``bars[:i+1]``
    at every position, but in a single linear pass. Used to mirror the live cycle's
    "min_bars_for_inference" / "MONTH_BARS for non-degraded" gating decisions.
    """
    n = len(bars)
    out = [0] * n
    if n == 0:
        return out
    out[0] = 1
    for i in range(1, n):
        delta = (bars[i]["ts"] - bars[i - 1]["ts"]).total_seconds()
        if delta > MAX_GAP_SECONDS:
            out[i] = 1
        else:
            out[i] = out[i - 1] + 1
    return out


async def backfill_predictions(
    session: AsyncSession,
    inference: RVInference,
    settings: Settings,
    *,
    start_ts: datetime | None = None,
    end_ts: datetime | None = None,
    update_bias: bool = True,
    fill_actuals: bool = True,
    commit_every: int = 200,
    progress_cb: Callable[[int, int], Awaitable[None] | None] | None = None,
) -> dict:
    """Run inference for every bar in (start_ts, end_ts] that has no prediction yet.

    Mirrors the live ``run_prediction`` logic exactly (HAR context, log-bias
    calibration, ``degraded`` flag, idempotent insert) but over a batch of
    historical bars in a single pass: features and HAR rolling means are
    computed once on the whole loaded buffer, then sliced per target bar.

    Args:
        start_ts: lower bound (exclusive). When None, defaults to
            ``MAX(predictions.ts)`` so we backfill the gap after the last
            recorded prediction. When the table is empty, we start from the
            first bar that has enough history (``min_bars_for_inference``).
        end_ts: upper bound (inclusive). When None, ``MAX(bars_5m.ts)``.
        update_bias: if True, refresh the multiplicative log-bias from existing
            (prediction, rv_actual) pairs before running — same behaviour as
            ``run_prediction``.
        fill_actuals: if True, run ``backfill_actual_rv`` after inserting new
            predictions, so the dashboard sees both lines in one shot.
        commit_every: batch size for intermediate commits (keeps memory and
            transaction size bounded for multi-day backfills).
        progress_cb: optional ``(filled, total) -> None`` callback. May be sync
            or async. Called every ``commit_every`` iterations and at the end.

    Returns:
        Dict with ``filled``, ``skipped_existing``, ``skipped_features``,
        ``skipped_har``, ``skipped_short_tail``, ``degraded``, ``total_targets``,
        ``actuals_filled``.
    """
    if update_bias:
        lb3, lb12, ls3, ls12, bias_n = await _load_log_bias_stats(session)
        if bias_n >= settings.min_pairs_for_bias_calibration:
            inference.update_log_bias(lb3=lb3, lb12=lb12, ls3=ls3, ls12=ls12)
            logger.info(
                "Bias calibration active: n=%d lb3=%.4f lb12=%.4f",
                bias_n, float(lb3 or 0), float(lb12 or 0),
            )

    # Resolve the time window we want to fill.
    if end_ts is None:
        end_ts = await session.scalar(select(func.max(Bar5m.ts)))
    if end_ts is None:
        return {"filled": 0, "skipped_existing": 0, "total_targets": 0,
                "msg": "no bars in DB"}

    if start_ts is None:
        start_ts = await session.scalar(select(func.max(Prediction.ts)))

    # Load enough history BEFORE start_ts to populate the inference window
    # (240 bars) and HAR rolling means (up to MONTH_BARS = 6336 bars). One
    # bar = 5 minutes; we add a small safety margin for tail trimming.
    history_bars = MONTH_BARS + _SEQ_LEN + 32
    if start_ts is None:
        bars = await _load_bars_range(session, None, end_ts)
    else:
        history_start = start_ts - pd.Timedelta(minutes=5 * history_bars)
        bars = await _load_bars_range(session, history_start, end_ts)

    if len(bars) < settings.min_bars_for_inference:
        return {"filled": 0, "skipped_existing": 0, "total_targets": 0,
                "msg": f"need at least {settings.min_bars_for_inference} bars, "
                       f"have {len(bars)}"}

    # Existing predictions inside the target window — for idempotency.
    existing_window_start = bars[0]["ts"]
    existing_stmt = select(Prediction.ts).where(
        Prediction.ts >= existing_window_start,
        Prediction.ts <= end_ts,
    )
    existing_ts: set = {row[0] for row in (await session.execute(existing_stmt)).all()}

    # Compute features once on the whole buffer.
    engine = FeatureEngine(buffer_size=len(bars) + 32, min_bars=settings.min_bars_for_inference)
    for b in bars:
        engine.add_bar(b)
    df = engine._get_feature_df()

    feature_cols = inference.feature_columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"FeatureEngine produced no values for: {missing_cols}. "
            "Likely a model/feature_engine version mismatch."
        )

    # Pre-compute HAR rolling weekly/monthly means once for every bar.
    # Matches FeatureEngine.compute_har_context exactly.
    har_w_series: list[pd.Series] = []
    har_m_series: list[pd.Series] = []
    for col in HAR_RV_BASE:
        s = df[col].astype(float)
        w = s.rolling(WEEK_BARS, min_periods=WEEK_BARS // 2).mean().bfill().ffill()
        m = s.rolling(MONTH_BARS, min_periods=MONTH_BARS // 2).mean().bfill().ffill()
        har_w_series.append(w)
        har_m_series.append(m)

    cont_tail = _continuous_tail_lengths(bars)
    feature_df = df[feature_cols]
    feature_arr = feature_df.to_numpy(dtype=np.float64, copy=False)
    feature_nan_mask = feature_df.isna().to_numpy()

    # Build the list of target indices to predict.
    target_indices: list[int] = []
    for i, b in enumerate(bars):
        ts = b["ts"]
        if start_ts is not None and ts <= start_ts:
            continue
        if ts in existing_ts:
            continue
        target_indices.append(i)

    stats = {
        "filled": 0,
        "skipped_existing": len(existing_ts),
        "skipped_features": 0,
        "skipped_har": 0,
        "skipped_short_tail": 0,
        "degraded": 0,
        "total_targets": len(target_indices),
        "actuals_filled": 0,
    }

    if not target_indices:
        if fill_actuals:
            stats["actuals_filled"] = await backfill_actual_rv(session)
        return stats

    async def _emit_progress(done: int) -> None:
        if progress_cb is None:
            return
        result = progress_cb(done, len(target_indices))
        if hasattr(result, "__await__"):
            await result  # type: ignore[func-returns-value]

    pending = 0
    for k_idx, target_idx in enumerate(target_indices, start=1):
        tail_len = cont_tail[target_idx]
        if tail_len < settings.min_bars_for_inference or tail_len < _SEQ_LEN:
            stats["skipped_short_tail"] += 1
            continue

        win_start = target_idx - _SEQ_LEN + 1
        if feature_nan_mask[win_start:target_idx + 1].any():
            stats["skipped_features"] += 1
            continue
        window = feature_arr[win_start:target_idx + 1]

        har_vec: list[float] = []
        valid_har = True
        for k in range(len(HAR_RV_BASE)):
            w_val = har_w_series[k].iloc[target_idx]
            m_val = har_m_series[k].iloc[target_idx]
            if pd.isna(w_val):
                valid_har = False
                break
            if pd.isna(m_val):
                m_val = w_val
            har_vec.extend([float(w_val), float(m_val)])
        if not valid_har:
            stats["skipped_har"] += 1
            continue
        har = np.array(har_vec, dtype=np.float64)

        degraded = False
        if tail_len < MONTH_BARS:
            degraded = True
        for k, val in enumerate(har):
            if k < len(inference.har_bounds):
                lo, hi = inference.har_bounds[k]
                if val < lo or val > hi:
                    degraded = True
                    break
        if degraded:
            stats["degraded"] += 1

        result = inference.predict(window, har)
        pred = Prediction(
            id=uuid.uuid4(),
            ts=bars[target_idx]["ts"],
            rv_3bar=result.get("rv_3bar"),
            rv_12bar=result.get("rv_12bar"),
            model_ver=inference.model_ver,
            degraded=degraded,
        )
        session.add(pred)
        stats["filled"] += 1
        pending += 1

        if pending >= commit_every:
            await session.commit()
            pending = 0
            await _emit_progress(stats["filled"])

    if pending:
        await session.commit()
    await _emit_progress(stats["filled"])

    logger.info(
        "Backfill predictions: filled=%d, skipped_existing=%d, skipped_features=%d, "
        "skipped_har=%d, skipped_short_tail=%d, degraded=%d (of %d targets)",
        stats["filled"], stats["skipped_existing"], stats["skipped_features"],
        stats["skipped_har"], stats["skipped_short_tail"], stats["degraded"],
        stats["total_targets"],
    )

    if fill_actuals:
        stats["actuals_filled"] = await backfill_actual_rv(session)

    return stats
