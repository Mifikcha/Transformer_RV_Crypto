"""
Ingestion worker: fetch bars from Bybit and persist to bars_5m.

- bootstrap_history: one-time paginated backfill (~25 days / ~7200 bars).
- ingest_latest_bar: incremental append of the last closed bar.
- heal_funding_oi: correct funding_rate / open_interest that bootstrap
  wrote as a single current-time snapshot for all historical bars.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
from datetime import datetime, timezone

from sqlalchemy import select, func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from view.bybit_client import BybitClient
from view.models import Bar5m

logger = logging.getLogger(__name__)

BARS_PER_REQUEST = 200
BAR_INTERVAL_SEC = 300  # 5 min


# ---------------------------------------------------------------------------
# Forward-fill lookup for historical time-series (funding rate, OI)
# ---------------------------------------------------------------------------

class _TimeSeriesLookup:
    """Binary-search + forward-fill lookup over sorted (ts, value) pairs."""

    def __init__(self, data: list[tuple[datetime, float]]) -> None:
        self._ts = [d[0] for d in data]
        self._values = [d[1] for d in data]

    def __bool__(self) -> bool:
        return len(self._ts) > 0

    def get(self, ts: datetime) -> float | None:
        """Return the most recent value at or before *ts*."""
        if not self._ts:
            return None
        idx = bisect.bisect_right(self._ts, ts) - 1
        if idx < 0:
            return self._values[0] if self._values else None
        return self._values[idx]


async def _build_funding_lookup(
    client: BybitClient, start_ms: int, end_ms: int,
) -> _TimeSeriesLookup:
    """Paginate ``/v5/market/funding/history`` and return a lookup."""
    all_items: list[tuple[datetime, float]] = []
    cursor_end = end_ms
    for _ in range(20):  # safety limit
        items = await client.fetch_funding_history(
            limit=200, start=start_ms, end=cursor_end,
        )
        if not items:
            break
        for item in items:
            all_items.append((item["ts"], item["funding_rate"]))
        oldest_ts = items[0]["ts"]
        cursor_end = int(oldest_ts.timestamp() * 1000) - 1
        if len(items) < 200 or cursor_end <= start_ms:
            break
        await asyncio.sleep(0.3)
    seen: set[datetime] = set()
    deduped: list[tuple[datetime, float]] = []
    for ts, val in sorted(all_items, key=lambda x: x[0]):
        if ts not in seen:
            seen.add(ts)
            deduped.append((ts, val))
    logger.info("Funding lookup: %d records (%s .. %s)",
                len(deduped),
                deduped[0][0] if deduped else "?",
                deduped[-1][0] if deduped else "?")
    return _TimeSeriesLookup(deduped)


async def _build_oi_lookup(
    client: BybitClient, start_ms: int, end_ms: int,
) -> _TimeSeriesLookup:
    """Paginate ``/v5/market/open-interest`` (5 min) with cursor."""
    all_items: list[tuple[datetime, float]] = []
    cursor: str | None = None
    for _ in range(50):  # safety limit
        items, cursor = await client.fetch_open_interest_history(
            interval="5min", limit=200, start=start_ms, end=end_ms, cursor=cursor,
        )
        if not items:
            break
        for item in items:
            all_items.append((item["ts"], item["open_interest"]))
        if not cursor:
            break
        await asyncio.sleep(0.3)
    seen: set[datetime] = set()
    deduped: list[tuple[datetime, float]] = []
    for ts, val in sorted(all_items, key=lambda x: x[0]):
        if ts not in seen:
            seen.add(ts)
            deduped.append((ts, val))
    logger.info("OI lookup: %d records (%s .. %s)",
                len(deduped),
                deduped[0][0] if deduped else "?",
                deduped[-1][0] if deduped else "?")
    return _TimeSeriesLookup(deduped)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

async def bootstrap_history(
    client: BybitClient,
    session: AsyncSession,
    days: int = 25,
) -> None:
    """
    Backfill bars_5m with historical data.
    Uses per-bar historical funding rate and open interest.
    """
    target_bars = days * 288
    logger.info("Bootstrap: fetching ~%d bars (%d days)...", target_bars, days)

    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = end_ms - days * 86_400 * 1000

    funding_lookup = await _build_funding_lookup(client, start_ms, end_ms)
    oi_lookup = await _build_oi_lookup(client, start_ms, end_ms)

    # Single-value fallback if historical APIs returned nothing.
    fallback_funding: float | None = None
    fallback_oi: float | None = None
    if not funding_lookup:
        try:
            fallback_funding = await client.fetch_funding_rate()
        except Exception:
            pass
    if not oi_lookup:
        try:
            fallback_oi = await client.fetch_open_interest()
        except Exception:
            pass

    total_inserted = 0
    cursor_end_ms = end_ms

    while total_inserted < target_bars:
        perp_bars = await client.fetch_klines_perp(limit=BARS_PER_REQUEST, end=cursor_end_ms)
        if not perp_bars:
            logger.warning("Bootstrap: no more perp bars returned, stopping.")
            break

        spot_bars = await client.fetch_klines_spot(limit=BARS_PER_REQUEST, end=cursor_end_ms)
        spot_by_ts = {b["ts"]: b for b in spot_bars}

        rows: list[dict] = []
        for pb in perp_bars:
            sb = spot_by_ts.get(pb["ts"], {})
            fr = funding_lookup.get(pb["ts"]) if funding_lookup else fallback_funding
            oi = oi_lookup.get(pb["ts"]) if oi_lookup else fallback_oi
            rows.append({
                "ts": pb["ts"],
                "open_perp": pb["open"],
                "high_perp": pb["high"],
                "low_perp": pb["low"],
                "close_perp": pb["close"],
                "volume_perp": pb["volume"],
                "turnover_perp": pb["turnover"],
                "open_spot": sb.get("open"),
                "high_spot": sb.get("high"),
                "low_spot": sb.get("low"),
                "close_spot": sb.get("close"),
                "volume_spot": sb.get("volume"),
                "turnover_spot": sb.get("turnover"),
                "funding_rate": fr,
                "open_interest": oi,
            })

        if rows:
            stmt = pg_insert(Bar5m).values(rows)
            stmt = stmt.on_conflict_do_nothing(index_elements=["ts"])
            await session.execute(stmt)
            await session.commit()
            total_inserted += len(rows)
            oldest_ts = min(r["ts"] for r in rows)
            logger.info("Bootstrap: inserted %d bars (total %d), oldest=%s",
                        len(rows), total_inserted, oldest_ts)

        earliest_ts = min(b["ts"] for b in perp_bars)
        cursor_end_ms = int(earliest_ts.timestamp() * 1000) - 1
        await asyncio.sleep(0.5)

    logger.info("Bootstrap complete: %d bars inserted.", total_inserted)


# ---------------------------------------------------------------------------
# Gap healing (bar gaps)
# ---------------------------------------------------------------------------

async def heal_gap(client: BybitClient, session: AsyncSession) -> int:
    """
    Find the most recent internal gap in bars_5m and fill it from the Bybit API.
    Uses per-bar historical funding/OI for the gap period.
    """
    stmt = select(Bar5m.ts).order_by(Bar5m.ts.desc()).limit(2000)
    result = await session.execute(stmt)
    timestamps = [r[0] for r in result.fetchall()]
    if len(timestamps) < 2:
        return 0
    timestamps.reverse()

    gap_before = None
    gap_after = None
    for i in range(len(timestamps) - 1, 0, -1):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if delta > BAR_INTERVAL_SEC * 1.5:
            gap_before = timestamps[i - 1]
            gap_after = timestamps[i]
            break

    if gap_before is None:
        logger.info("Gap healing: no internal gaps found in recent 2000 bars.")
        return 0

    gap_minutes = (gap_after - gap_before).total_seconds() / 60
    logger.info(
        "Gap healing: found %.0f-min gap (%s -> %s). Filling...",
        gap_minutes, gap_before, gap_after,
    )

    gap_start_ms = int(gap_before.timestamp() * 1000)
    gap_end_ms = int(gap_after.timestamp() * 1000)

    funding_lookup = await _build_funding_lookup(client, gap_start_ms, gap_end_ms)
    oi_lookup = await _build_oi_lookup(client, gap_start_ms, gap_end_ms)

    fallback_funding: float | None = None
    fallback_oi: float | None = None
    if not funding_lookup:
        try:
            fallback_funding = await client.fetch_funding_rate()
        except Exception:
            pass
    if not oi_lookup:
        try:
            fallback_oi = await client.fetch_open_interest()
        except Exception:
            pass

    end_ms = gap_end_ms
    start_ms = gap_start_ms
    total_inserted = 0

    while end_ms > start_ms:
        perp_bars = await client.fetch_klines_perp(limit=BARS_PER_REQUEST, end=end_ms)
        if not perp_bars:
            break

        spot_bars = await client.fetch_klines_spot(limit=BARS_PER_REQUEST, end=end_ms)
        spot_by_ts = {b["ts"]: b for b in spot_bars}

        rows: list[dict] = []
        for pb in perp_bars:
            if pb["ts"] <= gap_before or pb["ts"] >= gap_after:
                continue
            sb = spot_by_ts.get(pb["ts"], {})
            fr = funding_lookup.get(pb["ts"]) if funding_lookup else fallback_funding
            oi = oi_lookup.get(pb["ts"]) if oi_lookup else fallback_oi
            rows.append({
                "ts": pb["ts"],
                "open_perp": pb["open"],
                "high_perp": pb["high"],
                "low_perp": pb["low"],
                "close_perp": pb["close"],
                "volume_perp": pb["volume"],
                "turnover_perp": pb["turnover"],
                "open_spot": sb.get("open"),
                "high_spot": sb.get("high"),
                "low_spot": sb.get("low"),
                "close_spot": sb.get("close"),
                "volume_spot": sb.get("volume"),
                "turnover_spot": sb.get("turnover"),
                "funding_rate": fr,
                "open_interest": oi,
            })

        if rows:
            stmt = pg_insert(Bar5m).values(rows).on_conflict_do_nothing(index_elements=["ts"])
            await session.execute(stmt)
            await session.commit()
            total_inserted += len(rows)

        earliest_ts = min(b["ts"] for b in perp_bars)
        if earliest_ts <= gap_before:
            break
        end_ms = int(earliest_ts.timestamp() * 1000) - 1
        await asyncio.sleep(0.5)

    logger.info("Gap healing complete: %d bars inserted.", total_inserted)
    return total_inserted


# ---------------------------------------------------------------------------
# Funding / OI healing (correct stale snapshot values in existing bars)
# ---------------------------------------------------------------------------

async def heal_funding_oi(client: BybitClient, session: AsyncSession) -> int:
    """
    Update ``funding_rate`` and ``open_interest`` for all bars in the DB
    using historical Bybit API data.  This corrects the bug where bootstrap
    stamped a single current-time value onto every historical bar.
    """
    min_ts = await session.scalar(select(func.min(Bar5m.ts)))
    max_ts = await session.scalar(select(func.max(Bar5m.ts)))
    if min_ts is None or max_ts is None:
        return 0

    start_ms = int(min_ts.timestamp() * 1000)
    end_ms = int(max_ts.timestamp() * 1000) + 1

    logger.info("heal_funding_oi: building lookups for %s .. %s", min_ts, max_ts)
    funding_lookup = await _build_funding_lookup(client, start_ms, end_ms)
    oi_lookup = await _build_oi_lookup(client, start_ms, end_ms)

    if not funding_lookup and not oi_lookup:
        logger.warning("heal_funding_oi: no historical funding/OI data from API")
        return 0

    BATCH = 500
    offset = 0
    total_updated = 0

    while True:
        stmt = (
            select(Bar5m)
            .order_by(Bar5m.ts.asc())
            .offset(offset)
            .limit(BATCH)
        )
        result = await session.execute(stmt)
        bars = result.scalars().all()
        if not bars:
            break

        batch_updated = 0
        for bar in bars:
            changed = False
            if funding_lookup:
                fr = funding_lookup.get(bar.ts)
                if fr is not None and bar.funding_rate != fr:
                    bar.funding_rate = fr
                    changed = True
            if oi_lookup:
                oi = oi_lookup.get(bar.ts)
                if oi is not None and bar.open_interest != oi:
                    bar.open_interest = oi
                    changed = True
            if changed:
                batch_updated += 1

        if batch_updated:
            await session.commit()
            total_updated += batch_updated

        offset += BATCH

    if total_updated:
        logger.info("heal_funding_oi: updated %d bars", total_updated)
    else:
        logger.info("heal_funding_oi: all bars already have correct values")
    return total_updated


# ---------------------------------------------------------------------------
# Incremental ingestion (latest closed bars)
# ---------------------------------------------------------------------------

async def ingest_latest_bar(client: BybitClient, session: AsyncSession) -> None:
    """
    Incremental ingestion with gap healing.

    Fetches a recent kline batch and upserts all closed bars newer than the
    latest DB timestamp, preventing 5m gaps from network hiccups.
    """
    await asyncio.sleep(15)

    last_ts = await session.scalar(select(func.max(Bar5m.ts)))

    perp_bars = await client.fetch_klines_perp(limit=BARS_PER_REQUEST)
    spot_bars = await client.fetch_klines_spot(limit=BARS_PER_REQUEST)
    if len(perp_bars) < 2:
        logger.warning("Ingestion: not enough perp bars returned.")
        return

    perp_closed = perp_bars[:-1]
    spot_closed = spot_bars[:-1] if len(spot_bars) >= 2 else spot_bars
    spot_by_ts = {b["ts"]: b for b in spot_closed}

    candidate_ts = [pb["ts"] for pb in perp_closed if (last_ts is None or pb["ts"] > last_ts)]
    if not candidate_ts:
        logger.debug("Ingestion: no new closed bars to insert.")
        return
    start_ms = int(min(candidate_ts).timestamp() * 1000) - 24 * 3600 * 1000
    end_ms = int(max(candidate_ts).timestamp() * 1000) + 1
    funding_lookup = await _build_funding_lookup(client, start_ms, end_ms)
    oi_lookup = await _build_oi_lookup(client, start_ms, end_ms)

    fallback_funding: float | None = None
    fallback_oi: float | None = None
    if not funding_lookup:
        try:
            fallback_funding = await client.fetch_funding_rate()
        except Exception:
            pass
    if not oi_lookup:
        try:
            fallback_oi = await client.fetch_open_interest()
        except Exception:
            pass

    rows: list[dict] = []
    for pb in perp_closed:
        if last_ts is not None and pb["ts"] <= last_ts:
            continue
        sb = spot_by_ts.get(pb["ts"], {})
        fr = funding_lookup.get(pb["ts"]) if funding_lookup else fallback_funding
        oi = oi_lookup.get(pb["ts"]) if oi_lookup else fallback_oi
        rows.append({
            "ts": pb["ts"],
            "open_perp": pb["open"],
            "high_perp": pb["high"],
            "low_perp": pb["low"],
            "close_perp": pb["close"],
            "volume_perp": pb["volume"],
            "turnover_perp": pb["turnover"],
            "open_spot": sb.get("open"),
            "high_spot": sb.get("high"),
            "low_spot": sb.get("low"),
            "close_spot": sb.get("close"),
            "volume_spot": sb.get("volume"),
            "turnover_spot": sb.get("turnover"),
            "funding_rate": fr,
            "open_interest": oi,
        })

    if not rows:
        logger.debug("Ingestion: no new closed bars to insert.")
        return

    stmt = pg_insert(Bar5m).values(rows).on_conflict_do_nothing(index_elements=["ts"])
    await session.execute(stmt)
    await session.commit()
    logger.info(
        "Ingested %d bars: %s -> %s",
        len(rows),
        rows[0]["ts"],
        rows[-1]["ts"],
    )
    q = await session.execute(
        text(
            """
            WITH s AS (
              SELECT funding_rate, open_interest, ts
              FROM bars_5m
              WHERE ts > NOW() - INTERVAL '24 hours'
            )
            SELECT COUNT(DISTINCT funding_rate), COUNT(DISTINCT open_interest)
            FROM s
            """
        )
    )
    fr_n, oi_n = q.first() or (0, 0)
    if int(fr_n or 0) < 2:
        logger.warning("Data quality alert: distinct(funding_rate) last24h=%s (<2)", fr_n)
    if int(oi_n or 0) < 50:
        logger.warning("Data quality alert: distinct(open_interest) last24h=%s (<50)", oi_n)
