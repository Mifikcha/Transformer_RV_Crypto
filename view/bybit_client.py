from __future__ import annotations

import logging
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)


def _ms_to_utc(ms: int | str) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


class BybitClient:
    def __init__(self, base_url: str = "https://api.bybit.com", symbol: str = "BTCUSDT") -> None:
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol

    async def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                data = await resp.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error: {data.get('retMsg', 'unknown')} (path={path})")
        return data["result"]

    async def fetch_klines(
        self,
        category: str,
        limit: int = 200,
        start: int | None = None,
        end: int | None = None,
    ) -> list[dict]:
        """Fetch OHLCV kline bars. Returns list sorted by ts ascending."""
        params: dict = {
            "category": category,
            "symbol": self.symbol,
            "interval": "5",
            "limit": str(limit),
        }
        if start is not None:
            params["start"] = str(start)
        if end is not None:
            params["end"] = str(end)

        result = await self._get("/v5/market/kline", params)
        rows = result.get("list", [])
        bars = []
        for row in rows:
            bars.append({
                "ts": _ms_to_utc(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "turnover": float(row[6]),
            })
        bars.sort(key=lambda b: b["ts"])
        return bars

    async def fetch_klines_perp(
        self, limit: int = 200, start: int | None = None, end: int | None = None,
    ) -> list[dict]:
        return await self.fetch_klines("linear", limit=limit, start=start, end=end)

    async def fetch_klines_spot(
        self, limit: int = 200, start: int | None = None, end: int | None = None,
    ) -> list[dict]:
        return await self.fetch_klines("spot", limit=limit, start=start, end=end)

    async def fetch_funding_rate(self) -> float:
        result = await self._get("/v5/market/funding/history", {
            "category": "linear",
            "symbol": self.symbol,
            "limit": "1",
        })
        items = result.get("list", [])
        if not items:
            raise RuntimeError("No funding rate data returned")
        return float(items[0]["fundingRate"])

    async def fetch_open_interest(self) -> float:
        result = await self._get("/v5/market/open-interest", {
            "category": "linear",
            "symbol": self.symbol,
            "intervalTime": "5min",
            "limit": "1",
        })
        items = result.get("list", [])
        if not items:
            raise RuntimeError("No open interest data returned")
        return float(items[0]["openInterest"])

    # ------------------------------------------------------------------
    # Historical funding / OI for correct bootstrap data
    # ------------------------------------------------------------------

    async def fetch_funding_history(
        self, limit: int = 200,
        start: int | None = None, end: int | None = None,
    ) -> list[dict]:
        """Fetch historical funding rates sorted ascending by timestamp."""
        params: dict = {
            "category": "linear",
            "symbol": self.symbol,
            "limit": str(limit),
        }
        if start is not None:
            params["startTime"] = str(start)
        if end is not None:
            params["endTime"] = str(end)
        result = await self._get("/v5/market/funding/history", params)
        items = result.get("list", [])
        out = [
            {
                "ts": _ms_to_utc(item["fundingRateTimestamp"]),
                "funding_rate": float(item["fundingRate"]),
            }
            for item in items
        ]
        out.sort(key=lambda x: x["ts"])
        return out

    async def fetch_open_interest_history(
        self,
        interval: str = "5min",
        limit: int = 200,
        start: int | None = None,
        end: int | None = None,
        cursor: str | None = None,
    ) -> tuple[list[dict], str | None]:
        """Fetch one page of historical open interest with cursor support."""
        params: dict = {
            "category": "linear",
            "symbol": self.symbol,
            "intervalTime": interval,
            "limit": str(limit),
        }
        if start is not None:
            params["startTime"] = str(start)
        if end is not None:
            params["endTime"] = str(end)
        if cursor:
            params["cursor"] = cursor
        result = await self._get("/v5/market/open-interest", params)
        items = result.get("list", [])
        out = [
            {
                "ts": _ms_to_utc(item["timestamp"]),
                "open_interest": float(item["openInterest"]),
            }
            for item in items
        ]
        out.sort(key=lambda x: x["ts"])
        return out, result.get("nextPageCursor")

    async def fetch_latest_bar(self) -> dict:
        """
        Fetch the most recent *closed* bar by combining perp + spot + funding + OI.
        Uses result.list[1] (index 1) because index 0 is the current unclosed bar.
        """
        perp_bars = await self.fetch_klines_perp(limit=2)
        spot_bars = await self.fetch_klines_spot(limit=2)
        funding = await self.fetch_funding_rate()
        oi = await self.fetch_open_interest()

        if len(perp_bars) < 2:
            raise RuntimeError("Not enough perp bars returned (need at least 2)")

        # Second-to-last = last fully closed bar (bars sorted asc, so index -2)
        perp = perp_bars[-2]
        spot = spot_bars[-2] if len(spot_bars) >= 2 else spot_bars[-1]

        bar = {
            "ts": perp["ts"],
            "open_perp": perp["open"],
            "high_perp": perp["high"],
            "low_perp": perp["low"],
            "close_perp": perp["close"],
            "volume_perp": perp["volume"],
            "turnover_perp": perp["turnover"],
            "open_spot": spot["open"],
            "high_spot": spot["high"],
            "low_spot": spot["low"],
            "close_spot": spot["close"],
            "volume_spot": spot["volume"],
            "turnover_spot": spot["turnover"],
            "funding_rate": funding,
            "open_interest": oi,
        }
        logger.info("Fetched bar ts=%s close_perp=%.2f funding=%.6f OI=%.0f",
                     bar["ts"], bar["close_perp"], bar["funding_rate"], bar["open_interest"])
        return bar
