from __future__ import annotations

import asyncio

from view.bybit_client import BybitClient


async def _main() -> None:
    client = BybitClient()
    bar = await client.fetch_latest_bar()
    print(
        f"OK: ts={bar['ts']}, close={bar['close_perp']:.2f}, "
        f"funding={bar['funding_rate']:.6f}, OI={bar['open_interest']:.0f}"
    )


def main() -> int:
    asyncio.run(_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

