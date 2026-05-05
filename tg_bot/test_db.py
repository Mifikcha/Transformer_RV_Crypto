from __future__ import annotations

import asyncio

from tg_bot.config import Settings
from tg_bot.db import build_engine, init_db


async def _main() -> None:
    s = Settings()
    engine = build_engine(s.database_url)
    await init_db(engine)
    await engine.dispose()
    print("OK: DB schema initialized")


def main() -> int:
    asyncio.run(_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

