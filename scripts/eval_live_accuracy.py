"""
Evaluate live bot accuracy over different sample sizes.

Usage:
  python -m scripts.eval_live_accuracy
  python -m scripts.eval_live_accuracy --sizes 100 200 500 1000
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
from sqlalchemy import text

from tg_bot.config import Settings
from tg_bot.db import build_engine, build_session_factory


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return 0.0
    c = np.corrcoef(y_true, y_pred)
    return float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=[100, 200, 500, 1000])
    args = parser.parse_args()

    settings = Settings()
    engine = build_engine(settings.database_url)
    session_factory = build_session_factory(engine)

    async with session_factory() as session:
        for n in args.sizes:
            stmt = text(
                f"""
                SELECT p.rv_3bar, p.rv_12bar, a.rv_3bar, a.rv_12bar
                FROM predictions p
                JOIN rv_actual a ON a.ts = p.ts
                WHERE p.rv_3bar IS NOT NULL
                  AND p.rv_12bar IS NOT NULL
                  AND a.rv_3bar IS NOT NULL
                  AND a.rv_12bar IS NOT NULL
                  AND COALESCE(p.degraded, FALSE) = FALSE
                ORDER BY p.ts DESC
                LIMIT {int(n)}
                """
            )
            rows = (await session.execute(stmt)).fetchall()
            if len(rows) < 5:
                print(f"N={n}: not enough data ({len(rows)})")
                continue
            p3 = np.array([r[0] for r in rows], dtype=float)
            p12 = np.array([r[1] for r in rows], dtype=float)
            a3 = np.array([r[2] for r in rows], dtype=float)
            a12 = np.array([r[3] for r in rows], dtype=float)
            print(
                f"N={len(rows):4d} | "
                f"R2_15m={_r2(a3,p3):+.4f} R2_1h={_r2(a12,p12):+.4f} | "
                f"r_15m={_pearson(a3,p3):+.4f} r_1h={_pearson(a12,p12):+.4f} | "
                f"MAE_15m={_mae(a3,p3):.6f} MAE_1h={_mae(a12,p12):.6f} | "
                f"bias_15m={_bias(a3,p3):+.6f} bias_1h={_bias(a12,p12):+.6f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
