"""Standalone CLI: fill ``predictions`` (and ``rv_actual``) for the gap between
the last recorded prediction and the latest bar.

Decoupled from the Telegram bot — runs the same inference pipeline (HAR
context, log-bias calibration, ``degraded`` flag) that ``run_prediction``
uses inside the live cycle, but in a single batch pass over historical bars.

Usage:
    python -m scripts.backfill_predictions
    python -m scripts.backfill_predictions --since 2026-04-17
    python -m scripts.backfill_predictions --since 2026-04-17T05:10:00 --no-actuals
    python -m scripts.backfill_predictions --no-bias            # skip log-bias calibration
    python -m scripts.backfill_predictions --commit-every 500   # larger DB transactions
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

from tg_bot.config import Settings
from tg_bot.db import build_engine, build_session_factory
from tg_bot.inference import RVInference
from tg_bot.prediction_worker import backfill_predictions

logger = logging.getLogger("scripts.backfill_predictions")


def _parse_iso(s: str) -> datetime:
    """Parse ISO timestamp; default to UTC if naive."""
    s = s.strip()
    # Allow "YYYY-MM-DD" by promoting to start-of-day.
    if len(s) == 10:
        s = f"{s}T00:00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    inference = RVInference(settings.model_paths_list, settings.features_path)

    engine = build_engine(settings.database_url)
    session_factory = build_session_factory(engine)

    start_ts = _parse_iso(args.since) if args.since else None
    end_ts = _parse_iso(args.until) if args.until else None

    last_logged = 0

    def _progress(done: int, total: int) -> None:
        nonlocal last_logged
        if total <= 0:
            return
        if done - last_logged >= args.log_every or done == total:
            pct = 100.0 * done / total
            logger.info("  progress: %d/%d (%.1f%%)", done, total, pct)
            last_logged = done

    try:
        async with session_factory() as session:
            stats = await backfill_predictions(
                session,
                inference,
                settings,
                start_ts=start_ts,
                end_ts=end_ts,
                update_bias=not args.no_bias,
                fill_actuals=not args.no_actuals,
                commit_every=args.commit_every,
                progress_cb=_progress,
            )
    finally:
        await engine.dispose()

    logger.info("Backfill complete:")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--since",
        help="Lower bound (exclusive) for backfill, ISO format (e.g. 2026-04-17 "
             "or 2026-04-17T05:10:00). Default: MAX(predictions.ts).",
    )
    parser.add_argument(
        "--until",
        help="Upper bound (inclusive) for backfill. Default: MAX(bars_5m.ts).",
    )
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="Skip log-bias calibration from existing (pred, actual) pairs.",
    )
    parser.add_argument(
        "--no-actuals",
        action="store_true",
        help="Skip backfill_actual_rv after predictions are filled.",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=200,
        help="Commit batch size (rows). Default: 200.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Progress log throttle (rows). Default: 50.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
