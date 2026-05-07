from __future__ import annotations

import sys
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.garch_models._common import (
    get_symbol,
    get_symbol_lower,
    now_tag,
    output_dir,
    python_exe,
    tee_print,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def _run_stage(cmd: list[str], *, f_log) -> None:
    tee_print(f_log, "")
    tee_print(f_log, "=" * 100)
    tee_print(f_log, f"[TS ] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    tee_print(f_log, f"[CMD] {' '.join(cmd)}")
    tee_print(f_log, "=" * 100)
    p = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR.parents[2]),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        tee_print(f_log, line.rstrip("\n"))
    rc = p.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="Run GARCH-family baselines (standalone).")
    p.add_argument("--symbol", type=str, default=None, help="Override SYMBOL (e.g. ETHUSDT).")
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    if args.symbol:
        os.environ["SYMBOL"] = args.symbol.strip().upper()
    symbol = get_symbol()
    symbol_lower = get_symbol_lower()

    out = output_dir(symbol_lower)
    log_path = out / f"run_garch_models_{symbol_lower}_{now_tag()}.txt"

    py = python_exe()
    stages = [
        ("prepare_daily_variance_target", [py, str(SCRIPT_DIR / "prepare_daily_variance_target.py")]),
        ("garch11", [py, str(SCRIPT_DIR / "train_garch_11.py"), "--n-splits", str(int(args.n_splits))]),
        ("egarch11", [py, str(SCRIPT_DIR / "train_egarch_11.py"), "--n-splits", str(int(args.n_splits))]),
        ("gjr_garch11", [py, str(SCRIPT_DIR / "train_gjr_garch_11.py"), "--n-splits", str(int(args.n_splits))]),
    ]

    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f_log:
        tee_print(f_log, f"[INFO] SYMBOL={symbol}")
        tee_print(f_log, f"[INFO] Output dir: {out}")
        tee_print(f_log, f"[INFO] n_splits={int(args.n_splits)}")

        for stage_name, cmd in stages:
            tee_print(f_log, f"\n[STAGE] {stage_name}")
            _run_stage(cmd, f_log=f_log)

        tee_print(f_log, f"\n[DONE] elapsed_sec={time.time() - started:.2f}")

    # Convenience: print the latest overall metrics if present.
    summary = {}
    for key, fname in [
        ("garch11", "metrics_garch11.json"),
        ("egarch11", "metrics_egarch11.json"),
        ("gjr_garch11", "metrics_gjr_garch11.json"),
    ]:
        pth = out / fname
        if pth.is_file():
            try:
                payload = json.loads(pth.read_text(encoding="utf-8"))
                summary[key] = payload.get("overall", {})
            except Exception:
                summary[key] = {}
    print(json.dumps({"symbol": symbol, "output_dir": str(out), "log": str(log_path), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

