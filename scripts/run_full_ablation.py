"""
Unified ablation runner with filtering controls.

Examples:
  python scripts/run_full_ablation.py
  python scripts/run_full_ablation.py --quick
  python scripts/run_full_ablation.py --only B1
  python scripts/run_full_ablation.py --skip A1,C1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = Path(PROJECT_ROOT) / "logs"
FULL_ABLATION_LOG = LOGS_DIR / "log_full_ablation.txt"
PY = sys.executable


GROUP_ORDER = ("A1", "A2", "A3", "A4", "A5", "B1", "C1", "D1")

GROUP_COMMANDS: dict[str, list[str]] = {
    "A1": [PY, "scripts/run_architecture_comparison.py"],
    "A2": [PY, "scripts/run_ablation_har.py"],
    "A3": [PY, "scripts/run_ablation_seq_len.py"],
    "A4": [PY, "scripts/run_ablation_patch_size.py"],
    "A5": [PY, "scripts/run_ablation_capacity.py"],
    "B1": [PY, "scripts/run_ablation_loss.py"],
    "C1": [PY, "scripts/run_ablation_ensemble.py"],
    "D1": [PY, "scripts/run_ablation_features.py"],
}


def _progress(step: int, total: int, message: str) -> None:
    pct = int((step / max(total, 1)) * 100)
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{step}/{total}] ({pct:>3}%) {message}")


def _normalize_groups(csv_arg: str | None) -> set[str]:
    if not csv_arg:
        return set()
    groups = {x.strip().upper() for x in csv_arg.split(",") if x.strip()}
    invalid = sorted(g for g in groups if g not in GROUP_ORDER)
    if invalid:
        raise ValueError(f"Unknown groups: {invalid}. Allowed: {list(GROUP_ORDER)}")
    return groups


def _resolve_schedule(only: set[str], skip: set[str]) -> list[str]:
    groups = [g for g in GROUP_ORDER if (not only or g in only) and g not in skip]
    if not groups:
        raise ValueError("No groups left to run after applying --only/--skip.")
    return groups


def _command_file_exists(cmd: list[str]) -> bool:
    if len(cmd) < 2:
        return True
    path = cmd[1]
    if os.path.isabs(path):
        return os.path.exists(path)
    return os.path.exists(os.path.join(PROJECT_ROOT, path))


def _ensure_empty_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _group_log_path(group: str) -> Path:
    return LOGS_DIR / f"ablation_{group}.txt"


def _run(cmd: list[str], *, quick: bool, group: str) -> None:
    run_cmd = list(cmd)
    if quick and "--quick" not in run_cmd:
        run_cmd.append("--quick")

    stage_log = _group_log_path(group)
    _ensure_empty_log(stage_log)
    started = time.time()
    print(f"[RUN] {' '.join(run_cmd)} -> {stage_log.as_posix()}")

    with open(stage_log, "a", encoding="utf-8") as f_stage, open(FULL_ABLATION_LOG, "a", encoding="utf-8") as f_full:
        header = (
            f"\n{'='*90}\n"
            f"[GROUP] {group}\n"
            f"[CMD  ] {' '.join(run_cmd)}\n"
            f"[CWD  ] {PROJECT_ROOT}\n"
            f"[TS   ] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*90}\n\n"
        )
        f_stage.write(header)
        f_full.write(header)
        f_stage.flush()
        f_full.flush()

        p = subprocess.Popen(
            run_cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            f_stage.write(line)
            f_full.write(line)
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, run_cmd)
    print(f"[OK ] {' '.join(run_cmd)} (elapsed {time.time() - started:.1f}s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation groups in a unified order.")
    p.add_argument("--quick", action="store_true", help="Pass --quick to subgroup runners when supported.")
    p.add_argument("--only", type=str, default=None, help="Comma-separated subset of groups to run, e.g. B1 or A2,B1.")
    p.add_argument("--skip", type=str, default=None, help="Comma-separated groups to skip, e.g. A1,C1.")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip groups whose script files do not exist instead of failing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    only = _normalize_groups(args.only)
    skip = _normalize_groups(args.skip)
    schedule = _resolve_schedule(only=only, skip=skip)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_empty_log(FULL_ABLATION_LOG)
    _progress(1, len(schedule) + 1, f"Initialized runner | groups={schedule} | quick={args.quick}")

    for i, group in enumerate(schedule, start=2):
        cmd = GROUP_COMMANDS[group]
        if not _command_file_exists(cmd):
            msg = f"[MISS] {group}: script not found ({cmd[1]})"
            if args.allow_missing:
                _progress(i, len(schedule) + 1, msg + " -> skipped")
                continue
            raise FileNotFoundError(msg)
        _progress(i, len(schedule) + 1, f"Running group {group}")
        _run(cmd, quick=args.quick, group=group)

    _progress(len(schedule) + 1, len(schedule) + 1, f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
