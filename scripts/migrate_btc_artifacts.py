"""
One-shot migration: relocate existing BTC-only artifacts into the new
symbol-aware ``btcusdt/`` subdirectories.

Why
---
After the multi-symbol refactor every consumer of training artifacts
(transformer, baselines, feature_selection, view bot) expects them under
``<root>/<symbol_lower>/``. Repos that already trained on BTCUSDT have
their artifacts at the old top-level paths -- this script moves them so
the new defaults pick them up without retraining.

What it migrates
----------------
- ``feature_selection/output/*.csv|*.log`` and ``output/eda/`` ->
  ``feature_selection/output/btcusdt/``
- ``transformer/output/models/fold_rv_*.pt`` ->
  ``transformer/output/models/btcusdt/``
- ``transformer/output/predictions/*`` ->
  ``transformer/output/predictions/btcusdt/``
- ``transformer/output/pictures/*.png`` ->
  ``transformer/output/pictures/btcusdt/``
- ``model/fold_rv_*.pt`` (project-level "last run" copy) ->
  ``model/btcusdt/``
- ``log_tranformer`` -> ``log_tranformer_btcusdt``

Usage
-----
::

  python scripts/migrate_btc_artifacts.py --dry-run   # preview only
  python scripts/migrate_btc_artifacts.py             # actually move
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYMBOL_DIR = "btcusdt"


def _move(src: Path, dst: Path, *, dry_run: bool) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[SKIP] target exists: {dst}")
        return False
    print(f"[MOVE] {src} -> {dst}")
    if not dry_run:
        shutil.move(str(src), str(dst))
    return True


def _migrate_dir_contents(src_dir: Path, dst_dir: Path, *, dry_run: bool,
                          skip_names: set[str] | None = None) -> int:
    """Move every immediate child of ``src_dir`` into ``dst_dir``.

    Children whose name matches ``skip_names`` are left in place
    (used so we don't recursively move ``btcusdt/`` into itself).
    """
    if not src_dir.is_dir():
        return 0
    skip_names = skip_names or set()
    moved = 0
    for child in list(src_dir.iterdir()):
        if child.name in skip_names:
            continue
        target = dst_dir / child.name
        if _move(child, target, dry_run=dry_run):
            moved += 1
    return moved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print actions without moving anything.")
    args = ap.parse_args()

    dry_run = args.dry_run
    print(f"[INFO] project_root = {PROJECT_ROOT}")
    print(f"[INFO] dry_run      = {dry_run}")

    total = 0

    # 1) feature_selection/output/* -> feature_selection/output/btcusdt/
    fs_root = PROJECT_ROOT / "feature_selection" / "output"
    fs_dst = fs_root / SYMBOL_DIR
    total += _migrate_dir_contents(fs_root, fs_dst, dry_run=dry_run, skip_names={SYMBOL_DIR})

    # 2) transformer/output/{models,predictions,pictures,logs}/* -> .../btcusdt/
    for sub in ("models", "predictions", "pictures", "logs"):
        src = PROJECT_ROOT / "transformer" / "output" / sub
        dst = src / SYMBOL_DIR
        total += _migrate_dir_contents(src, dst, dry_run=dry_run, skip_names={SYMBOL_DIR})

    # 3) project-level model/ -> model/btcusdt/
    model_root = PROJECT_ROOT / "model"
    model_dst = model_root / SYMBOL_DIR
    total += _migrate_dir_contents(model_root, model_dst, dry_run=dry_run, skip_names={SYMBOL_DIR})

    # 4) top-level log_tranformer -> log_tranformer_btcusdt
    legacy_log = PROJECT_ROOT / "log_tranformer"
    new_log = PROJECT_ROOT / "log_tranformer_btcusdt"
    if _move(legacy_log, new_log, dry_run=dry_run):
        total += 1

    print(f"[DONE] migrated entries: {total}{' (dry-run)' if dry_run else ''}")
    if total == 0:
        print("Nothing to migrate. The repository already follows the symbol-aware layout.")
        sys.exit(0)


if __name__ == "__main__":
    main()
