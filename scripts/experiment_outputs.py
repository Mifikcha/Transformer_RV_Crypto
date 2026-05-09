"""
Duplicate experiment / figure-data CSV writes into:
  - scripts/output/
  - ~deep_lom/raw/Преддипломная/Графики/data/

Графики по CSV (generate_all_figures.py) читают и scripts/output/, и scripts/output/graphs/.

Primary writers keep saving under transformer/output/experiments/ (etc.);
mirror uses the same basename (flat layout in both mirrors).
"""

from __future__ import annotations

import shutil
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def scripts_output_dir() -> Path:
    return project_root() / "scripts" / "output"


def thesis_graphs_data_dir() -> Path:
    return (
        project_root()
        / "~deep_lom"
        / "raw"
        / "Преддипломная"
        / "Графики"
        / "data"
    )


def mirror_saved_csv(primary_path: str | Path) -> None:
    """Copy an existing CSV into scripts/output and thesis Графики/data."""
    src = Path(primary_path).resolve()
    if not src.is_file():
        return
    if src.suffix.lower() != ".csv":
        return
    name = src.name
    for dest_dir in (scripts_output_dir(), thesis_graphs_data_dir()):
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / name)
