#!/usr/bin/env python3
"""
Генерация иллюстраций для глав 2–3 диплома.

Запуск (из этого каталога или по абсолютному пути к скрипту):
  python generate_all_figures.py

Графики 4–11: CSV ищутся в ``<repo>/scripts/output/`` и ``<repo>/scripts/output/graphs/`` (зеркало экспериментов);
иначе — fallback на табличные константы из Глава_3_Экспериментальная_часть_v2.md.

Ожидаемые файлы (после прогонов ``run_architecture_comparison``, абляций, ``train-rv``):
  architecture_comparison.csv, architecture_comparison_folds.csv,
  ablation_seq_len.csv, ablation_loss.csv, ablation_features.csv,
  predictions_walkforward_transformer_rv.csv;
  опционально multi_window_experiment.csv (fallback для кривой seq_len).

Блок spike (рис. 12): spike_warning/output/test_predictions.csv или зеркало в scripts/output[graphs]/test_predictions.csv.
METRICS/BUNDLE ищутся в spike_warning/output/.

Переопределение каталога с CSV: переменная окружения FIGURES_SCRIPTS_OUTPUT (абсолютный путь).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent


def _resolve_repo_root(start: Path) -> Path:
    """Каталог репозитория: где есть ``scripts/experiment_outputs.py``."""
    for d in (start, *start.parents):
        if (d / "scripts" / "experiment_outputs.py").is_file():
            return d
    raise RuntimeError(f"Не найден корень репозитория (нет scripts/experiment_outputs.py), старт: {start}")


ROOT = _resolve_repo_root(HERE)
SPIKE_DIR = ROOT / "spike_warning" / "output"
SPIKE_METRICS = SPIKE_DIR / "classifier_metrics.json"
SPIKE_BUNDLE = SPIKE_DIR / "spike_classifier_bundle.joblib"


def _csv_search_dirs(repo_root: Path) -> list[Path]:
    env = os.environ.get("FIGURES_SCRIPTS_OUTPUT", "").strip()
    if env:
        return [Path(env).expanduser().resolve()]
    flat = (repo_root / "scripts" / "output").resolve()
    return [flat, flat / "graphs"]


def _find_csv(name: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        p = (d / name).resolve()
        if p.is_file():
            return p
    return None


def _read_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Не удалось прочитать {path}: {exc}")
        return None


@dataclass
class ScriptsOutputBundle:
    repo_root: Path
    scripts_out: Path
    csv_search_dirs: tuple[Path, ...] = ()
    arch: pd.DataFrame | None = None
    folds: pd.DataFrame | None = None
    ablation_seq_len: pd.DataFrame | None = None
    ablation_loss: pd.DataFrame | None = None
    ablation_features: pd.DataFrame | None = None
    multi_window: pd.DataFrame | None = None
    pred_rv: pd.DataFrame | None = None
    spike_predictions: Path | None = None


def load_scripts_output_bundle(repo_root: Path) -> ScriptsOutputBundle:
    dirs = _csv_search_dirs(repo_root)
    so = dirs[0]
    bundle = ScriptsOutputBundle(
        repo_root=repo_root,
        scripts_out=so,
        csv_search_dirs=tuple(dirs),
        arch=_read_csv(_find_csv("architecture_comparison.csv", dirs)),
        folds=_read_csv(_find_csv("architecture_comparison_folds.csv", dirs)),
        ablation_seq_len=_read_csv(_find_csv("ablation_seq_len.csv", dirs)),
        ablation_loss=_read_csv(_find_csv("ablation_loss.csv", dirs)),
        ablation_features=_read_csv(_find_csv("ablation_features.csv", dirs)),
        multi_window=_read_csv(_find_csv("multi_window_experiment.csv", dirs)),
        pred_rv=_read_csv(_find_csv("predictions_walkforward_transformer_rv.csv", dirs)),
    )
    sp = SPIKE_DIR / "test_predictions.csv"
    if sp.is_file():
        bundle.spike_predictions = sp
    else:
        alt = _find_csv("test_predictions.csv", dirs)
        if alt is not None:
            bundle.spike_predictions = alt
    return bundle


# Табл. 3.x — короткие имена горизонтов ↔ столбцы таргетов в CSV
HORIZON_TARGETS: tuple[tuple[str, str], ...] = (
    ("15m (3-bar)", "rv_3bar_fwd"),
    ("1h (12-bar)", "rv_12bar_fwd"),
    ("4h (48-bar)", "rv_48bar_fwd"),
    ("24h (288-bar)", "rv_288bar_fwd"),
)

# Подписи моделей (как в тексте) → значение model_type в architecture_comparison.csv
LABEL_TO_MODEL_TYPE: dict[str, str] = {
    "Historical Mean": "baseline:historical_mean",
    "HAR-RV-J": "baseline:har_rv_j",
    "HAR-RV": "baseline:har_rv",
    "Linear Ridge": "baseline:linear_ridge",
    "Persistence": "baseline:persistence",
    "LightGBM": "baseline:lightgbm",
    "LSTM": "baseline:lstm",
    "Patch Transformer": "patch_encoder",
}

FIG8_ORDER: tuple[str, ...] = tuple(LABEL_TO_MODEL_TYPE.keys())


def _transformer_arch_rows(arch: pd.DataFrame | None) -> pd.DataFrame | None:
    if arch is None or arch.empty or "model_type" not in arch.columns:
        return None
    ts = arch[~arch["model_type"].astype(str).str.startswith("baseline:")].copy()
    if ts.empty or "qlike_mean" not in ts.columns:
        return None
    ts = ts[pd.to_numeric(ts["qlike_mean"], errors="coerce").notna()]
    return ts if len(ts) else None


def _baseline_metric(arch: pd.DataFrame | None, model_type: str, key: str) -> float | None:
    if arch is None or arch.empty or key not in arch.columns:
        return None
    row = arch.loc[arch["model_type"].astype(str) == model_type]
    if row.empty:
        return None
    v = pd.to_numeric(row.iloc[0][key], errors="coerce")
    return float(v) if pd.notna(v) else None


def _seq_len_curve_df(bundle: ScriptsOutputBundle) -> tuple[pd.DataFrame | None, str]:
    if bundle.ablation_seq_len is not None and not bundle.ablation_seq_len.empty:
        df = bundle.ablation_seq_len.copy()
        if "variant_name" in df.columns and "qlike_mean" in df.columns:
            extr = df["variant_name"].astype(str).str.extract(r"seq_(\d+)", expand=False)
            if extr.notna().any():
                df["_seq_x"] = pd.to_numeric(extr, errors="coerce")
                df = df.dropna(subset=["_seq_x"])
                return df.sort_values("_seq_x"), "_seq_x"
    if bundle.multi_window is not None and not bundle.multi_window.empty:
        if "seq_len" in bundle.multi_window.columns and "qlike_mean" in bundle.multi_window.columns:
            return bundle.multi_window.sort_values("seq_len"), "seq_len"
    return None, ""


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.35,
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    path = HERE / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Fig 1–2: schematic diagrams (matplotlib, PNG — без PlantUML)
# -----------------------------------------------------------------------------
def fig01_ml_pipeline_png() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.2)
    ax.axis("off")

    boxes = [
        (0.1, 1.0, "Bybit API\n5m OHLCV"),
        (1.35, 1.0, "Очистка"),
        (2.35, 1.0, "Признаки"),
        (3.35, 1.0, "Таргеты RV"),
        (4.35, 1.0, "Walk-forward\nembargo"),
        (5.55, 1.0, "Patch\nTransformer"),
        (6.85, 1.0, "Spike\nclassifier"),
        (8.05, 1.0, "Inference"),
        (9.25, 1.0, "PostgreSQL"),
        (10.55, 1.0, "Бот /\nDashboard"),
    ]
    colors = ["#cfe8ff"] + ["#ffffff"] * 6 + ["#e8e8e8", "#d5f5ff"]

    for (x, y, txt), c in zip(boxes, colors):
        fb = FancyBboxPatch(
            (x, y), 0.95, 0.85, boxstyle="round,pad=0.05",
            linewidth=1.2, facecolor=c, edgecolor="#333",
        )
        ax.add_patch(fb)
        ax.text(x + 0.48, y + 0.42, txt, ha="center", va="center", fontsize=8.5)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.95
        x2 = boxes[i + 1][0]
        ax.annotate(
            "",
            xy=(x2, 1.42),
            xytext=(x1, 1.42),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.4),
        )

    ax.text(7.0, 2.05, "ML-пайплайн: от сырых данных до production", ha="center", fontsize=12, weight="bold")
    ax.text(
        7.0, 0.35,
        "Примечание: также доступны исходники PlantUML: 01_ml_pipeline.puml",
        ha="center",
        fontsize=8,
        color="#555",
    )
    _save(fig, "01_ml_pipeline.png")


def fig02_production_png() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    row1 = [
        (0.2, 1.55, "Bybit API", "#cfe8ff"),
        (1.7, 1.55, "Ingestion\nWorker", "#fff"),
        (3.2, 1.55, "PostgreSQL\nbars_5m", "#ddd"),
        (5.0, 1.55, "Predictor\n(FeatureEngine +\nRVInference)", "#fff"),
        (7.0, 1.55, "PostgreSQL\npredictions /\nrv_actual", "#ddd"),
        (9.2, 1.55, "Notifier", "#fff"),
        (10.6, 1.55, "Telegram /\nDashboard", "#d5f5ff"),
    ]

    for x, y, txt, c in row1:
        fb = FancyBboxPatch(
            (x, y), 1.25, 1.05, boxstyle="round,pad=0.06",
            linewidth=1.1, facecolor=c, edgecolor="#222",
        )
        ax.add_patch(fb)
        ax.text(x + 0.62, y + 0.52, txt, ha="center", va="center", fontsize=8)

    arrows = [(1.45, 2.05, 3.15), (4.45, 2.05, 4.95), (6.25, 2.05, 6.95), (8.25, 2.05, 9.15), (10.45, 2.05, 10.55)]
    for x1, ym, x2 in arrows:
        ax.annotate("", xy=(x2, ym), xytext=(x1, ym), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))

    # Spike overlay
    fb_sw = FancyBboxPatch(
        (5.3, 0.25), 4.4, 0.65, boxstyle="round,pad=0.04",
        linewidth=1.5, facecolor="#ffe0ef", edgecolor="#c2185b", linestyle="--",
    )
    ax.add_patch(fb_sw)
    ax.text(7.5, 0.57, "Spike Warning (integrate.py) — слой над predictions", ha="center", va="center", fontsize=9)

    ax.plot([7.5, 7.5], [1.5, 0.9], color="#c2185b", lw=1.2, linestyle=":")

    ax.text(6.0, 2.85, "Production-архитектура (обмен через PostgreSQL)", fontsize=12, weight="bold")
    ax.text(6.0, -0.15, "Исходник PlantUML: 02_production_architecture.puml", fontsize=8, color="#555")
    _save(fig, "02_production_architecture.png")


# -----------------------------------------------------------------------------
# Thesis tables (fallback если нет CSV)
# -----------------------------------------------------------------------------
ARCH_QLIKE = {
    "patch_encoder": -4.228,
    "decoder_only": -4.228,
    "patch_decoder": -4.226,
    "vanilla_enc_dec": -4.223,
}
SEQ_LEN = {48: -4.223, 120: -4.225, 240: -4.228, 480: -4.226, 576: -4.225}
ALPHA_ROWS = [
    (0.0, -4.228, 4.4e-5),
    (0.3, -4.224, 8.2e-5),
    (0.5, -4.221, 1.15e-4),
    (0.7, -4.218, 1.35e-4),
    (1.0, -4.212, 2.01e-4),
]
FEAT_DELTA = {
    "price_sma": -0.002,
    "volume_std": -0.001,
    "derivatives": 0.001,
    "time": 0.004,
    "volatility_rv": 0.049,
}

MODELS_TAB39 = [
    ("Historical Mean", -0.466, -4.060),
    ("HAR-RV-J", 0.329, -4.145),
    ("HAR-RV", 0.380, -4.172),
    ("Linear Ridge", 0.429, -4.184),
    ("Persistence", 0.457, -4.183),
    ("LightGBM", 0.530, -4.213),
    ("LSTM", 0.576, -4.216),
    ("Patch Transformer", 0.516, -4.228),
]

R2_HORIZON = {
    "Historical Mean": [-0.435, -0.452, -0.461, -0.515],
    "HAR-RV-J": [0.427, 0.382, 0.301, 0.207],
    "HAR-RV": [0.502, 0.446, 0.346, 0.228],
    "Linear Ridge": [0.548, 0.480, 0.401, 0.288],
    "Persistence": [0.732, 0.618, 0.388, 0.091],
    "LightGBM": [0.790, 0.662, 0.554, 0.115],
    "LSTM": [0.704, 0.649, 0.578, 0.374],
    "Patch Transformer": [0.683, 0.545, 0.479, 0.357],
}

FOLD_R2_TRANS = [0.416, 0.425, 0.538, 0.643, 0.559]
LSTM_OFFSET = 0.576 - 0.516
LGB_OFFSET = 0.530 - 0.516


def fig04_arch_qlike(bundle: ScriptsOutputBundle) -> None:
    trans = _transformer_arch_rows(bundle.arch)
    if trans is not None and len(trans):
        names = trans["model_type"].astype(str).tolist()
        vals = pd.to_numeric(trans["qlike_mean"], errors="coerce").tolist()
        best_idx = int(np.nanargmin(vals)) if len(vals) else 0
        best_mt = names[best_idx]
        colors = ["#2e7d32" if n == best_mt else "#1565c0" for n in names]
        lstm_line = _baseline_metric(bundle.arch, "baseline:lstm", "qlike_mean")
        lstm_label = f"LSTM baseline QLIKE = {lstm_line:.3f}" if lstm_line is not None else "LSTM baseline"
        subtitle = "(данные: scripts/output/architecture_comparison.csv)"
    else:
        names = list(ARCH_QLIKE.keys())
        vals = [ARCH_QLIKE[n] for n in names]
        colors = ["#2e7d32" if n == "patch_encoder" else "#1565c0" for n in names]
        lstm_line = -4.216
        lstm_label = "LSTM baseline QLIKE = −4.216"
        subtitle = "(fallback: табличные константы)"

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(names, vals, color=colors, edgecolor="#222")
    if lstm_line is not None and np.isfinite(lstm_line):
        ax.axvline(lstm_line, color="#d32f2f", linestyle="--", lw=1.8, label=lstm_label)
    ax.set_xlabel("QLIKE (ниже = лучше)")
    ax.set_title(f"Табл. 3.1 — QLIKE по трансформерным архитектурам (BTCUSDT, WF)\n{subtitle}", fontsize=10)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    _save(fig, "04_qlike_architectures.png")


def fig05_seq_len(bundle: ScriptsOutputBundle) -> None:
    curve_df, xcol = _seq_len_curve_df(bundle)
    lstm_line = _baseline_metric(bundle.arch, "baseline:lstm", "qlike_mean")

    if curve_df is not None and len(curve_df) and xcol:
        xs = pd.to_numeric(curve_df[xcol], errors="coerce").astype(int).tolist()
        ys = pd.to_numeric(curve_df["qlike_mean"], errors="coerce").tolist()
        subtitle = "(данные: scripts/output/ablation_seq_len.csv или multi_window_experiment.csv)"
        if lstm_line is None or not np.isfinite(lstm_line):
            lstm_line = -4.216
    else:
        xs = sorted(SEQ_LEN.keys())
        ys = [SEQ_LEN[x] for x in xs]
        lstm_line = lstm_line if lstm_line is not None and np.isfinite(lstm_line) else -4.216
        subtitle = "(fallback: табличные константы)"

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(xs, ys, "o-", color="#1565c0", lw=2, markersize=9)
    ax.axhline(float(lstm_line), color="#d32f2f", linestyle="--", lw=1.6, label="LSTM baseline")
    ax.axvline(240, color="#757575", linestyle=":", lw=1.2)
    if 240 in xs:
        y240 = ys[xs.index(240)]
        ax.annotate(
            "seq_len=240",
            xy=(240, y240),
            xytext=(max(xs) * 0.55, min(ys) + 0.25 * (max(ys) - min(ys))),
            arrowprops=dict(arrowstyle="->", color="#555"),
        )
    ax.set_xticks(xs)
    ax.set_xlabel("seq_len (число 5m-баров)")
    ax.set_ylabel("QLIKE")
    ax.set_title(f"Табл. 3.3 — QLIKE vs длина контекста\n{subtitle}", fontsize=10)
    ax.legend()
    _save(fig, "05_qlike_vs_seq_len.png")


def fig06_alpha_dual(bundle: ScriptsOutputBundle) -> None:
    subtitle = "(fallback: табличные константы)"
    alphas = [r[0] for r in ALPHA_ROWS]
    qlikes = [r[1] for r in ALPHA_ROWS]
    biases = [abs(r[2]) for r in ALPHA_ROWS]

    df = bundle.ablation_loss
    if df is not None and not df.empty and "loss_type" in df.columns:
        sub = df.loc[df["loss_type"].astype(str) == "rv_log_aware"].copy()
        if len(sub) >= 2 and "loss_alpha" in sub.columns and "qlike_mean" in sub.columns:
            sub = sub.sort_values("loss_alpha")
            alphas = pd.to_numeric(sub["loss_alpha"], errors="coerce").tolist()
            qlikes = pd.to_numeric(sub["qlike_mean"], errors="coerce").tolist()
            bias_col = "bias_rv_3bar_fwd" if "bias_rv_3bar_fwd" in sub.columns else "bias_mean"
            biases = pd.to_numeric(sub[bias_col], errors="coerce").abs().tolist()
            subtitle = "(данные: scripts/output/ablation_loss.csv)"

    fig, ax1 = plt.subplots(figsize=(7.5, 4.4))
    ax2 = ax1.twinx()
    ax1.plot(alphas, qlikes, "o-", color="#1565c0", lw=2, label="QLIKE")
    ax2.plot(alphas, biases, "s-", color="#ef6c00", lw=2, label="|bias| (15m)")
    ax1.set_xlabel(r"$\alpha$ (доля Huber)")
    ax1.set_ylabel("QLIKE", color="#1565c0")
    ax2.set_ylabel("|bias| RV (15m)", color="#ef6c00")
    ax1.set_title(f"Табл. 3.6 — Компромисс QLIKE vs |bias| при смешанном лоссе\n{subtitle}", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#ef6c00")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    _save(fig, "06_alpha_qlike_bias.png")


def fig07_feat_groups(bundle: ScriptsOutputBundle) -> None:
    subtitle = "(fallback: табличные константы)"
    items = sorted(FEAT_DELTA.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#c62828" if v > 0 else "#1565c0" for v in vals]

    df = bundle.ablation_features
    if df is not None and not df.empty and "variant_name" in df.columns and "qlike_mean" in df.columns:
        base_rows = df.loc[df["variant_name"].astype(str) == "none"]
        if not base_rows.empty:
            base_q = float(pd.to_numeric(base_rows.iloc[0]["qlike_mean"], errors="coerce"))
            deltas: list[tuple[str, float]] = []
            for _, row in df.iterrows():
                vn = str(row["variant_name"])
                if vn == "none" or not vn.startswith("drop_"):
                    continue
                g = vn.removeprefix("drop_")
                if g == "volume":
                    g = "volume_std"
                dq = float(pd.to_numeric(row["qlike_mean"], errors="coerce")) - base_q
                deltas.append((g, dq))
            if deltas:
                items = sorted(deltas, key=lambda kv: kv[1])
                labels = [k for k, _ in items]
                vals = [v for _, v in items]
                colors = ["#c62828" if v > 0 else "#1565c0" for v in vals]
                subtitle = "(данные: scripts/output/ablation_features.csv)"

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.barh(labels, vals, color=colors, edgecolor="#222")
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("ΔQLIKE (положительное — ухудшение при удалении группы)")
    ax.set_title(f"Табл. 3.7 — Ablation групп признаков (BTCUSDT, K=3)\n{subtitle}", fontsize=10)
    ax.invert_yaxis()

    red = mpatches.Patch(color="#c62828", label="ухудшение (+ΔQLIKE)")
    blue = mpatches.Patch(color="#1565c0", label="улучшение (−ΔQLIKE)")
    ax.legend(handles=[red, blue], loc="lower right")
    _save(fig, "07_delta_qlike_feature_groups.png")


def fig08_models_dual_bar(bundle: ScriptsOutputBundle) -> None:
    subtitle = "(fallback: табличные константы)"
    names = [m[0] for m in MODELS_TAB39]
    r2 = [m[1] for m in MODELS_TAB39]
    ql = [m[2] for m in MODELS_TAB39]

    arch = bundle.arch
    if arch is not None and not arch.empty:
        r2_c = []
        ql_c = []
        ok = True
        for label in FIG8_ORDER:
            mt = LABEL_TO_MODEL_TYPE[label]
            hit = arch.loc[arch["model_type"].astype(str) == mt]
            if hit.empty or "r2_mean" not in hit.columns or "qlike_mean" not in hit.columns:
                ok = False
                break
            r2_c.append(float(pd.to_numeric(hit.iloc[0]["r2_mean"], errors="coerce")))
            ql_c.append(float(pd.to_numeric(hit.iloc[0]["qlike_mean"], errors="coerce")))
        if ok:
            names = list(FIG8_ORDER)
            r2, ql = r2_c, ql_c
            subtitle = "(данные: scripts/output/architecture_comparison.csv)"

    colors = ["#6a1b9a" if "Transformer" in n else "#78909c" for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5))
    x = np.arange(len(names))
    ax1.bar(x, r2, color=colors, edgecolor="#222")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right")
    ax1.set_ylabel(r"$R^2_{\mathrm{mean}}$")
    ax1.set_title("Табл. 3.9 — среднее $R^2$ по горизонтам")

    ax2.bar(x, ql, color=colors, edgecolor="#222")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right")
    ax2.set_ylabel("QLIKE")
    ax2.set_title("Табл. 3.9 — QLIKE (меньше = лучше)")
    fig.suptitle(f"Сравнение моделей на BTCUSDT (walk-forward)\n{subtitle}", fontsize=11, weight="bold")
    fig.tight_layout()
    _save(fig, "08_models_r2_qlike.png")


def fig09_r2_horizons(bundle: ScriptsOutputBundle) -> None:
    labs = ["15m", "1h", "4h", "24h"]
    x = np.arange(len(labs))
    highlight = {"Patch Transformer": "#6a1b9a", "LSTM": "#1565c0", "LightGBM": "#ef6c00"}
    subtitle = "(fallback: табличные константы)"
    series_src: dict[str, list[float]] = {k: list(v) for k, v in R2_HORIZON.items()}

    arch = bundle.arch
    r2_cols = [f"r2_{t}" for _, t in HORIZON_TARGETS]
    if arch is not None and not arch.empty and all(c in arch.columns for c in r2_cols):
        built: dict[str, list[float]] = {}
        for label, mt in LABEL_TO_MODEL_TYPE.items():
            hit = arch.loc[arch["model_type"].astype(str) == mt]
            if hit.empty:
                continue
            row = hit.iloc[0]
            built[label] = [float(pd.to_numeric(row[c], errors="coerce")) for c in r2_cols]
        # Полный прогон сравнения даёт 8 моделей; при явном неполном наборе остаёмся на fallback.
        if len(built) >= 6:
            series_src = built
            subtitle = "(данные: scripts/output/architecture_comparison.csv)"

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for name, series in series_src.items():
        style = dict(lw=2.2, marker="o", markersize=5)
        if name in highlight:
            ax.plot(x, series, label=name, color=highlight[name], **style)
        else:
            ax.plot(x, series, alpha=0.35, lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel(r"$R^2$")
    ax.set_title(f"Табл. 3.10 — $R^2$ по горизонтам (BTCUSDT)\n{subtitle}", fontsize=10)
    ax.legend(loc="best")
    if "LightGBM" in series_src and len(series_src["LightGBM"]) >= 4:
        v = series_src["LightGBM"][3]
        if np.isfinite(v) and v < 0.25:
            ax.annotate(
                "LightGBM: падение на 24h",
                xy=(3, v), xytext=(1.6, min(0.05, v - 0.02)),
                arrowprops=dict(arrowstyle="->", color="#555"),
                fontsize=9,
            )
    _save(fig, "09_r2_by_horizon.png")


def _simulate_log_scatter(rho: float, bias_nat: float, mae_nat: float, n: int = 1200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Упрощённая выборка (log-масштаб графика): corr≈rho, bias≈bias_nat, MAE≈mae_nat."""
    rng = np.random.default_rng(seed)
    la = rng.normal(-4.0, 0.42, size=n)
    eps_t = rng.normal(0, 1, size=n)
    lp = rho * la + math.sqrt(max(1e-8, 1 - rho * rho)) * eps_t

    scale_mae = mae_nat / np.mean(np.abs(np.exp(lp) - np.exp(la)))
    lp = la + (lp - la) * scale_mae + bias_nat * 0.85
    actual = np.exp(la)
    pred = np.exp(lp)
    pred *= np.exp(bias_nat - np.mean(pred - actual))
    return actual.clip(1e-9), pred.clip(1e-9)


def fig10_pred_vs_actual(bundle: ScriptsOutputBundle) -> None:
    df = bundle.pred_rv
    cols_ok = df is not None and not df.empty
    if cols_ok:
        for _, col in HORIZON_TARGETS:
            if f"actual_{col}" not in df.columns or f"pred_{col}" not in df.columns:
                cols_ok = False
                break

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    subtitle = (
        "Predicted vs actual RV (лог-масштаб; данные: scripts/output/predictions_walkforward_transformer_rv.csv)"
        if cols_ok
        else "Predicted vs actual RV (лог-масштаб; модельные облака по corr/bias/MAE из табл. 3.11 — fallback)"
    )

    if cols_ok:
        rng = np.random.default_rng(42)
        max_pts = 9000
        assert df is not None
        for ax, (title, col) in zip(axes.ravel(), HORIZON_TARGETS):
            act = df[f"actual_{col}"].to_numpy(dtype=float)
            pr = df[f"pred_{col}"].to_numpy(dtype=float)
            m = np.isfinite(act) & np.isfinite(pr) & (act > 0) & (pr > 0)
            act, pr = act[m], pr[m]
            if act.size > max_pts:
                pick = rng.choice(act.size, size=max_pts, replace=False)
                act, pr = act[pick], pr[pick]
            q = pd.Series(act).quantile([0.33, 0.66]).values
            cats = np.digitize(act, q)

            la = np.log(act)
            lp = np.log(pr)
            scatter = ax.scatter(la, lp, c=cats, cmap="viridis", alpha=0.35, s=12, edgecolors="none")
            mn = float(min(la.min(), lp.min()))
            mx = float(max(la.max(), lp.max()))
            ax.plot([mn, mx], [mn, mx], "k--", lw=1.4, label="y = x")
            coef = np.polyfit(la, lp, 1)
            xs = np.linspace(mn, mx, 50)
            ax.plot(xs, np.poly1d(coef)(xs), color="#d32f2f", lw=2, label="OLS fit")
            ax.set_xlabel(r"$\ln(\mathrm{RV}_{\mathrm{actual}})$")
            ax.set_ylabel(r"$\ln(\mathrm{RV}_{\mathrm{pred}})$")
            ax.set_title(f"Walk-forward OOS — {title}")
            ax.legend(loc="upper left", fontsize=8)
            plt.colorbar(scatter, ax=ax, label="квантиль факта", shrink=0.65)
    else:
        stats_h = [
            ("15m (3-bar)", 0.833, -4.4e-5, 4.57e-4),
            ("1h (12-bar)", 0.749, -1.09e-4, 1.104e-3),
            ("4h (48-bar)", 0.705, -8.5e-5, 2.38e-3),
            ("24h (288-bar)", 0.642, 2.01e-4, 5.846e-3),
        ]
        for ax, (title, rho, bias, mae) in zip(axes.ravel(), stats_h):
            act, pr = _simulate_log_scatter(rho, bias, mae, seed=hash(title) % 2**32)
            q = pd.Series(act).quantile([0.33, 0.66]).values
            cats = np.digitize(act, q)

            la = np.log(act)
            lp = np.log(pr)
            scatter = ax.scatter(la, lp, c=cats, cmap="viridis", alpha=0.35, s=12, edgecolors="none")
            mn = min(la.min(), lp.min())
            mx = max(la.max(), lp.max())
            ax.plot([mn, mx], [mn, mx], "k--", lw=1.4, label="y = x")
            coef = np.polyfit(la, lp, 1)
            xs = np.linspace(mn, mx, 50)
            ax.plot(xs, np.poly1d(coef)(xs), color="#d32f2f", lw=2, label="OLS fit")
            ax.set_xlabel(r"$\ln(\mathrm{RV}_{\mathrm{actual}})$")
            ax.set_ylabel(r"$\ln(\mathrm{RV}_{\mathrm{pred}})$")
            ax.set_title(f"Табл. 3.11 — {title}")
            ax.legend(loc="upper left", fontsize=8)
            plt.colorbar(scatter, ax=ax, label="квантиль факта", shrink=0.65)

    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    _save(fig, "10_pred_vs_actual_log_four_panel.png")


def _fold_r2_series(folds: pd.DataFrame, model_type: str) -> np.ndarray | None:
    sub = folds.loc[folds["model_type"].astype(str) == model_type].copy()
    if sub.empty or "r2_mean" not in sub.columns:
        return None
    sub["r2_mean"] = pd.to_numeric(sub["r2_mean"], errors="coerce")
    sub = sub[np.isfinite(sub["r2_mean"])]
    if sub.empty:
        return None
    if "seed" in sub.columns and sub["seed"].notna().any():
        sub = sub.groupby("fold_id", as_index=False)["r2_mean"].mean()
    elif "fold_id" in sub.columns:
        sub = sub.sort_values("fold_id")
    else:
        sub = sub.sort_index()
    return sub["r2_mean"].to_numpy(dtype=float)


def fig11_boxplot_folds(bundle: ScriptsOutputBundle) -> None:
    folds = bundle.folds
    use_csv = folds is not None and not folds.empty
    trans = lstm = lgb = None
    if use_csv:
        trans = _fold_r2_series(folds, "patch_encoder")
        lstm = _fold_r2_series(folds, "baseline:lstm")
        lgb = _fold_r2_series(folds, "baseline:lightgbm")
        use_csv = trans is not None and lstm is not None and lgb is not None and len(trans) > 0

    if use_csv and trans is not None and lstm is not None and lgb is not None:
        data = [trans, lstm, lgb]
        subtitle = "(данные: scripts/output/architecture_comparison_folds.csv)"
    else:
        trans = np.array(FOLD_R2_TRANS, dtype=float)
        lstm = trans + LSTM_OFFSET
        lgb = trans + LGB_OFFSET
        data = [trans, lstm, lgb]
        subtitle = (
            "Межфолдовое распределение $R^2$: fallback по табл. 3.12;\n"
            "LSTM/LightGBM — синтетический параллельный сдвиг (+0.060 / +0.014 к среднему)"
        )
        if folds is None or folds.empty:
            print("[WARN] Нет architecture_comparison_folds.csv — рис. 11 из констант.")

    lens = [len(d) for d in data]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bp = ax.boxplot(data, tick_labels=["Transformer", "LSTM", "LightGBM"], patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#e1bee7", "#bbdefb", "#ffe0b2"]):
        patch.set_facecolor(c)
    xs_sc = np.concatenate([np.full(s, i + 1, dtype=float) for i, s in enumerate(lens)])
    ys_sc = np.concatenate(data)
    ax.scatter(xs_sc, ys_sc, alpha=0.35, color="#333", s=22)
    ax.set_ylabel(r"$R^2_{\mathrm{mean}}$ по фолдам")
    ax.set_title(f"Межфолдовое распределение $R^2$\n{subtitle}", fontsize=9)
    ax.grid(axis="y")
    _save(fig, "11_boxplot_r2_by_fold.png")


# -----------------------------------------------------------------------------
# Spike block (12)
# -----------------------------------------------------------------------------
def _draw_spike_pr_ax(ax, df: pd.DataFrame, y: pd.Series) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    for col, label, c in [
        ("proba_lr", "Logistic Regression", "#37474f"),
        ("proba_rf", "Random Forest", "#1565c0"),
        ("proba_lgb", "LightGBM", "#2e7d32"),
    ]:
        if col not in df.columns:
            continue
        pr, rc, _ = precision_recall_curve(y, df[col].values)
        ap = float(average_precision_score(y, df[col].values))
        ax.plot(rc, pr, lw=2, label=f"{label} (AP={ap:.3f})", color=c)
    ax.axhline(float(y.mean()), color="#9e9e9e", linestyle="--", lw=1, label=f"baseline rate={y.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR-кривые на тесте")
    ax.legend(loc="upper right", fontsize=8)


def fig12_spike_panel(bundle: ScriptsOutputBundle) -> None:
    pred_path = bundle.spike_predictions
    if pred_path is None or not pred_path.is_file():
        print("[WARN] Нет test_predictions.csv (spike_warning/output или scripts/output) — блок spike пропущен.")
        return

    df = pd.read_csv(pred_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    y = df["y_true"].astype(int)

    thr_lgb = 0.5
    if SPIKE_METRICS.is_file():
        thr_lgb = float(json.loads(SPIKE_METRICS.read_text(encoding="utf-8")).get("best_threshold", 0.5))

    try:
        import joblib

        bundle = joblib.load(SPIKE_BUNDLE)
        model_name = bundle.get("model_name", "lgb")
        feat_cols = bundle.get("feature_columns", [])
        model = bundle["model"]
    except Exception as e:
        print("[WARN] bundle не загружен:", e)
        model_name = "lgb"
        feat_cols = []
        model = None

    prob = df["proba_lgb"].values if "proba_lgb" in df.columns else df["proba_rf"].values
    pred = (prob >= thr_lgb).astype(int)

    # --- 12a отдельно ---
    fig_a, ax_a = plt.subplots(figsize=(6.5, 5))
    _draw_spike_pr_ax(ax_a, df, y)
    fig_a.tight_layout()
    _save(fig_a, "12a_spike_pr_curves.png")

    # --- 12b ---
    fig_b, ax_cm = plt.subplots(figsize=(5.2, 4.8))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
    ax_cm.set_yticklabels(["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j, i, str(v), ha="center", va="center", color="black", fontsize=14)
    ax_cm.set_title(f"Матрица ошибок LightGBM (thr={thr_lgb:.3f})")
    plt.colorbar(im, ax=ax_cm, shrink=0.82)
    fig_b.tight_layout()
    _save(fig_b, "12b_spike_confusion_matrix.png")

    # --- 12c ---
    fig_c, ax_fi = plt.subplots(figsize=(7.2, 6))
    if model is not None and hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1][:24]
        ax_fi.barh(np.arange(len(order))[::-1], imp[order][::-1], color="#1565c0")
        ax_fi.set_yticks(np.arange(len(order)))
        ax_fi.set_yticklabels([feat_cols[i] if i < len(feat_cols) else str(i) for i in order][::-1], fontsize=7)
        ax_fi.set_title(f"Feature importance ({model_name})")
    else:
        ax_fi.text(0.5, 0.5, "Нет feature_importances_", ha="center")
        ax_fi.axis("off")
    fig_c.tight_layout()
    _save(fig_c, "12c_spike_feature_importance.png")

    # --- 12d ---
    fig_d, ax_tl = plt.subplots(figsize=(9.5, 4.5))
    close_col = "close_perp" if "close_perp" in df.columns else None
    if close_col:
        kind = []
        for i in range(len(df)):
            if int(y.iloc[i]) == 1 and pred[i] == 1:
                kind.append("TP")
            elif int(y.iloc[i]) == 0 and pred[i] == 1:
                kind.append("FP")
            elif int(y.iloc[i]) == 1 and pred[i] == 0:
                kind.append("FN")
            else:
                kind.append("TN")
        plot_df = df.copy()
        plot_df["_k"] = kind
        ax_tl.plot(plot_df["ts"], plot_df[close_col], color="#bdbdbd", lw=0.8, label="BTC close")

        def scatter_mask(mask, color, label, z):
            sub = plot_df.loc[mask]
            ax_tl.scatter(sub["ts"], sub[close_col], s=z, c=color, label=label, alpha=0.85, edgecolors="none")

        scatter_mask(plot_df["_k"] == "TP", "#2e7d32", "TP", 28)
        scatter_mask(plot_df["_k"] == "FP", "#fbc02d", "FP", 22)
        scatter_mask(plot_df["_k"] == "FN", "#c62828", "FN", 26)

        ax_tl.legend(loc="upper left", fontsize=8)
        ax_tl.set_title("Цена и TP / FP / FN (LightGBM, порог из classifier_metrics)")
        ax_tl.tick_params(axis="x", rotation=22)
    fig_d.tight_layout()
    _save(fig_d, "12d_spike_timeline_price_markers.png")

    # --- сводная 12 ---
    fig = plt.figure(figsize=(12.5, 10))
    ax_pr = fig.add_subplot(2, 2, 1)
    _draw_spike_pr_ax(ax_pr, df, y)
    ax_pr.set_title("(a) PR-кривые")

    ax_cm2 = fig.add_subplot(2, 2, 2)
    im2 = ax_cm2.imshow(cm, cmap="Blues")
    ax_cm2.set_xticks([0, 1])
    ax_cm2.set_yticks([0, 1])
    ax_cm2.set_xticklabels(["Pred 0", "Pred 1"])
    ax_cm2.set_yticklabels(["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        ax_cm2.text(j, i, str(v), ha="center", va="center", color="black", fontsize=14)
    ax_cm2.set_title(f"(b) Матрица ошибок (thr={thr_lgb:.3f})")
    plt.colorbar(im2, ax=ax_cm2, shrink=0.8)

    ax_fi2 = fig.add_subplot(2, 2, 3)
    if model is not None and hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1][:20]
        ax_fi2.barh(np.arange(len(order))[::-1], imp[order][::-1], color="#1565c0")
        ax_fi2.set_yticks(np.arange(len(order)))
        ax_fi2.set_yticklabels([feat_cols[i] if i < len(feat_cols) else str(i) for i in order][::-1], fontsize=7)
        ax_fi2.set_title(f"(c) Feature importance ({model_name})")
    else:
        ax_fi2.axis("off")

    ax_tl2 = fig.add_subplot(2, 2, 4)
    if close_col:
        ax_tl2.plot(plot_df["ts"], plot_df[close_col], color="#bdbdbd", lw=0.8)
        scatter_mask = lambda mask, color, label, z: ax_tl2.scatter(
            plot_df.loc[mask, "ts"],
            plot_df.loc[mask, close_col],
            s=z, c=color, label=label, alpha=0.85, edgecolors="none",
        )
        scatter_mask(plot_df["_k"] == "TP", "#2e7d32", "TP", 28)
        scatter_mask(plot_df["_k"] == "FP", "#fbc02d", "FP", 22)
        scatter_mask(plot_df["_k"] == "FN", "#c62828", "FN", 26)
        ax_tl2.legend(loc="upper left", fontsize=8)
        ax_tl2.set_title("(d) Цена и исходы классификации")
        ax_tl2.tick_params(axis="x", rotation=25)

    fig.suptitle("Spike-классификатор: сводная фигура (spike_warning/output/)", fontsize=12, weight="bold")
    fig.tight_layout()
    _save(fig, "12_spike_classifier_panel.png")


def main() -> None:
    bundle = load_scripts_output_bundle(ROOT)
    print("OUTPUT PNG:", HERE)
    print("CSV каталоги (поиск по порядку):", ", ".join(str(d) for d in bundle.csv_search_dirs))

    def _csv_ok(df: pd.DataFrame | None) -> str:
        return "ok" if df is not None and not df.empty else "—"

    spike_note = bundle.spike_predictions.name if bundle.spike_predictions else "—"
    print(
        "  architecture_comparison:", _csv_ok(bundle.arch),
        "| folds:", _csv_ok(bundle.folds),
        "| ablation_seq_len:", _csv_ok(bundle.ablation_seq_len),
        "| ablation_loss:", _csv_ok(bundle.ablation_loss),
        "| ablation_features:", _csv_ok(bundle.ablation_features),
        "| predictions_rv:", _csv_ok(bundle.pred_rv),
        "| spike test_predictions:", spike_note,
    )
    fig01_ml_pipeline_png()
    fig02_production_png()
    fig04_arch_qlike(bundle)
    fig05_seq_len(bundle)
    fig06_alpha_dual(bundle)
    fig07_feat_groups(bundle)
    fig08_models_dual_bar(bundle)
    fig09_r2_horizons(bundle)
    fig10_pred_vs_actual(bundle)
    fig11_boxplot_folds(bundle)
    fig12_spike_panel(bundle)
    print("Готово. PNG сохранены в", HERE)


if __name__ == "__main__":
    main()
