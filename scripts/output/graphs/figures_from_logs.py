#!/usr/bin/env python3
"""
Числовые графики (04–11) из метрик, извлечённых из logs/*.txt.
Схемы 01–02 и блок spike (12) здесь не строятся — см. generate_all_figures.py.

Источники данных (ручной разбор + JSON из transformer_train_rv_BTCUSDT.txt):
  — ablation_A1.txt / log_full_ablation.txt: строки [ARCH DONE] (архитектуры)
  — ablation_A3.txt: сводная таблица seq_len (patch_decoder)
  — ablation_B1.txt: alpha vs QLIKE / bias (rv_3bar)
  — ablation_D1.txt: абляция групп признаков, K=3
  — log_full_ablation.txt: BASELINE DONE + per-horizon R² бейзлайнов
  — transformer_train_rv_BTCUSDT.txt: Patch Transformer R² по горизонтам, по фолдам, corr/bias для scatter
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _resolve_repo_root(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / "scripts" / "experiment_outputs.py").is_file():
            return d
    raise RuntimeError(f"Не найден корень репозитория от {start}")


ROOT = _resolve_repo_root(HERE)
LOGS = ROOT / "logs"

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


def load_transformer_fold_r2() -> list[float]:
    txt = (LOGS / "transformer_train_rv_BTCUSDT.txt").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"RV metrics per fold:\s*(\[.+?\])\s*\n", txt, re.DOTALL)
    if not m:
        raise RuntimeError("Не найден блок RV metrics per fold в transformer_train_rv_BTCUSDT.txt")
    rows = json.loads(m.group(1))
    return [float(r["r2_mean"]) for r in sorted(rows, key=lambda x: x["fold_id"])]


# --- Извлечено из logs/ablation_A1.txt строки [ARCH DONE] ---
ARCH_QLIKE_LOGS = {
    "patch_encoder": -4.227540,
    "decoder_only": -4.227027,
    "vanilla_enc_dec": -4.222398,
    "patch_decoder": -4.229543,
}

# --- ablation_A3.txt (patch_decoder, сводная строка qlike_mean) ---
SEQ_LEN_QLOG = {48: -4.226904, 120: -4.228371, 240: -4.229542, 480: -4.228738, 576: -4.228472}

# --- log_full_ablation.txt [BASELINE DONE] baseline:lstm ---
LSTM_QLIKE_BASELINE = -4.217527

# --- ablation_B1.txt (rv_log_aware; alpha 0.3 в логе отсутствует) ---
ALPHA_LOSS_ROWS = [
    (0.0, -4.227540, 4.4e-5),
    (0.5, -4.226437, 1.19e-4),
    (0.7, -4.225271, 1.98e-4),
    (0.9, -4.223640, 2.36e-4),
    (1.0, -4.223898, 1.88e-4),
]

# --- ablation_D1.txt: D1.0 qlike_mean = -4.210267; Δ = qlike(drop) - qlike(full) ---
QLIKE_FULL_D1 = -4.210267
FEAT_DELTA_LOGS = {
    "drop_derivatives": -4.213128 - QLIKE_FULL_D1,
    "drop_volume": -4.210425 - QLIKE_FULL_D1,
    "drop_price_sma": -4.209722 - QLIKE_FULL_D1,
    "drop_time": -4.197934 - QLIKE_FULL_D1,
    "drop_volatility_rv": -4.172297 - QLIKE_FULL_D1,
}

# --- log_full_ablation + transformer_train_rv (для трансформера) ---
MODELS_TAB_LOGS = [
    ("Historical Mean", -0.465611, -4.060313),
    ("HAR-RV-J", 0.329294, -4.144692),
    ("HAR-RV", 0.380470, -4.171783),
    ("Linear Ridge", 0.429374, -4.184415),
    ("Persistence", 0.457081, -4.182994),
    ("LightGBM", 0.529955, -4.212539),
    ("LSTM", 0.587713, -4.217527),
    ("Patch Transformer", 0.516007, -4.227540),
]

R2_HORIZON_LOGS = {
    "Historical Mean": [-0.4347, -0.4516, -0.4613, -0.5149],
    "HAR-RV-J": [0.4266, 0.3824, 0.3009, 0.2072],
    "HAR-RV": [0.5020, 0.4457, 0.3460, 0.2282],
    "Linear Ridge": [0.5483, 0.4803, 0.4007, 0.2882],
    "Persistence": [0.7316, 0.6182, 0.3879, 0.0906],
    "LightGBM": [0.7897, 0.6618, 0.5536, 0.1147],
    "LSTM": [0.7109, 0.6505, 0.5842, 0.4052],
    "Patch Transformer": [0.6834, 0.5449, 0.4790, 0.3568],
}


def fig04_arch_qlike() -> None:
    names = sorted(ARCH_QLIKE_LOGS.keys(), key=lambda k: ARCH_QLIKE_LOGS[k])
    vals = [ARCH_QLIKE_LOGS[n] for n in names]
    best_name = min(ARCH_QLIKE_LOGS, key=ARCH_QLIKE_LOGS.get)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#2e7d32" if n == best_name else "#1565c0" for n in names]
    ax.barh(names, vals, color=colors, edgecolor="#222")
    ax.axvline(LSTM_QLIKE_BASELINE, color="#d32f2f", linestyle="--", lw=1.8, label=f"LSTM baseline QLIKE = {LSTM_QLIKE_BASELINE:.3f}")
    ax.set_xlabel("QLIKE (ниже = лучше)")
    ax.set_title("QLIKE по архитектурам (logs: ablation_A1, [ARCH DONE])")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    _save(fig, "04_qlike_architectures.png")


def fig05_seq_len() -> None:
    xs = sorted(SEQ_LEN_QLOG.keys())
    ys = [SEQ_LEN_QLOG[x] for x in xs]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(xs, ys, "o-", color="#1565c0", lw=2, markersize=9)
    ax.axhline(LSTM_QLIKE_BASELINE, color="#d32f2f", linestyle="--", lw=1.6, label=f"LSTM baseline ({LSTM_QLIKE_BASELINE:.3f})")
    ax.axvline(240, color="#757575", linestyle=":", lw=1.2)
    ax.annotate("seq_len=240", xy=(240, SEQ_LEN_QLOG[240]), xytext=(320, SEQ_LEN_QLOG[240] + 0.004),
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xticks(xs)
    ax.set_xlabel("seq_len (число 5m-баров)")
    ax.set_ylabel("QLIKE (среднее по горизонтам)")
    ax.set_title("QLIKE vs длина контекста (logs: ablation_A3, patch_decoder)")
    ax.legend()
    _save(fig, "05_qlike_vs_seq_len.png")


def fig06_alpha_dual() -> None:
    alphas = [r[0] for r in ALPHA_LOSS_ROWS]
    qlikes = [r[1] for r in ALPHA_LOSS_ROWS]
    biases = [abs(r[2]) for r in ALPHA_LOSS_ROWS]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.4))
    ax2 = ax1.twinx()
    ax1.plot(alphas, qlikes, "o-", color="#1565c0", lw=2, label="QLIKE")
    ax2.plot(alphas, biases, "s-", color="#ef6c00", lw=2, label="|bias| (15m)")
    ax1.set_xlabel(r"$\alpha$ (доля Huber в rv_log_aware)")
    ax1.set_ylabel("QLIKE", color="#1565c0")
    ax2.set_ylabel("|bias| RV (15m)", color="#ef6c00")
    ax1.set_title("Компромисс QLIKE vs |bias| (logs: ablation_B1; α=0.3 в логе нет)")
    ax1.tick_params(axis="y", labelcolor="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#ef6c00")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    _save(fig, "06_alpha_qlike_bias.png")


def fig07_feat_groups() -> None:
    items = sorted(FEAT_DELTA_LOGS.items(), key=lambda kv: kv[1])
    labels = [k.replace("drop_", "") for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#c62828" if v > 0 else "#1565c0" for v in vals]

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.barh(labels, vals, color=colors, edgecolor="#222")
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("ΔQLIKE = QLIKE(ablated) − QLIKE(full); K=3 фолда")
    ax.set_title("Абляция групп признаков (logs: ablation_D1; full qlike={:.4f})".format(QLIKE_FULL_D1))
    ax.invert_yaxis()
    red = mpatches.Patch(color="#c62828", label="ухудшение (+Δ)")
    blue = mpatches.Patch(color="#1565c0", label="улучшение (−Δ)")
    ax.legend(handles=[red, blue], loc="lower right")
    _save(fig, "07_delta_qlike_feature_groups.png")


def fig08_models_dual_bar() -> None:
    names = [m[0] for m in MODELS_TAB_LOGS]
    r2 = [m[1] for m in MODELS_TAB_LOGS]
    ql = [m[2] for m in MODELS_TAB_LOGS]
    colors = ["#6a1b9a" if "Transformer" in n else "#78909c" for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5))
    x = np.arange(len(names))
    ax1.bar(x, r2, color=colors, edgecolor="#222")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right")
    ax1.set_ylabel(r"$R^2_{\mathrm{mean}}$")
    ax1.set_title(r"$R^2_{\mathrm{mean}}$ (logs: WF-сводки)")

    ax2.bar(x, ql, color=colors, edgecolor="#222")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right")
    ax2.set_ylabel("QLIKE")
    ax2.set_title("QLIKE (меньше = лучше)")
    fig.suptitle("Сравнение моделей BTCUSDT (logs: log_full_ablation + transformer_train_rv)", fontsize=12, weight="bold")
    fig.tight_layout()
    _save(fig, "08_models_r2_qlike.png")


def fig09_r2_horizons() -> None:
    labs = ["15m", "1h", "4h", "24h"]
    x = np.arange(len(labs))
    highlight = {"Patch Transformer": "#6a1b9a", "LSTM": "#1565c0", "LightGBM": "#ef6c00"}

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for name, series in R2_HORIZON_LOGS.items():
        style = dict(lw=2.2, marker="o", markersize=5)
        if name in highlight:
            ax.plot(x, series, label=name, color=highlight[name], **style)
        else:
            ax.plot(x, series, alpha=0.35, lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel(r"$R^2$")
    ax.set_title("$R^2$ по горизонтам (logs: log_full_ablation, трансформер — transformer_train_rv)")
    ax.legend(loc="best")
    ax.annotate(
        "LightGBM: резкое падение на 24h",
        xy=(3, R2_HORIZON_LOGS["LightGBM"][3]),
        xytext=(1.55, 0.08),
        arrowprops=dict(arrowstyle="->", color="#555"),
        fontsize=9,
    )
    _save(fig, "09_r2_by_horizon.png")


def _simulate_log_scatter(rho: float, bias_nat: float, mae_nat: float, n: int = 1200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
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


def fig10_pred_vs_actual() -> None:
    stats_h = [
        ("15m (3-bar)", 0.8330, -4.4e-5, 4.57e-4),
        ("1h (12-bar)", 0.7487, -1.09e-4, 1.104e-3),
        ("4h (48-bar)", 0.7051, -8.5e-5, 2.38e-3),
        ("24h (288-bar)", 0.6420, 2.01e-4, 5.846e-3),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
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
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        plt.colorbar(scatter, ax=ax, label="квантиль факта", shrink=0.65)

    fig.suptitle(
        "Predicted vs actual (logs: transformer_train_rv PREDICTION DIAGNOSTICS corr/bias/MAE; облака симулированы)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "10_pred_vs_actual_log_four_panel.png")


def fig11_boxplot_folds() -> None:
    trans = np.array(load_transformer_fold_r2(), dtype=float)
    # В сохранённых логах нет показателей r2_mean по фолдам для LSTM/LGB — строим синтетический «ящик»
    # из mean ± std из WF-сводки (log_full_ablation).
    lstm_m, lstm_s = 0.587713, 0.083306
    lgb_m, lgb_s = 0.529955, 0.121347
    qs = np.linspace(-1, 1, 5)
    lstm = lstm_m + lstm_s * qs
    lgb = lgb_m + lgb_s * qs

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    data = [trans, lstm, lgb]
    bp = ax.boxplot(data, tick_labels=["Transformer", "LSTM*", "LightGBM*"], patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#e1bee7", "#bbdefb", "#ffe0b2"]):
        patch.set_facecolor(c)
    ax.scatter(np.repeat(np.arange(1, 4), len(trans)), np.concatenate([trans, lstm, lgb]), alpha=0.35, color="#333", s=22)
    ax.set_ylabel(r"$R^2_{\mathrm{mean}}$ по фолдам")
    ax.set_title(
        r"$R^2$ по фолдам: Transformer — JSON из transformer_train_rv; "
        r"LSTM*/LightGBM* — линейная сетка [mean−std, mean+std] по ±std из WF-сводки (пофолдовых значений в логах нет)",
        fontsize=8.5,
    )
    ax.grid(axis="y")
    _save(fig, "11_boxplot_r2_by_fold.png")


def main() -> None:
    print("OUTPUT:", HERE)
    fig04_arch_qlike()
    fig05_seq_len()
    fig06_alpha_dual()
    fig07_feat_groups()
    fig08_models_dual_bar()
    fig09_r2_horizons()
    fig10_pred_vs_actual()
    fig11_boxplot_folds()
    print("Готово: 04-11 пересобраны из логов ->", HERE)


if __name__ == "__main__":
    main()
