"""
EDA for feature informativeness in RV forecasting.

Outputs:
- feature_selection/output/eda_summary.csv
- feature_selection/output/high_corr_pairs.csv
- feature_selection/output/eda/*.png
"""

from __future__ import annotations

import io
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_SCRIPT_DIR)
_BASELINES = os.path.join(_BASE, "baselines")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _BASELINES not in sys.path:
    sys.path.insert(0, _BASELINES)

from utils import (
    RV_TARGET_COLS,
    get_default_data_path,
    get_feature_columns,
    load_dataset,
    walk_forward_split,
)


def _out(msg: str, log_file: io.TextIOWrapper | None) -> None:
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    val = a.corr(b, method="spearman")
    return float(0.0 if pd.isna(val) else val)


def _plot_matrix(
    matrix: pd.DataFrame,
    title: str,
    out_path: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plt.figure(figsize=(max(8, matrix.shape[1] * 0.6), max(8, matrix.shape[0] * 0.2)))
    im = plt.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title(title)
    plt.yticks(np.arange(matrix.shape[0]), matrix.index)
    plt.xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run(
    data_path: str | None = None,
    output_dir: str | None = None,
    log_file: io.TextIOWrapper | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    path = data_path or get_default_data_path()
    out_dir = output_dir or os.path.join(_SCRIPT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    eda_plot_dir = os.path.join(out_dir, "eda")
    os.makedirs(eda_plot_dir, exist_ok=True)

    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    target_cols = [c for c in RV_TARGET_COLS if c in df.columns]
    if not target_cols:
        raise ValueError("No RV target columns found in dataset.")

    X_raw = df[feat_cols].astype(float)
    X = X_raw.fillna(0.0)
    Y = df[target_cols].astype(float).fillna(0.0)

    missing_pct = X_raw.isna().mean() * 100.0
    feat_std = X.std(ddof=0)

    result = pd.DataFrame(
        {
            "feature": feat_cols,
            "missing_pct": [float(missing_pct[c]) for c in feat_cols],
            "std": [float(feat_std[c]) for c in feat_cols],
        }
    )
    result["near_zero_variance"] = result["std"] < 1e-8

    # Univariate Spearman
    for tgt in target_cols:
        result[f"spearman_{tgt}"] = [_safe_spearman(X[c], Y[tgt]) for c in feat_cols]

    # Mutual Information
    for tgt in target_cols:
        mi = mutual_info_regression(X.values, Y[tgt].values, random_state=42)
        result[f"mi_{tgt}"] = mi

    spearman_cols = [f"spearman_{c}" for c in target_cols]
    mi_cols = [f"mi_{c}" for c in target_cols]
    result["mean_abs_spearman"] = result[spearman_cols].abs().mean(axis=1)
    result["mean_mi"] = result[mi_cols].mean(axis=1)

    # Stability over walk-forward validation windows
    splits = walk_forward_split(df, n_splits=n_splits)
    for tgt in target_cols:
        per_fold_spearman: list[pd.Series] = []
        for _, val_idx in splits:
            x_fold = X.iloc[val_idx]
            y_fold = Y[tgt].iloc[val_idx]
            s = x_fold.apply(lambda col: _safe_spearman(col, y_fold), axis=0)
            per_fold_spearman.append(s)

        fold_df = pd.DataFrame(per_fold_spearman)
        result[f"spearman_std_{tgt}"] = [
            float(fold_df[c].std(ddof=0)) if c in fold_df.columns else 0.0
            for c in feat_cols
        ]

    spearman_std_cols = [f"spearman_std_{c}" for c in target_cols]
    result["spearman_std_mean"] = result[spearman_std_cols].mean(axis=1)

    # Redundancy: strong feature-feature correlations
    corr_abs = X.corr(method="spearman").abs()
    high_corr_rows: list[dict[str, float | str]] = []
    for i, c1 in enumerate(feat_cols):
        for c2 in feat_cols[i + 1 :]:
            v = float(corr_abs.loc[c1, c2])
            if v >= 0.9:
                high_corr_rows.append(
                    {"feature_a": c1, "feature_b": c2, "abs_spearman": v}
                )
    if high_corr_rows:
        high_corr_df = pd.DataFrame(high_corr_rows).sort_values(
            "abs_spearman", ascending=False
        )
    else:
        high_corr_df = pd.DataFrame(columns=["feature_a", "feature_b", "abs_spearman"])

    # Save tabular outputs
    result = result.sort_values("mean_abs_spearman", ascending=False).reset_index(drop=True)
    result.to_csv(os.path.join(out_dir, "eda_summary.csv"), index=False)
    high_corr_df.to_csv(os.path.join(out_dir, "high_corr_pairs.csv"), index=False)

    # Plot 1: feature vs target Spearman heatmap
    hm = result.set_index("feature")[spearman_cols]
    _plot_matrix(
        hm,
        title="Spearman(feature, RV target)",
        out_path=os.path.join(eda_plot_dir, "spearman_feature_vs_target.png"),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    # Plot 2: MI top-20 per target
    for tgt in target_cols:
        col = f"mi_{tgt}"
        top = result.sort_values(col, ascending=False).head(20)
        plt.figure(figsize=(12, 6))
        plt.barh(top["feature"][::-1], top[col][::-1])
        plt.title(f"Mutual Information Top-20 ({tgt})")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_plot_dir, f"mi_top20_{tgt}.png"), dpi=150)
        plt.close()

    # Plot 3: inter-feature correlation heatmap
    _plot_matrix(
        corr_abs,
        title="Feature redundancy: |Spearman(feature_i, feature_j)|",
        out_path=os.path.join(eda_plot_dir, "feature_redundancy_heatmap.png"),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    # Plot 4: instability barplot
    unstable = result.sort_values("spearman_std_mean", ascending=False).head(20)
    plt.figure(figsize=(12, 6))
    plt.barh(unstable["feature"][::-1], unstable["spearman_std_mean"][::-1])
    plt.title("Most unstable features (std of Spearman over walk-forward folds)")
    plt.xlabel("spearman_std_mean")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_plot_dir, "spearman_instability_top20.png"), dpi=150)
    plt.close()

    _out("\n" + "=" * 80, log_file)
    _out("  EDA FOR FEATURE INFORMATIVENESS (RV)", log_file)
    _out("=" * 80, log_file)
    _out(f"  Dataset rows: {len(df):,}", log_file)
    _out(f"  Feature count: {len(feat_cols)}", log_file)
    _out(f"  Targets: {', '.join(target_cols)}", log_file)
    _out("\n  Top-15 by mean_abs_spearman:", log_file)
    for _, r in result.head(15).iterrows():
        _out(
            f"    {r['feature']:<30} spearman={r['mean_abs_spearman']:.4f} mi={r['mean_mi']:.5f} stbl_std={r['spearman_std_mean']:.4f}",
            log_file,
        )

    if not high_corr_df.empty:
        _out("\n  Top high-correlation feature pairs (|rho| >= 0.9):", log_file)
        for _, r in high_corr_df.head(20).iterrows():
            _out(
                f"    {r['feature_a']:<25} ~ {r['feature_b']:<25} |rho|={r['abs_spearman']:.4f}",
                log_file,
            )
    else:
        _out("\n  No strongly correlated feature pairs (|rho| >= 0.9).", log_file)

    _out(f"\n  Saved: {os.path.join(out_dir, 'eda_summary.csv')}", log_file)
    _out(f"  Saved: {os.path.join(out_dir, 'high_corr_pairs.csv')}", log_file)
    _out(f"  Plots dir: {eda_plot_dir}", log_file)
    _out("=" * 80 + "\n", log_file)

    return result
