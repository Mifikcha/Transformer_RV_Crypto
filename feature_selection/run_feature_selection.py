"""
Feature selection pipeline for RV regression:
Step 0 EDA -> builtin importance -> permutation importance -> group ablation -> summary.

Run from project root:
  python feature_selection/run_feature_selection.py

Results are logged to feature_selection/output/feature_selection.log and CSVs in the same dir.
"""

from __future__ import annotations

import io
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_SCRIPT_DIR)
_BASELINES = os.path.join(_BASE, "baselines")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _BASELINES not in sys.path:
    sys.path.insert(0, _BASELINES)

import numpy as np
import pandas as pd

import builtin_importance
import eda
import group_ablation
import permutation_importance

DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")


def _out(msg: str, log_file: io.TextIOWrapper | None) -> None:
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def _normalized_rank_ascending(series: pd.Series) -> pd.Series:
    """Rank 1 = best (highest value). Return rank in [0,1] with 0 = best."""
    r = series.rank(ascending=False, method="average")
    n = r.dropna().count()
    if n <= 0:
        return r
    return (r - 1) / (n - 1) if n > 1 else pd.Series(0.0, index=series.index)


def main(
    data_path: str | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    output_dir: str | None = None,
) -> None:
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "feature_selection.log")
    log_file: io.TextIOWrapper | None = open(log_path, "w", encoding="utf-8")

    try:
        _out("\n" + "=" * 80, log_file)
        _out("  FEATURE SELECTION PIPELINE (RV regression)", log_file)
        _out("=" * 80, log_file)
        _out(f"  Output dir: {out_dir}", log_file)
        _out(f"  Log file: {log_path}", log_file)

        _out("\nStep 0: EDA for feature informativeness", log_file)
        df_eda = eda.run(
            data_path=data_path,
            output_dir=out_dir,
            log_file=log_file,
            n_splits=n_splits,
        )
        df_eda.to_csv(os.path.join(out_dir, "eda_summary.csv"), index=False)

        _out("\nStep 1: Builtin importance (LightGBMRegressor gain)", log_file)
        df_builtin = builtin_importance.run(
            data_path=data_path, n_splits=n_splits, log_file=log_file
        )
        df_builtin.to_csv(os.path.join(out_dir, "builtin_importance.csv"), index=False)

        _out("\nStep 2: Permutation importance (neg_MSE drop)", log_file)
        df_perm = permutation_importance.run(
            data_path=data_path,
            n_splits=n_splits,
            n_repeats=n_repeats,
            log_file=log_file,
        )
        df_perm.to_csv(os.path.join(out_dir, "permutation_importance.csv"), index=False)

        _out("\nStep 3: Group ablation (RV regression)", log_file)
        df_ablation = group_ablation.run(
            data_path=data_path, n_splits=n_splits, log_file=log_file
        )
        df_ablation.to_csv(os.path.join(out_dir, "group_ablation.csv"), index=False)

        # --- SUMMARY ---
        by_feature = df_builtin.set_index("feature")["mean_importance"].reindex(
            df_perm["feature"]
        )
        by_feature = by_feature.fillna(0)
        rank_builtin = _normalized_rank_ascending(by_feature)
        rank_perm = _normalized_rank_ascending(
            df_perm.set_index("feature")["mean_perm_importance"]
        )
        common = rank_builtin.index.intersection(rank_perm.index)
        rank_combined = (
            rank_builtin.reindex(common).fillna(0.5)
            + rank_perm.reindex(common).fillna(0.5)
        ) / 2
        rank_combined = rank_combined.sort_values().dropna()
        result_rank = pd.DataFrame(
            {
                "feature": rank_combined.index,
                "combined_rank_norm": rank_combined.values,
            }
        )
        result_rank["rank"] = np.arange(1, len(result_rank) + 1, dtype=int)

        total_imp = float(df_builtin["mean_importance"].sum())
        df_builtin_sorted = df_builtin.sort_values(
            "mean_importance", ascending=False
        ).reset_index(drop=True)
        cumsum = df_builtin_sorted["mean_importance"].cumsum()
        threshold = 0.9 * total_imp
        n_top = int((cumsum <= threshold).sum()) + 1
        n_top = min(n_top, len(df_builtin_sorted))
        recommended = df_builtin_sorted.head(n_top)["feature"].tolist()

        result_rank.to_csv(os.path.join(out_dir, "summary_ranking.csv"), index=False)
        pd.DataFrame({"feature": recommended}).to_csv(
            os.path.join(out_dir, "recommended_features.csv"), index=False
        )

        _out("\n" + "=" * 80, log_file)
        _out("  SUMMARY", log_file)
        _out("=" * 80, log_file)
        _out(
            "\n  Combined feature ranking (average of normalized builtin + permutation rank):",
            log_file,
        )
        _out("  Top 25:", log_file)
        for _, row in result_rank.head(25).iterrows():
            _out(
                f"    {row['rank']:3.0f}  {row['feature']:<35}  (norm_rank={row['combined_rank_norm']:.4f})",
                log_file,
            )

        _out(
            "\n  Group ranking by ablation (higher delta_mse / delta_r2 means more important):",
            log_file,
        )
        ablation_sorted = df_ablation.sort_values(
            ["delta_mse_mean", "delta_r2_mean"], ascending=False
        )
        for _, r in ablation_sorted.iterrows():
            _out(
                f"    {r['group']:<18}  delta_mse={r['delta_mse_mean']:+.6f}  delta_r2={r['delta_r2_mean']:+.6f}  delta_qlike={r['delta_qlike_mean']:+.6f}",
                log_file,
            )

        _out(
            f"\n  Recommended compact feature set (cumulative builtin importance >= 90%): {n_top} features",
            log_file,
        )
        _out(
            "  " + ", ".join(recommended[:15]) + ("  ..." if len(recommended) > 15 else ""),
            log_file,
        )
        if len(recommended) > 15:
            _out("  " + ", ".join(recommended[15:]), log_file)
        _out("=" * 80 + "\n", log_file)
        _out(f"  Results saved to: {out_dir}", log_file)
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
