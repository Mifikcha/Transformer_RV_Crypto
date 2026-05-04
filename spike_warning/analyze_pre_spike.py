from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spike_warning.common import OUTPUT_DIR, ensure_output_dir, save_csv


def analyze_pre_spike_conditions(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    spike_mask = features_df["spike_event"].fillna(0).astype(int) == 1
    pre_marker = spike_mask.shift(-1, fill_value=False)

    feature_cols = [
        c
        for c in features_df.columns
        if c not in {"ts", "spike_in_next_4h", "spike_event", "spike_threshold", "max_rv_next_4h", "time_to_spike"}
        and not c.startswith("_meta_")
    ]

    pre_spike = features_df.loc[pre_marker, feature_cols]
    pre_no_spike_all = features_df.loc[~pre_marker, feature_cols]
    n_spikes = len(pre_spike)
    sample_n = min(len(pre_no_spike_all), max(n_spikes * 5, 1))
    pre_no_spike = pre_no_spike_all.sample(n=sample_n, random_state=42) if sample_n > 0 else pre_no_spike_all

    comparison = pd.DataFrame(
        {
            "pre_spike_mean": pre_spike.mean(numeric_only=True),
            "pre_no_spike_mean": pre_no_spike.mean(numeric_only=True),
        }
    )
    comparison["ratio"] = comparison["pre_spike_mean"] / comparison["pre_no_spike_mean"].replace(0, np.nan)
    comparison = comparison.sort_values("ratio", ascending=False)

    corr = features_df[feature_cols].corrwith(features_df["spike_in_next_4h"]).abs().sort_values(ascending=False)
    return comparison, corr


def render_plots(df: pd.DataFrame, comparison: pd.DataFrame, corr: pd.Series) -> None:
    ensure_output_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Pre-spike analysis", fontsize=14)

    top_features = list(corr.head(6).index)
    if top_features:
        axis = axes[0, 0]
        vals = [df.loc[df["spike_event"] == 1, f].dropna().values for f in top_features]
        axis.boxplot(vals, tick_labels=top_features, showfliers=False)
        axis.tick_params(axis="x", rotation=25)
        axis.set_title("Top-6 feature distributions (spike events)")

    axis = axes[0, 1]
    corr.head(10).sort_values().plot(kind="barh", ax=axis, color="#4C72B0")
    axis.set_title("Top correlations |feature, spike_in_next_4h|")
    axis.set_xlabel("Absolute correlation")

    axis = axes[1, 0]
    if "hour" in df.columns:
        spike_hours = df.loc[df["spike_event"] == 1, "hour"].dropna()
        axis.hist(spike_hours, bins=np.arange(-0.5, 24.5, 1), color="#DD8452", edgecolor="black")
        axis.set_xticks(range(0, 24, 2))
    axis.set_title("Spike events by UTC hour")
    axis.set_xlabel("Hour")
    axis.set_ylabel("Count")

    axis = axes[1, 1]
    if not comparison.empty:
        rel = comparison["ratio"].replace([np.inf, -np.inf], np.nan).dropna().head(10).sort_values()
        rel.plot(kind="barh", ax=axis, color="#55A868")
    axis.set_title("Top mean-ratio features (pre_spike / pre_no_spike)")
    axis.set_xlabel("Ratio")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pre_spike_analysis.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze pre-spike conditions")
    p.add_argument(
        "--features-path",
        default=str(OUTPUT_DIR / "spike_features.csv"),
        help="CSV from extract_features.py",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.features_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    comparison, correlations = analyze_pre_spike_conditions(df)
    save_csv(comparison.reset_index().rename(columns={"index": "feature"}), "comparison_table.csv")
    save_csv(correlations.reset_index().rename(columns={"index": "feature", 0: "abs_corr"}), "feature_correlations.csv")
    render_plots(df, comparison, correlations)

    print(f"Saved: {OUTPUT_DIR / 'comparison_table.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'feature_correlations.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'pre_spike_analysis.png'}")


if __name__ == "__main__":
    main()

