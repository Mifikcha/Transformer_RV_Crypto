from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from spike_warning.common import OUTPUT_DIR, ensure_output_dir


def evaluate_classifier(y_true: pd.Series, y_proba: np.ndarray, model_name: str) -> tuple[dict, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        best_threshold = 0.5
        y_pred = (y_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        return {
            "model": model_name,
            "roc_auc": float("nan"),
            "avg_precision": float("nan"),
            "best_threshold": best_threshold,
            "best_f1": float("nan"),
            "precision_at_best": float("nan"),
            "recall_at_best": float("nan"),
            "brier_score": float("nan"),
            "tp": int(cm[1, 1]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tn": int(cm[0, 0]),
            "classification_report": {},
        }, y_pred

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    y_pred = (y_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tn = int(cm[0, 0])

    return {
        "model": model_name,
        "roc_auc": float(roc_auc),
        "avg_precision": float(avg_precision),
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "precision_at_best": float(precision[best_idx]),
        "recall_at_best": float(recall[best_idx]),
        "brier_score": float(brier),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_warnings": int(tp + fp),
        "true_spike_rate": float(tp / max(tp + fp, 1)),
        "spikes_caught": float(tp / max(tp + fn, 1)),
        "classification_report": report,
    }, y_pred


def evaluate_at_multiple_thresholds(y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = (y_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": t,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "n_warnings_per_day": (tp + fp) / (len(y_true) / 288),
            }
        )
    return pd.DataFrame(rows)


def render_plots(
    model_name: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    best_threshold: float,
    ts: pd.Series,
    close_perp: pd.Series | None,
) -> None:
    ensure_output_dir()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[0, 0].plot(recall, precision, color="#1f77b4")
    axes[0, 0].set_title(f"PR curve ({model_name}), threshold={best_threshold:.3f}")
    axes[0, 0].set_xlabel("Recall")
    axes[0, 0].set_ylabel("Precision")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    axes[0, 1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, cm[i, j], ha="center", va="center")
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_xticklabels(["No spike", "Spike"])
    axes[0, 1].set_yticklabels(["No spike", "Spike"])
    axes[0, 1].set_title("Confusion matrix")

    # Calibration plot
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    axes[1, 0].plot(prob_pred, prob_true, marker="o", label="Model")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Calibration")
    axes[1, 0].set_xlabel("Predicted probability")
    axes[1, 0].set_ylabel("Actual frequency")
    axes[1, 0].legend(loc="upper left")

    # Timeline TP/FP/FN
    if close_perp is not None:
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        axes[1, 1].plot(ts, close_perp, color="#999999", linewidth=1, label="Price")
        axes[1, 1].scatter(ts[tp_mask], close_perp[tp_mask], s=20, color="#2ca02c", label="TP")
        axes[1, 1].scatter(ts[fp_mask], close_perp[fp_mask], s=20, color="#ffbf00", label="FP")
        axes[1, 1].scatter(ts[fn_mask], close_perp[fn_mask], s=20, color="#d62728", label="FN")
        axes[1, 1].set_title("Timeline: TP/FP/FN on price")
        axes[1, 1].legend(loc="upper left")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.05, 0.5, "close_perp not available for timeline plot")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "evaluation_plots.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate spike warning classifier")
    p.add_argument("--test-predictions-path", default=str(OUTPUT_DIR / "test_predictions.csv"))
    p.add_argument("--preferred-model", default="", help="Optional: lr/rf/lgb")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_output_dir()

    df = pd.read_csv(args.test_predictions_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    y_true = df["y_true"].astype(int)

    model_candidates = [c.replace("proba_", "") for c in df.columns if c.startswith("proba_")]
    if not model_candidates:
        raise RuntimeError("No probability columns found in test_predictions.csv.")
    model_name = args.preferred_model if args.preferred_model in model_candidates else model_candidates[0]
    y_proba = df[f"proba_{model_name}"].to_numpy(dtype=float)

    results, y_pred = evaluate_classifier(y_true, y_proba, model_name)
    thresh_table = evaluate_at_multiple_thresholds(y_true, y_proba)
    thresh_table.to_csv(OUTPUT_DIR / "threshold_tradeoff.csv", index=False)

    with open(OUTPUT_DIR / "classifier_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    close = df["close_perp"] if "close_perp" in df.columns else None
    render_plots(
        model_name=model_name,
        y_true=y_true,
        y_proba=y_proba,
        y_pred=y_pred,
        best_threshold=float(results["best_threshold"]),
        ts=df["ts"],
        close_perp=close,
    )

    print(f"Saved: {OUTPUT_DIR / 'classifier_metrics.json'}")
    print(f"Saved: {OUTPUT_DIR / 'threshold_tradeoff.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'evaluation_plots.png'}")
    print(results)


if __name__ == "__main__":
    main()

