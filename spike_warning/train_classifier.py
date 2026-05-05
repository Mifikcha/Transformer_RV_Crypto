from __future__ import annotations

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spike_warning.common import OUTPUT_DIR, ensure_output_dir

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional at runtime
    lgb = None


def _split_temporal(df: pd.DataFrame, test_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * (1 - test_ratio - val_ratio))
    val_end = int(n * (1 - test_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {
        "ts",
        "spike_in_next_4h",
        "spike_in_next_4h_raw",
        "spike_event",
        "spike_now_raw",
        "spike_now_event",
        "spike_threshold",
        "max_rv_next_4h",
        "time_to_spike",
    }
    return [c for c in df.columns if c not in drop and not c.startswith("_meta_")]


def _safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    # ROC AUC is undefined when test contains only one class.
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _split_temporal_with_class_guard(
    df: pd.DataFrame,
    label_col: str,
    test_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split that guarantees at least 2 classes in train when possible."""
    n = len(df)
    if n < 3:
        return _split_temporal(df, test_ratio, val_ratio)

    test_len = max(int(n * test_ratio), 1)
    val_len = max(int(n * val_ratio), 1)
    train_end = max(1, n - test_len - val_len)
    val_end = min(n - test_len, train_end + val_len)

    y = df[label_col].fillna(0).astype(int)
    if y.iloc[:train_end].nunique() < 2 and y.nunique() >= 2:
        # Move train_end to the earliest point where both classes become visible.
        pos_idx = np.flatnonzero(y.to_numpy() == 1)
        neg_idx = np.flatnonzero(y.to_numpy() == 0)
        first_pos = int(pos_idx[0]) if len(pos_idx) else None
        first_neg = int(neg_idx[0]) if len(neg_idx) else None
        if first_pos is not None and first_neg is not None:
            required_end = max(first_pos, first_neg) + 1
            # Keep at least 1 row for test.
            train_end = min(max(required_end, 1), n - 1)

            # Recompute val/test lengths on remaining tail.
            tail = n - train_end
            if tail <= 1:
                val_end = train_end
            else:
                denom = max(test_ratio + val_ratio, 1e-12)
                test_share = test_ratio / denom
                new_test_len = max(1, int(round(tail * test_share)))
                new_test_len = min(new_test_len, tail - 1)
                val_end = n - new_test_len
                if val_end < train_end:
                    val_end = train_end

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _build_importance_plot(model_name: str, model, feature_cols: list[str]) -> None:
    ensure_output_dir()
    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            values = np.abs(np.asarray(clf.coef_).ravel())
    if values is None:
        return

    imp = pd.Series(values, index=feature_cols).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Feature importance ({model_name})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train spike warning classifier")
    p.add_argument("--features-path", default=str(OUTPUT_DIR / "spike_features.csv"))
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_output_dir()

    df = pd.read_csv(args.features_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    feature_cols = _feature_columns(df)
    train_df, val_df, test_df = _split_temporal_with_class_guard(
        df, "spike_in_next_4h", args.test_ratio, args.val_ratio
    )
    if train_df.empty or test_df.empty:
        raise RuntimeError("Not enough rows for temporal split.")

    y_train = train_df["spike_in_next_4h"].fillna(0).astype(int)
    y_val = val_df["spike_in_next_4h"].fillna(0).astype(int)
    y_test = test_df["spike_in_next_4h"].fillna(0).astype(int)
    if y_train.nunique() < 2:
        counts = y_train.value_counts(dropna=False).to_dict()
        raise RuntimeError(
            "Train split still contains a single class after class-guard split. "
            f"Class counts: {counts}. Re-run define_spikes with different parameters "
            "or extend history window."
        )
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pos_weight = float((y_train == 0).sum()) / float(max((y_train == 1).sum(), 1))

    models: dict[str, object] = {}
    proba: dict[str, np.ndarray] = {}
    model_scores: dict[str, dict[str, float]] = {}

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1, random_state=42)),
        ]
    )
    lr.fit(X_train, y_train)
    proba["lr"] = lr.predict_proba(X_test)[:, 1]
    models["lr"] = lr

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    proba["rf"] = rf.predict_proba(X_test)[:, 1]
    models["rf"] = rf

    if lgb is not None:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.04,
            scale_pos_weight=pos_weight,
            min_child_samples=25,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        if len(X_val) > 0:
            lgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
            )
        else:
            lgb_model.fit(X_train, y_train)
        proba["lgb"] = lgb_model.predict_proba(X_test)[:, 1]
        models["lgb"] = lgb_model

    for name, p in proba.items():
        model_scores[name] = {
            "avg_precision": float(average_precision_score(y_test, p)),
            "roc_auc": _safe_roc_auc(y_test, p),
        }

    best_name = max(model_scores, key=lambda k: model_scores[k]["avg_precision"])
    best_model = models[best_name]

    pred_df = pd.DataFrame({"ts": test_df["ts"], "y_true": y_test.values})
    for name, p in proba.items():
        pred_df[f"proba_{name}"] = p
    if "_meta_close_perp" in test_df.columns:
        pred_df["close_perp"] = test_df["_meta_close_perp"].values
    pred_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    bundle = {
        "model_name": best_name,
        "model": best_model,
        "feature_columns": feature_cols,
        "scores": model_scores,
        "seed": 42,
    }
    joblib.dump(bundle, OUTPUT_DIR / "spike_classifier_bundle.joblib")

    _build_importance_plot(best_name, best_model, feature_cols)

    with open(OUTPUT_DIR / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_name,
                "scores": model_scores,
                "pos_weight": pos_weight,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
                "val_class_counts": {str(k): int(v) for k, v in y_val.value_counts().to_dict().items()},
                "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"Saved: {OUTPUT_DIR / 'spike_classifier_bundle.joblib'}")
    print(f"Saved: {OUTPUT_DIR / 'test_predictions.csv'}")
    print(f"Best model: {best_name}")
    print(model_scores)


if __name__ == "__main__":
    main()

