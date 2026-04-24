"""LightGBM regression baseline (multi-output RV forecasting)."""

from __future__ import annotations

import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from utils import (
    RV_TARGET_COLS,
    compute_regression_metrics,
    get_default_data_path,
    get_feature_columns,
    get_regression_target_columns,
    load_dataset,
    print_regression_metrics,
    walk_forward_split,
)


def run(
    data_path: str | None = None,
    n_splits: int = 5,
    target_columns: tuple[str, ...] = RV_TARGET_COLS,
) -> list[dict]:
    path = data_path or get_default_data_path()
    df = load_dataset(path)
    feat_cols = get_feature_columns(df)
    tgt_cols = get_regression_target_columns(df, target_columns=target_columns)

    X = df[feat_cols].astype(float).fillna(0.0).values
    y = df[tgt_cols].astype(float).values
    splits = walk_forward_split(df, n_splits=n_splits)
    metrics_per_fold: list[dict] = []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        base = lgb.LGBMRegressor(
            objective="regression",
            max_depth=6,
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
        model = MultiOutputRegressor(base)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        metrics_per_fold.append(
            compute_regression_metrics(y_true=y_test, y_pred=y_pred, target_columns=tgt_cols)
        )

    print_regression_metrics(metrics_per_fold, "LightGBM", tgt_cols)
    return metrics_per_fold


if __name__ == "__main__":
    run()
