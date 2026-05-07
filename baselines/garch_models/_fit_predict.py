from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.garch_models._common import ensure_arch_available, metrics_for_variance


@dataclass(frozen=True)
class FitResult:
    model_name: str
    params: dict[str, float]


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    metrics: dict[str, float]


def _walk_forward_splits(n: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits on daily rows."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n < n_splits + 5:
        raise ValueError(f"Not enough rows for n_splits={n_splits}: n={n}")
    seg = n // (n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        train_end = (k + 1) * seg
        test_end = (k + 2) * seg if k < (n_splits - 1) else n
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(train_end, min(test_end, n), dtype=int)
        if len(test_idx) == 0 or len(train_idx) < 50:
            continue
        splits.append((train_idx, test_idx))
    if not splits:
        raise ValueError("No valid splits constructed.")
    return splits


def fit_forecast_variance(
    *,
    returns: np.ndarray,
    realized_var: np.ndarray,
    n_splits: int,
    mean: str,
    vol: str,
    p: int,
    q: int,
    o: int = 0,
    dist: str = "normal",
) -> tuple[list[FoldResult], np.ndarray]:
    """
    Fit ARCH model on train returns and forecast 1-step ahead variance for each test point.

    Returns:
      - list of FoldResult (metrics per fold)
      - y_pred_all: array of len N filled with NaN except test indices predicted
    """
    ensure_arch_available()
    from arch import arch_model  # type: ignore

    r = np.asarray(returns, dtype=float)
    y = np.asarray(realized_var, dtype=float)
    if len(r) != len(y):
        raise ValueError("returns and realized_var length mismatch")

    n = len(r)
    splits = _walk_forward_splits(n, n_splits)
    y_pred_all = np.full(n, np.nan, dtype=float)

    fold_results: list[FoldResult] = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        # Proper out-of-sample walk-forward:
        # for each test time t, fit on [0..t-1] and forecast variance for t.
        # This is slower but correct, and daily data size is small enough.
        scale = 100.0  # percent scaling for optimizer stability
        var_test = np.full(len(test_idx), np.nan, dtype=float)
        for j, t in enumerate(test_idx):
            r_hist = r[:t] * scale
            if len(r_hist) < 50:
                continue
            am = arch_model(
                r_hist,
                mean=mean,
                vol=vol,
                p=int(p),
                q=int(q),
                o=int(o),
                dist=dist,
                rescale=False,
            )
            res = am.fit(disp="off")
            fc = res.forecast(horizon=1, reindex=False)
            # The last row corresponds to the 1-step-ahead forecast beyond the sample.
            v_scaled = float(np.asarray(fc.variance.values, dtype=float)[-1, 0])
            var_test[j] = v_scaled / (scale**2)

        y_pred_all[test_idx] = var_test
        metrics = metrics_for_variance(y_true=y[test_idx], y_pred=var_test)
        # Additional OOS R² in log-space relative to a simple expanding-mean benchmark
        # that uses ONLY training data (more appropriate for forecasting evaluation).
        eps = 1e-12
        yt_pos = np.clip(y[test_idx].astype(float), eps, None)
        yp_pos = np.clip(var_test.astype(float), eps, None)
        yt_log = np.log(yt_pos)
        yp_log = np.log(yp_pos)
        ytr_log = np.log(np.clip(y[train_idx].astype(float), eps, None))
        mu_train = float(np.mean(ytr_log))
        ss_res = float(np.sum((yt_log - yp_log) ** 2))
        ss_base = float(np.sum((yt_log - mu_train) ** 2))
        metrics["r2_log_oos"] = float(1.0 - ss_res / ss_base) if ss_base > 0 else float("nan")
        fold_results.append(
            FoldResult(
                fold=fold,
                train_size=int(len(train_idx)),
                test_size=int(len(test_idx)),
                metrics=metrics,
            )
        )

    return fold_results, y_pred_all

