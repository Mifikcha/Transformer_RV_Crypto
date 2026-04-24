"""
Feature groups for selection pipeline: semantic families of predictors.
Used by builtin_importance, permutation_importance, and group_ablation.
"""

from __future__ import annotations

# All feature columns grouped by semantic family (matches dataset from 1. Dataset + 2. Target)
FEATURE_GROUPS: dict[str, list[str]] = {
    "ohlcv": [
        "open_perp", "high_perp", "low_perp", "close_perp",
        "volume_perp", "turnover_perp",
        "open_spot", "high_spot", "low_spot", "close_spot",
        "volume_spot", "turnover_spot",
    ],
    "time": [
        "hour", "weekday",
        "sin_hour", "cos_hour", "sin_day", "cos_day",
        "min_to_asia_open", "min_to_eu_open", "min_to_ny_open",
        "is_asia_session", "is_eu_session", "is_ny_session",
    ],
    "log_returns": [
        "log_return_1min", "log_return_3min", "log_return_5min",
        "log_return_15min", "log_return_60min",
        "cum_log_return_15min", "cum_log_return_60min", "cum_log_return_240min",
    ],
    "price_derived": [
        "delta_price_5min",
        "sma_15min", "delta_sma_15min",
        "sma_60min", "delta_sma_60min",
        "sma_240min", "delta_sma_240min",
        "basis",
    ],
    "volatility": [
        "atr_14",
        "realized_vol_15min", "realized_vol_60min", "realized_vol_240min",
        "vol_parkinson", "vol_garman_klass", "vol_rogers_satchell",
        "rv_parkinson_15min", "rv_parkinson_60min", "rv_parkinson_240min",
        "rv_gk_15min", "rv_gk_60min", "rv_gk_240min",
        "rv_rs_15min", "rv_rs_60min", "rv_rs_240min",
    ],
    "volume_stats": [
        "rolling_vol_mean_60min", "rolling_vol_mean_240min",
        "rolling_vol_std_60min", "z_score_vol_60min",
        "rolling_vol_std_240min", "z_score_vol_240min",
        "anomalous_vol_60min", "anomalous_ratio_60min",
        "anomalous_vol_240min", "anomalous_ratio_240min",
        "volume_perp_clipped",
    ],
    "derivatives": [
        "fundingRate", "funding_missing", "time_to_funding_min",
        "openInterest", "oi_missing", "delta_oi",
        "basis_diff",
    ],
}


def get_group_for_feature(col: str) -> str:
    """Return the group name for a feature column, or 'other' if not in any group."""
    for group_name, cols in FEATURE_GROUPS.items():
        if col in cols:
            return group_name
    return "other"


def get_all_group_features() -> list[str]:
    """Return flat list of all feature names that belong to a defined group."""
    out: list[str] = []
    for cols in FEATURE_GROUPS.values():
        out.extend(cols)
    return out
