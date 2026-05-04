"""Блок 1 — Price + RV: Lightweight Charts (TradingView-style crosshair) + метрики SMA."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.lwc_dual_pane import render_tv_price_rv


def render_price_rv_chart(
    bars_df: pd.DataFrame,
    rv_pred_df: pd.DataFrame,
    rv_actual_df: pd.DataFrame,
    selected_horizon: str = "rv_3bar",
    n_bars: int = 500,
    *,
    tz_display: str = "UTC",
    sma_period: int = 24,
    chart_key: str = "lwc_default",
) -> None:
    """Две панели LWC: общий вертикальный крест; время только на нижней оси."""
    if bars_df.empty:
        st.warning("Нет данных баров для отображения.")
        return

    render_tv_price_rv(
        bars_df,
        rv_pred_df,
        rv_actual_df,
        selected_horizon=selected_horizon,
        n_bars=int(n_bars),
        sma_period=int(sma_period),
        tz_display=tz_display,
        component_key=chart_key,
    )


def compute_rv_sma_and_pct_vs_sma(
    predictions_df: pd.DataFrame,
    col: str,
    sma_period: int,
) -> tuple[float | None, float | None, float | None]:
    """Последнее значение прогноза, SMA по последним sma_period точкам, % относительно SMA."""
    if predictions_df.empty or col not in predictions_df.columns:
        return None, None, None
    df = predictions_df.sort_values("ts") if "ts" in predictions_df.columns else predictions_df
    s = df[col].dropna().tail(max(sma_period, 1))
    if s.empty:
        return None, None, None
    last = float(s.iloc[-1])
    win = min(len(s), sma_period)
    sma = float(s.rolling(window=win, min_periods=1).mean().iloc[-1])
    if sma == 0:
        return last, sma, None
    pct = (last - sma) / sma * 100.0
    return last, sma, pct
