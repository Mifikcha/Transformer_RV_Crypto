"""Блок 1 — Price + RV Overlay через streamlit-lightweight-charts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    import streamlit_lightweight_charts as stlc
    _HAS_STLC = True
except ImportError:
    _HAS_STLC = False


_CHART_LAYOUT = {
    "layout": {
        "background": {"color": "#0d1117"},
        "textColor": "#e0e0e0",
    },
    "grid": {
        "vertLines": {"color": "#1e2a3a"},
        "horzLines": {"color": "#1e2a3a"},
    },
    "crosshair": {"mode": 1},
    "rightPriceScale": {"borderColor": "#30363d"},
    "timeScale": {"borderColor": "#30363d", "timeVisible": True},
}


def _prepare_candles(bars_df: pd.DataFrame, n_bars: int) -> list[dict]:
    df = bars_df.tail(n_bars).copy()
    df["time"] = df["ts"].apply(
        lambda t: int(t.timestamp()) if hasattr(t, "timestamp") else int(t)
    )
    return df[["time", "open_perp", "high_perp", "low_perp", "close_perp"]].rename(
        columns={
            "open_perp": "open",
            "high_perp": "high",
            "low_perp": "low",
            "close_perp": "close",
        }
    ).to_dict("records")


def _prepare_line(df: pd.DataFrame, col: str, n_bars: int) -> list[dict]:
    sub = df[["ts", col]].tail(n_bars).dropna(subset=[col]).copy()
    sub["time"] = sub["ts"].apply(
        lambda t: int(t.timestamp()) if hasattr(t, "timestamp") else int(t)
    )
    return sub[["time", col]].rename(columns={col: "value"}).to_dict("records")


def render_price_rv_chart(
    bars_df: pd.DataFrame,
    rv_pred_df: pd.DataFrame,
    rv_actual_df: pd.DataFrame,
    selected_horizon: str = "rv_3bar",
    n_bars: int = 500,
) -> None:
    """Рендерим двухпанельный график: свечи + RV-панель."""
    if bars_df.empty:
        st.warning("Нет данных баров для отображения.")
        return

    if not _HAS_STLC:
        st.error(
            "Библиотека `streamlit-lightweight-charts` не установлена. "
            "Запустите: `pip install streamlit-lightweight-charts`"
        )
        return

    candle_data = _prepare_candles(bars_df, n_bars)
    rv_pred_data = _prepare_line(rv_pred_df, selected_horizon, n_bars) if not rv_pred_df.empty else []
    rv_actual_data = _prepare_line(rv_actual_df, selected_horizon, n_bars) if not rv_actual_df.empty else []

    series_list = [{"type": "Candlestick", "data": candle_data, "options": {}}]

    rv_series: list[dict] = []
    if rv_pred_data:
        rv_series.append({
            "type": "Area",
            "data": rv_pred_data,
            "options": {
                "lineColor": "#4fc3f7",
                "topColor": "rgba(79,195,247,0.25)",
                "bottomColor": "rgba(79,195,247,0.0)",
                "lineWidth": 2,
                "title": "Predicted RV",
            },
        })
    if rv_actual_data:
        rv_series.append({
            "type": "Line",
            "data": rv_actual_data,
            "options": {
                "color": "#ff7043",
                "lineWidth": 1,
                "title": "Actual RV",
            },
        })

    charts = [
        {"chart": {**_CHART_LAYOUT, "height": 380}, "series": series_list},
    ]
    if rv_series:
        charts.append({"chart": {**_CHART_LAYOUT, "height": 180}, "series": rv_series})

    stlc.renderLightweightCharts(charts, key=f"price_rv_chart_{selected_horizon}")
