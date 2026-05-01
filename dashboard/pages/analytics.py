"""Страница Analytics — Блоки 4, 5, 6: Model Performance, Market Heat, Cross-Asset."""

from __future__ import annotations

import streamlit as st

from components.heatmap import compute_market_heat, render_cross_asset, render_heat_panel
from components.performance import HORIZON_LABEL, render_model_performance
from components.styles import inject_styles
from config import settings
from db import query_df
from queries import (
    QUERY_BARS,
    QUERY_CROSS_ASSET_RV,
    QUERY_PRED_VS_ACTUAL,
    QUERY_PREDICTIONS,
)

PERF_HORIZON_OPTIONS = list(HORIZON_LABEL.keys())   # ["3bar", "12bar", "48bar", "288bar"]

TIMEFRAME_MAP = {"1д": 1, "3д": 3, "7д": 7, "14д": 14, "30д": 30}
HOURS_MAP = {"6ч": 6, "12ч": 12, "24ч": 24, "3д": 72, "7д": 168}


def render() -> None:
    inject_styles()

    st.markdown("## Analytics")

    # ── Панель управления ────────────────────────────────────────────────────
    ctrl_cols = st.columns([1, 1, 1])
    with ctrl_cols[0]:
        symbol = st.selectbox("Пара", settings.symbols, key="an_symbol")
    with ctrl_cols[1]:
        horizon_key = st.selectbox(
            "Горизонт (модель)",
            PERF_HORIZON_OPTIONS,
            format_func=lambda k: HORIZON_LABEL[k],
            key="an_horizon",
        )
    with ctrl_cols[2]:
        perf_days_label = st.selectbox(
            "Период для метрик",
            list(TIMEFRAME_MAP.keys()),
            index=2,
            key="an_perf_days",
        )
    perf_days = TIMEFRAME_MAP[perf_days_label]

    # ── Блок 4: Model Performance ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Model Performance Monitor</div>', unsafe_allow_html=True)

    with st.spinner("Загрузка pred/actual…"):
        pred_actual_df = query_df(QUERY_PRED_VS_ACTUAL, days=perf_days)

    render_model_performance(pred_actual_df, horizon_key=horizon_key)

    # ── Блок 5: Market Heat ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Market Heat Indicators</div>', unsafe_allow_html=True)

    heat_col, _ = st.columns([3, 1])
    with heat_col:
        heat_period = st.select_slider(
            "Период данных (heat)",
            options=list(HOURS_MAP.keys()),
            value="24ч",
            key="an_heat_period",
        )
    heat_hours = HOURS_MAP[heat_period]

    with st.spinner("Загрузка market heat…"):
        bars_df = query_df(QUERY_BARS, hours=heat_hours)
        predictions_df = query_df(QUERY_PREDICTIONS, hours=heat_hours)

    heat = compute_market_heat(bars_df=bars_df, predictions_df=predictions_df)
    render_heat_panel(heat)

    # ── Блок 6: Cross-Asset Correlation ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Cross-Asset RV Correlation (BTC / ETH)</div>', unsafe_allow_html=True)

    with st.spinner("Загрузка cross-asset данных…"):
        try:
            cross_df = query_df(
                QUERY_CROSS_ASSET_RV,
                btc_table="predictions",
                eth_table="predictions",
            )
            # Если в таблице нет колонки symbol — корреляция с самой собой (нет смысла)
            # Пользователь может настроить btc_table / eth_table в config.py
        except Exception as exc:
            cross_df = __import__("pandas").DataFrame()
            st.caption(f"Cross-asset: {exc}")

    render_cross_asset(cross_df)
