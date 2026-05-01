"""Страница Overview — Блоки 1, 2, 3: Price+RV, Term Structure, Regime."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from components.price_chart import render_price_rv_chart
from components.regime_gauge import (
    classify_regime,
    render_regime_gauge,
    render_regime_timeline,
)
from components.styles import REGIME_COLORS, TEXT_MUTED, inject_styles
from components.term_structure import render_term_structure
from config import settings
from db import query_df
from queries import (
    QUERY_BARS,
    QUERY_PREDICTIONS,
    QUERY_REGIME_HISTORY,
    QUERY_REGIME_TIMELINE,
    QUERY_RV_ACTUAL,
    QUERY_TERM_STRUCTURE_HISTORY,
)

# Маппинги
HORIZON_OPTIONS = ["15 мин", "1 час", "4 часа", "1 день"]
HORIZON_MAP = {
    "15 мин": "rv_3bar",
    "1 час": "rv_12bar",
    "4 часа": "rv_48bar",
    "1 день": "rv_288bar",
}
HORIZON_LABEL_MAP = {v: k for k, v in HORIZON_MAP.items()}

TIMEFRAME_MAP = {
    "6ч": 6, "12ч": 12, "24ч": 24,
    "3д": 72, "7д": 168, "30д": 720,
}


def render() -> None:
    inject_styles()

    st.markdown("## Overview")

    # ── Панель управления ────────────────────────────────────────────────────
    ctrl_cols = st.columns([1, 1, 2, 1])
    with ctrl_cols[0]:
        symbol = st.selectbox("Пара", settings.symbols, key="ov_symbol")
    with ctrl_cols[1]:
        horizon_label = st.selectbox(
            "Горизонт RV", HORIZON_OPTIONS, key="ov_horizon"
        )
        selected_horizon = HORIZON_MAP[horizon_label]
    with ctrl_cols[2]:
        timeframe = st.select_slider(
            "Период",
            options=list(TIMEFRAME_MAP.keys()),
            value="24ч",
            key="ov_timeframe",
        )
    with ctrl_cols[3]:
        n_bars = st.number_input("Баров на графике", min_value=50, max_value=2000, value=500, step=50)

    hours = TIMEFRAME_MAP[timeframe]

    # ── Загрузка данных ──────────────────────────────────────────────────────
    with st.spinner("Загрузка данных…"):
        bars_df = query_df(QUERY_BARS, hours=hours)
        predictions_df = query_df(QUERY_PREDICTIONS, hours=hours)
        rv_actual_df = query_df(QUERY_RV_ACTUAL, hours=hours)
        term_history_df = query_df(QUERY_TERM_STRUCTURE_HISTORY)
        regime_history_df = query_df(QUERY_REGIME_HISTORY)
        regime_timeline_df = query_df(QUERY_REGIME_TIMELINE)

    if not bars_df.empty:
        last_ts = bars_df["ts"].iloc[-1]
        now_utc = datetime.now(timezone.utc)
        lag_hours = (now_utc - last_ts.to_pydatetime()).total_seconds() / 3600
        if lag_hours > 1:
            st.warning(
                f"Данные в БД устарели: последний бар {last_ts} (лаг ~{lag_hours:.1f}ч). "
                "Чтобы видеть актуальные данные, запустите ingestion/predictor из `view`."
            )

    # ── Блок 1: Price + RV Overlay ──────────────────────────────────────────
    st.markdown("---")
    chart_col, metrics_col = st.columns([4, 1])

    with chart_col:
        st.markdown('<div class="section-title">Price + RV Overlay</div>', unsafe_allow_html=True)
        render_price_rv_chart(
            bars_df=bars_df,
            rv_pred_df=predictions_df,
            rv_actual_df=rv_actual_df,
            selected_horizon=selected_horizon,
            n_bars=int(n_bars),
        )

    with metrics_col:
        st.markdown("**Текущий прогноз**")
        if not predictions_df.empty:
            latest_pred = predictions_df.iloc[-1]
            latest_actual = rv_actual_df.iloc[-1] if not rv_actual_df.empty else None

            for h_label, h_col in HORIZON_MAP.items():
                pred_val = latest_pred.get(h_col) if hasattr(latest_pred, "get") else getattr(latest_pred, h_col, None)
                actual_val = None
                if latest_actual is not None and h_col in rv_actual_df.columns:
                    actual_val = latest_actual.get(h_col) if hasattr(latest_actual, "get") else getattr(latest_actual, h_col, None)

                delta_str = None
                if pred_val is not None and actual_val is not None:
                    delta_str = f"{pred_val - actual_val:+.6f}"

                st.metric(
                    label=f"RV {h_label}",
                    value=f"{pred_val:.6f}" if pred_val is not None else "—",
                    delta=delta_str if delta_str else "pending",
                )
        else:
            st.info("Нет данных predictions.")

        # Статус деградации модели
        if not predictions_df.empty and "degraded" in predictions_df.columns:
            if predictions_df["degraded"].iloc[-1]:
                st.warning("⚠ Модель в режиме деградации")
            if "model_ver" in predictions_df.columns:
                ver = predictions_df["model_ver"].iloc[-1]
                st.caption(f"Версия модели: {ver}")

    # ── Блок 2: Term Structure ───────────────────────────────────────────────
    st.markdown("---")
    ts_col, regime_col = st.columns([3, 2])

    with ts_col:
        st.markdown('<div class="section-title">Volatility Term Structure</div>', unsafe_allow_html=True)
        if not predictions_df.empty:
            render_term_structure(
                latest_pred=predictions_df.iloc[-1],
                historical_df=term_history_df,
            )
        else:
            st.info("Нет данных для term structure.")

    # ── Блок 3: Regime ────────────────────────────────────────────────────────
    with regime_col:
        st.markdown('<div class="section-title">Volatility Regime</div>', unsafe_allow_html=True)
        if not predictions_df.empty and not regime_history_df.empty:
            current_rv = float(predictions_df["rv_3bar"].iloc[-1])
            historical_rvs = regime_history_df["rv_3bar"].dropna()
            render_regime_gauge(current_rv=current_rv, historical_rvs=historical_rvs)
        else:
            st.info("Нет данных для классификации режима.")

    # Regime timeline
    st.markdown('<div class="section-title">Regime Timeline (7 дней)</div>', unsafe_allow_html=True)
    if not regime_timeline_df.empty and not regime_history_df.empty:
        render_regime_timeline(
            timeline_df=regime_timeline_df,
            historical_rvs=regime_history_df["rv_3bar"].dropna(),
        )
    else:
        st.info("Нет данных для timeline.")
