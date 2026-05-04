"""Страница Overview — Price+RV, Term Structure, Regime."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from components.price_chart import compute_rv_sma_and_pct_vs_sma, render_price_rv_chart
from components.regime_gauge import classify_regime, render_regime_pill, render_regime_timeline
from components.styles import inject_styles
from components.term_structure import render_term_structure
from config import settings
from db import query_df
from spike_warning.integrate import get_dashboard_spike_signal
from queries import (
    QUERY_BARS,
    QUERY_PREDICTIONS,
    QUERY_REGIME_HISTORY,
    QUERY_REGIME_TIMELINE,
    QUERY_RV_ACTUAL,
    QUERY_TERM_STRUCTURE_HISTORY,
)

HORIZON_OPTIONS = ["15 мин", "1 час"]
HORIZON_MAP = {
    "15 мин": "rv_3bar",
    "1 час": "rv_12bar",
}

TIMEFRAME_MAP = {
    "6ч": 6, "12ч": 12, "24ч": 24,
    "3д": 72, "7д": 168, "30д": 720,
}

TZ_OPTIONS = ["UTC", "MSK (UTC+3)", "EKB (UTC+5)"]


def _forecast_block_html(label: str, pred: float | None, sma: float | None, pct: float | None) -> str:
    pred_html = f'<div style="font-size:1.35em;font-weight:600;">{pred:.6f}</div>' if pred is not None else '<div style="font-size:1.1em;">—</div>'
    sma_html = (
        f'<div style="color:#8b949e;font-size:12px;margin-top:4px;">SMA: {sma:.6f}</div>'
        if sma is not None
        else ""
    )
    pct_html = ""
    if pct is not None:
        col = "#81c784" if pct >= 0 else "#e57373"
        pct_html = f'<div style="color:{col};font-size:13px;margin-top:2px;">{pct:+.1f}% к SMA</div>'
    return (
        f'<div style="margin-bottom:16px;">'
        f'<div style="color:#aaa;font-size:12px;text-transform:uppercase;">{label}</div>'
        f"{pred_html}{sma_html}{pct_html}"
        f"</div>"
    )


def render() -> None:
    inject_styles()

    st.markdown("## Overview")

    ctrl_cols = st.columns([1, 1, 1.5, 1, 1.2, 1])
    with ctrl_cols[0]:
        st.selectbox("Пара", settings.symbols, key="ov_symbol")
    with ctrl_cols[1]:
        horizon_label = st.selectbox("Горизонт RV", HORIZON_OPTIONS, key="ov_horizon")
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
    with ctrl_cols[4]:
        tz_display = st.selectbox("Время", TZ_OPTIONS, index=0, key="ov_tz")
    with ctrl_cols[5]:
        sma_period = st.number_input("SMA (баров)", min_value=3, max_value=500, value=24, step=1)

    hours = TIMEFRAME_MAP[timeframe]

    with st.spinner("Загрузка данных…"):
        bars_df = query_df(QUERY_BARS, hours=hours)
        predictions_df = query_df(QUERY_PREDICTIONS, hours=hours)
        rv_actual_df = query_df(QUERY_RV_ACTUAL, hours=hours)
        term_history_df = query_df(QUERY_TERM_STRUCTURE_HISTORY)
        regime_history_df = query_df(QUERY_REGIME_HISTORY)
        regime_timeline_df = query_df(QUERY_REGIME_TIMELINE)

    now_utc = datetime.now(timezone.utc)

    def _lag_hours(df, col: str = "ts") -> float | None:
        if df is None or df.empty or col not in df.columns:
            return None
        last_ts = df[col].iloc[-1]
        try:
            py = last_ts.to_pydatetime()
        except AttributeError:
            return None
        if py.tzinfo is None:
            py = py.replace(tzinfo=timezone.utc)
        try:
            return (now_utc - py).total_seconds() / 3600
        except TypeError:
            return None

    bars_lag = _lag_hours(bars_df)
    pred_lag = _lag_hours(predictions_df)
    actual_lag = _lag_hours(rv_actual_df)

    if bars_lag is not None and bars_lag > 1:
        st.warning(
            f"bars_5m устарели: последний бар {bars_df['ts'].iloc[-1]} (лаг ~{bars_lag:.1f}ч). "
            "Запустите ingestion."
        )
    # Предсказания «протухают» гораздо быстрее, чем бары: 30+ мин без новых строк
    # уже означает, что prediction_worker остановился или падает с ошибкой.
    if pred_lag is not None and pred_lag > 0.5:
        st.warning(
            f"predictions устарели: последняя запись {predictions_df['ts'].iloc[-1]} "
            f"(лаг ~{pred_lag:.1f}ч). График волатильности будет иметь разрывы — "
            "нажмите «Догнать прогноз» в боковой панели или запустите "
            "`python -m scripts.backfill_predictions`."
        )
    # rv_actual считается с лагом 1 час (12 баров вперёд), поэтому терпим до ~2ч.
    if actual_lag is not None and actual_lag > 2:
        st.warning(
            f"rv_actual устарели: последняя запись {rv_actual_df['ts'].iloc[-1]} "
            f"(лаг ~{actual_lag:.1f}ч). Линия Actual RV будет неполной — "
            "она досчитается автоматически после «Догнать прогноз»."
        )

    st.markdown("---")
    chart_col, right_col = st.columns([3.1, 1])

    with chart_col:
        st.markdown('<div class="section-title">Price + RV Overlay</div>', unsafe_allow_html=True)
        _tzk = tz_display.replace(" ", "_").replace("(", "").replace(")", "")
        render_price_rv_chart(
            bars_df=bars_df,
            rv_pred_df=predictions_df,
            rv_actual_df=rv_actual_df,
            selected_horizon=selected_horizon,
            n_bars=int(n_bars),
            tz_display=tz_display,
            sma_period=int(sma_period),
            chart_key=f"ov_{hours}_{selected_horizon}_{int(n_bars)}_{int(sma_period)}_{_tzk}",
        )

    with right_col:
        st.markdown("**Текущий прогноз**")
        if not predictions_df.empty:
            p15, sma15, pct15 = compute_rv_sma_and_pct_vs_sma(predictions_df, "rv_3bar", int(sma_period))
            p1h, sma1h, pct1h = compute_rv_sma_and_pct_vs_sma(predictions_df, "rv_12bar", int(sma_period))

            st.markdown(
                _forecast_block_html("RV 15 мин", p15, sma15, pct15)
                + _forecast_block_html("RV 1 час", p1h, sma1h, pct1h),
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title" style="margin-top:8px;">Volatility Regime</div>', unsafe_allow_html=True)
            if not regime_history_df.empty:
                current_rv = float(predictions_df["rv_3bar"].iloc[-1])
                hist = regime_history_df["rv_3bar"].dropna()
                regime_name, _ = classify_regime(current_rv, hist)
                render_regime_pill(regime_name)
            else:
                render_regime_pill("NORMAL")

            if "degraded" in predictions_df.columns and predictions_df["degraded"].iloc[-1]:
                st.warning("Модель в режиме деградации")
            if "model_ver" in predictions_df.columns:
                st.caption(f"Версия: {predictions_df['model_ver'].iloc[-1]}")
        else:
            st.info("Нет данных predictions.")

        st.markdown('<div class="section-title" style="margin-top:8px;">Spike Warning</div>', unsafe_allow_html=True)
        try:
            spike = get_dashboard_spike_signal()
            if not spike.get("available"):
                st.caption("Spike модель недоступна (нужно обучить pipeline).")
            else:
                spike_prob = float(spike["probability"])
                if spike_prob >= 0.50:
                    color = "#f44336"
                    label = f"⚠ ВЫСОКАЯ ({spike_prob:.0%})"
                elif spike_prob >= 0.30:
                    color = "#ff9800"
                    label = f"ПОВЫШЕННАЯ ({spike_prob:.0%})"
                else:
                    color = "#4caf50"
                    label = f"НИЗКАЯ ({spike_prob:.0%})"
                st.markdown(
                    f'<div style="font-size:20px; color:{color}; font-weight:bold;">{label}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Вероятность spike в ближайшие 4 часа")
        except Exception:
            st.caption("Spike модель временно недоступна.")

    st.markdown("---")
    st.markdown('<div class="section-title">Volatility Term Structure</div>', unsafe_allow_html=True)
    if not predictions_df.empty:
        render_term_structure(
            latest_pred=predictions_df.iloc[-1],
            historical_df=term_history_df,
        )
    else:
        st.info("Нет данных для term structure.")

    st.markdown('<div class="section-title">Regime Timeline (7 дней)</div>', unsafe_allow_html=True)
    if not regime_timeline_df.empty and not regime_history_df.empty:
        render_regime_timeline(
            timeline_df=regime_timeline_df,
            historical_rvs=regime_history_df["rv_3bar"].dropna(),
            tz_display=tz_display,
        )
    else:
        st.info("Нет данных для timeline.")
