"""Блок 3 — Regime Classification."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.styles import PLOTLY_TEMPLATE, REGIME_COLORS, TEXT_MUTED


def classify_regime(current_rv: float, historical_rvs: pd.Series) -> tuple[str, str]:
    """Вернуть (regime_name, color) по квантилям за 30 дней."""
    if historical_rvs.empty or historical_rvs.std() == 0:
        return "NORMAL", REGIME_COLORS["NORMAL"]

    p25 = historical_rvs.quantile(0.25)
    p75 = historical_rvs.quantile(0.75)
    p95 = historical_rvs.quantile(0.95)

    if current_rv <= p25:
        name = "LOW"
    elif current_rv <= p75:
        name = "NORMAL"
    elif current_rv <= p95:
        name = "HIGH"
    else:
        name = "EXTREME"

    return name, REGIME_COLORS[name]


def _compute_percentile(current_rv: float, historical_rvs: pd.Series) -> float:
    if historical_rvs.empty:
        return 50.0
    return float((historical_rvs <= current_rv).mean() * 100)


def render_regime_gauge(
    current_rv: float,
    historical_rvs: pd.Series,
) -> None:
    """Отрисовка текущего режима + цветной индикатор."""
    regime, color = classify_regime(current_rv, historical_rvs)
    pct = _compute_percentile(current_rv, historical_rvs)

    html = f"""
    <div style="text-align:center;padding:16px 0;">
        <div style="font-size:48px;font-weight:bold;color:{color};letter-spacing:3px;">{regime}</div>
        <div style="font-size:16px;color:{TEXT_MUTED};margin-top:6px;">
            RV&nbsp;=&nbsp;{current_rv:.6f}&nbsp;&nbsp;|&nbsp;&nbsp;Перцентиль:&nbsp;{pct:.0f}%
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_regime_timeline(
    timeline_df: pd.DataFrame,
    historical_rvs: pd.Series,
) -> None:
    """Горизонтальная полоса режимов за 7 дней (scatter+fill)."""
    if timeline_df.empty:
        st.info("Нет данных для timeline режимов.")
        return

    regime_labels = timeline_df["rv_3bar"].apply(
        lambda x: classify_regime(x, historical_rvs)[0]
    )
    colors_list = regime_labels.map(REGIME_COLORS)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline_df["ts"],
        y=timeline_df["rv_3bar"],
        mode="lines",
        line=dict(color="#4fc3f7", width=1),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.1)",
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>RV: %{y:.6f}<extra></extra>",
    ))

    for regime, color in REGIME_COLORS.items():
        mask = regime_labels == regime
        if mask.any():
            fig.add_trace(go.Scatter(
                x=timeline_df["ts"][mask],
                y=timeline_df["rv_3bar"][mask],
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.6),
                name=regime,
                hoverinfo="skip",
            ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=160,
        margin=dict(l=50, r=20, t=20, b=30),
        xaxis_title="",
        yaxis_title="RV 3bar",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
