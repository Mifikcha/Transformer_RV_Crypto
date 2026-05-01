"""Блок 2 — Volatility Term Structure."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.styles import PLOTLY_TEMPLATE

HORIZONS = ["15 мин", "1 час", "4 часа", "1 день"]
HORIZON_COLS = ["rv_3bar", "rv_12bar", "rv_48bar", "rv_288bar"]


def render_term_structure(
    latest_pred: dict | pd.Series,
    historical_df: pd.DataFrame,
) -> None:
    """Кривая волатильности по горизонтам + медиана за 30 дней."""
    current = []
    median_vals = []

    for col in HORIZON_COLS:
        val = latest_pred.get(col) if isinstance(latest_pred, dict) else getattr(latest_pred, col, None)
        current.append(val if val is not None else 0.0)
        if not historical_df.empty and col in historical_df.columns:
            median_vals.append(historical_df[col].median())
        else:
            median_vals.append(0.0)

    is_inverted = (current[0] > 0 and current[-1] > 0 and current[0] > current[-1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=HORIZONS,
        y=current,
        mode="lines+markers",
        name="Текущий прогноз",
        line=dict(color="#4fc3f7", width=3),
        marker=dict(size=10),
    ))
    fig.add_trace(go.Scatter(
        x=HORIZONS,
        y=median_vals,
        mode="lines+markers",
        name="Медиана 30д",
        line=dict(color="#8b949e", width=2, dash="dash"),
        marker=dict(size=6),
    ))

    if is_inverted:
        fig.update_layout(plot_bgcolor="rgba(244, 67, 54, 0.06)")
        fig.add_annotation(
            text="⚠ ИНВЕРСИЯ — краткосрочная vol > долгосрочной",
            xref="paper", yref="paper",
            x=0.5, y=1.10,
            showarrow=False,
            font=dict(color="#ff5722", size=13),
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=280,
        yaxis_title="Realized Volatility",
        xaxis_title="Горизонт прогноза",
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
