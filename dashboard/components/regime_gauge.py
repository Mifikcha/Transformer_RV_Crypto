"""Блок 3 — Regime Classification."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.styles import PLOTLY_TEMPLATE, REGIME_COLORS, TEXT_MUTED

_TZ_MAP = {
    "UTC": ZoneInfo("UTC"),
    "MSK (UTC+3)": ZoneInfo("Europe/Moscow"),
    "EKB (UTC+5)": ZoneInfo("Asia/Yekaterinburg"),
}

# Плашки: зелёный / жёлтый / оранжевый / красный (фон, текст)
PILL_STYLE = {
    "LOW": ("#2e7d32", "#ffffff"),
    "NORMAL": ("#f9a825", "#1a1a1a"),
    "HIGH": ("#ef6c00", "#ffffff"),
    "EXTREME": ("#c62828", "#ffffff"),
}


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


def render_regime_pill(regime: str) -> None:
    """Компактная цветная плашка режима без дополнительного текста."""
    bg, fg = PILL_STYLE.get(regime, PILL_STYLE["NORMAL"])
    st.markdown(
        f'<div style="display:inline-block;padding:4px 10px;border-radius:6px;'
        f'background:{bg};color:{fg};font-size:12px;font-weight:700;letter-spacing:0.5px;">'
        f"{regime}</div>",
        unsafe_allow_html=True,
    )


def render_regime_gauge(
    current_rv: float,
    historical_rvs: pd.Series,
) -> None:
    """Крупный режим (оставлен для совместимости; на Overview не используется)."""
    regime, color = classify_regime(current_rv, historical_rvs)
    html = f"""
    <div style="text-align:center;padding:16px 0;">
        <div style="font-size:48px;font-weight:bold;color:{color};letter-spacing:3px;">{regime}</div>
        <div style="font-size:16px;color:{TEXT_MUTED};margin-top:6px;">
            RV&nbsp;=&nbsp;{current_rv:.6f}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _ts_display(s: pd.Series, tz_key: str) -> pd.Series:
    ts = pd.to_datetime(s)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert("UTC")
    tz = _TZ_MAP.get(tz_key, ZoneInfo("UTC"))
    return ts.dt.tz_convert(tz)


def render_regime_timeline(
    timeline_df: pd.DataFrame,
    historical_rvs: pd.Series,
    *,
    tz_display: str = "UTC",
) -> None:
    """RV за 7 дней с маркерами режима."""
    if timeline_df.empty:
        st.info("Нет данных для timeline режимов.")
        return

    df = timeline_df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["ts_disp"] = _ts_display(df["ts"], tz_display)

    regime_labels = df["rv_3bar"].apply(lambda x: classify_regime(x, historical_rvs)[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts_disp"],
        y=df["rv_3bar"],
        mode="lines",
        line=dict(color="#4fc3f7", width=1),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.1)",
        showlegend=False,
        hovertemplate="%{x}<br>RV: %{y:.6f}<extra></extra>",
    ))

    for regime, color in REGIME_COLORS.items():
        mask = regime_labels == regime
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df["ts_disp"][mask],
                y=df["rv_3bar"][mask],
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
