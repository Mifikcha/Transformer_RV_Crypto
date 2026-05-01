"""Блок 5 — Market Heat indicators + Блок 6 — Cross-Asset Correlation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.styles import COLOR_GREEN, COLOR_ORANGE, COLOR_RED, PLOTLY_TEMPLATE, TEXT_MUTED


# ---------------------------------------------------------------------------
# Вычисление индикаторов перегрева
# ---------------------------------------------------------------------------

def _overall_heat(zscore: float, vol_ratio: float, funding_extreme: bool) -> str:
    score = 0
    if zscore > 2:
        score += 2
    elif zscore > 1:
        score += 1
    if vol_ratio > 2:
        score += 2
    elif vol_ratio > 1.5:
        score += 1
    if funding_extreme:
        score += 1

    if score >= 4:
        return "ПЕРЕГРЕВ"
    elif score >= 2:
        return "ПОВЫШЕННЫЙ"
    return "НОРМА"


def compute_market_heat(
    bars_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> dict:
    """Вычислить индикаторы перегрева рынка."""
    result: dict = {
        "rv_zscore": 0.0,
        "volume_ratio": 1.0,
        "funding_rate": 0.0,
        "funding_extreme": False,
        "rv_acceleration": 0.0,
        "overall_heat": "НОРМА",
    }

    if predictions_df.empty or "rv_3bar" not in predictions_df.columns:
        return result

    rv_series = predictions_df["rv_3bar"].dropna()
    if rv_series.empty:
        return result

    rv_current = float(rv_series.iloc[-1])
    rv_mean = float(rv_series.mean())
    rv_std = float(rv_series.std())
    result["rv_zscore"] = (rv_current - rv_mean) / rv_std if rv_std > 0 else 0.0

    # Volume anomaly (7 дней = 2016 баров по 5м)
    if not bars_df.empty and "volume_perp" in bars_df.columns:
        vol_current = float(bars_df["volume_perp"].iloc[-1])
        vol_median = float(bars_df["volume_perp"].tail(2016).median())
        result["volume_ratio"] = vol_current / vol_median if vol_median > 0 else 1.0

    # Funding rate
    if not bars_df.empty and "funding_rate" in bars_df.columns:
        fr = float(bars_df["funding_rate"].iloc[-1])
        result["funding_rate"] = fr
        result["funding_extreme"] = abs(fr) > 0.0005

    # RV acceleration (1 час = 12 баров по 5м)
    if len(rv_series) > 13:
        rv_prev = float(rv_series.iloc[-13])
        result["rv_acceleration"] = (rv_current - rv_prev) / rv_prev if rv_prev > 0 else 0.0

    result["overall_heat"] = _overall_heat(
        result["rv_zscore"], result["volume_ratio"], result["funding_extreme"]
    )
    return result


# ---------------------------------------------------------------------------
# Рендер панели перегрева
# ---------------------------------------------------------------------------

def _heat_color(value: float, thresholds: tuple[float, float]) -> str:
    lo, hi = thresholds
    if value >= hi:
        return COLOR_RED
    if value >= lo:
        return COLOR_ORANGE
    return COLOR_GREEN


def render_heat_panel(heat: dict) -> None:
    """4 компактных индикатора в ряд."""
    cols = st.columns(4)

    with cols[0]:
        color = _heat_color(abs(heat["rv_zscore"]), (1.0, 2.0))
        st.markdown(f"""
        <div style="text-align:center;padding:8px;">
            <div style="font-size:13px;color:{TEXT_MUTED};">RV Z-Score</div>
            <div style="font-size:30px;font-weight:bold;color:{color};">{heat['rv_zscore']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        color = _heat_color(heat["volume_ratio"], (1.5, 2.0))
        st.markdown(f"""
        <div style="text-align:center;padding:8px;">
            <div style="font-size:13px;color:{TEXT_MUTED};">Volume / Median</div>
            <div style="font-size:30px;font-weight:bold;color:{color};">{heat['volume_ratio']:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        color = COLOR_RED if heat["funding_extreme"] else COLOR_GREEN
        fr_pct = heat["funding_rate"] * 100
        st.markdown(f"""
        <div style="text-align:center;padding:8px;">
            <div style="font-size:13px;color:{TEXT_MUTED};">Funding Rate</div>
            <div style="font-size:30px;font-weight:bold;color:{color};">{fr_pct:.3f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        acc = heat["rv_acceleration"]
        color = _heat_color(abs(acc), (0.2, 0.5))
        st.markdown(f"""
        <div style="text-align:center;padding:8px;">
            <div style="font-size:13px;color:{TEXT_MUTED};">RV Δ (1ч)</div>
            <div style="font-size:30px;font-weight:bold;color:{color};">{acc:+.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    # Итоговый статус
    heat_colors = {"НОРМА": COLOR_GREEN, "ПОВЫШЕННЫЙ": COLOR_ORANGE, "ПЕРЕГРЕВ": COLOR_RED}
    overall = heat["overall_heat"]
    color = heat_colors.get(overall, COLOR_GREEN)
    st.markdown(f"""
    <div style="text-align:center;margin-top:8px;padding:8px;
                border:1px solid {color};border-radius:6px;background:rgba(0,0,0,0.2);">
        <span style="font-size:14px;color:{TEXT_MUTED};">Итоговый статус:&nbsp;</span>
        <span style="font-size:16px;font-weight:bold;color:{color};">{overall}</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Блок 6 — Cross-Asset Correlation (BTC/ETH)
# ---------------------------------------------------------------------------

def render_cross_asset(df: pd.DataFrame) -> None:
    """Скользящая корреляция RV между BTC и ETH (окно 288 баров = 24ч)."""
    if df.empty or "btc_rv" not in df.columns or "eth_rv" not in df.columns:
        st.info("Нет данных для cross-asset корреляции (нужны данные по двум символам).")
        return

    rolling_corr = df["btc_rv"].rolling(288, min_periods=50).corr(df["eth_rv"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"],
        y=rolling_corr,
        mode="lines",
        line=dict(color="#4fc3f7", width=2),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.1)",
        name="BTC-ETH RV corr (24h)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Corr: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.8, line_dash="dash", line_color="#f44336",
                  annotation_text="Системный стресс", annotation_position="top right")
    fig.add_hline(y=0, line_dash="dot", line_color="#555")

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=250,
        margin=dict(l=50, r=20, t=30, b=30),
        yaxis_range=[-0.1, 1.05],
        yaxis_title="Корреляция RV",
        xaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)
