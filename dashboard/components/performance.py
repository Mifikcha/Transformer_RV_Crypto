"""Блок 4 — Model Performance Monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.styles import PLOTLY_TEMPLATE

HORIZON_LABEL = {
    "3bar": "15 мин",
    "12bar": "1 час",
    "48bar": "4 часа",
    "288bar": "1 день",
}


def _compute_r2(actual: pd.Series, predicted: pd.Series) -> float:
    """R² = 1 - SS_res/SS_tot."""
    actual = actual.dropna()
    predicted = predicted.dropna()
    common = actual.index.intersection(predicted.index)
    if len(common) < 2:
        return float("nan")
    a = actual.loc[common].values
    p = predicted.loc[common].values
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _compute_qlike(actual: pd.Series, predicted: pd.Series) -> float:
    """QLIKE = E[log(pred) + actual/pred]."""
    mask = (actual > 0) & (predicted > 0)
    if mask.sum() < 2:
        return float("nan")
    a = actual[mask].values
    p = predicted[mask].values
    return float(np.mean(np.log(p) + a / p))


def render_model_performance(
    pred_actual_df: pd.DataFrame,
    horizon_key: str = "3bar",
    window_bars: int = 288,
) -> None:
    """Rolling R², bias + scatter predicted vs actual."""
    pred_col = f"pred_{horizon_key}"
    actual_col = f"actual_{horizon_key}"
    h_label = HORIZON_LABEL.get(horizon_key, horizon_key)

    if pred_actual_df.empty or pred_col not in pred_actual_df.columns:
        st.info(f"Нет данных pred/actual для горизонта {h_label}.")
        return

    df = pred_actual_df[["ts", pred_col, actual_col]].dropna().copy()
    if df.empty:
        st.info(f"Нет перекрывающихся данных predictions + rv_actual для {h_label}.")
        return

    # --- Скользящие метрики ---
    def rolling_r2(sub: pd.DataFrame) -> pd.Series:
        result = []
        vals = sub[[actual_col, pred_col]].values
        for i in range(len(vals)):
            start = max(0, i - window_bars + 1)
            chunk_a = pd.Series(vals[start : i + 1, 0])
            chunk_p = pd.Series(vals[start : i + 1, 1])
            result.append(_compute_r2(chunk_a, chunk_p))
        return pd.Series(result, index=sub.index)

    df["rolling_r2"] = rolling_r2(df)
    df["rolling_bias"] = (df[pred_col] - df[actual_col]).rolling(window_bars, min_periods=10).mean()

    # --- График 1: Rolling R² и Bias ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=(f"Rolling R² ({h_label}, окно {window_bars} баров)", "Rolling Bias"),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
    )

    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["rolling_r2"],
        name="R²",
        line=dict(color="#4fc3f7", width=2),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>R²: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="#555", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["rolling_bias"],
        name="Bias",
        line=dict(color="#ff9800", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,152,0,0.1)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Bias: %{y:.6f}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="#555", row=2, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=380,
        margin=dict(l=50, r=20, t=50, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- График 2: Scatter predicted vs actual ---
    st.markdown('<div class="section-title">Scatter: Predicted vs Actual</div>', unsafe_allow_html=True)

    sample = df.sample(min(len(df), 3000), random_state=42)
    min_val = float(min(sample[actual_col].min(), sample[pred_col].min()))
    max_val = float(max(sample[actual_col].max(), sample[pred_col].max()))

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scattergl(
        x=sample[actual_col],
        y=sample[pred_col],
        mode="markers",
        marker=dict(size=3, color="#4fc3f7", opacity=0.25),
        name="точки",
        hovertemplate="Actual: %{x:.6f}<br>Pred: %{y:.6f}<extra></extra>",
    ))
    fig_sc.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="#f44336", dash="dash", width=1.5),
        name="Ideal (y=x)",
    ))

    # Сводные метрики
    r2_full = _compute_r2(df[actual_col], df[pred_col])
    qlike_full = _compute_qlike(df[actual_col], df[pred_col])
    bias_full = float((df[pred_col] - df[actual_col]).mean())

    fig_sc.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis_title="Actual RV",
        yaxis_title="Predicted RV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        annotations=[dict(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text=f"R²={r2_full:.4f} | QLIKE={qlike_full:.4f} | Bias={bias_full:+.6f}",
            showarrow=False, font=dict(size=11, color="#aaa"),
            align="left", bgcolor="rgba(0,0,0,0.4)",
        )],
    )
    st.plotly_chart(fig_sc, use_container_width=True)
