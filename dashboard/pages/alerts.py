"""Страница Alerts — Блок 7: Alert History."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.styles import COLOR_GREEN, COLOR_ORANGE, COLOR_RED, inject_styles
from db import query_df
from queries import QUERY_ALERTS

DAYS_OPTIONS = {"24ч": 1, "3 дня": 3, "7 дней": 7, "30 дней": 30}

ALERT_COLORS = {
    "SPIKE": COLOR_RED,
    "HIGH": COLOR_ORANGE,
    "NORMAL": COLOR_GREEN,
}


def _color_row(row: pd.Series) -> list[str]:
    atype = str(row.get("alert_type", "")).upper()
    color = ALERT_COLORS.get(atype, "")
    style = f"color: {color};" if color else ""
    return [style] * len(row)


def render() -> None:
    inject_styles()

    st.markdown("## Alert History")

    # ── Панель управления ────────────────────────────────────────────────────
    ctrl_cols = st.columns([1, 1, 2])
    with ctrl_cols[0]:
        period_label = st.selectbox("Период", list(DAYS_OPTIONS.keys()), index=2, key="al_period")
    with ctrl_cols[1]:
        filter_type = st.selectbox(
            "Тип алерта",
            ["Все", "SPIKE", "HIGH", "NORMAL"],
            key="al_type_filter",
        )

    days = DAYS_OPTIONS[period_label]

    # ── Загрузка ─────────────────────────────────────────────────────────────
    with st.spinner("Загрузка алертов…"):
        alerts_df = query_df(QUERY_ALERTS, days=days)

    if alerts_df.empty:
        st.info(f"Алертов за последние {period_label} не найдено.")
        return

    # ── Фильтрация ───────────────────────────────────────────────────────────
    if filter_type != "Все" and "alert_type" in alerts_df.columns:
        alerts_df = alerts_df[alerts_df["alert_type"].str.upper() == filter_type]

    # ── Форматирование ───────────────────────────────────────────────────────
    display = alerts_df.copy()

    if "sent_at" in display.columns:
        display["sent_at"] = pd.to_datetime(display["sent_at"]).dt.strftime("%Y-%m-%d %H:%M")
    if "prediction_ts" in display.columns:
        display["prediction_ts"] = pd.to_datetime(display["prediction_ts"]).dt.strftime("%Y-%m-%d %H:%M")
    if "pred_rv" in display.columns:
        display["pred_rv"] = display["pred_rv"].apply(
            lambda x: f"{x:.6f}" if pd.notna(x) else "—"
        )
    if "actual_rv" in display.columns:
        display["actual_rv"] = display["actual_rv"].apply(
            lambda x: f"{x:.6f}" if pd.notna(x) else "pending"
        )

    # ── Статистика ───────────────────────────────────────────────────────────
    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.metric("Всего алертов", len(alerts_df))
    with stat_cols[1]:
        if "alert_type" in alerts_df.columns:
            spike_count = (alerts_df["alert_type"].str.upper() == "SPIKE").sum()
            st.metric("SPIKE алертов", spike_count)
    with stat_cols[2]:
        if "sent_at" in alerts_df.columns:
            last_alert = pd.to_datetime(alerts_df["sent_at"]).max()
            st.metric("Последний алерт", last_alert.strftime("%Y-%m-%d %H:%M") if pd.notna(last_alert) else "—")

    st.markdown("---")

    # ── Таблица ──────────────────────────────────────────────────────────────
    cols_order = [c for c in ["sent_at", "alert_type", "prediction_ts", "pred_rv", "actual_rv", "model_ver"]
                  if c in display.columns]

    styled = (
        display[cols_order]
        .style
        .apply(_color_row, axis=1)
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    # ── Экспорт ──────────────────────────────────────────────────────────────
    csv_data = display[cols_order].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Скачать CSV",
        data=csv_data,
        file_name=f"alerts_{period_label}.csv",
        mime="text/csv",
    )
