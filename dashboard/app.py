"""Точка входа Streamlit-дэшборда.

Запуск:
    cd dashboard
    streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import time

import streamlit as st

st.set_page_config(
    page_title="RV Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Импорт после set_page_config
from config import settings
from db import check_connection
from history_loader import pull_history
from predictions_loader import run_backfill


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 RV Dashboard")
    st.markdown("---")

    page = st.radio(
        "Навигация",
        ["Overview", "Analytics", "Alerts"],
        key="nav_page",
    )

    st.markdown("---")

    # Статус подключения к БД
    if check_connection():
        st.success("БД: подключено")
    else:
        st.error("БД: нет соединения")

    st.markdown("---")

    # Автообновление
    auto_refresh = st.checkbox(
        "Автообновление",
        value=True,
        key="auto_refresh",
        help=f"Обновлять каждые {settings.refresh_interval_sec} сек",
    )
    if auto_refresh:
        st.caption(f"Интервал: {settings.refresh_interval_sec} сек")

    st.markdown("---")
    st.markdown("### История баров")
    bootstrap_days = st.slider(
        "Глубина bootstrap (дней)",
        min_value=3,
        max_value=60,
        value=25,
        step=1,
        key="bootstrap_days",
    )
    if st.button("Подтянуть историю", use_container_width=True):
        with st.spinner("Подтягиваю историю из Bybit..."):
            ok, msg = pull_history(days=int(bootstrap_days))
        if ok:
            st.success(msg)
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(msg)

    st.markdown("---")
    st.markdown("### Прогнозы")
    st.caption(
        "Прогоняет инференс по всем барам, для которых ещё нет prediction "
        "(пробел между MAX(predictions.ts) и последним баром). "
        "После заполнения автоматически досчитывает rv_actual там, где есть ≥12 будущих баров."
    )
    if st.button("Догнать прогноз", use_container_width=True, key="btn_backfill_pred"):
        progress_bar = st.progress(0.0, text="Загружаю модель и бары...")

        def _on_progress(done: int, total: int) -> None:
            if total <= 0:
                return
            pct = min(1.0, done / total)
            progress_bar.progress(pct, text=f"Прогноз: {done}/{total} ({pct * 100:.1f}%)")

        with st.spinner("Прогоняю инференс по историческим барам..."):
            ok, msg, _stats = run_backfill(progress_cb=_on_progress)
        progress_bar.empty()
        if ok:
            st.success(msg)
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(msg)

    st.markdown("---")
    st.caption("RV Forecast Bot · v0.1")


# ---------------------------------------------------------------------------
# Роутинг страниц
# ---------------------------------------------------------------------------
if page == "Overview":
    from pages.overview import render
    render()

elif page == "Analytics":
    from pages.analytics import render
    render()

elif page == "Alerts":
    from pages.alerts import render
    render()


# ---------------------------------------------------------------------------
# Автообновление
# ---------------------------------------------------------------------------
# В режиме истории auto-refresh замораживаем — иначе слайдер каждые N секунд
# скакал бы обратно к ts_max при перерасчёте границ из свежих баров.
if auto_refresh and not st.session_state.get("replay_active", False):
    time.sleep(settings.refresh_interval_sec)
    st.cache_data.clear()
    st.rerun()
