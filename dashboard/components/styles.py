"""Глобальные CSS-стили и цветовая схема дэшборда."""

import streamlit as st

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED = "#8b949e"

COLOR_GREEN = "#4caf50"
COLOR_BLUE = "#4fc3f7"
COLOR_ORANGE = "#ff9800"
COLOR_RED = "#f44336"
COLOR_ACCENT = "#ff7043"

REGIME_COLORS = {
    "LOW": COLOR_GREEN,
    "NORMAL": COLOR_BLUE,
    "HIGH": COLOR_ORANGE,
    "EXTREME": COLOR_RED,
}

HEAT_COLORS = {
    "НОРМА": COLOR_GREEN,
    "ПОВЫШЕННЫЙ": COLOR_ORANGE,
    "ПЕРЕГРЕВ": COLOR_RED,
}

PLOTLY_TEMPLATE = "plotly_dark"

GLOBAL_CSS = f"""
<style>
    /* Основной фон */
    .stApp {{
        background-color: {DARK_BG};
        color: {TEXT_PRIMARY};
    }}

    /* Убираем верхний отступ */
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}

    /* Карточки / панели */
    .dash-card {{
        background: {PANEL_BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }}

    /* Метрики */
    [data-testid="stMetric"] {{
        background: {PANEL_BG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 10px 14px;
    }}

    /* Заголовки секций */
    .section-title {{
        font-size: 14px;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {PANEL_BG};
        border-right: 1px solid {BORDER};
    }}

    /* Selectbox и slider */
    .stSelectbox > div > div {{
        background-color: {PANEL_BG};
        border-color: {BORDER};
    }}

    /* Таблицы */
    [data-testid="stDataFrame"] {{
        background-color: {PANEL_BG};
    }}

    /* Скрыть Streamlit header menu */
    #MainMenu, footer, header {{visibility: hidden;}}
</style>
"""


def inject_styles() -> None:
    """Внедрить глобальные CSS-стили."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, color: str = TEXT_PRIMARY, subtitle: str = "") -> str:
    """Вернуть HTML-строку компактной карточки метрики."""
    sub_html = f'<div style="font-size:12px;color:{TEXT_MUTED};">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="dash-card" style="text-align:center;">
        <div style="font-size:13px;color:{TEXT_MUTED};">{label}</div>
        <div style="font-size:28px;font-weight:bold;color:{color};">{value}</div>
        {sub_html}
    </div>
    """
