from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from config import settings

# Заменить asyncpg на psycopg2 для синхронного подключения Streamlit
_sync_url = settings.database_url.replace("+asyncpg", "+psycopg2").replace(
    "postgresql+asyncpg", "postgresql+psycopg2"
)
# Если URL уже начинается с postgresql:// без драйвера — добавляем psycopg2
if _sync_url.startswith("postgresql://"):
    _sync_url = _sync_url.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(_sync_url, pool_pre_ping=True, pool_size=5, max_overflow=2)


@st.cache_data(ttl=settings.refresh_interval_sec, show_spinner=False)
def query_df(sql: str, **params: object) -> pd.DataFrame:
    """Выполнить параметризованный SQL, вернуть DataFrame.

    Параметры передаются как keyword-аргументы и подставляются через str.format().
    Используем format()-подстановку только для строковых частей (имена таблиц,
    INTERVAL). Числовые данные от пользователя — через SQLAlchemy bindparams.
    """
    formatted_sql = sql.format(**params) if params else sql
    with engine.connect() as conn:
        return pd.read_sql(text(formatted_sql), conn)


def check_connection() -> bool:
    """Проверить подключение к PostgreSQL."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
