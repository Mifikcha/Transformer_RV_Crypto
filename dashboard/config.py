from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    # Переиспользуем DATABASE_URL из основного .env
    # Streamlit требует sync-драйвер: postgresql:// (не postgresql+asyncpg://)
    database_url: str = "postgresql://user:pass@localhost:5432/rv_bot"

    symbols: list[str] = ["BTCUSDT", "ETHUSDT"]
    refresh_interval_sec: int = 300       # автообновление каждые 5 мин
    default_lookback_hours: int = 24
    regime_lookback_days: int = 30
    performance_lookback_days: int = 7

    # Таблицы по символу (Вариант A: одна таблица = один символ)
    # Для Варианта B (колонка symbol) — поменять запросы в queries.py
    btc_predictions_table: str = "predictions"
    eth_predictions_table: str = "predictions"


settings = DashboardSettings()
