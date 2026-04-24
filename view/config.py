from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Bybit
    bybit_base_url: str = "https://api.bybit.com"
    symbol: str = "BTCUSDT"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/rv_bot"

    # Telegram
    telegram_bot_token: str = ""
    telegram_channel_id: str = ""
    telegram_message_thread_id: int | None = None
    telegram_alert_chat_ids: str = ""
    allowed_chat_ids: str = ""

    # Model
    model_path: str = "transformer/output/models/fold_rv_4.pt"
    model_paths: str = ""
    features_path: str = "feature_selection/output/recommended_features.csv"

    # Schedule
    fetch_interval_seconds: int = 300
    predict_interval_seconds: int = 300
    notify_interval_seconds: int = 300

    # Alerts
    rv_spike_multiplier: float = 2.0
    rv_lookback_hours_for_median: int = 24

    # Buffer
    bar_buffer_size: int = 7000
    min_bars_for_inference: int = 250
    min_pairs_for_bias_calibration: int = 100

    @property
    def alert_chat_ids_list(self) -> list[int]:
        if not self.telegram_alert_chat_ids:
            return []
        return [int(x.strip()) for x in self.telegram_alert_chat_ids.split(",") if x.strip()]

    @property
    def allowed_chat_ids_set(self) -> set[int]:
        if not self.allowed_chat_ids:
            return set()
        return {int(x.strip()) for x in self.allowed_chat_ids.split(",") if x.strip()}

    @property
    def model_paths_list(self) -> list[str]:
        if self.model_paths.strip():
            return [x.strip() for x in self.model_paths.split(",") if x.strip()]
        return [self.model_path]
