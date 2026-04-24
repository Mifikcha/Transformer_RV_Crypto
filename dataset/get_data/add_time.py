import pandas as pd
import numpy as np
from datetime import datetime, time

# Пути (подставь свои)
INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_combined_2020-2026.parquet"
OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_combined_with_time_features.parquet"

print("=== Добавление временных признаков ===")

# 1. Загрузка
df = pd.read_parquet(INPUT_PATH)
print(f"Загружено строк: {len(df):,}")
print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

# Убедимся, что ts — datetime с tz
df['ts'] = pd.to_datetime(df['ts'], utc=True)

# 2. Базовые временные признаки (из плана 1.2)
df['hour']     = df['ts'].dt.hour
df['weekday']  = df['ts'].dt.weekday  # 0 = понедельник
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
df['sin_day']  = np.sin(2 * np.pi * df['weekday'] / 7)
df['cos_day']  = np.cos(2 * np.pi * df['weekday'] / 7)

# 3. Время до сессий (крипта 24/7, но добавляем традиционные сессии в UTC)
# Азиатская: 00:00–09:00 UTC
# Лондонская (европейская): 08:00–16:30 UTC (зимой/летом примерно одинаково)
# Нью-Йоркская: 13:30–20:00 UTC (летом), 14:30–21:00 UTC (зимой) — берём среднее 14:00–21:00 UTC

def minutes_to_session_open(ts):
    """Время до ближайшего открытия сессии в минутах (0–1440)"""
    hour = ts.hour
    minute = ts.minute
    
    # Азиатская (00:00 UTC)
    if hour < 0:  # невозможно
        next_asia = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        next_asia = (ts + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Европейская (08:00 UTC)
    if hour < 8:
        next_eu = ts.replace(hour=8, minute=0, second=0, microsecond=0)
    else:
        next_eu = (ts + pd.Timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    
    # Нью-Йоркская (14:00 UTC — усреднённо)
    if hour < 14:
        next_ny = ts.replace(hour=14, minute=0, second=0, microsecond=0)
    else:
        next_ny = (ts + pd.Timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0)
    
    minutes_asia = (next_asia - ts).total_seconds() / 60
    minutes_eu   = (next_eu   - ts).total_seconds() / 60
    minutes_ny   = (next_ny   - ts).total_seconds() / 60
    
    return minutes_asia, minutes_eu, minutes_ny

# Применяем
print("Расчёт времени до сессий...")
df[['min_to_asia_open', 'min_to_eu_open', 'min_to_ny_open']] = df['ts'].apply(
    lambda ts: pd.Series(minutes_to_session_open(ts))
)

# Индикаторы активных сессий (можно использовать как категориальные)
df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
df['is_eu_session']   = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
df['is_ny_session']   = ((df['hour'] >= 14) & (df['hour'] < 21)).astype(int)

# 4. Проверки
print("\nПроверка добавленных признаков:")
print(df[['ts', 'hour', 'weekday', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day',
          'min_to_ny_open', 'is_ny_session']].tail(8))

print(f"\nПропуски в новых признаках: {df.filter(regex='sin|cos|min_to|is_').isna().sum().sum()}")

# 5. Сохранение
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nСохранено: {OUTPUT_PATH}")
print("Временные признаки добавлены. Готово к следующему блоку фич.")