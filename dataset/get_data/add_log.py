import pandas as pd
import numpy as np

# Пути (подставь свои)
INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_combined_with_time_features.parquet"
OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_price_derivatives.parquet"

print("=== Добавление производных от цены ===")

# 1. Загрузка
df = pd.read_parquet(INPUT_PATH)
print(f"Загружено строк: {len(df):,}")
print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

# 2. Основная цена — close_perp (по плану perp основной)
c = df['close_perp']

# 3. Лог-доходности на лагах (в шагах 5m = минуты / 5)
# Лаги в минутах из плана: 1, 3, 5, 15, 60 → в шагах: 0.2, 0.6, 1, 3, 12 → округляем до целых
lags_minutes = [1, 3, 5, 15, 60]
lags_steps = [max(1, round(m / 5)) for m in lags_minutes]  # [1, 1, 1, 3, 12]

for lag_min, lag_step in zip(lags_minutes, lags_steps):
    df[f'log_return_{lag_min}min'] = np.log(c / c.shift(lag_step))

# 4. Кумулятивная доходность на окнах (log(c / c.shift(N)))
# Окна из плана: 60 мин (и можно добавить 15, 240 для полноты)
cum_windows_min = [15, 60, 240]
cum_windows_steps = [round(m / 5) for m in cum_windows_min]  # [3, 12, 48]

for win_min, win_step in zip(cum_windows_min, cum_windows_steps):
    df[f'cum_log_return_{win_min}min'] = np.log(c / c.shift(win_step))

# 5. Простая дельта цены (опционально, как baseline)
df['delta_price_5min'] = c.diff()

# 6. Проверки
print("\nПроверка новых признаков (последние 5 строк):")
print(df.filter(regex='log_return|cum_log_return|delta_price').tail(5))

print("\nПропуски в новых признаках:")
print(df.filter(regex='log_return|cum_log_return|delta_price').isna().sum())

# 7. Сохранение
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nСохранено: {OUTPUT_PATH}")
print("Производные от цены добавлены. Готово к волатильности/объёмам.")