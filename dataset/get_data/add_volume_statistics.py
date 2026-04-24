import pandas as pd
import numpy as np

# Пути (подставь свои)
INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_volatility.parquet"
OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_volume_stats.parquet"

print("=== Добавление объёмных статистик ===")

# 1. Загрузка
df = pd.read_parquet(INPUT_PATH)
print(f"Загружено строк: {len(df):,}")
print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

# Основной объём
v = df['volume_perp']

# 2. Окна в минутах (как в плане и твоём описании)
windows_min = [60, 240]
windows_steps = [round(m / 5) for m in windows_min]  # [12, 48]

# 3. Скользящие объёмы (rolling mean)
print("Расчёт rolling volume mean...")
for win_min, win_step in zip(windows_min, windows_steps):
    df[f'rolling_vol_mean_{win_min}min'] = v.rolling(win_step).mean()

# 4. Z-score объёма
print("Расчёт z-score...")
for win_min, win_step in zip(windows_min, windows_steps):
    mean_col = f'rolling_vol_mean_{win_min}min'
    std_col = f'rolling_vol_std_{win_min}min'
    df[std_col] = v.rolling(win_step).std()
    # Защита от деления на 0
    df[f'z_score_vol_{win_min}min'] = (v - df[mean_col]) / df[std_col].replace(0, np.nan)

# 5. Доля аномального объёма
print("Расчёт доли аномального объёма...")
for win_min, win_step in zip(windows_min, windows_steps):
    mean_col = f'rolling_vol_mean_{win_min}min'
    std_col = f'rolling_vol_std_{win_min}min'
    
    # Аномальный = 1 если volume > mean + 2*std
    df[f'anomalous_vol_{win_min}min'] = (v > (df[mean_col] + 2 * df[std_col])).astype(int)
    
    # Доля = rolling mean этой метки (на том же окне)
    df[f'anomalous_ratio_{win_min}min'] = df[f'anomalous_vol_{win_min}min'].rolling(win_step).mean()

# 6. Проверки
vol_cols = [c for c in df.columns if 'rolling_vol' in c or 'z_score_vol' in c or 'anomalous' in c]
print("\nПроверка новых признаков (последние 5 строк):")
print(df[vol_cols].tail(5))

print("\nПропуски в новых признаках:")
print(df[vol_cols].isna().sum())

# 7. Сохранение
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nСохранено: {OUTPUT_PATH}")
print("Объёмные статистики добавлены. Готово к funding/OI или таргетам.")