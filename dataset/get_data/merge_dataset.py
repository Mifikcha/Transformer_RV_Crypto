import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths as paths  # noqa: E402

# Пути берутся из _paths.py (SYMBOL → файлы с нужным префиксом).
PERP_PATH = paths.PERP_CLEAN
SPOT_PATH = paths.SPOT_CLEAN
OUTPUT_PATH = paths.COMBINED

print("=== Объединение perp и spot (5m) ===")

# 1. Загрузка
print("Загрузка perp...")
df_perp = pd.read_parquet(PERP_PATH)
print("Загрузка spot...")
df_spot = pd.read_parquet(SPOT_PATH)

# 2. Проверки базовых требований (по плану 1.4)
def basic_checks(df, name):
    print(f"\nПроверка {name}:")
    print(f"  Строк: {len(df):,}")
    print(f"  Период: {df['ts'].min()} → {df['ts'].max()}")
    print(f"  Пропуски в close: {df['close'].isna().sum():,}")
    print(f"  Монотонность ts: {'OK' if df['ts'].is_monotonic_increasing else 'НАРУШЕНА!'}")
    print(f"  Дубли ts: {df['ts'].duplicated().sum():,}")
    print(f"  Цены разумные: {df['close'].between(100, 500000).all()}")

basic_checks(df_perp, "perp")
basic_checks(df_spot, "spot")

# 3. Переименование колонок для ясности
df_perp = df_perp.rename(columns={
    'open': 'open_perp', 'high': 'high_perp', 'low': 'low_perp',
    'close': 'close_perp', 'volume': 'volume_perp', 'turnover': 'turnover_perp'
})

df_spot = df_spot.rename(columns={
    'open': 'open_spot', 'high': 'high_spot', 'low': 'low_spot',
    'close': 'close_spot', 'volume': 'volume_spot', 'turnover': 'turnover_spot'
})

# 4. Объединение (merge_asof backward)
print("\nСинхронизация через merge_asof (backward)...")
df_combined = pd.merge_asof(
    df_perp.sort_values('ts'),
    df_spot.sort_values('ts'),
    on='ts',
    direction='backward'
)

# Удаляем возможные дублирующие колонки (если есть)
df_combined = df_combined.loc[:, ~df_combined.columns.str.endswith('_drop')]

# 5. Заполнение пропусков (ffill)
print("Заполнение пропусков (ffill)...")
df_combined = df_combined.ffill()

# После ffill
initial_rows = len(df_combined)
df_combined = df_combined.dropna(subset=['close_spot'])
print(f"Удалено строк без spot: {initial_rows - len(df_combined):,}")
print(f"Осталось строк: {len(df_combined):,}")

# Критические пропуски после ffill
critical_na = df_combined[['close_perp', 'close_spot', 'volume_perp']].isna().sum()
if critical_na.sum() > 0:
    print("ВНИМАНИЕ! Остались критические пропуски:\n", critical_na)
else:
    print("Критические поля без пропусков — OK")

# 6. Добавляем basis (по плану 1.3)
df_combined['basis'] = (df_combined['close_perp'] - df_combined['close_spot']) / df_combined['close_spot'].replace(0, np.nan)

# 7. Финальные проверки
print("\nФинальные характеристики объединённого датасета:")
print(f"  Строк: {len(df_combined):,}")
print(f"  Период: {df_combined['ts'].min()} → {df_combined['ts'].max()}")
print(f"  Пропуски в close_perp: {df_combined['close_perp'].isna().sum():,}")
print(f"  Пропуски в basis: {df_combined['basis'].isna().sum():,}")
print(f"  Монотонность ts: {'OK' if df_combined['ts'].is_monotonic_increasing else 'НАРУШЕНА!'}")

# 8. Сохранение
df_combined.to_parquet(OUTPUT_PATH, index=False)
print(f"\nСохранено: {OUTPUT_PATH}")
print("Готово к добавлению признаков!")