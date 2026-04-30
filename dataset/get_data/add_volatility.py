import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths as paths  # noqa: E402

INPUT_PATH = paths.WITH_PRICE_DERIVATIVES
OUTPUT_PATH = paths.WITH_VOLATILITY


def parkinson_estimator(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980): HL range. sqrt( (ln(H/L))^2 / (4 ln 2) )."""
    ratio_hl = (high / low).clip(lower=1.0001)
    ln_hl = np.log(ratio_hl)
    return np.sqrt(ln_hl**2 / (4 * np.log(2)))


def garman_klass_estimator(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Garman-Klass (1980): OHLC. Inner term clipped before sqrt."""
    ratio_hl = (high / low).clip(lower=1.0001)
    ratio_co = (close / open_).clip(lower=1e-6, upper=1e6)
    ln_hl = np.log(ratio_hl)
    ln_co = np.log(ratio_co)
    inner = 0.5 * ln_hl**2 - (2 * np.log(2) - 1) * ln_co**2
    return np.sqrt(np.maximum(0.0, inner))


def rogers_satchell_estimator(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Rogers-Satchell (1991): sqrt( ln(H/C)ln(H/O) + ln(L/C)ln(L/O) ), non-negative."""
    h = high.clip(lower=1e-12)
    l = low.clip(lower=1e-12)
    o = open_.clip(lower=1e-12)
    c = close.clip(lower=1e-12)
    inner = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(np.maximum(0.0, inner))


print("=== Добавление оценок волатильности ===")

# 1. Загрузка
df = pd.read_parquet(INPUT_PATH)
print(f"Загружено строк: {len(df):,}")
print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

# Основные колонки для расчётов
c = df['close_perp']
h = df['high_perp']
l = df['low_perp']
o = df['open_perp']

# 2. Окна в минутах (как в плане и твоём описании)
windows_min = [15, 60, 240]
windows_steps = [round(m / 5) for m in windows_min]  # [3, 12, 48]

# 3. Дельта средних скользящих (SMA.diff())
print("Расчёт SMA и их дельты...")
for win_min, win_step in zip(windows_min, windows_steps):
    sma = c.rolling(win_step).mean()
    df[f'sma_{win_min}min'] = sma
    df[f'delta_sma_{win_min}min'] = sma.diff()

# 4. ATR (14-периодный, стандартный)
print("Расчёт ATR...")
tr = np.maximum(
    h - l,
    np.maximum(
        abs(h - c.shift()),
        abs(l - c.shift())
    )
)
df['atr_14'] = tr.rolling(14).mean()

# 5. Реализованная волатильность
# Используем log_return_5min (уже есть из предыдущего шага)
log_ret_5 = df['log_return_5min']

print("Расчёт realized volatility...")
for win_min, win_step in zip(windows_min, windows_steps):
    # sqrt( sum( log_return^2 ) за окно ) → аннуализировать не будем (пока), оставим как есть
    rv = np.sqrt((log_ret_5**2).rolling(win_step).sum())
    df[f'realized_vol_{win_min}min'] = rv

# 5b. OHLC-оценщики (Parkinson, Garman-Klass, Rogers-Satchell) и rolling RV на тех же горизонтах
print("Расчёт OHLC volatility estimators...")
df['vol_parkinson'] = parkinson_estimator(h, l)
df['vol_garman_klass'] = garman_klass_estimator(o, h, l, c)
df['vol_rogers_satchell'] = rogers_satchell_estimator(o, h, l, c)

for win_bars, win_name in [(3, '15min'), (12, '60min'), (48, '240min')]:
    df[f'rv_parkinson_{win_name}'] = np.sqrt(
        (df['vol_parkinson'] ** 2).rolling(win_bars, min_periods=win_bars).sum()
    )
    df[f'rv_gk_{win_name}'] = np.sqrt(
        (df['vol_garman_klass'] ** 2).rolling(win_bars, min_periods=win_bars).sum()
    )
    df[f'rv_rs_{win_name}'] = np.sqrt(
        (df['vol_rogers_satchell'] ** 2).rolling(win_bars, min_periods=win_bars).sum()
    )

new_ohlc_rv_cols = [
    col for col in df.columns
    if col.startswith('rv_parkinson_') or col.startswith('rv_gk_') or col.startswith('rv_rs_')
]
df[new_ohlc_rv_cols] = df[new_ohlc_rv_cols].bfill().ffill()

# 5c. Relative / stationary price features (avoid StandardScaler drift).
# These replace absolute-price columns (open_perp, high_perp, low_perp,
# sma_*min, atr_14) whose distribution shifts when BTC price moves away
# from the training range.  Available after feature selection picks them.
prev_close = c.shift(1)
df['open_ret'] = np.log((o / prev_close).clip(lower=1e-12))
df['high_ret'] = np.log((h / c).clip(lower=1e-12))
df['low_ret'] = np.log((l / c).clip(lower=1e-12))
df['close_open_ret'] = np.log((c / o).clip(lower=1e-12, upper=1e6))
df['sma_60min_ratio'] = c / df['sma_60min'].clip(lower=1e-12)
df['sma_240min_ratio'] = c / df['sma_240min'].clip(lower=1e-12)
df['atr_14_norm'] = df['atr_14'] / c.clip(lower=1e-12)

# 6. Проверки
print("\nПроверка новых признаков (последние 5 строк):")
vol_cols = [
    col for col in df.columns
    if 'sma' in col or 'delta_sma' in col or 'atr' in col or 'realized_vol' in col
    or col.startswith('vol_parkinson') or col.startswith('vol_garman')
    or col.startswith('vol_rogers') or col.startswith('rv_parkinson_')
    or col.startswith('rv_gk_') or col.startswith('rv_rs_')
]
print(df[vol_cols].tail(5))

print("\nПропуски в новых признаках:")
print(df[vol_cols].isna().sum())

# 7. Сохранение
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nСохранено: {OUTPUT_PATH}")
print("Оценки волатильности добавлены. Готово к объёмным статистикам или funding/OI.")