import ccxt
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime

# ─── Настройки ───────────────────────────────────────────────────────────────
INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_volume_stats.parquet"
OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_derivatives.parquet"

SYMBOL = "BTCUSDT"
CATEGORY = "linear"

exchange = ccxt.bybit({'enableRateLimit': True})

print("=== Добавление funding rate, OI, ΔOI, basis динамики ===")

# 1. Загрузка текущего датасета
df = pd.read_parquet(INPUT_PATH)
print(f"Загружено строк: {len(df):,}")
print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

df['ts'] = pd.to_datetime(df['ts'], utc=True)
df = df.sort_values('ts').reset_index(drop=True)

min_ts_ms = int(df['ts'].min().timestamp() * 1000)

# ─── Funding rate через ccxt ─────────────────────────────────────────────────
print("\nЗагрузка funding rate history...")
funding_raw = []
since = min_ts_ms

while True:
    try:
        data = exchange.fetch_funding_rate_history(
            'BTC/USDT:USDT',
            since=since,
            limit=200,
            params={'category': 'linear'}
        )
        if not data:
            break
        funding_raw.extend(data)
        since = data[-1]['timestamp'] + 1
        print(f"Funding загружено {len(funding_raw)} записей...")
        time.sleep(1.2)
    except Exception as e:
        print(f"Funding ошибка: {e}")
        break

if funding_raw:
    df_funding = pd.DataFrame(funding_raw)[['timestamp', 'fundingRate']]
    df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'], unit='ms', utc=True)
    df_funding = df_funding.sort_values('timestamp').reset_index(drop=True)
else:
    df_funding = pd.DataFrame(columns=['timestamp', 'fundingRate'])
    print("Funding: данных нет")

# Merge funding
df = pd.merge_asof(
    df,
    df_funding,
    left_on='ts',
    right_on='timestamp',
    direction='backward'
).drop(columns=['timestamp'], errors='ignore')

df['fundingRate'] = df['fundingRate'].ffill()
df['funding_missing'] = df['fundingRate'].isna().astype(int)

# Время до следующего funding (00:00, 08:00, 16:00 UTC)
def time_to_next_funding(ts):
    h = ts.hour
    if h < 8:
        next_h = 8
    elif h < 16:
        next_h = 16
    else:
        next_h = 24
    next_ts = ts.replace(hour=next_h % 24, minute=0, second=0, microsecond=0)
    if next_h == 24:
        next_ts += pd.Timedelta(days=1)
    return (next_ts - ts).total_seconds() / 60

df['time_to_funding_min'] = df['ts'].apply(time_to_next_funding)

# ─── Open Interest через ручной запрос к Bybit V5 API ────────────────────────
print("\nЗагрузка open interest history (Bybit V5 API)...")
oi_list = []
cursor = None

while True:
    params = {
        'category': CATEGORY,
        'symbol': SYMBOL,
        'intervalTime': '5min',
        'limit': 200
    }
    if cursor:
        params['cursor'] = cursor

    url = "https://api.bybit.com/v5/market/open-interest"
    try:
        resp = requests.get(url, params=params).json()
        if resp.get('retCode') != 0:
            print(f"OI ошибка: {resp.get('retMsg', 'Неизвестная ошибка')}")
            break

        data = resp.get('result', {}).get('list', [])
        if not data:
            break

        oi_list.extend(data)
        cursor = resp['result'].get('nextPageCursor')
        if not cursor:
            break

        print(f"OI загружено {len(oi_list)} записей...")
        time.sleep(0.7)
    except Exception as e:
        print(f"OI запрос ошибка: {e}")
        break

if oi_list:
    df_oi = pd.DataFrame(oi_list)[['timestamp', 'openInterest']]
    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'].astype(int), unit='ms', utc=True)
    df_oi['openInterest'] = df_oi['openInterest'].astype(float)
    df_oi = df_oi.sort_values('timestamp').reset_index(drop=True)
else:
    df_oi = pd.DataFrame(columns=['timestamp', 'openInterest'])
    print("OI: данных нет")

# Merge OI
df = pd.merge_asof(
    df,
    df_oi,
    left_on='ts',
    right_on='timestamp',
    direction='backward'
).drop(columns=['timestamp'], errors='ignore')

df['openInterest'] = df['openInterest'].ffill()
df['oi_missing'] = df['openInterest'].isna().astype(int)

# ΔOI
df['delta_oi'] = df['openInterest'].diff().fillna(0)

# ─── Basis динамика ──────────────────────────────────────────────────────────
if 'basis' in df.columns:
    df['basis_diff'] = df['basis'].diff().fillna(0)
else:
    print("ВНИМАНИЕ: колонка 'basis' отсутствует — basis_diff не добавлен")

# ─── Проверки ────────────────────────────────────────────────────────────────
deriv_cols = ['fundingRate', 'time_to_funding_min', 'openInterest', 'delta_oi', 'basis_diff',
              'funding_missing', 'oi_missing']

print("\nПроверка новых признаков (последние 5 строк):")
print(df[deriv_cols].tail(5))

print("\nПропуски в новых признаках:")
print(df[deriv_cols].isna().sum())

print("\nПример fundingRate (не NaN):")
print(df[df['fundingRate'].notna()][['ts', 'fundingRate']].tail(3))

print("\nПример OI (не NaN):")
print(df[df['openInterest'].notna()][['ts', 'openInterest', 'delta_oi']].tail(3))

# ─── Сохранение ──────────────────────────────────────────────────────────────
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nСохранено: {OUTPUT_PATH}")
print("Деривативные фичи добавлены. Готово к таргетам!")