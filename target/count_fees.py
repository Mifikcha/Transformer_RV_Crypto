import pandas as pd
import numpy as np

df = pd.read_csv('btcusdt_5m_with_horizons.csv', parse_dates=['ts'], index_col='ts')
df.sort_index(inplace=True)

# === Параметры издержек (Bybit Non-VIP, BTCUSDT perp, 2026) ===
commissions_roundtrip = 0.0009      # 0.09% — среднее (taker-heavy)
effective_half_spread = 0.0003      # 0.03% — proxy из литературы / order book
slippage_proxy = 0.0007             # 0.07% — для средних ордеров

c_base = commissions_roundtrip + effective_half_spread + slippage_proxy
print(f"Базовый c (round-trip): {c_base:.4f} ({c_base*100:.2f}%)")

# Варианты для sensitivity
c_scenarios = {
    'optimistic': 0.0010,   # maker-heavy, низкий slippage
    'base':       c_base,   # ~0.19%
    'pessimistic':0.0020    # taker + высокий slippage
}

print("\nСценарии издержек:")
for name, val in c_scenarios.items():
    print(f"  {name}: {val:.4f} ({val*100:.2f}%)")

# Эмпирический proxy спреда из OHLC (дополнительно, для проверки)
df['bar_range_pct'] = (df['high_perp'] - df['low_perp']) / df['close_perp']
avg_bar_range = df['bar_range_pct'].mean()
proxy_half_spread_from_ohlc = avg_bar_range / 4  # грубый proxy (range ≈ 4 * half-spread в спокойном рынке)
print(f"\nЭмпирический proxy half-spread из OHLC: {proxy_half_spread_from_ohlc:.4f} ({proxy_half_spread_from_ohlc*100:.2f}%)")

# Сравнение с ожидаемыми доходностями (на H=60 мин)
h = 60
ret_col = f'delta_log_{h}min'
std_ret = df[ret_col].std()
mean_ret = df[ret_col].mean()

print(f"\nНа H=60 мин: mean log-ret = {mean_ret:.6f}, std = {std_ret:.6f}")
print(f"Типичное движение (1 std): ±{std_ret*100:.2f}%")
print(f"c / std_ret = {c_base / std_ret:.2f} → c составляет ~{c_base / std_ret * 100:.0f}% от типичного движения")

# Сохранение в конфиг
config_step3 = {
    "platform": "Bybit perpetual BTCUSDT",
    "account_type": "Non-VIP",
    "commissions_roundtrip": commissions_roundtrip,
    "effective_half_spread": effective_half_spread,
    "slippage_proxy": slippage_proxy,
    "c_base": c_base,
    "c_scenarios": c_scenarios,
    "note": "Используется базовый c=0.0019 для основного таргета. Sensitivity в бейзлайнах."
}

print("\nКонфигурация шага 3:")
print(config_step3)

# Опционально: сохранить
# import yaml
# with open('config_step3.yaml', 'w') as f:
#     yaml.dump(config_step3, f)