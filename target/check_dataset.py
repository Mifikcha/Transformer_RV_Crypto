import pandas as pd
import numpy as np
df = pd.read_csv('btcusdt_5m_final_cleaned.csv', parse_dates=['ts'], index_col='ts')
df.sort_index(inplace=True)
print(f"Δt: {df.index.to_series().diff().median()}")
print(f"Период: {df.index.min()} to {df.index.max()}")
print(f"Средняя волатильность: {df['log_return_1min'].std()}")
print(df.describe())