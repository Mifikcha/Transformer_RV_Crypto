import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка исходного датасета
df = pd.read_csv('btcusdt_5m_final_cleaned.csv', parse_dates=['ts'], index_col='ts')
df.sort_index(inplace=True)

delta_t = pd.Timedelta(minutes=5)  # из шага 1

# Список горизонтов для проверки (в минутах)
H_minutes = [30, 60, 120, 240]

# Словарь для хранения количества шагов и будущих цен
future_shifts = {}

for h in H_minutes:
    shift_steps = int(pd.Timedelta(minutes=h) / delta_t)
    future_col = f'future_close_{h}min'
    df[future_col] = df['close_perp'].shift(-shift_steps)  # используем close_perp как основную цену
    future_shifts[h] = shift_steps
    print(f"H = {h} мин → {shift_steps} шагов")

# Вычисление будущих лог-доходностей и добавление в df
for h in H_minutes:
    future_col = f'future_close_{h}min'
    ret_col = f'delta_log_{h}min'
    df[ret_col] = np.log(df[future_col] / df['close_perp'])

# Визуализация распределений будущих доходностей
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, h in enumerate(H_minutes):
    ret_col = f'delta_log_{h}min'
    df[ret_col].dropna().hist(bins=100, ax=axes[i], alpha=0.7)
    axes[i].set_title(f'Распределение log-return на {h} мин (mean: {df[ret_col].mean():.6f}, std: {df[ret_col].std():.6f})')
    axes[i].set_xlabel('Log-return')
    axes[i].set_ylabel('Частота')
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Сохранение конфигурации (можно в YAML или просто в переменные)
config = {
    "delta_t_minutes": 5,
    "primary_H_minutes": 60,
    "tested_H_minutes": H_minutes,
    "shifts": future_shifts
}

print("Конфигурация горизонтов сохранена:")
print(config)

# Сохранение обновлённого датасета с новыми колонками (future_close_* и delta_log_*)
# Используем новое имя, чтобы не перезаписать оригинал
df.to_csv('btcusdt_5m_with_horizons.csv', index=True)
print("Обновлённый датасет сохранён как 'btcusdt_5m_with_horizons.csv'")