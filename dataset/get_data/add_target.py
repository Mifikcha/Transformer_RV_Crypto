import pandas as pd
import numpy as np

# Пути (подставь свои)
INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_derivatives.parquet"
OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_final_with_targets.parquet"

H_MINUTES = 60  # Горизонт предсказания (п.2.1)
DELTA_T = 5  # Шаг датасета в минутах
H_STEPS = H_MINUTES // DELTA_T  # Шаги для shift (12 для 60 мин при 5m)
C = 0.0005  # Издержки (комиссия + спред + проскальзывание, п.2.3)

print("=== Добавление таргетов ===")

# 1. Загрузка
df = pd.read_parquet(INPUT_PATH)
print(f"Загружено строк: {len(df):,}")
print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

# 2. Базовый таргет (регрессия ожидаемой доходности, п.2.2)
c = df['close_perp']
df['target_return'] = np.log(c.shift(-H_STEPS) / c)  # \Delta p_{t \to t+H}

# 3. Торгуемый таргет (классы long/short/flat с учётом издержек, п.2.3)
df['target_class'] = np.where(
    df['target_return'] > C, 'long',
    np.where(df['target_return'] < -C, 'short', 'flat')
)

# 4. Валидность таргета (1 если есть данные на t+H, без NaN в будущем)
df['is_valid_target'] = df['target_return'].notna().astype(int)

# 5. Проверки
target_cols = ['target_return', 'target_class', 'is_valid_target']
print("\nПроверка новых таргетов (последние 5 строк, где valid):")
print(df[df['is_valid_target'] == 1][target_cols].tail(5))

print("\nРаспределение классов:")
print(df['target_class'].value_counts(normalize=True) * 100)

print("\nПропуски в таргетах:")
print(df[target_cols].isna().sum())

# 6. Сохранение
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nСохранено: {OUTPUT_PATH}")
print("Таргеты добавлены. Датасет готов к моделям!")