# Spike Warning

Pipeline раннего предупреждения спайков волатильности (binary classification).

## Шаги

1. `python -m spike_warning.define_spikes`
2. `python -m spike_warning.extract_features`
3. `python -m spike_warning.analyze_pre_spike`
4. `python -m spike_warning.train_classifier`
5. `python -m spike_warning.evaluate`

Все артефакты сохраняются в `spike_warning/output/`.

## Что используется

- Источник данных: PostgreSQL (`bars_5m`, `predictions`, `rv_actual`)
- Модель: лучший из `LogisticRegression`, `RandomForest`, `LightGBM` по `average_precision`
- Воспроизводимость: `random_state=42`

## Интеграция

- Дашборд: `spike_warning.integrate.get_dashboard_spike_signal`
- Бот: `spike_warning.integrate.get_spike_probability_async`

Если модель не обучена и `spike_classifier_bundle.joblib` отсутствует, интеграция мягко
деградирует (блок предупреждения показывает статус "нет модели").

