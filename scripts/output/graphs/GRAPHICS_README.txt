Каталог иллюстраций для глав 2–3 (генерация: python generate_all_figures.py).

Файлы
--------
Глава 2 / общая логика
  01_ml_pipeline.png          — блок-схема ML-пайплайна (matplotlib)
  01_ml_pipeline.puml          — то же в PlantUML (экспорт PNG через PlantUML / IDEA)

Глава 3 — production
  02_production_architecture.png / .puml

Глава 3 — графики по таблицам из Глава_3 … _v2 (CSV экспериментов в репозитории отсутствуют — числа прошиты в скрипт)
  04_qlike_architectures.png
  05_qlike_vs_seq_len.png
  06_alpha_qlike_bias.png
  07_delta_qlike_feature_groups.png
  08_models_r2_qlike.png
  09_r2_by_horizon.png
  10_pred_vs_actual_log_four_panel.png  — модельные облака по corr/bias/MAE табл. 3.11 (не сырые предикции)
  11_boxplot_r2_by_fold.png             — фолды табл. 3.12 для Transformer; LSTM/LGB — параллельный сдвиг среднего

Spike (данные: spike_warning/output/test_predictions.csv, classifier_metrics.json, bundle joblib)
  12_spike_classifier_panel.png
  12a_spike_pr_curves.png
  12b_spike_confusion_matrix.png
  12c_spike_feature_importance.png
  12d_spike_timeline_price_markers.png

Скриншоты Telegram/Dashboard — по запросу автора в тексте игнорированы.

Подстановка реальных CSV
------------------------
Если появятся файлы вида transformer/output/experiments/*.csv, расширьте generate_all_figures.py:
читать датафреймы и переопределять словари ARCH_QLIKE, SEQ_LEN, … .
