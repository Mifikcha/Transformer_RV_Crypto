# RV Forecast Dashboard

Streamlit-дэшборд для мониторинга прогнозов реализованной волатильности.

## Стек

- **Streamlit** — UI-фреймворк
- **streamlit-lightweight-charts** — свечной график (TradingView Lightweight Charts)
- **Plotly** — аналитические графики
- **SQLAlchemy + psycopg2** — синхронное подключение к PostgreSQL
- **pandas** — обработка данных

## Предварительные требования

### 1. БД — добавить колонки rv_48bar / rv_288bar

Таблицы `predictions` и `rv_actual` сейчас хранят только `rv_3bar`, `rv_12bar`.
Для отображения 4 горизонтов выполните в PostgreSQL:

```sql
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS rv_48bar FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS rv_288bar FLOAT;

ALTER TABLE rv_actual ADD COLUMN IF NOT EXISTS rv_48bar FLOAT;
ALTER TABLE rv_actual ADD COLUMN IF NOT EXISTS rv_288bar FLOAT;
```

### 2. .env

Дэшборд читает `../.env` (на уровень выше `dashboard/`).
Убедитесь что `DATABASE_URL` указывает на рабочую базу:

```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/rv_bot
```

Драйвер автоматически заменяется на `psycopg2` для синхронного подключения.

## Установка

```bash
cd dashboard
pip install -r requirements.txt
```

## Запуск

```bash
cd dashboard
streamlit run app.py --server.port 8501
```

Откройте браузер: [http://localhost:8501](http://localhost:8501)

## Запуск через Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY dashboard/ .
COPY .env ../.env
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

```bash
docker build -t rv-dashboard -f Dockerfile.dashboard .
docker run -p 8501:8501 --env-file .env rv-dashboard
```

## Структура

```
dashboard/
├── app.py                    # Точка входа (навигация, автообновление)
├── config.py                 # Настройки (читает .env)
├── db.py                     # SQLAlchemy sync engine, query_df()
├── queries.py                # SQL-запросы всех блоков
├── pages/
│   ├── overview.py           # Price+RV, Term Structure, Regime
│   ├── analytics.py          # Model Performance, Market Heat, Cross-Asset
│   └── alerts.py             # История алертов
├── components/
│   ├── price_chart.py        # Lightweight Charts (свечи + RV overlay)
│   ├── term_structure.py     # Кривая волатильности по горизонтам
│   ├── regime_gauge.py       # Классификация режима + timeline
│   ├── performance.py        # Rolling R², bias, scatter
│   ├── heatmap.py            # Market heat + cross-asset корреляция
│   └── styles.py             # CSS, цвета, inject_styles()
└── requirements.txt
```

## Страницы

### Overview
- **Блок 1** — Свечной график + RV overlay (predicted / actual) через Lightweight Charts
- **Блок 2** — Volatility Term Structure (кривая по 4 горизонтам + медиана 30д)
- **Блок 3** — Regime Classification (LOW / NORMAL / HIGH / EXTREME) + timeline 7д

### Analytics
- **Блок 4** — Model Performance: rolling R², rolling bias, scatter predicted vs actual
- **Блок 5** — Market Heat: RV Z-Score, Volume/Median, Funding Rate, RV acceleration
- **Блок 6** — Cross-Asset RV Correlation BTC/ETH (требует два источника данных)

### Alerts
- **Блок 7** — История spike-алертов из бота с фильтрами, экспортом в CSV

## Настройка multi-symbol

По умолчанию дэшборд использует одну таблицу `predictions` для обоих символов.
Для полноценного multi-symbol добавьте колонку `symbol`:

```sql
ALTER TABLE bars_5m     ADD COLUMN IF NOT EXISTS symbol TEXT DEFAULT 'BTCUSDT';
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS symbol TEXT DEFAULT 'BTCUSDT';
ALTER TABLE rv_actual   ADD COLUMN IF NOT EXISTS symbol TEXT DEFAULT 'BTCUSDT';
```

Затем обновите SQL-запросы в `queries.py`, добавив `WHERE symbol = '{symbol}'`.

## Автообновление

Включается чекбоксом в sidebar. Интервал: `REFRESH_INTERVAL_SEC` (по умолчанию 300 сек).
Данные кэшируются через `@st.cache_data(ttl=300)`.
