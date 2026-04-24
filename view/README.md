# RV Bot -- Руководство по запуску и тестированию

## Пререквизиты

- Python 3.11+
- Docker Desktop (для PostgreSQL / TimescaleDB)
- Обученная модель `fold_rv_4.pt` (или любой fold из `transformer/output/models/`)
- Файл `feature_selection/output/recommended_features.csv`
- Telegram-бот (токен от @BotFather)

---

## Фаза 0: Подготовка (одноразово)

### 0.1 Обучить модель (если ещё не обучена)

Модель и фичи **gitignored** (`transformer/output/`, `feature_selection/output/`).
Если их нет на диске -- сначала запустить пайплайн:

```bash
# Из корня проекта:
python scripts/run_rv_pipeline.py
```

Или только нужные шаги:

```bash
# 1. Feature selection (создаёт recommended_features.csv)
python feature_selection/run_feature_selection.py

# 2. Обучение трансформера (создаёт fold_rv_*.pt)
python transformer/run_transformer.py --mode train-rv
```

После этого должны появиться:
- `transformer/output/models/fold_rv_4.pt` (или другой fold)
- `feature_selection/output/recommended_features.csv`

### 0.2 Установить зависимости

```bash
pip install -r requirements_live.txt
```

### 0.3 Создать Telegram-бота

1. Открыть @BotFather в Telegram
2. `/newbot` -- дать имя, получить токен
3. Создать канал/группу для прогнозов, добавить туда бота как администратора
4. Узнать chat_id канала (отправить сообщение, вызвать `https://api.telegram.org/bot<TOKEN>/getUpdates`)
5. Узнать свой user_id (через @userinfobot)

### 0.4 Создать `.env` файл

```bash
copy .env.example .env
```

Заполнить реальными значениями:

```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/rv_bot
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHANNEL_ID=-1001234567890
TELEGRAM_ALERT_CHAT_IDS=ваш_user_id
ALLOWED_CHAT_IDS=ваш_user_id
MODEL_PATH=transformer/output/models/fold_rv_4.pt
FEATURES_PATH=feature_selection/output/recommended_features.csv
```

---

## Фаза 1: Запуск базы данных

### 1.1 Поднять PostgreSQL/TimescaleDB через Docker

```bash
docker compose up db -d
```

### 1.2 Проверить что БД работает

```bash
docker compose exec db psql -U user -d rv_bot -c "SELECT version();"
```

Должна вернуться версия PostgreSQL 16.x.

---

## Фаза 2: Тестирование по частям (без Telegram)

Каждый шаг можно запускать отдельно для проверки.

### 2.1 Тест Feature Engine (самое важное)

Проверяет что фичи считаются так же, как в историческом датасете:

```python
# Сохранить как view/test_features.py и запустить:
# python -m view.test_features

import pandas as pd
import numpy as np
from view.feature_engine import FeatureEngine, FEATURE_COLS

df_hist = pd.read_csv(
    "target/btcusdt_5m_final_with_targets.csv", parse_dates=["ts"]
).sort_values("ts").reset_index(drop=True)

engine = FeatureEngine(buffer_size=7000, min_bars=250)
results = []
for i, row in df_hist.iterrows():
    engine.add_bar(row.to_dict())
    if len(engine.bars) >= 250 and len(results) < 500:
        feats = engine.compute_features()
        if feats is not None:
            results.append({"ts": row["ts"], **feats})
    if len(results) >= 500:
        break

if not results:
    print("FAIL -- compute_features() так и не вернул результат")
    exit(1)

df_engine = pd.DataFrame(results).set_index("ts")
df_hist_idx = df_hist.set_index("ts")
overlap = df_engine.index.intersection(df_hist_idx.index)

print(f"Проверяем {len(overlap)} точек совпадения...")

TOLERANCE = 1e-5
failed = []
for feat in FEATURE_COLS:
    if feat not in df_hist_idx.columns:
        print(f"  SKIP {feat} -- нет в историческом CSV")
        continue
    a = df_engine.loc[overlap, feat].astype(float)
    b = df_hist_idx.loc[overlap, feat].astype(float)
    mae = np.abs(a - b).mean()
    if mae > TOLERANCE:
        failed.append(f"  {feat}: MAE={mae:.2e}")

if failed:
    print("FAIL -- расхождения найдены:")
    print("\n".join(failed))
else:
    print(f"OK -- все фичи совпадают с историческим датасетом (tolerance={TOLERANCE})")
```

```bash
python -m view.test_features
```

Ожидаемый результат: `OK -- все фичи совпадают`.

### 2.2 Тест Bybit клиента (требует интернет)

```python
# python -c "..."
import asyncio
from view.bybit_client import BybitClient

async def test():
    client = BybitClient()
    bar = await client.fetch_latest_bar()
    print(f"ts={bar['ts']}, close={bar['close_perp']:.2f}, "
          f"funding={bar['funding_rate']:.6f}, OI={bar['open_interest']:.0f}")

asyncio.run(test())
```

### 2.3 Тест инференса (требует .pt модель)

```python
import numpy as np
import pandas as pd
from view.feature_engine import FeatureEngine
from view.inference import RVInference

df = pd.read_csv("target/btcusdt_5m_final_with_targets.csv",
                  parse_dates=["ts"]).sort_values("ts").tail(7000)

engine = FeatureEngine()
for _, row in df.iterrows():
    engine.add_bar(row.to_dict())

window = engine.get_window(240)
har = engine.compute_har_context()
if har is None:
    har = np.zeros(6)

inf = RVInference("transformer/output/models/fold_rv_4.pt")
result = inf.predict(window, har)
print(f"rv_3bar={result['rv_3bar']:.6f}, rv_12bar={result['rv_12bar']:.6f}")
assert 1e-6 < result["rv_3bar"] < 0.1
assert 1e-6 < result["rv_12bar"] < 0.1
print("OK")
```

### 2.4 Тест создания таблиц

```python
import asyncio
from view.config import Settings
from view.db import build_engine, init_db

async def test():
    s = Settings()
    engine = build_engine(s.database_url)
    await init_db(engine)
    print("OK -- таблицы созданы")
    await engine.dispose()

asyncio.run(test())
```

---

## Фаза 3: Полный запуск

### Вариант A: Всё в одном процессе (рекомендуется для начала)

```bash
python -m view.main
```

Что произойдёт:
1. Подключение к PostgreSQL, создание таблиц
2. Если в `bars_5m` < 250 строк -- **bootstrap**: загрузка ~7200 баров с Bybit (~3-5 мин)
3. Запуск планировщика: каждые 5 минут ingestion + prediction + notification
4. Запуск Telegram-бота (polling)

В логах должно появиться:
```
INFO  view.db: Database tables created/verified.
INFO  view.ingestion_worker: Bootstrap: fetching ~7200 bars...
INFO  view.inference: Model loaded: patch_encoder, features=32, horizons=2
INFO  view.main: Scheduler started: ingest=300s, predict=300s, notify=300s
INFO  view.main: Starting Telegram bot polling...
```

### Вариант B: Через Docker Compose (продакшен)

```bash
docker compose up -d
```

Поднимет 5 контейнеров: db, ingestion, predictor, notifier, bot.

---

## Фаза 4: Проверка что всё работает

### 4.1 Проверить данные в БД

```bash
docker compose exec db psql -U user -d rv_bot
```

```sql
-- Сколько баров загружено?
SELECT COUNT(*) FROM bars_5m;

-- Последний бар?
SELECT ts, close_perp FROM bars_5m ORDER BY ts DESC LIMIT 1;

-- Есть ли прогнозы?
SELECT ts, rv_3bar, rv_12bar FROM predictions ORDER BY created_at DESC LIMIT 5;

-- Фактическая RV?
SELECT * FROM rv_actual ORDER BY ts DESC LIMIT 5;

-- Отправлены ли уведомления?
SELECT alert_type, sent_at FROM notification_log ORDER BY sent_at DESC LIMIT 5;
```

### 4.2 Проверить Telegram-бота

Открыть чат с ботом и отправить команды:

| Команда | Что проверяет |
|---------|---------------|
| `/rv` | Последний прогноз, режим волатильности |
| `/data` | Статус данных, задержка последнего бара |
| `/model` | Версия модели |
| `/history 24h` | Прогнозы за последние 24 часа vs факт |
| `/accuracy` | R² и MAE на живых данных vs train best |
| `/regime` | Текущий режим + распределение за 7 дней |

### 4.3 Проверить что чужие не имеют доступа

Попросить кого-то (чей user_id НЕ в `ALLOWED_CHAT_IDS`) отправить `/rv` боту.
Должен получить ответ "Нет доступа".

---

## Порядок отладки проблем

| Симптом | Что проверить |
|---------|---------------|
| `Connection refused` при старте | Docker с БД запущен? `docker compose ps` |
| `No module named 'view'` | Запускать из корня проекта |
| `FileNotFoundError: fold_rv_4.pt` | Модель обучена? Путь в `.env` верный? |
| Bootstrap висит | Bybit API доступен? Проверить VPN/прокси |
| Прогнозы = 0 или NaN | Feature Engine расходится -- запустить тест 2.1 |
| Бот не отвечает | Токен верный? Бот добавлен в чат? ALLOWED_CHAT_IDS содержит ваш ID? |
| `/accuracy` говорит "Недостаточно данных" | Нужно подождать пока накопится >= 10 пар прогноз/факт |

---

## Структура файлов

```
project/
├── .env                    ← ваши настройки (создать из .env.example)
├── .env.example            ← шаблон
├── docker-compose.yml      ← PostgreSQL + сервисы
├── requirements_live.txt   ← зависимости
└── view/
    ├── __init__.py
    ├── config.py           ← настройки из .env
    ├── db.py               ← подключение к PostgreSQL
    ├── models.py           ← таблицы: bars_5m, predictions, rv_actual, notification_log
    ├── bybit_client.py     ← HTTP-клиент Bybit API
    ├── feature_engine.py   ← вычисление 32 фич (критичный модуль)
    ├── inference.py        ← загрузка модели и инференс
    ├── ingestion_worker.py ← загрузка баров с Bybit в БД
    ├── prediction_worker.py← фичи + инференс + запись прогноза
    ├── notification_worker.py ← отправка в Telegram
    ├── bot.py              ← Telegram-бот с командами
    └── main.py             ← точка входа (всё вместе)
```
