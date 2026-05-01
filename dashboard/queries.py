"""SQL-запросы для всех блоков дэшборда.

Используем str.format()-подстановку для строковых частей (INTERVAL, имена таблиц).
Числовые параметры — тоже через format, т.к. данные приходят из UI-контролов
(не из пользовательского ввода), поэтому SQL-инъекции исключены.
"""

QUERY_BARS = """
    SELECT ts, open_perp, high_perp, low_perp, close_perp,
           volume_perp, funding_rate, open_interest
    FROM bars_5m
    WHERE ts >= (
        SELECT COALESCE(MAX(ts), NOW()) - INTERVAL '{hours} hours'
        FROM bars_5m
    )
    ORDER BY ts
"""

QUERY_PREDICTIONS = """
    SELECT ts, rv_3bar, rv_12bar,
           rv_48bar, rv_288bar,
           model_ver, degraded
    FROM predictions
    WHERE ts >= (
        SELECT COALESCE(MAX(ts), NOW()) - INTERVAL '{hours} hours'
        FROM predictions
    )
    ORDER BY ts
"""

QUERY_RV_ACTUAL = """
    SELECT ts, rv_3bar, rv_12bar,
           rv_48bar, rv_288bar
    FROM rv_actual
    WHERE ts >= (
        SELECT COALESCE(MAX(ts), NOW()) - INTERVAL '{hours} hours'
        FROM rv_actual
    )
    ORDER BY ts
"""

QUERY_TERM_STRUCTURE_HISTORY = """
    SELECT rv_3bar, rv_12bar, rv_48bar, rv_288bar
    FROM predictions
    WHERE ts >= (
        SELECT COALESCE(MAX(ts), NOW()) - INTERVAL '30 days'
        FROM predictions
    )
      AND rv_3bar IS NOT NULL
"""

QUERY_PRED_VS_ACTUAL = """
    SELECT p.ts,
           p.rv_3bar   AS pred_3bar,   a.rv_3bar   AS actual_3bar,
           p.rv_12bar  AS pred_12bar,  a.rv_12bar  AS actual_12bar,
           p.rv_48bar  AS pred_48bar,  a.rv_48bar  AS actual_48bar,
           p.rv_288bar AS pred_288bar, a.rv_288bar AS actual_288bar
    FROM predictions p
    JOIN rv_actual a ON a.ts = p.ts
    WHERE p.ts >= (
        SELECT COALESCE(MAX(ts), NOW()) - INTERVAL '{days} days'
        FROM predictions
    )
    ORDER BY p.ts
"""

QUERY_ALERTS = """
    SELECT n.sent_at, n.alert_type,
           p.ts      AS prediction_ts,
           p.rv_3bar AS pred_rv,
           a.rv_3bar AS actual_rv,
           p.model_ver
    FROM notification_log n
    JOIN predictions p ON p.id = n.prediction_id
    LEFT JOIN rv_actual a ON a.ts = p.ts
    WHERE n.sent_at >= (
        SELECT COALESCE(MAX(sent_at), NOW()) - INTERVAL '{days} days'
        FROM notification_log
    )
    ORDER BY n.sent_at DESC
    LIMIT 100
"""

QUERY_CROSS_ASSET_RV = """
    SELECT b.ts,
           b.rv_3bar AS btc_rv,
           e.rv_3bar AS eth_rv
    FROM {btc_table} b
    JOIN {eth_table} e ON e.ts = b.ts
    WHERE b.ts >= (
        SELECT COALESCE(MAX(ts), NOW()) - INTERVAL '7 days'
        FROM {btc_table}
    )
      AND b.rv_3bar IS NOT NULL
      AND e.rv_3bar IS NOT NULL
    ORDER BY b.ts
"""

QUERY_REGIME_HISTORY = """
    SELECT ts, rv_3bar
    FROM predictions
    WHERE ts >= (
        SELECT COALESCE(MAX(ts), NOW()) - INTERVAL '30 days'
        FROM predictions
    )
      AND rv_3bar IS NOT NULL
    ORDER BY ts
"""

QUERY_REGIME_TIMELINE = """
    SELECT ts, rv_3bar
    FROM predictions
    WHERE ts >= (
        SELECT COALESCE(MAX(ts), NOW()) - INTERVAL '7 days'
        FROM predictions
    )
      AND rv_3bar IS NOT NULL
    ORDER BY ts
"""
