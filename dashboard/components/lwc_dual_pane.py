"""Две панели Lightweight Charts + синхронный крест (TradingView-style) через components.html."""

from __future__ import annotations

import json
from html import escape as html_escape

import pandas as pd
import streamlit.components.v1 as components

# LWC 4.x — API addCandlestickSeries / subscribeCrosshairMove
_LWC_CDN = "https://unpkg.com/lightweight-charts@4.2.1/dist/lightweight-charts.standalone.production.js"

_TZ_IANA = {
    "UTC": "UTC",
    "MSK (UTC+3)": "Europe/Moscow",
    "EKB (UTC+5)": "Asia/Yekaterinburg",
}


def _ts_unix(row_ts) -> int:
    if hasattr(row_ts, "timestamp"):
        return int(row_ts.timestamp())
    return int(pd.Timestamp(row_ts).timestamp())


def _ts_utc_series(s: pd.Series) -> pd.Series:
    """Привести метки времени к UTC (aware) для merge_asof."""
    t = pd.to_datetime(s)
    if t.dt.tz is None:
        return t.dt.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")
    return t.dt.tz_convert("UTC")


def _floor_5m_utc(s: pd.Series) -> pd.Series:
    """Ключ 5m-свечи в UTC (как у bars_5m / predictions)."""
    return _ts_utc_series(s).dt.floor("5min")


# Окно merge_asof: слишком узкое даёт пустой нижний график; слишком широкое — «залипание» старых
# значений (одна устаревшая точка тянется константой через все бары и маскирует факт того, что
# predictions/rv_actual давно не обновлялись). 30 минут = 6 баров — допускаем мелкие опоздания
# воркеров, но не позволяем «протащить» вчерашнее значение через 24 часа.
_ASOF_TOLERANCE = pd.Timedelta("30min")


def render_tv_price_rv(
    bars_df: pd.DataFrame,
    rv_pred_df: pd.DataFrame,
    rv_actual_df: pd.DataFrame,
    *,
    selected_horizon: str,
    n_bars: int,
    sma_period: int,
    tz_display: str,
    component_key: str,
) -> None:
    """Свечи (верх) + RV pred / actual / SMA (низ), общий вертикальный крест."""
    if bars_df.empty:
        return

    b = bars_df.tail(n_bars).copy()
    b["ts"] = _ts_utc_series(b["ts"])
    b = b.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    p = rv_pred_df.copy() if not rv_pred_df.empty else pd.DataFrame()
    a = rv_actual_df.copy() if not rv_actual_df.empty else pd.DataFrame()
    if not p.empty:
        p["ts"] = _ts_utc_series(p["ts"])
    if not a.empty:
        a["ts"] = _ts_utc_series(a["ts"])

    merged = b.copy()
    merged["_bk"] = _floor_5m_utc(merged["ts"])

    # 1) Точное совпадение по 5m-бакету UTC
    if not p.empty:
        pq = p[[selected_horizon]].copy()
        pq["_bk"] = _floor_5m_utc(p["ts"])
        pq = pq.rename(columns={selected_horizon: "rv_pred"}).sort_values("_bk")
        pq = pq.drop_duplicates(subset=["_bk"], keep="last")
        merged = merged.merge(pq[["_bk", "rv_pred"]], on="_bk", how="left")
    else:
        merged["rv_pred"] = float("nan")

    if not a.empty:
        aq = a[[selected_horizon]].copy()
        aq["_bk"] = _floor_5m_utc(a["ts"])
        aq = aq.rename(columns={selected_horizon: "rv_act"}).sort_values("_bk")
        aq = aq.drop_duplicates(subset=["_bk"], keep="last")
        merged = merged.merge(aq[["_bk", "rv_act"]], on="_bk", how="left")
    else:
        merged["rv_act"] = float("nan")

    merged = merged.drop(columns=["_bk"], errors="ignore")

    # 2) merge_asof: индекс левой таблицы после sort без reset может не совпасть с результатом combine_first → NaN.
    #    Сбрасываем индекс, мержим на отсортированном кадре, затем fillna/combine_first по позициям.
    ms = merged.sort_values("ts").reset_index(drop=True)

    if not p.empty:
        pr = (
            p[["ts", selected_horizon]]
            .sort_values("ts")
            .drop_duplicates(subset=["ts"], keep="last")
            .rename(columns={selected_horizon: "_rv_asof"})
        )
        m2 = pd.merge_asof(
            ms,
            pr,
            on="ts",
            direction="backward",
            tolerance=_ASOF_TOLERANCE,
        )
        ms["rv_pred"] = ms["rv_pred"].combine_first(m2["_rv_asof"])
        # Никаких fallback без tolerance: иначе одна устаревшая точка из глубокой истории
        # «протягивается» по всем барам и маскирует, что воркер давно не пишет предсказания.
        # Убираем типичный мусорный уровень «~1» на фоне RV порядка 1e-3…1e-2
        ms.loc[ms["rv_pred"] >= 0.98, "rv_pred"] = float("nan")

    if not a.empty:
        ar = (
            a[["ts", selected_horizon]]
            .sort_values("ts")
            .drop_duplicates(subset=["ts"], keep="last")
            .rename(columns={selected_horizon: "_ra_asof"})
        )
        m3 = pd.merge_asof(
            ms,
            ar,
            on="ts",
            direction="backward",
            tolerance=_ASOF_TOLERANCE,
        )
        ms["rv_act"] = ms["rv_act"].combine_first(m3["_ra_asof"])
        ms.loc[ms["rv_act"] >= 0.98, "rv_act"] = float("nan")

    merged = ms.sort_values("ts")
    merged["rv_pred"] = pd.to_numeric(merged["rv_pred"], errors="coerce")
    merged["rv_act"] = pd.to_numeric(merged["rv_act"], errors="coerce")
    merged["rv_sma"] = merged["rv_pred"].rolling(window=sma_period, min_periods=1).mean()

    candles: list[dict] = []
    pred_pts: list[dict] = []
    act_pts: list[dict] = []
    sma_pts: list[dict] = []
    by_time: dict[str, dict] = {}

    for _, row in merged.iterrows():
        t = _ts_unix(row["ts"])
        tk = str(t)
        candles.append({
            "time": t,
            "open": float(row["open_perp"]),
            "high": float(row["high_perp"]),
            "low": float(row["low_perp"]),
            "close": float(row["close_perp"]),
        })
        entry: dict = {"c": float(row["close_perp"])}
        if pd.notna(row.get("rv_pred")):
            pv = float(row["rv_pred"])
            pred_pts.append({"time": t, "value": pv})
            entry["p"] = pv
        if pd.notna(row.get("rv_act")):
            av = float(row["rv_act"])
            act_pts.append({"time": t, "value": av})
            entry["a"] = av
        if pd.notna(row.get("rv_sma")):
            sv = float(row["rv_sma"])
            sma_pts.append({"time": t, "value": sv})
            entry["s"] = sv
        by_time[tk] = entry

    payload = {
        "candles": candles,
        "pred": pred_pts,
        "actual": act_pts,
        "sma": sma_pts,
        "byTime": by_time,
        "tzIana": _TZ_IANA.get(tz_display, "UTC"),
        "smaPeriod": int(sma_period),
    }
    json_str = json.dumps(payload, separators=(",", ":"))
    # Экранируем для встраивания в inline-script
    json_safe = json_str.replace("</", "<\\/")

    _sid = html_escape(str(component_key), quote=True)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<script src="{_LWC_CDN}"></script>
<style>
  html, body {{ margin: 0; padding: 0; background: #131722; overflow: hidden; font-family: system-ui, sans-serif; }}
  #p {{ height: 380px; width: 100%; }}
  #v {{ height: 220px; width: 100%; border-top: 1px solid #2a2e39; }}
</style></head>
<body><!--chart:{_sid}-->
<div id="p"></div>
<div id="v"></div>
<script>
const DATA = {json_safe};
const L = window.LightweightCharts;
const dashed = 2;
const crossMode = (L.CrosshairMode !== undefined) ? L.CrosshairMode.Normal : 0;
const cross = {{
  mode: crossMode,
  vertLine: {{ color: '#758696', width: 1, style: dashed, labelBackgroundColor: '#363a45' }},
  horzLine: {{ color: '#758696', width: 1, style: dashed, labelBackgroundColor: '#363a45' }},
}};
const chartOpts = (height, timeVisible) => ({{
  height,
  layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
  grid: {{ vertLines: {{ color: '#2a2e39' }}, horzLines: {{ color: '#2a2e39' }} }},
  crosshair: cross,
  rightPriceScale: {{ borderColor: '#2a2e39' }},
  timeScale: {{
    borderColor: '#2a2e39',
    timeVisible: true,
    secondsVisible: false,
    visible: timeVisible,
  }},
  localization: {{
    locale: 'ru-RU',
    timeFormatter: (t) => {{
      if (t === undefined || t === null) return '';
      const sec = (typeof t === 'number') ? t : (t.timestamp !== undefined ? t.timestamp : null);
      if (sec === null) return '';
      const d = new Date(sec * 1000);
      try {{
        return d.toLocaleString('ru-RU', {{
          timeZone: DATA.tzIana,
          day: '2-digit', month: 'short',
          hour: '2-digit', minute: '2-digit', hour12: false
        }});
      }} catch (e) {{
        return d.toUTCString().slice(5, 22);
      }}
    }},
  }},
}});

const elP = document.getElementById('p');
const elV = document.getElementById('v');
const chartP = L.createChart(elP, chartOpts(380, false));
const chartV = L.createChart(elV, chartOpts(220, true));

const candle = chartP.addCandlestickSeries({{
  upColor: '#26a69a', downColor: '#ef5350',
  borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350',
}});

// RV порядка 1e-3…1e-2 — дефолтный priceFormat (precision=2, minMove=0.01)
// показывает всё как "0.00" и квантует тики в нули. Нужно 6 знаков и шаг 1e-6.
const rvPriceFormat = {{ type: 'price', precision: 6, minMove: 0.000001 }};

const pred = chartV.addAreaSeries({{
  lineColor: '#4fc3f7', topColor: 'rgba(79,195,247,0.25)', bottomColor: 'rgba(79,195,247,0.05)',
  lineWidth: 2, title: 'Predicted RV',
  priceFormat: rvPriceFormat,
  priceLineVisible: false, lastValueVisible: true,
}});
const act = chartV.addLineSeries({{
  color: '#ab47bc', lineWidth: 2, title: 'Actual RV',
  priceFormat: rvPriceFormat,
  priceLineVisible: false, lastValueVisible: true,
}});
const sma = chartV.addLineSeries({{
  color: '#ffeb3b', lineWidth: 2, lineStyle: 0, title: 'SMA(' + DATA.smaPeriod + ')',
  priceFormat: rvPriceFormat,
  priceLineVisible: false, lastValueVisible: true,
}});

candle.setData(DATA.candles);
pred.setData(DATA.pred || []);
act.setData(DATA.actual || []);
sma.setData(DATA.sma || []);

chartP.timeScale().fitContent();
chartV.timeScale().fitContent();
try {{
  chartV.priceScale('right').applyOptions({{
    autoScale: true,
    scaleMargins: {{ top: 0.1, bottom: 0.1 }},
  }});
}} catch (e) {{}}

function timeKey(t) {{
  if (t === undefined || t === null) return null;
  if (typeof t === 'number') return String(t);
  if (typeof t === 'object' && t.timestamp !== undefined) return String(t.timestamp);
  return null;
}}

let isSyncing = false;
function withSync(fn) {{
  if (isSyncing) return;
  isSyncing = true;
  try {{ fn(); }} finally {{ isSyncing = false; }}
}}

function syncFromPrice(param) {{
  const k = timeKey(param.time);
  if (!k || !DATA.byTime[k]) {{
    chartV.clearCrosshairPosition();
    return;
  }}
  const row = DATA.byTime[k];
  if (row.p !== undefined) {{
    chartV.setCrosshairPosition(row.p, param.time, pred);
  }} else if (row.a !== undefined) {{
    chartV.setCrosshairPosition(row.a, param.time, act);
  }} else if (row.s !== undefined) {{
    chartV.setCrosshairPosition(row.s, param.time, sma);
  }} else {{
    chartV.clearCrosshairPosition();
  }}
}}

function syncFromVol(param) {{
  const k = timeKey(param.time);
  if (!k || !DATA.byTime[k]) {{
    chartP.clearCrosshairPosition();
    return;
  }}
  const row = DATA.byTime[k];
  chartP.setCrosshairPosition(row.c, param.time, candle);
}}

chartP.subscribeCrosshairMove((param) => withSync(() => syncFromPrice(param)));
chartV.subscribeCrosshairMove((param) => withSync(() => syncFromVol(param)));

let rangeSync = false;
function linkRange(a, b) {{
  a.timeScale().subscribeVisibleLogicalRangeChange((range) => {{
    if (rangeSync || !range) return;
    rangeSync = true;
    try {{ b.timeScale().setVisibleLogicalRange(range); }} catch (e) {{}}
    finally {{ rangeSync = false; }}
  }});
}}
linkRange(chartP, chartV);
linkRange(chartV, chartP);

window.addEventListener('resize', () => {{
  chartP.applyOptions({{ width: elP.clientWidth }});
  chartV.applyOptions({{ width: elV.clientWidth }});
}});
chartP.applyOptions({{ width: elP.clientWidth }});
chartV.applyOptions({{ width: elV.clientWidth }});
</script>
</body></html>"""

    components.html(html, height=610, scrolling=False)
