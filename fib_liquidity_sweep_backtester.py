#!/usr/bin/env python3
"""
fib-liquidity-sweep-backtester  (v2)
Backtest de una estrategia intradía de retroceso Fibonacci + barrido de liquidez
(estilo SMC/ICT) sobre XAUUSD, con R neta de costes, MFE/MAE y validación
in-sample / out-of-sample.

REGLAS BASE (setup Fibo + Liquidez)
--------------------------------------
Contexto (HTF, por defecto 15m):
  - Swings = fractales de N barras a cada lado (por defecto 3), confirmados solo
    cuando han cerrado las N barras de la derecha (sin mirar al futuro).
  - Impulso activo = último swing alto y último swing bajo confirmados.
      * bajo posterior al alto -> impulso bajista A(alto)->B(bajo) -> CORTOS
      * alto posterior al bajo -> impulso alcista A(bajo)->B(alto) -> LARGOS
  - Tamaño mínimo del impulso: min_leg_atr x ATR(14) del HTF.
  - Se invalida si el precio vuelve más allá de A (100%).
Fibonacci (0 = B, 1 = A): el sweep debe alcanzar zone_min (0.618); TP en tp_ext (-0.382).
Gatillo 5m: la mecha barre un nivel de liquidez y el CUERPO cierra de vuelta dentro;
  por defecto la vela cierra a favor del trade.
Ejecución: entrada en la apertura de la vela siguiente; SL tras la mecha + sl_buffer.
  Si en una vela se tocan SL y TP se asume SL. Costes (spread + comisión) restados:
  la R es NETA y siempre sobre el stop inicial.

NOVEDADES v3 (multi-activo)
---------------------------
  --asset xau|btc|eth   presets de costes: oro en $ por onza; cripto en % del precio
                        (por defecto 0,10% ida y vuelta: comisión taker + deslizamiento)
  --ltf / --htf         temporalidades de entrada y de impulso (p.ej. 15min / 1h)
  --data a.csv b.csv    varios CSV se concatenan (p.ej. dos tramos de años)
  --ladder h1           escalera H1: sube temporalidad y añade tendencia; la tabla
                        incluye cuántos años salen positivos

PALANCAS (v2)
--------------------
  --max-cost-frac 0.1   solo trades donde coste <= 10% del riesgo (stop mínimo)
  --trend-tf 4h         solo a favor de la tendencia mayor (cierre vs EMA --trend-ema)
  --liquidity           fractal (defecto) | pdhl (máx/mín día anterior) |
                        asia (rango asiático) | key (pdhl + asia)
  --context none        ignora el impulso/Fibo: opera cualquier sweep de nivel clave
                        (necesita --tp-r)
  --tp-r 2              TP fijo en R en lugar del nivel Fibo
  --session 7-17        horas UTC permitidas para la vela del sweep

DIAGNÓSTICO
-----------
  - MFE/MAE de cada trade (en R) en trades.csv.
  - "Curva de TP": esperanza neta si el TP fuera k R, calculada dejando correr cada
    trade hasta su stop original. Indica dónde está el objetivo óptimo.
  - Monte Carlo (2000 reordenaciones) del drawdown máximo.
  - --ladder: aplica las palancas en orden y muestra IS / OOS de cada escalón.

USO
---
  python fib_liquidity_sweep_backtester.py --synthetic                      # prueba de humo
  python fib_liquidity_sweep_backtester.py --data xau_m1.csv --split 2025-07-01
  python fib_liquidity_sweep_backtester.py --data xau_m1.csv --split 2025-07-01 --ladder
  python fib_liquidity_sweep_backtester.py --data xau_m1.csv --split 2025-07-01 \
         --liquidity key --trend-tf 4h --max-cost-frac 0.1

Requisitos: pandas, numpy (matplotlib opcional).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

TP_CURVE_KS = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]


# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
@dataclass
class Config:
    htf: str = "15min"
    ltf: str = "5min"
    htf_fractal_n: int = 3
    ltf_fractal_n: int = 2
    zone_min: float = 0.618
    zone_max: float = 1.0
    tp_ext: float = -0.382
    tp_r: float | None = None         # TP fijo en R (anula tp_ext)
    min_leg_atr: float = 3.0
    atr_period: int = 14
    sl_buffer: float = 0.30
    require_body_dir: bool = True
    min_rr: float = 1.5
    max_rr: float = 40.0
    spread: float = 0.30
    commission: float = 0.07
    cost_pct: float | None = None     # coste ida y vuelta en % del precio (anula spread+commission)
    sl_buffer_pct: float | None = None  # colchón del SL en % del precio (anula sl_buffer)
    max_cost_frac: float | None = None  # coste / riesgo máximo admitido
    be_at_r: float | None = None
    session: tuple = (0, 24)
    max_bars_in_trade: int = 288
    trades_per_leg: int = 1
    context: str = "fib"              # fib | none
    liquidity: str = "fractal"        # fractal | pdhl | asia | key
    trend_tf: str | None = None       # p.ej. "4h", "1D"
    trend_ema: int = 50
    day_roll_utc: int = 22            # cambio de día de trading (cierre NY aprox.)
    asia_hours: int = 9               # 22:00-07:00 UTC


# -----------------------------------------------------------------------------
# Datos
# -----------------------------------------------------------------------------
def load_csv(path: str, data_tz: str = "UTC", fmt: str = "auto") -> pd.DataFrame:
    if fmt == "histdata":
        df = pd.read_csv(path, sep=";", header=None,
                         names=["dt", "open", "high", "low", "close", "volume"])
        ts = pd.to_datetime(df["dt"].astype(str), format="%Y%m%d %H%M%S")
        if data_tz == "UTC":
            data_tz = "Etc/GMT+5"
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            head = fh.readline()
        sep = max([",", ";", "\t"], key=head.count)
        df = pd.read_csv(path, sep=sep)
        df.columns = [str(c).strip().strip("<>").strip().lower() for c in df.columns]
        cols = set(df.columns)
        if {"date", "time"} <= cols:
            ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        else:
            tcol = next((c for c in ["timestamp", "datetime", "time", "date",
                                     "gmt time", "local time", "time (utc)"] if c in cols), None)
            if tcol is None:
                raise ValueError(f"No encuentro columna de tiempo en {list(df.columns)}")
            raw = df[tcol]
            if pd.api.types.is_numeric_dtype(raw):
                ts = pd.to_datetime(raw, unit="ms" if raw.iloc[0] > 1e11 else "s")
            elif re.match(r"^\d{2}\.\d{2}\.\d{4}", str(raw.iloc[0])):
                ts = pd.to_datetime(raw, format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")
                if ts.isna().all():
                    ts = pd.to_datetime(raw, dayfirst=True, errors="coerce")
            else:
                ts = pd.to_datetime(raw, errors="coerce")
        for c in ["open", "high", "low", "close"]:
            if c not in df.columns:
                raise ValueError(f"Falta la columna {c}; columnas: {list(df.columns)}")

    ts = pd.Series(pd.DatetimeIndex(ts))
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    elif data_tz and data_tz.upper() != "UTC":
        ts = (ts.dt.tz_localize(data_tz, ambiguous="NaT", nonexistent="shift_forward")
                .dt.tz_convert("UTC").dt.tz_localize(None))
    out = pd.DataFrame({k: pd.to_numeric(df[k], errors="coerce").values
                        for k in ["open", "high", "low", "close"]},
                       index=pd.DatetimeIndex(ts.values, name="time"))
    out = out[~out.index.isna()].dropna()
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out[out["high"] > out["low"]]


def synthetic_data(days: int = 500, seed: int = 7) -> pd.DataFrame:
    """Paseo aleatorio M1 con colas gruesas y vol intradía: sin ventaja posible."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 1440, freq="1min")
    idx = idx[idx.dayofweek < 5]
    hrs = idx.hour.values
    vol = np.where((hrs >= 7) & (hrs < 17), 0.65, 0.35)
    close = 2000 + np.cumsum(rng.standard_t(4, len(idx)) * vol / np.sqrt(2))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.standard_t(4, len(idx))) * vol * 0.4
    low = np.minimum(open_, close) - np.abs(rng.standard_t(4, len(idx))) * vol * 0.4
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                        index=pd.DatetimeIndex(idx, name="time"))


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (df.resample(rule, label="left", closed="left")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
              .dropna())


# -----------------------------------------------------------------------------
# Indicadores y niveles
# -----------------------------------------------------------------------------
def fractals(high, low, n):
    h, l = pd.Series(high), pd.Series(low)
    fh = ((h > h.shift(1).rolling(n).max()) &
          (h >= h[::-1].shift(1).rolling(n).max()[::-1])).fillna(False).values
    fl = ((l < l.shift(1).rolling(n).min()) &
          (l <= l[::-1].shift(1).rolling(n).min()[::-1])).fillna(False).values
    return fh, fl


def atr(df, period):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().values


def trend_array(df1: pd.DataFrame, ltf_index: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    """+1 / -1 según cierre de la última vela CERRADA del trend_tf vs su EMA."""
    if not cfg.trend_tf:
        return np.zeros(len(ltf_index))
    t = resample(df1, cfg.trend_tf)
    ema = t["close"].ewm(span=cfg.trend_ema, adjust=False).mean()
    sign = np.sign(t["close"] - ema)
    sign[: cfg.trend_ema] = 0  # sin datos suficientes
    known = pd.Series(sign.values, index=t.index + pd.Timedelta(cfg.trend_tf))
    known = known[~known.index.duplicated(keep="last")]
    return known.reindex(ltf_index, method="ffill").fillna(0).values


def key_levels(ltf: pd.DataFrame, cfg: Config) -> dict:
    """Niveles de liquidez 'reales' y si siguen intactos antes de cada vela."""
    ti = ltf.index
    h, l = ltf["high"], ltf["low"]
    shifted = ti + pd.Timedelta(hours=24 - cfg.day_roll_utc)
    day = pd.Series(shifted.normalize(), index=ti)
    hsd = (shifted - shifted.normalize()).total_seconds().values / 3600  # horas desde inicio
    g_h, g_l = h.groupby(day.values), l.groupby(day.values)

    dhi, dlo = g_h.max(), g_l.min()
    pdh = day.map(dhi.shift(1)).values
    pdl = day.map(dlo.shift(1)).values
    run_hi = g_h.cummax().groupby(day.values).shift(1).fillna(-np.inf).values
    run_lo = g_l.cummin().groupby(day.values).shift(1).fillna(np.inf).values

    asia = hsd < cfg.asia_hours
    ahi = h.where(asia).groupby(day.values).max()
    alo = l.where(asia).groupby(day.values).min()
    asia_hi = np.where(~asia, day.map(ahi).values, np.nan)
    asia_lo = np.where(~asia, day.map(alo).values, np.nan)
    post_h = h.where(~asia, -np.inf)
    post_l = l.where(~asia, np.inf)
    run_hi_post = post_h.groupby(day.values).cummax().groupby(day.values).shift(1).fillna(-np.inf).values
    run_lo_post = post_l.groupby(day.values).cummin().groupby(day.values).shift(1).fillna(np.inf).values

    return {
        "pdh": np.where(run_hi < pdh, pdh, np.nan),         # aún no tocado hoy
        "pdl": np.where(run_lo > pdl, pdl, np.nan),
        "asia_hi": np.where(run_hi_post < asia_hi, asia_hi, np.nan),
        "asia_lo": np.where(run_lo_post > asia_lo, asia_lo, np.nan),
    }


# -----------------------------------------------------------------------------
# Motor
# -----------------------------------------------------------------------------
def backtest(ltf: pd.DataFrame, htf: pd.DataFrame, cfg: Config,
             trend: np.ndarray | None = None) -> pd.DataFrame:
    if cfg.context == "none" and cfg.tp_r is None:
        raise ValueError("--context none necesita --tp-r")
    times = ltf.index.values
    o, h, l, c = (ltf[k].values for k in ["open", "high", "low", "close"])
    N = len(ltf)
    trend = np.zeros(N) if trend is None else trend
    hours = pd.DatetimeIndex(times).hour.values
    s0, s1 = cfg.session

    # swings HTF -> eventos con hora de confirmación
    n = cfg.htf_fractal_n
    hfh, hfl = fractals(htf["high"].values, htf["low"].values, n)
    hatr = atr(htf, cfg.atr_period)
    hidx, freq = htf.index, pd.Timedelta(cfg.htf)
    events = []
    for i in np.where(hfh)[0]:
        if i + n < len(htf):
            events.append((hidx[i + n] + freq, "H", htf["high"].values[i], hidx[i], hatr[i]))
    for i in np.where(hfl)[0]:
        if i + n < len(htf):
            events.append((hidx[i + n] + freq, "L", htf["low"].values[i], hidx[i], hatr[i]))
    events.sort(key=lambda e: e[0])
    ev_times = np.array([np.datetime64(e[0]) for e in events])

    # fractales LTF (último confirmado utilizable)
    m = cfg.ltf_fractal_n
    lfh, lfl = fractals(h, l, m)
    fh_px, fh_ix, fl_px, fl_ix = (np.full(N, np.nan) for _ in range(4))
    for i in np.where(lfh)[0]:
        if i + m + 1 < N:
            fh_px[i + m + 1], fh_ix[i + m + 1] = h[i], i
    for i in np.where(lfl)[0]:
        if i + m + 1 < N:
            fl_px[i + m + 1], fl_ix[i + m + 1] = l[i], i
    fh_px, fh_ix, fl_px, fl_ix = (pd.Series(a).ffill().values for a in (fh_px, fh_ix, fl_px, fl_ix))

    kl = key_levels(ltf, cfg) if cfg.liquidity != "fractal" else None
    use_pd = cfg.liquidity in ("pdhl", "key")
    use_asia = cfg.liquidity in ("asia", "key")

    def sweep(d, t, b_ix):
        """¿Barre la vela t un nivel de liquidez y cierra dentro? -> (ok, nivel, tipo)"""
        if cfg.require_body_dir and ((d == -1 and c[t] >= o[t]) or (d == 1 and c[t] <= o[t])):
            return False, None, None
        cands = []
        if cfg.liquidity == "fractal":
            px, ix = (fh_px[t], fh_ix[t]) if d == -1 else (fl_px[t], fl_ix[t])
            if not np.isnan(px) and (b_ix is None or ix > b_ix):
                cands.append((px, "fractal"))
        else:
            if use_pd:
                cands.append((kl["pdh"][t] if d == -1 else kl["pdl"][t], "PDH" if d == -1 else "PDL"))
            if use_asia:
                cands.append((kl["asia_hi"][t] if d == -1 else kl["asia_lo"][t],
                              "AsiaHi" if d == -1 else "AsiaLo"))
        for lv, kind in cands:
            if lv is None or np.isnan(lv):
                continue
            if d == -1 and h[t] > lv and c[t] < lv:
                return True, lv, kind
            if d == 1 and l[t] < lv and c[t] > lv:
                return True, lv, kind
        return False, None, None

    ev_ptr = 0
    SH = SL = None
    leg_key, run_ext, b_ix = None, None, 0
    leg_count: dict = {}
    busy_until = -1
    trades = []

    for t in range(N - 1):
        ts = times[t]
        while ev_ptr < len(events) and ev_times[ev_ptr] <= ts:
            e = events[ev_ptr]
            if e[1] == "H":
                SH = e
            else:
                SL = e
            ev_ptr += 1

        leg_ok = False
        if SH is not None and SL is not None:
            if SL[3] > SH[3]:
                d_leg, A, B = -1, SH, SL
            else:
                d_leg, A, B = 1, SL, SH
            A_p, B_p, B_atr = A[2], B[2], B[4]
            key = (A[3], B[3])
            if key != leg_key:
                leg_key = key
                w0 = np.searchsorted(times, np.datetime64(B[3]))
                w1 = max(np.searchsorted(times, np.datetime64(B[3] + freq)), w0 + 1)
                seg = l[w0:w1] if d_leg == -1 else h[w0:w1]
                b_ix = w0 + (int(np.argmin(seg)) if d_leg == -1 else int(np.argmax(seg))) if len(seg) else w0
                if t > b_ix + 1:
                    run_ext = h[b_ix + 1:t].max() if d_leg == -1 else l[b_ix + 1:t].min()
                else:
                    run_ext = -np.inf if d_leg == -1 else np.inf
            fib = lambda x: B_p + x * (A_p - B_p)
            leg_ok = True

        if t > busy_until and s0 <= hours[t] < s1:
            cand = None  # (d, nivel, tipo, tp)
            if cfg.context == "fib" and leg_ok:
                zone_px, inval_px = fib(cfg.zone_min), fib(cfg.zone_max)
                in_zone = ((h[t] >= zone_px and h[t] < inval_px and run_ext < inval_px) if d_leg == -1
                           else (l[t] <= zone_px and l[t] > inval_px and run_ext > inval_px))
                if (in_zone and leg_count.get(leg_key, 0) < cfg.trades_per_leg
                        and not np.isnan(B_atr) and abs(A_p - B_p) >= cfg.min_leg_atr * B_atr):
                    ok, lv, kind = sweep(d_leg, t, b_ix if cfg.liquidity == "fractal" else None)
                    if ok:
                        cand = (d_leg, lv, kind, fib(cfg.tp_ext))
            elif cfg.context == "none":
                for d in (-1, 1):
                    ok, lv, kind = sweep(d, t, None)
                    if ok:
                        cand = (d, lv, kind, None)
                        break

            if cand is not None:
                d, lv, kind, tp = cand
                if trend[t] == 0 and cfg.trend_tf:
                    cand = None
                elif cfg.trend_tf and trend[t] != d:
                    cand = None
            if cand is not None:
                entry = o[t + 1]
                cost = (entry * cfg.cost_pct / 100) if cfg.cost_pct is not None else (cfg.spread + cfg.commission)
                buf = (entry * cfg.sl_buffer_pct / 100) if cfg.sl_buffer_pct is not None else cfg.sl_buffer
                sl = h[t] + buf if d == -1 else l[t] - buf
                risk = abs(sl - entry)
                if cfg.tp_r is not None:
                    tp = entry + d * cfg.tp_r * risk
                ok_side = (sl > entry > tp) if d == -1 else (sl < entry < tp)
                rr = abs(tp - entry) / risk if risk > 0 else 0
                cost_ok = cfg.max_cost_frac is None or (risk > 0 and cost / risk <= cfg.max_cost_frac)
                rr_ok = cfg.tp_r is not None or (cfg.min_rr <= rr <= cfg.max_rr)
                if ok_side and rr_ok and cost_ok:
                    sim = _simulate(d, entry, sl, tp, t + 1, o, h, l, c, N, cfg)
                    free = _free_run(d, entry, sl, t + 1, o, h, l, c, N, cfg)
                    j_exit, px_exit, reason, mfe, mae = sim
                    R = (d * (px_exit - entry) - cost) / risk
                    trades.append({
                        "signal_time": pd.Timestamp(ts), "entry_time": pd.Timestamp(times[t + 1]),
                        "exit_time": pd.Timestamp(times[j_exit]),
                        "dir": "LONG" if d == 1 else "SHORT", "liq": kind,
                        "entry": round(entry, 3), "sl": round(sl, 3), "tp": round(tp, 3),
                        "exit": round(px_exit, 3), "reason": reason,
                        "risk_$": round(risk, 3), "cost_frac": round(cost / risk, 3),
                        "rr_plan": round(rr, 2), "R": round(R, 3),
                        "mfe_R": round(mfe / risk, 2), "mae_R": round(mae / risk, 2),
                        "mfe_free_R": round(free[0] / risk, 2), "end_free_R": round(free[1], 3),
                        "hour_utc": int(hours[t]),
                    })
                    if cfg.context == "fib":
                        leg_count[leg_key] = leg_count.get(leg_key, 0) + 1
                    busy_until = j_exit

        if leg_ok:
            run_ext = max(run_ext, h[t]) if d_leg == -1 else min(run_ext, l[t])

    return pd.DataFrame(trades)


def _simulate(d, entry, sl, tp, j0, o, h, l, c, N, cfg):
    """Gestión real. Devuelve (índice salida, precio, motivo, MFE $, MAE $)."""
    risk = abs(entry - sl)
    cur_sl, be_done = sl, False
    mfe = mae = 0.0
    last = min(N - 1, j0 + cfg.max_bars_in_trade)
    for j in range(j0, last + 1):
        fav = (entry - l[j]) if d == -1 else (h[j] - entry)
        adv = (h[j] - entry) if d == -1 else (entry - l[j])
        mae = max(mae, adv)
        hit_sl = (h[j] >= cur_sl) if d == -1 else (l[j] <= cur_sl)
        if j > j0 and ((d == -1 and o[j] >= cur_sl) or (d == 1 and o[j] <= cur_sl)):
            return j, o[j], "BE" if be_done else "SL", mfe, mae
        if hit_sl:
            return j, cur_sl, "BE" if be_done else "SL", mfe, mae
        mfe = max(mfe, fav)
        if (d == -1 and l[j] <= tp) or (d == 1 and h[j] >= tp):
            px = tp if j == j0 else (min(tp, o[j]) if d == -1 else max(tp, o[j]))
            return j, px, "TP", mfe, mae
        if cfg.be_at_r and not be_done and fav >= cfg.be_at_r * risk:
            cur_sl, be_done = entry, True
    return last, c[last], "TIME", mfe, mae


def _free_run(d, entry, sl, j0, o, h, l, c, N, cfg):
    """Sin TP ni BE: MFE antes de tocar el stop original. -> (MFE $, R final bruto)"""
    risk = abs(entry - sl)
    mfe = 0.0
    last = min(N - 1, j0 + cfg.max_bars_in_trade)
    for j in range(j0, last + 1):
        if (d == -1 and h[j] >= sl) or (d == 1 and l[j] <= sl):
            return mfe, -1.0
        mfe = max(mfe, (entry - l[j]) if d == -1 else (h[j] - entry))
    return mfe, d * (c[last] - entry) / risk


# -----------------------------------------------------------------------------
# Estadísticas
# -----------------------------------------------------------------------------
def max_dd(R):
    eq = np.cumsum(R)
    return float((eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min())


def summarize(tr: pd.DataFrame, risk_pcts=(0.5, 1.0), mc: int = 2000) -> dict:
    if tr.empty:
        return {"trades": 0}
    R = tr["R"].values
    wins, losses = R[R > 0.1], R[R < -0.1]
    streak = cur = 0
    for r in R:
        cur = cur + 1 if r < -0.1 else 0
        streak = max(streak, cur)
    months = max((tr["entry_time"].max() - tr["entry_time"].min()).days / 30.44, 1)
    out = {
        "trades": int(len(R)),
        "trades_por_mes": round(len(R) / months, 1),
        "win_rate_%": round(100 * len(wins) / len(R), 1),
        "be_%": round(100 * np.mean(np.abs(R) <= 0.1), 1),
        "ganancia_media_R": round(float(wins.mean()), 2) if len(wins) else 0,
        "perdida_media_R": round(float(losses.mean()), 2) if len(losses) else 0,
        "esperanza_R_por_trade": round(float(R.mean()), 3),
        "coste_medio_R": round(float(tr["cost_frac"].mean()), 3),
        "total_R": round(float(R.sum()), 1),
        "R_por_mes": round(float(R.sum()) / months, 2),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) else None,
        "max_drawdown_R": round(max_dd(R), 1),
        "racha_perdedora_max": int(streak),
        "salidas": tr["reason"].value_counts().to_dict(),
    }
    if mc and len(R) >= 20:
        rng = np.random.default_rng(0)
        dds = np.array([max_dd(rng.permutation(R)) for _ in range(mc)])
        out["MC_maxDD_R_mediana"] = round(float(np.median(dds)), 1)
        out["MC_maxDD_R_peor_5%"] = round(float(np.percentile(dds, 5)), 1)
    for p in risk_pcts:
        eq = np.cumprod(1 + R * p / 100)
        out[f"retorno_%_riesgo_{p}%"] = round(100 * (eq[-1] - 1), 1)
        out[f"maxDD_%_riesgo_{p}%"] = round(100 * (eq / np.maximum.accumulate(eq) - 1).min(), 1)
    return out


def tp_curve(tr: pd.DataFrame) -> pd.DataFrame:
    """Esperanza neta si el TP fuera k R (trade libre hasta el stop original)."""
    if tr.empty:
        return pd.DataFrame()
    rows = []
    for k in TP_CURVE_KS:
        hit = tr["mfe_free_R"] >= k
        R = np.where(hit, k, tr["end_free_R"]) - tr["cost_frac"]
        rows.append({"TP_R": k, "alcanzan_%": round(100 * hit.mean(), 1),
                     "esperanza_neta_R": round(float(R.mean()), 3),
                     "total_R": round(float(R.sum()), 1)})
    return pd.DataFrame(rows)


def years_positive(tr: pd.DataFrame) -> str:
    if tr.empty:
        return "-"
    y = tr.groupby(tr["entry_time"].dt.year)["R"].sum()
    return f"{int((y > 0).sum())}/{len(y)}"


def breakdown(tr: pd.DataFrame) -> dict:
    t = tr.copy()
    t["año"] = t["entry_time"].dt.year.astype(str)
    t["sesion"] = pd.cut(t["hour_utc"], [-1, 6, 12, 16, 24],
                         labels=["Asia 0-7", "Londres 7-13", "NY 13-17", "Tarde 17-24"]).astype(str)
    g = lambda col: (t.groupby(col)["R"].agg(trades="count", esperanza="mean", total="sum")
                       .round(2).to_dict(orient="index"))
    return {"por_año": g("año"), "por_sesion": g("sesion"), "por_direccion": g("dir"),
            "por_liquidez": g("liq")}


def print_block(title, s):
    print(f"\n=== {title} ===")
    for k, v in s.items():
        print(f"  {k:28s} {v}")


def _split(tr, split):
    if tr.empty or not split:
        return [("TOTAL", tr)]
    return [("TOTAL", tr), ("IN-SAMPLE", tr[tr["entry_time"] < split]),
            ("OUT-OF-SAMPLE", tr[tr["entry_time"] >= split])]


# -----------------------------------------------------------------------------
# Ejecución
# -----------------------------------------------------------------------------
class Runner:
    """Cachea los remuestreos para lanzar muchas configuraciones rápido."""
    def __init__(self, df1):
        self.df1, self.cache = df1, {}

    def frame(self, rule):
        if rule not in self.cache:
            self.cache[rule] = resample(self.df1, rule)
        return self.cache[rule]

    def run(self, cfg: Config) -> pd.DataFrame:
        ltf = self.frame(cfg.ltf)
        tr = trend_array(self.df1, ltf.index, cfg)
        return backtest(ltf, self.frame(cfg.htf), cfg, tr)


LADDER_H1 = [
    ("A Base 15m/5m", {"htf": "15min", "ltf": "5min"}),
    ("B 1h/15m", {"htf": "1h", "ltf": "15min"}),
    ("C 1h/15m + tendencia 4h", {"htf": "1h", "ltf": "15min", "trend_tf": "4h"}),
    ("D 1h/15m + tendencia diaria", {"htf": "1h", "ltf": "15min", "trend_tf": "1D", "trend_ema": 20}),
    ("E 4h/1h + tendencia diaria", {"htf": "4h", "ltf": "1h", "trend_tf": "1D", "trend_ema": 20,
                                     "min_leg_atr": 2.0}),
]

LADDER = [
    ("0 Base (setup original)", {}),
    ("1 + coste <= 10% R", {"max_cost_frac": 0.10}),
    ("2 + tendencia 4h EMA50", {"max_cost_frac": 0.10, "trend_tf": "4h"}),
    ("3 + liquidez clave (PDH/PDL/Asia) en zona Fibo 1h",
     {"max_cost_frac": 0.10, "trend_tf": "4h", "liquidity": "key", "zone_min": 0.5,
      "htf": "1h", "min_leg_atr": 2.0}),
    ("4 Sweep de nivel clave sin Fibo, TP 2R",
     {"max_cost_frac": 0.10, "trend_tf": "4h", "liquidity": "key", "context": "none", "tp_r": 2.0}),
    ("5 = 4 solo Londres+NY (7-17 UTC)",
     {"max_cost_frac": 0.10, "trend_tf": "4h", "liquidity": "key", "context": "none",
      "tp_r": 2.0, "session": (7, 17)}),
]


ASSETS = {
    "xau": {"spread": 0.30, "commission": 0.07, "cost_pct": None, "sl_buffer": 0.30,
            "sl_buffer_pct": None, "day_roll_utc": 22},
    "btc": {"cost_pct": 0.10, "sl_buffer_pct": 0.02, "day_roll_utc": 0},
    "eth": {"cost_pct": 0.10, "sl_buffer_pct": 0.02, "day_roll_utc": 0},
}


def main():
    ap = argparse.ArgumentParser(description="fib-liquidity-sweep-backtester: Fibonacci + liquidity sweep en XAUUSD")
    ap.add_argument("--data", nargs="+", help="uno o varios CSV (se concatenan)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--asset", default="xau", choices=list(ASSETS))
    ap.add_argument("--ltf", default="5min")
    ap.add_argument("--cost-pct", type=float, help="coste ida y vuelta en %% del precio")
    ap.add_argument("--format", default="auto", choices=["auto", "histdata"])
    ap.add_argument("--data-tz", default="UTC")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--split", help="fecha in-sample / out-of-sample, p.ej. 2025-07-01")
    ap.add_argument("--htf", default="15min")
    ap.add_argument("--zone-min", type=float, default=0.618)
    ap.add_argument("--tp-ext", type=float, default=-0.382)
    ap.add_argument("--tp-r", type=float)
    ap.add_argument("--be", type=float)
    ap.add_argument("--session", default="0-24")
    ap.add_argument("--spread", type=float)
    ap.add_argument("--commission", type=float)
    ap.add_argument("--sl-buffer", type=float)
    ap.add_argument("--min-rr", type=float, default=1.5)
    ap.add_argument("--min-leg-atr", type=float, default=3.0)
    ap.add_argument("--max-cost-frac", type=float)
    ap.add_argument("--trend-tf"); ap.add_argument("--trend-ema", type=int, default=50)
    ap.add_argument("--liquidity", default="fractal", choices=["fractal", "pdhl", "asia", "key"])
    ap.add_argument("--context", default="fib", choices=["fib", "none"])
    ap.add_argument("--no-body-dir", action="store_true")
    ap.add_argument("--ladder", nargs="?", const="v2", choices=["v2", "h1"],
                    help="escalera de palancas: v2 (por defecto) o h1 (temporalidad + tendencia)")
    ap.add_argument("--grid", action="store_true", help="barrido de sensibilidad")
    ap.add_argument("--out", default="resultados")
    a = ap.parse_args()

    if a.synthetic:
        df = synthetic_data()
        print(f"Datos sintéticos: {len(df):,} velas M1 (paseo aleatorio, sin ventaja posible)")
    elif a.data:
        parts = [load_csv(p, a.data_tz, a.format) for p in a.data]
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        print(f"Cargadas {len(df):,} velas  {df.index[0]} -> {df.index[-1]} (UTC)  [{a.asset.upper()}]")
    else:
        ap.error("indica --data fichero.csv o --synthetic")
    if a.start:
        df = df[df.index >= a.start]
    if a.end:
        df = df[df.index < a.end]

    s0, s1 = (int(x) for x in a.session.split("-"))
    preset = dict(ASSETS[a.asset])
    if a.spread is not None: preset["spread"] = a.spread
    if a.commission is not None: preset["commission"] = a.commission
    if a.sl_buffer is not None: preset["sl_buffer"], preset["sl_buffer_pct"] = a.sl_buffer, None
    if a.cost_pct is not None: preset["cost_pct"] = a.cost_pct
    cfg = Config(htf=a.htf, ltf=a.ltf, zone_min=a.zone_min, tp_ext=a.tp_ext, tp_r=a.tp_r, be_at_r=a.be,
                 session=(s0, s1), min_rr=a.min_rr, min_leg_atr=a.min_leg_atr, **preset,
                 max_cost_frac=a.max_cost_frac, trend_tf=a.trend_tf, trend_ema=a.trend_ema,
                 liquidity=a.liquidity, context=a.context, require_body_dir=not a.no_body_dir)
    os.makedirs(a.out, exist_ok=True)
    runner = Runner(df)

    if a.ladder or a.grid:
        if a.ladder:
            steps = LADDER_H1 if a.ladder == "h1" else LADDER
            configs = [(name, replace(cfg, **kw)) for name, kw in steps]
        else:
            configs = [(f"zona={zm} tp={te} be={be} htf={hf}",
                        replace(cfg, zone_min=zm, tp_ext=te, be_at_r=be, htf=hf))
                       for zm, te, be, hf in itertools.product(
                           [0.5, 0.618], [0.0, -0.272, -0.382, -0.618], [None, 1.0], ["15min", "1h"])]
        rows = []
        for name, c2 in configs:
            tr = runner.run(c2)
            for part, sub in _split(tr, a.split):
                s = summarize(sub, mc=0)
                rows.append({"config": name, "tramo": part, "trades": s.get("trades", 0),
                             "win%": s.get("win_rate_%"), "esperanza_R": s.get("esperanza_R_por_trade"),
                             "coste_R": s.get("coste_medio_R"), "PF": s.get("profit_factor"),
                             "maxDD_R": s.get("max_drawdown_R"), "total_R": s.get("total_R"),
                             "años+": years_positive(sub)})
            print(f"  hecho: {name} ({len(tr)} trades)")
        g = pd.DataFrame(rows)
        g.to_csv(os.path.join(a.out, f"ladder_{a.ladder}_{a.asset}.csv" if a.ladder else "grid.csv"), index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print("\n" + g.to_string(index=False))
        print("\nDecide mirando SOLO in-sample; el out-of-sample es el examen final.")
        return

    tr = runner.run(cfg)
    tr.to_csv(os.path.join(a.out, "trades.csv"), index=False)
    print(f"\nConfig: {json.dumps(asdict(cfg), default=str)}")
    report = {"config": asdict(cfg)}
    for part, sub in _split(tr, a.split):
        s = summarize(sub)
        print_block(f"RESULTADOS {part}", s)
        report[part] = s
    if not tr.empty:
        ins = tr[tr["entry_time"] < a.split] if a.split else tr
        curve = tp_curve(ins)
        print(f"\n=== CURVA DE TP ({'in-sample' if a.split else 'total'}): esperanza neta si TP = k R ===")
        print(curve.to_string(index=False))
        print(f"\n  MFE (R) percentiles 25/50/75/90: "
              f"{np.percentile(tr['mfe_free_R'], [25, 50, 75, 90]).round(2).tolist()}")
        b = breakdown(tr)
        report["desglose"] = b
        report["curva_tp"] = curve.to_dict(orient="records")
        for k, v in b.items():
            print_block(k.upper().replace("_", " "), v)
        _plot(tr, os.path.join(a.out, "equity.png"), a.split)
    with open(os.path.join(a.out, "resumen.json"), "w") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nFicheros en ./{a.out}/  (trades.csv, resumen.json, equity.png)")


def _plot(tr, path, split):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(tr["entry_time"], tr["R"].cumsum(), lw=1.4)
    ax.axhline(0, color="grey", lw=0.6)
    if split:
        ax.axvline(pd.Timestamp(split), color="red", ls="--", lw=1, label="inicio out-of-sample")
        ax.legend()
    ax.set_title("Curva de equity en R (neto de costes)")
    ax.set_ylabel("R acumulado")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()