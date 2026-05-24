#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
birzha_v2.py  --  STANDALONE technical-analysis preprocessor (sections 1-19).

Usage:
    python birzha_v2.py file.csv
    python birzha_v2.py file.csv --output report.txt

Flask integration:
    from birzha_v2 import run_v2
    result = run_v2("path/to/data.csv")
    print(result["report_text"])

Dependencies: pandas, numpy  (stdlib: argparse, math, os, sys, datetime)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════
ZIGZAG_DEV_MAJOR = 5.0
ZIGZAG_DEV_MINOR = 1.0
VP_BINS = 100
VP_VA_THRESH = 0.70
BB_PERIOD = 20
BB_STD = 2.0
KC_EMA_PERIOD = 20
KC_ATR_PERIOD = 10
KC_MULT = 1.5
FIBO_RET = [0.236, 0.382, 0.500, 0.618, 0.786]
FIBO_EXT = [1.272, 1.618]

SECTION_EMOJIS = {
    1: "📈", 2: "🌊", 3: "🔺", 4: "🕯", 5: "📉", 6: "📊",
    7: "📐", 8: "🔊", 9: "📊", 10: "⚖", 11: "📍", 12: "🔄",
    13: "💧", 14: "📊", 15: "⚡", 16: "💰", 17: "🔬", 18: "🔗",
    19: "🧠",
}
SECTION_TITLES = {
    1: "ТРЕНДЫ", 2: "ВОЛНОВОЙ АНАЛИЗ", 3: "ГРАФИЧЕСКИЕ ПАТТЕРНЫ",
    4: "СВЕЧНОЙ АНАЛИЗ", 5: "ДИВЕРГЕНЦИИ", 6: "УРОВНИ",
    7: "ФИБОНАЧЧИ", 8: "VSA", 9: "ОБЪЁМНЫЕ ЗОНЫ",
    10: "ВАЙКОФФ", 11: "ЗОНЫ СБОРА СТОПОВ", 12: "ТЕМП РЫНКА",
    13: "ИМБАЛАНСЫ / FVG", 14: "BOLLINGER / KELTNER / SQUEEZE",
    15: "ЭФФЕКТИВНОСТЬ", 16: "ПОТОК", 17: "МИКРОСТРУКТУРА",
    18: "КОРРЕЛЯЦИИ", 19: "ВЫВОД",
}

RU_EN_MAP = {
    "CVD (Цена откр.)": "CVD (Open)", "CVD (Макс.)": "CVD (High)",
    "CVD (Макс.": "CVD (High)", "CVD (Мин.)": "CVD (Low)",
    "CVD (Мин.": "CVD (Low)", "CVD (Закрыть)": "CVD (Close)",
    "CVD (Закрыть": "CVD (Close)",
    "MF (Money Flow Index)": "MF",
    "Signal line": "Signal",
}


# ════════════════════════════════════════════════════════════════
#  INDICATORS  (all computed from OHLCV if missing in CSV)
# ════════════════════════════════════════════════════════════════

def calc_ema(data: np.ndarray, period: int) -> np.ndarray:
    ema = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return ema
    ema[period - 1] = np.mean(data[:period])
    m = 2.0 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = data[i] * m + ema[i - 1] * (1 - m)
    return ema


def calc_sma(data: np.ndarray, period: int) -> np.ndarray:
    sma = np.full_like(data, np.nan, dtype=float)
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    return sma


def calc_atr(high, low, close, period=14):
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = np.full_like(tr, np.nan)
    if len(tr) < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calc_rsi(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.full(len(close), np.nan)
    al = np.full(len(close), np.nan)
    if len(gain) < period:
        return ag
    ag[period] = np.mean(gain[:period])
    al[period] = np.mean(loss[:period])
    for i in range(period + 1, len(close)):
        ag[i] = (ag[i - 1] * (period - 1) + gain[i - 1]) / period
        al[i] = (al[i - 1] * (period - 1) + loss[i - 1]) / period
    rs = ag / np.where(al == 0, 1e-10, al)
    rsi = np.full(len(close), np.nan)
    v = ~np.isnan(rs)
    rsi[v] = 100.0 - 100.0 / (1.0 + rs[v])
    return rsi


def calc_alma(data, period=20, offset=0.85, sigma=6.0):
    alma = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return alma
    m = offset * (period - 1)
    s = period / sigma
    w = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
    w /= w.sum()
    for i in range(period - 1, len(data)):
        alma[i] = np.dot(w, data[i - period + 1:i + 1])
    return alma


def calc_macd(close, fast=12, slow=26, sig=9):
    ef = calc_ema(close, fast)
    es = calc_ema(close, slow)
    ml = ef - es
    v = ~np.isnan(ml)
    sl = np.full_like(close, np.nan)
    fv = int(np.argmax(v)) if np.any(v) else len(close)
    if np.sum(v) >= sig:
        sl_part = calc_ema(ml[fv:], sig)
        sl[fv:fv + len(sl_part)] = sl_part
    hist = ml - sl
    return ml, sl, hist


def calc_obv(close, volume):
    obv = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def calc_mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    mfi = np.full_like(close, np.nan)
    for i in range(period, len(close)):
        pos = neg = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0 and tp[j] > tp[j - 1]:
                pos += rmf[j]
            elif j > 0:
                neg += rmf[j]
        mfi[i] = 100.0 - 100.0 / (1.0 + pos / max(neg, 1e-10))
    return mfi


def calc_cmf(high, low, close, volume, period=20):
    hl = high - low
    hl[hl == 0] = 1e-10
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    cmf = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        sv = np.sum(volume[i - period + 1:i + 1])
        cmf[i] = np.sum(mfv[i - period + 1:i + 1]) / sv if sv > 0 else 0.0
    return cmf


def calc_ad(high, low, close, volume):
    hl = high - low
    hl[hl == 0] = 1e-10
    mfm = ((close - low) - (high - close)) / hl
    return np.cumsum(mfm * volume)


def calc_elder_force(close, volume, period=13):
    fi = np.zeros_like(close)
    fi[1:] = (close[1:] - close[:-1]) * volume[1:]
    return calc_ema(fi, period)


def calc_vwap(high, low, close, volume):
    tp = (high + low + close) / 3.0
    cv = np.cumsum(tp * volume)
    cumv = np.cumsum(volume)
    cumv[cumv == 0] = 1e-10
    return cv / cumv


# ════════════════════════════════════════════════════════════════
#  ZIGZAG
# ════════════════════════════════════════════════════════════════

def zigzag(high, low, dev_pct=5.0, times=None):
    n = len(high)
    if n < 3:
        return []
    dev = dev_pct / 100.0
    points = []
    fh_idx = fl_idx = 0
    fh = high[0]; fl = low[0]
    for i in range(1, n):
        if high[i] > fh:
            fh = high[i]; fh_idx = i
        if low[i] < fl:
            fl = low[i]; fl_idx = i
        up = (fh - fl) / fl if fl > 0 else 0
        if up >= dev:
            if fl_idx < fh_idx:
                points.append({"index": fl_idx, "price": fl, "type": "low"})
                points.append({"index": fh_idx, "price": fh, "type": "high"})
            else:
                points.append({"index": fh_idx, "price": fh, "type": "high"})
                points.append({"index": fl_idx, "price": fl, "type": "low"})
            break
    else:
        return []
    si = max(fh_idx, fl_idx) + 1
    last = points[-1]
    lt = last["type"]
    for i in range(si, n):
        if lt == "high":
            if low[i] < last["price"] * (1 - dev):
                points.append({"index": i, "price": low[i], "type": "low"})
                last = points[-1]; lt = "low"
            elif high[i] > last["price"]:
                last["index"] = i; last["price"] = high[i]
        else:
            if high[i] > last["price"] * (1 + dev):
                points.append({"index": i, "price": high[i], "type": "high"})
                last = points[-1]; lt = "high"
            elif low[i] < last["price"]:
                last["index"] = i; last["price"] = low[i]
    if times is not None:
        for p in points:
            p["time"] = str(times[p["index"]])
    return points


def classify_swings(zz):
    if len(zz) < 3:
        return zz
    ph = []; pl = []
    out = []
    for p in zz:
        p = dict(p)
        if p["type"] == "high":
            p["swing"] = "HH" if (ph and p["price"] > ph[-1]) else ("LH" if ph else "HH")
            ph.append(p["price"])
        else:
            p["swing"] = "HL" if (pl and p["price"] > pl[-1]) else ("LL" if pl else "HL")
            pl.append(p["price"])
        out.append(p)
    return out


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def detect_tf_hours(ts):
    ts = pd.to_datetime(ts)
    d = ts.diff().dropna().dt.total_seconds()
    d = d[d > 0]
    return float(d.median()) / 3600.0 if len(d) else 1.0


def tf_label(h):
    if h < 1: return f"{int(h * 60)}m"
    if h < 24: return f"{int(h)}H" if h == int(h) else f"{h}H"
    if h < 168: return f"{int(h / 24)}D"
    return f"{int(h / 168)}W"


def _pct(base, target):
    return round((target - base) / base * 100, 2) if base else 0.0


def pct_from(price, base):
    """Percent distance from base to price."""
    return (price - base) / base * 100 if base else 0.0


def fp(price, cur, dec=2):
    """Format price with pct offset from current."""
    p = round(price, dec)
    pct = (price - cur) / cur * 100 if cur else 0
    return f"{p:.{dec}f} ({pct:+.2f}%)"


def human_time(bars, tfh):
    bars = max(bars, 1.0)
    h = bars * tfh
    h = max(h, tfh)
    if h < 24:
        return f"~{max(round(h), round(tfh))} ч"
    d = round(h / 24, 1)
    return f"~{int(d)} дн" if d == int(d) else f"~{d} дн"


def safe_last(arr):
    v = arr[~np.isnan(arr)] if hasattr(arr, '__len__') else np.array([])
    return float(v[-1]) if len(v) else np.nan


def _slope_dir(arr, n=20):
    if len(arr) < n:
        return 0.0
    seg = arr[-n:]
    x = np.arange(len(seg))
    return float(np.polyfit(x, seg, 1)[0])


# ════════════════════════════════════════════════════════════════
#  CSV LOADING
# ════════════════════════════════════════════════════════════════

def load_csv(path):
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if len(df.columns) >= 6:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Cannot read CSV: {path}")
    df.columns = [c.strip() for c in df.columns]
    df.columns = [RU_EN_MAP.get(c, c) for c in df.columns]
    # dedup
    seen = {}; nc = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1; nc.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0; nc.append(c)
    df.columns = nc
    cl = {c.lower(): c for c in df.columns}
    for r in ["time", "open", "high", "low", "close", "volume"]:
        if r.lower() not in cl:
            raise ValueError(f"Missing column: {r}")
    rm = {cl[r.lower()]: r for r in ["time","open","high","low","close","volume"] if cl[r.lower()] != r}
    if rm:
        df = df.rename(columns=rm)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close","volume"])
    return df


# ════════════════════════════════════════════════════════════════
#  TFPack
# ════════════════════════════════════════════════════════════════

@dataclass
class TFPack:
    df: pd.DataFrame
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    open_: np.ndarray
    volume: np.ndarray
    times: np.ndarray
    atr: np.ndarray
    atr_last: float
    tfh: float
    tf_min: int = 0
    tfl: str
    n: int
    cur: float
    dec: int
    ticker: str
    exchange: str
    avail: dict = field(default_factory=dict)
    zz5: list = field(default_factory=list)
    zz1: list = field(default_factory=list)
    swings: list = field(default_factory=list)
    # context values set by section_1
    trend_dir: str = "боковик"
    trend_start_idx: int = 0
    trend_start_price: float = 0.0
    local_dir: str = "боковик"
    local_start_idx: int = 0
    global_high_idx: int = 0
    global_low_idx: int = 0
    # pivot levels from section_6
    pivot_supports: list = field(default_factory=list)
    pivot_resistances: list = field(default_factory=list)


def build_pack(path: str) -> TFPack:
    df = load_csv(path)
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    times = df["time"].values
    n = len(df)
    cur = float(close[-1])
    dec = 2 if cur >= 10 else (4 if cur >= 0.01 else 6)
    atr = calc_atr(high, low, close, 14)
    atr_last = safe_last(atr)
    if np.isnan(atr_last):
        atr_last = float(np.mean(high - low)) if n else 0.01
    tfh = detect_tf_hours(df["time"])
    tfl_ = tf_label(tfh)
    tf_min_ = int(round(tfh * 60))
    # ticker / exchange from filename
    fname = os.path.splitext(os.path.basename(path))[0]
    parts = fname.split("_")
    exchange = parts[0].strip().upper() if len(parts) >= 2 else ""
    ticker = parts[1].split(",")[0].strip().upper() if len(parts) >= 2 else fname.upper()
    # zigzag
    zz5 = zigzag(high, low, ZIGZAG_DEV_MAJOR, times)
    zz1 = zigzag(high, low, ZIGZAG_DEV_MINOR, times)
    swings = classify_swings(zz5)
    # detect available optional columns
    avail = {}
    for grp, cols in {
        "alma": ["ALMA 20", "ALMA 50", "ALMA 200"],
        "bb": ["Upper", "Basis", "Lower"],
        "vwap": ["VWAP"],
        "rsi": ["RSI"],
        "cvd": ["CVD (Close)"],
        "div": ["Regular Bullish", "Hidden Bullish", "Regular Bearish", "Hidden Bearish"],
        "macd": ["MACD", "Signal", "Histogram"],
        "stoch": ["%K", "%D"],
        "flow": ["MF", "CMF"],
        "wyckoff": ["Elder Force Index", "Accumulation/Distribution", "CMF"],
        "atr_col": ["ATR"],
    }.items():
        found = [c for c in cols if c in df.columns]
        if found:
            avail[grp] = found
    # global extremes
    ghi = int(np.argmax(high))
    gli = int(np.argmin(low))
    return TFPack(
        df=df, close=close, high=high, low=low, open_=open_, volume=volume,
        times=times, atr=atr, atr_last=atr_last, tfh=tfh, tf_min=tf_min_, tfl=tfl_,
        n=n, cur=cur, dec=dec, ticker=ticker, exchange=exchange,
        avail=avail, zz5=zz5, zz1=zz1, swings=swings,
        global_high_idx=ghi, global_low_idx=gli,
    )


# ════════════════════════════════════════════════════════════════
#  LINREG CHANNEL
# ════════════════════════════════════════════════════════════════

def linreg_channel(close, period=100):
    n = len(close)
    p = min(period, n)
    data = close[-p:]
    x = np.arange(p, dtype=float)
    coeffs = np.polyfit(x, data, 1)
    slope, intercept = coeffs
    center = slope * x + intercept
    res = data - center
    sigma = float(np.std(res))
    upper = center + 2 * sigma
    lower = center - 2 * sigma
    cw = float(upper[-1] - lower[-1])
    pos = (data[-1] - lower[-1]) / cw * 100 if cw > 0 else 50.0
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mid = float(np.mean(data))
    slope_pct = slope / mid * 100 if mid else 0.0
    return {
        "slope_pct": float(slope_pct), "pos": float(np.clip(pos, 0, 100)),
        "r2": float(r2), "upper": float(upper[-1]),
        "lower": float(lower[-1]), "center": float(center[-1]), "sigma": sigma,
    }


# ════════════════════════════════════════════════════════════════
#  PIVOT POINTS
# ════════════════════════════════════════════════════════════════

def calc_pivots(h, l, c):
    P = (h + l + c) / 3
    return {
        "P": round(P, 4), "R1": round(2*P - l, 4), "R2": round(P + (h-l), 4),
        "R3": round(h + 2*(P-l), 4), "S1": round(2*P - h, 4),
        "S2": round(P - (h-l), 4), "S3": round(l - 2*(h-P), 4),
    }


def pivots_for_period(df, period):
    if len(df) < 2:
        return None
    dfc = df.copy()
    dfc["_date"] = dfc["time"].dt.date
    if period == "D":
        dates = sorted(dfc["_date"].unique())
        if len(dates) < 2: return None
        mask = dfc["_date"] == dates[-2]
    elif period == "W":
        iso = dfc["time"].dt.isocalendar()
        dfc["_yw"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
        yws = sorted(dfc["_yw"].unique())
        if len(yws) < 2: return None
        mask = dfc["_yw"] == yws[-2]
    elif period == "M":
        dfc["_ym"] = dfc["time"].dt.year * 100 + dfc["time"].dt.month
        yms = sorted(dfc["_ym"].unique())
        if len(yms) < 2: return None
        mask = dfc["_ym"] == yms[-2]
    else:
        return None
    pd_ = dfc[mask]
    if pd_.empty:
        return None
    return calc_pivots(float(pd_["high"].max()), float(pd_["low"].min()),
                       float(pd_["close"].iloc[-1]))


# ════════════════════════════════════════════════════════════════
#  VOLUME PROFILE
# ════════════════════════════════════════════════════════════════

def volume_profile(df_slice, cur):
    lo_all = df_slice["low"].values.astype(float)
    hi_all = df_slice["high"].values.astype(float)
    vol_all = df_slice["volume"].values.astype(float)
    if len(df_slice) < 2:
        return {"POC": cur, "VAH": cur, "VAL": cur}
    pmin = float(lo_all.min()); pmax = float(hi_all.max())
    if pmax <= pmin:
        pmax = pmin + 1e-8
    edges = np.linspace(pmin, pmax, VP_BINS + 1)
    vpb = np.zeros(VP_BINS)
    for i in range(len(df_slice)):
        lo, hi, v = lo_all[i], hi_all[i], vol_all[i]
        il = max(0, min(np.searchsorted(edges, lo, side="right") - 1, VP_BINS - 1))
        ih = max(0, min(np.searchsorted(edges, hi, side="right") - 1, VP_BINS - 1))
        nb = ih - il + 1
        for b in range(il, ih + 1):
            vpb[b] += v / nb
    poc_i = int(np.argmax(vpb))
    poc = float(0.5 * (edges[poc_i] + edges[poc_i + 1]))
    total = vpb.sum(); target = total * VP_VA_THRESH
    cum = vpb[poc_i]; li = hi_i = poc_i
    while cum < target and (li > 0 or hi_i < VP_BINS - 1):
        el = vpb[li - 1] if li > 0 else 0
        eh = vpb[hi_i + 1] if hi_i < VP_BINS - 1 else 0
        if el >= eh and li > 0:
            li -= 1; cum += el
        elif hi_i < VP_BINS - 1:
            hi_i += 1; cum += eh
        else:
            li -= 1; cum += el
    val = float(edges[li]); vah = float(edges[hi_i + 1])
    return {"POC": poc, "VAH": vah, "VAL": val}


# ════════════════════════════════════════════════════════════════
#  PATTERN DETECTION
# ════════════════════════════════════════════════════════════════

def detect_patterns(swings, cur, atr_last):
    results = []
    if len(swings) < 4:
        return results
    sw = swings[-20:]
    tol = 0.015
    highs = [(i, p) for i, p in enumerate(sw) if p["type"] == "high"]
    lows = [(i, p) for i, p in enumerate(sw) if p["type"] == "low"]

    def near(a, b):
        return abs(a - b) / max(a, 0.001) < tol

    # Double Top
    if len(highs) >= 2:
        for ii in range(len(highs) - 1):
            i1, h1 = highs[ii]; i2, h2 = highs[ii + 1]
            if near(h1["price"], h2["price"]):
                bl = [l for j, l in lows if i1 < j < i2]
                if bl:
                    neck = min(l["price"] for l in bl)
                    tgt = neck - (h1["price"] - neck)
                    st = "В реализации" if cur < neck else "Формируется"
                    results.append({"name": "Двойная вершина", "dir": "медвежий",
                                    "status": st, "target": tgt}); break

    # Double Bottom
    if len(lows) >= 2:
        for ii in range(len(lows) - 1):
            i1, l1 = lows[ii]; i2, l2 = lows[ii + 1]
            if near(l1["price"], l2["price"]):
                bh = [h for j, h in highs if i1 < j < i2]
                if bh:
                    neck = max(h["price"] for h in bh)
                    tgt = neck + (neck - l1["price"])
                    st = "В реализации" if cur > neck else "Формируется"
                    results.append({"name": "Двойное дно", "dir": "бычий",
                                    "status": st, "target": tgt}); break

    # H&S
    if len(highs) >= 3:
        for ii in range(len(highs) - 2):
            _, hl = highs[ii]; _, hh = highs[ii+1]; _, hr = highs[ii+2]
            if hh["price"] > hl["price"] and hh["price"] > hr["price"] and near(hl["price"], hr["price"]):
                neck_ls = [l["price"] for j, l in lows if highs[ii][0] < j < highs[ii+2][0]]
                if neck_ls:
                    neck = np.mean(neck_ls)
                    tgt = neck - (hh["price"] - neck)
                    results.append({"name": "Голова и плечи", "dir": "медвежий",
                                    "status": "Формируется", "target": tgt}); break

    # Triangles
    if len(sw) >= 6:
        rh = [p["price"] for p in sw[-8:] if p["type"] == "high"]
        rl = [p["price"] for p in sw[-8:] if p["type"] == "low"]
        if len(rh) >= 2 and len(rl) >= 2:
            hs = rh[-1] - rh[0]; ls = rl[-1] - rl[0]
            if hs < 0 and ls > 0:
                mid = (np.mean(rh) + np.mean(rl)) / 2
                span = max(rh) - min(rl)
                results.append({"name": "Треугольник", "dir": "нейтральный",
                                "status": "Формируется", "target": mid + span if cur > mid else mid - span})
            elif abs(hs) < atr_last * 0.05 and ls > 0:
                res = np.mean(rh); tgt = res + (res - min(rl))
                results.append({"name": "Восходящий треугольник", "dir": "бычий",
                                "status": "Формируется", "target": tgt})

    # Wedge
    if len(sw) >= 6:
        rh = [p["price"] for p in sw[-8:] if p["type"] == "high"]
        rl = [p["price"] for p in sw[-8:] if p["type"] == "low"]
        if len(rh) >= 2 and len(rl) >= 2:
            if rh[-1] > rh[0] and rl[-1] > rl[0] and (rl[-1]-rl[0]) > (rh[-1]-rh[0]) * 0.5:
                results.append({"name": "Восходящий клин", "dir": "медвежий",
                                "status": "Формируется", "target": min(rl)})
            elif rh[-1] < rh[0] and rl[-1] < rl[0]:
                results.append({"name": "Нисходящий клин", "dir": "бычий",
                                "status": "Формируется", "target": max(rh)})

    # Flag
    if len(sw) >= 5:
        recent = sw[-6:]; prev = sw[-10:-6] if len(sw) >= 10 else sw[:4]
        rng = max(p["price"] for p in recent) - min(p["price"] for p in recent)
        if prev:
            imp = max(p["price"] for p in prev) - min(p["price"] for p in prev)
            if imp > 0 and rng / imp < 0.3 and rng > atr_last * 0.5:
                pp = [p["price"] for p in prev]
                if pp[-1] > pp[0]:
                    tgt = max(p["price"] for p in recent) + imp
                    results.append({"name": "Бычий флаг", "dir": "бычий",
                                    "status": "Формируется", "target": tgt})
                else:
                    tgt = min(p["price"] for p in recent) - imp
                    results.append({"name": "Медвежий флаг", "dir": "медвежий",
                                    "status": "Формируется", "target": tgt})
    return results


# ════════════════════════════════════════════════════════════════
#  CANDLE PATTERNS
# ════════════════════════════════════════════════════════════════

def detect_candles(open_, high, low, close, n_last=20):
    patterns = []
    n = len(close)
    start = max(0, n - n_last)
    bull = bear = 0
    bodies = []
    for i in range(start, n):
        body = abs(close[i] - open_[i])
        rng = high[i] - low[i]
        bodies.append(body / rng * 100 if rng > 0 else 0)
        if close[i] > open_[i]: bull += 1
        else: bear += 1

        if rng == 0:
            continue
        body_pct = body / rng
        upper = high[i] - max(open_[i], close[i])
        lower = min(open_[i], close[i]) - low[i]

        # Doji
        if body_pct < 0.1:
            patterns.append((i, "Doji"))
        # Hammer
        elif lower > body * 2 and upper < body * 0.3 and close[i] > open_[i]:
            patterns.append((i, "Hammer"))
        # Shooting star
        elif upper > body * 2 and lower < body * 0.3 and close[i] < open_[i]:
            patterns.append((i, "Shooting Star"))
        # Marubozu
        elif body_pct > 0.9:
            patterns.append((i, "Marubozu бычий" if close[i] > open_[i] else "Marubozu медвежий"))

    # Multi-candle
    for i in range(start + 1, n):
        # Engulfing
        if (close[i-1] < open_[i-1] and close[i] > open_[i] and
                close[i] > open_[i-1] and open_[i] < close[i-1]):
            patterns.append((i, "Бычье поглощение"))
        elif (close[i-1] > open_[i-1] and close[i] < open_[i] and
                close[i] < open_[i-1] and open_[i] > close[i-1]):
            patterns.append((i, "Медвежье поглощение"))
        # Harami
        if (close[i-1] < open_[i-1] and close[i] > open_[i] and
                close[i] < open_[i-1] and open_[i] > close[i-1]):
            patterns.append((i, "Бычий харами"))
        elif (close[i-1] > open_[i-1] and close[i] < open_[i] and
                close[i] > open_[i-1] and open_[i] < close[i-1]):
            patterns.append((i, "Медвежий харами"))

    # 3 soldiers / 3 crows
    for i in range(start + 2, n):
        if all(close[i-j] > open_[i-j] for j in range(3)) and close[i] > close[i-1] > close[i-2]:
            patterns.append((i, "Три солдата"))
        if all(close[i-j] < open_[i-j] for j in range(3)) and close[i] < close[i-1] < close[i-2]:
            patterns.append((i, "Три вороны"))

    # Morning / Evening star
    for i in range(start + 2, n):
        rng1 = high[i-2] - low[i-2]
        body1 = abs(close[i-2] - open_[i-2])
        body_mid = abs(close[i-1] - open_[i-1])
        rng_mid = high[i-1] - low[i-1]
        if rng1 > 0 and rng_mid > 0 and body_mid / rng_mid < 0.3 and body1 / rng1 > 0.5:
            if close[i-2] < open_[i-2] and close[i] > open_[i]:
                patterns.append((i, "Утренняя звезда"))
            elif close[i-2] > open_[i-2] and close[i] < open_[i]:
                patterns.append((i, "Вечерняя звезда"))

    avg_body = round(np.mean(bodies), 1) if bodies else 0
    # Character
    last5 = bodies[-5:] if len(bodies) >= 5 else bodies
    first5 = bodies[:5] if len(bodies) >= 5 else bodies
    if np.mean(last5) > np.mean(first5) * 1.3:
        character = "ускорение"
    elif np.mean(last5) < np.mean(first5) * 0.7:
        character = "замедление"
    elif bull > bear * 1.5 or bear > bull * 1.5:
        character = "развитие"
    else:
        character = "переходный"

    return {
        "bull": bull, "bear": bear, "avg_body_pct": avg_body,
        "patterns": patterns[-10:], "character": character,
    }


# ════════════════════════════════════════════════════════════════
#  ELLIOTT WAVE (simplified)
# ════════════════════════════════════════════════════════════════

def label_elliott(zz_points, trend_start_price, cur, is_up):
    """Simple Elliott labeling on zigzag points."""
    if len(zz_points) < 4:
        return {"waves": [], "current": "—", "targets": [], "pattern": "unknown"}
    # find anchor
    anchor = 0
    md = float("inf")
    for i, p in enumerate(zz_points):
        d = abs(p["price"] - trend_start_price)
        if d < md:
            md = d; anchor = i
    pts = zz_points[anchor:]
    # extract alternating
    alt = [pts[0]]
    for p in pts[1:]:
        if p["type"] != alt[-1]["type"]:
            alt.append(p)
        else:
            if p["type"] == "high" and p["price"] > alt[-1]["price"]:
                alt[-1] = p
            elif p["type"] == "low" and p["price"] < alt[-1]["price"]:
                alt[-1] = p
    waves = []
    labels_imp = ["0", "I", "II", "III", "IV", "V"]
    labels_corr = ["A", "B", "C"]
    for i, p in enumerate(alt[:6]):
        lbl = labels_imp[i] if i < 6 else str(i)
        waves.append({"label": lbl, "price": p["price"], "type": p["type"]})
    pattern = "impulse" if len(waves) >= 6 else "partial"
    current = waves[-1]["label"] if waves else "—"

    # If impulse complete and more points, add correction
    if len(alt) > 6:
        for i, p in enumerate(alt[6:9]):
            lbl = labels_corr[i] if i < 3 else f"C{i}"
            waves.append({"label": lbl, "price": p["price"], "type": p["type"]})
            current = lbl
        pattern = "impulse+correction"

    # targets
    targets = []
    prices = {w["label"]: w["price"] for w in waves}
    if current in ("III", "IV", "V"):
        if "0" in prices and "I" in prices:
            w1 = abs(prices["I"] - prices["0"])
            base = prices.get("IV", prices.get("II", cur))
            for r, nm in [(0.618, "0.618xW1"), (1.0, "=W1"), (1.618, "1.618xW1")]:
                t = base + w1 * r * (1 if is_up else -1)
                targets.append({"fib": f"W5 {nm}", "price": round(t, 4)})
    if current in ("A", "B", "C"):
        if "V" in prices and "0" in prices:
            imp_len = abs(prices["V"] - prices["0"])
            base = prices.get("A", prices.get("V", cur))
            for r, nm in [(0.382, "38.2%"), (0.618, "61.8%"), (1.0, "100%")]:
                t = base + imp_len * r * (-1 if is_up else 1)
                targets.append({"fib": f"Corr {nm}", "price": round(t, 4)})

    return {"waves": waves, "current": current, "targets": targets, "pattern": pattern}


# ════════════════════════════════════════════════════════════════
#  SECTIONS 1-18  (each returns (text, goals))
# ════════════════════════════════════════════════════════════════

def section_1(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ТРЕНДЫ"""
    goals = []
    n = p.n; cur = p.cur

    # Adaptive window by TF
    tf_min = p.tf_min
    if tf_min <= 15:
        window_days = 14
    elif tf_min <= 60:
        window_days = 60
    elif tf_min <= 240:
        window_days = 180
    else:
        window_days = 730
    bars_per_day = int(24 * 60 / max(tf_min, 1))
    max_bars = min(window_days * bars_per_day, n)

    # windowed extremes
    w_high = p.high[-max_bars:]
    w_low = p.low[-max_bars:]
    offset = n - max_bars
    hi_idx = int(np.argmax(w_high)) + offset
    lo_idx = int(np.argmin(w_low)) + offset
    hi_price = float(p.high[hi_idx]); lo_price = float(p.low[lo_idx])
    # direction
    dist_hi = abs(hi_price - cur); dist_lo = abs(lo_price - cur)
    if dist_hi >= dist_lo:
        senior_dir = "нисходящий"; ts_idx = hi_idx; ts_price = hi_price
    else:
        senior_dir = "восходящий"; ts_idx = lo_idx; ts_price = lo_price
    p.trend_dir = senior_dir; p.trend_start_idx = ts_idx; p.trend_start_price = ts_price
    p.trend_window_bars = max_bars

    # local direction from last 6 windowed swings
    w_zz5 = zigzag(w_high, w_low, ZIGZAG_DEV_MAJOR)
    w_swings = classify_swings(w_zz5)
    recent = w_swings[-6:] if len(w_swings) >= 6 else w_swings
    rh = [s["price"] for s in recent if s["type"] == "high"]
    rl = [s["price"] for s in recent if s["type"] == "low"]
    local_dir = "боковик"
    if len(rh) >= 2 and len(rl) >= 2:
        if rh[-1] > rh[0] and rl[-1] > rl[0]: local_dir = "восходящий"
        elif rh[-1] < rh[0] and rl[-1] < rl[0]: local_dir = "нисходящий"
    p.local_dir = local_dir

    # stage via linreg
    lr = linreg_channel(p.close, 100)
    slope = lr["slope_pct"]; r2 = lr["r2"]; pos = lr["pos"]
    if r2 > 0.4 and abs(slope) > 0.05: stage = "развитие"
    elif abs(slope) < 0.01: stage = "затухание"
    else: stage = "начальная"

    # ALMA 200
    if "alma" in p.avail and "ALMA 200" in p.avail["alma"]:
        a200 = safe_last(p.df["ALMA 200"].values.astype(float))
    elif n >= 200:
        a200 = safe_last(calc_alma(p.close, 200))
    else:
        a200 = np.nan
    alma_pos = "выше" if (not np.isnan(a200) and cur > a200) else "ниже" if not np.isnan(a200) else "N/A"

    # trend start as goal
    goals.append((ts_price, 1))
    # linreg channel boundaries
    goals.append((lr["upper"], 1))
    goals.append((lr["lower"], 1))

    ts_time = str(p.df["time"].iloc[ts_idx])[:16]
    lines = [
        f"Направление: {senior_dir} (старший), {local_dir} (локальный)",
        f"Стадия: {stage}",
        f"Начало тренда: {fp(ts_price, cur, p.dec)} ({ts_time})",
        f"LinReg канал: верх {fp(lr['upper'], cur, p.dec)}, низ {fp(lr['lower'], cur, p.dec)}, позиция {pos:.0f}%",
        f"R² = {r2:.3f}, slope = {slope:.4f}%/бар",
        f"Позиция vs ALMA 200: {alma_pos}" + (f" ({fp(a200, cur, p.dec)})" if not np.isnan(a200) else ""),
    ]
    return "\n".join(lines), goals


def section_2(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ВОЛНОВОЙ АНАЛИЗ"""
    goals = []
    is_up = p.cur > p.trend_start_price

    # Use windowed data (same window as S01) for zigzag
    max_bars = getattr(p, 'trend_window_bars', p.n)
    w_high = p.high[-max_bars:]
    w_low = p.low[-max_bars:]
    w_times = p.times[-max_bars:]
    w_zz5 = zigzag(w_high, w_low, ZIGZAG_DEV_MAJOR, w_times)
    w_zz1 = zigzag(w_high, w_low, ZIGZAG_DEV_MINOR, w_times)

    ew = label_elliott(w_zz5, p.trend_start_price, p.cur, is_up)
    lines = []
    if ew["waves"]:
        wstr = " → ".join(f"{w['label']}={fp(w['price'], p.cur, p.dec)}" for w in ew["waves"])
        lines.append(f"Разметка: {wstr}")
        lines.append(f"Текущая волна: {ew['current']}")
        lines.append(f"Паттерн: {ew['pattern']}")
        if ew["targets"]:
            lines.append("Цели:")
            for t in ew["targets"]:
                lines.append(f"  {t['fib']}: {fp(t['price'], p.cur, p.dec)}")
                goals.append((t["price"], 2))
    else:
        lines.append("Недостаточно данных для разволновки.")
    # subwaves from ZZ 1%
    sub_ew = label_elliott(w_zz1, p.trend_start_price, p.cur, is_up)
    if sub_ew["waves"] and len(sub_ew["waves"]) > len(ew["waves"]):
        lines.append(f"Подволны (ZZ 1%): текущая = {sub_ew['current']}")
    return "\n".join(lines), goals


def section_3(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ГРАФИЧЕСКИЕ ПАТТЕРНЫ"""
    goals = []
    pats = detect_patterns(p.swings, p.cur, p.atr_last)
    lines = []
    if pats:
        for pat in pats:
            tgt = pat.get("target", 0)
            lines.append(f"{pat['name']} [{pat['dir']}] — {pat['status']}, цель: {fp(tgt, p.cur, p.dec)}")
            goals.append((tgt, 3))
    else:
        lines.append("Активных графических паттернов не обнаружено.")
    return "\n".join(lines), goals


def section_4(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """СВЕЧНОЙ АНАЛИЗ"""
    cd = detect_candles(p.open_, p.high, p.low, p.close)
    lines = [
        f"Последние 20 баров: бычьих {cd['bull']}, медвежьих {cd['bear']}, avg тело {cd['avg_body_pct']:.1f}%",
        f"Характер: {cd['character']}",
    ]
    if cd["patterns"]:
        pnames = [f"{nm} (бар {idx})" for idx, nm in cd["patterns"][-5:]]
        lines.append("Паттерны: " + ", ".join(pnames))
    else:
        lines.append("Значимых свечных паттернов нет.")
    return "\n".join(lines), []


def section_5(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ДИВЕРГЕНЦИИ"""
    goals = []
    # RSI
    if "rsi" in p.avail:
        rsi_arr = p.df["RSI"].values.astype(float)
    else:
        rsi_arr = calc_rsi(p.close)
    rsi_val = safe_last(rsi_arr)
    if np.isnan(rsi_val): rsi_val = 50.0
    if rsi_val > 70: rsi_zone = "перекупленность"
    elif rsi_val < 30: rsi_zone = "перепроданность"
    else: rsi_zone = "нейтральная"
    rsi_trend = "растёт" if _slope_dir(rsi_arr, 10) > 0 else "падает"

    # MACD
    if "macd" in p.avail:
        hist = p.df["Histogram"].values.astype(float) if "Histogram" in p.df.columns else np.zeros(p.n)
    else:
        _, _, hist = calc_macd(p.close)
    h_last = safe_last(hist)
    macd_dir = "растёт" if (not np.isnan(h_last) and h_last > 0) else "падает"

    # CVD
    cvd_dir = "N/A"
    if "cvd" in p.avail and "CVD (Close)" in p.df.columns:
        cvd = p.df["CVD (Close)"].values.astype(float)
        cvd_dir = "растёт" if _slope_dir(cvd, 10) > 0 else "падает"
        # CVD divergence
        price_up = p.close[-1] > p.close[-min(20, p.n)]
        cvd_up = cvd[-1] > cvd[-min(20, p.n)] if len(cvd) >= 2 else False
        if price_up and not cvd_up:
            lines_div = "Медвежья дивергенция CVD/цена"
        elif not price_up and cvd_up:
            lines_div = "Бычья дивергенция CVD/цена"
        else:
            lines_div = "Дивергенций CVD нет"
    else:
        lines_div = "CVD отсутствует"

    # Divergence labels from CSV
    div_labels = []
    for col in ["Regular Bullish", "Hidden Bullish", "Regular Bearish", "Hidden Bearish"]:
        if col in p.df.columns:
            vals = p.df[col].values
            last_valid = vals[~pd.isna(vals)]
            if len(last_valid) > 0 and float(last_valid[-1]) != 0:
                div_labels.append(f"{col}: {float(last_valid[-1]):.4f}")

    lines = [
        f"RSI: {rsi_val:.1f} ({rsi_zone}), тренд {rsi_trend}",
        f"MACD гистограмма: {macd_dir}",
        f"CVD: {cvd_dir}",
        lines_div,
    ]
    if div_labels:
        lines.append("Метки дивергенций: " + "; ".join(div_labels))
    return "\n".join(lines), goals


def section_6(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """УРОВНИ"""
    goals = []
    cur = p.cur
    # Pivots
    piv = {}
    if p.tfh < 24:
        d = pivots_for_period(p.df, "D")
        if d: piv["D"] = d
    if p.tfh < 168:
        w = pivots_for_period(p.df, "W")
        if w: piv["W"] = w
    m = pivots_for_period(p.df, "M")
    if m: piv["M"] = m

    # Collect all pivot levels
    all_lvls = []
    for pref, data in piv.items():
        for lbl, val in data.items():
            all_lvls.append((val, f"{pref}.{lbl}"))

    # Swing S/R from ZZ 1%
    sh = sorted(set(pt["price"] for pt in p.zz1 if pt["type"] == "high"), reverse=True)
    sl = sorted(set(pt["price"] for pt in p.zz1 if pt["type"] == "low"))

    # Resistances
    res_raw = [(v, l) for v, l in all_lvls if v > cur]
    for s in sh:
        if s > cur: res_raw.append((s, "Swing"))
    res_raw.sort(key=lambda x: x[0])

    # Supports
    sup_raw = [(v, l) for v, l in all_lvls if v < cur]
    for s in sl:
        if s < cur: sup_raw.append((s, "Swing"))
    sup_raw.sort(key=lambda x: x[0], reverse=True)

    # Merge close levels
    md = 0.3 * p.atr_last
    def merge(lvls, lim=5):
        if not lvls: return []
        mg = [lvls[0]]
        for v, l in lvls[1:]:
            pv, pl = mg[-1]
            if abs(v - pv) < md:
                mg[-1] = ((pv + v) / 2, l if l.startswith("D.") else pl)
            else:
                mg.append((v, l))
        return mg[:lim]

    resistances = merge(res_raw)
    supports = merge(sup_raw)

    # Store for section 11
    p.pivot_resistances = [v for v, _ in resistances]
    p.pivot_supports = [v for v, _ in supports]

    lines = []
    # Pivot tables
    for pref, data in piv.items():
        name = {"D": "Дневной", "W": "Недельный", "M": "Месячный"}.get(pref, pref)
        lines.append(f"  {name}: P={fp(data['P'], cur, p.dec)}, "
                     f"R1={fp(data['R1'], cur, p.dec)}, S1={fp(data['S1'], cur, p.dec)}")

    lines.append("Ближайшие сопротивления:")
    for v, l in resistances:
        lines.append(f"  {l}: {fp(v, cur, p.dec)}")
        goals.append((v, 6))
    lines.append("Ближайшие поддержки:")
    for v, l in supports:
        lines.append(f"  {l}: {fp(v, cur, p.dec)}")
        goals.append((v, 6))
    return "\n".join(lines), goals


def section_7(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ФИБОНАЧЧИ"""
    goals = []
    # Use windowed high/low (same window as S01)
    max_bars = getattr(p, 'trend_window_bars', p.n)
    w_high = p.high[-max_bars:]
    w_low = p.low[-max_bars:]
    offset = p.n - max_bars
    w_hi_idx = int(np.argmax(w_high)) + offset
    w_lo_idx = int(np.argmin(w_low)) + offset
    hi = float(p.high[w_hi_idx])
    lo = float(p.low[w_lo_idx])
    is_up = w_lo_idx < w_hi_idx
    rng = hi - lo
    lines = [f"Сетка: {'Low→High' if is_up else 'High→Low'} ({fp(lo, p.cur, p.dec)} → {fp(hi, p.cur, p.dec)})"]
    lines.append("Ретрейсменты:")
    for r in FIBO_RET:
        if is_up:
            lvl = hi - rng * r
        else:
            lvl = lo + rng * r
        lines.append(f"  {r:.3f}: {fp(lvl, p.cur, p.dec)}")
        goals.append((lvl, 7))
    lines.append("Расширения:")
    for r in FIBO_EXT:
        if is_up:
            lvl = hi + rng * (r - 1)
        else:
            lvl = lo - rng * (r - 1)
        lines.append(f"  {r:.3f}: {fp(lvl, p.cur, p.dec)}")
        goals.append((lvl, 7))
    return "\n".join(lines), goals


def section_8(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """VSA"""
    cur_vol = float(p.volume[-1])
    avg20 = float(np.mean(p.volume[-20:])) if p.n >= 20 else float(np.mean(p.volume))
    vol_ratio = cur_vol / avg20 if avg20 > 0 else 1.0
    last_rng = float(p.high[-1] - p.low[-1])
    vl_ratio = cur_vol / last_rng if last_rng > 0 else 0
    obv = calc_obv(p.close, p.volume)
    obv_dir = "растёт" if _slope_dir(obv, 20) > 0 else "падает"
    cvd_dir = "N/A"
    if "cvd" in p.avail and "CVD (Close)" in p.df.columns:
        cvd = p.df["CVD (Close)"].values.astype(float)
        cvd_dir = "растёт" if _slope_dir(cvd, 10) > 0 else "падает"
    # anomalies
    if p.n >= 20:
        vol_std = float(np.std(p.volume[-20:]))
        vol_mean = float(np.mean(p.volume[-20:]))
        z = (cur_vol - vol_mean) / vol_std if vol_std > 0 else 0
        anom = f"z={z:.1f}" + (" АНОМАЛИЯ" if abs(z) > 2 else "")
    else:
        anom = "N/A"
    lines = [
        f"Объём текущий: {cur_vol:.0f}, средний(20): {avg20:.0f}, ratio: {vol_ratio:.2f}x",
        f"V/L ratio: {vl_ratio:.0f}",
        f"OBV: {obv_dir}",
        f"CVD: {cvd_dir}",
        f"Аномалии объёма: {anom}",
    ]
    return "\n".join(lines), []


def section_9(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ОБЪЁМНЫЕ ЗОНЫ"""
    goals = []
    # Profile A: from trend start
    pa = volume_profile(p.df.iloc[p.trend_start_idx:], p.cur)
    # Profile B: from last swing low/high
    last_swing_idx = p.swings[-1]["index"] if p.swings else max(0, p.n - 100)
    pb = volume_profile(p.df.iloc[last_swing_idx:], p.cur)

    def pos_str(poc, vah, val, cur):
        if cur > vah: return "выше VAH"
        if cur < val: return "ниже VAL"
        if cur > poc: return "между POC и VAH"
        return "между VAL и POC"

    lines = [
        f"Profile A (от начала тренда): POC={fp(pa['POC'], p.cur, p.dec)}, "
        f"VAH={fp(pa['VAH'], p.cur, p.dec)}, VAL={fp(pa['VAL'], p.cur, p.dec)}",
        f"  Позиция: {pos_str(pa['POC'], pa['VAH'], pa['VAL'], p.cur)}",
        f"Profile B (от последнего свинга): POC={fp(pb['POC'], p.cur, p.dec)}, "
        f"VAH={fp(pb['VAH'], p.cur, p.dec)}, VAL={fp(pb['VAL'], p.cur, p.dec)}",
        f"  Позиция: {pos_str(pb['POC'], pb['VAH'], pb['VAL'], p.cur)}",
    ]
    for vp in [pa, pb]:
        goals.append((vp["POC"], 9))
        goals.append((vp["VAH"], 9))
        goals.append((vp["VAL"], 9))
    return "\n".join(lines), goals


def section_10(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ВАЙКОФФ"""
    ad = calc_ad(p.high, p.low, p.close, p.volume)
    cmf = calc_cmf(p.high, p.low, p.close, p.volume)
    efi = calc_elder_force(p.close, p.volume)
    ad_slope = _slope_dir(ad, 20)
    cmf_last = safe_last(cmf)
    efi_last = safe_last(efi)
    # classify
    if ad_slope > 0 and (not np.isnan(cmf_last) and cmf_last > 0):
        phase = "Накопление (Accumulation)"
        stage = "фаза C-D (Spring / Mark Up)"
    elif ad_slope < 0 and (not np.isnan(cmf_last) and cmf_last < 0):
        phase = "Распределение (Distribution)"
        stage = "фаза C-D (UTAD / Mark Down)"
    else:
        phase = "Переходная"
        stage = "фаза B (Range)"
    # next node
    if "Accumulation" in phase:
        next_node = "SOS (Sign of Strength) — пробой сопротивления"
    elif "Distribution" in phase:
        next_node = "SOW (Sign of Weakness) — пробой поддержки"
    else:
        next_node = "Ожидание Spring/UTAD"
    lines = [
        f"Классификация: {phase}",
        f"Стадия: {stage}",
        f"A/D slope: {'растёт' if ad_slope > 0 else 'падает'}",
        f"CMF: {cmf_last:.3f}" if not np.isnan(cmf_last) else "CMF: N/A",
        f"EFI: {efi_last:.1f}" if not np.isnan(efi_last) else "EFI: N/A",
        f"Следующий узел: {next_node}",
    ]
    return "\n".join(lines), []


def section_11(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ЗОНЫ СБОРА СТОПОВ"""
    goals = []
    atr = p.atr_last
    lines = ["Зоны стопов (S/R + ATR offset):"]
    # Above resistances
    for r in p.pivot_resistances[:3]:
        zone = r + atr * 0.5
        lines.append(f"  Выше {fp(r, p.cur, p.dec)}: зона стопов ~{fp(zone, p.cur, p.dec)}")
        goals.append((zone, 11))
    # Below supports
    for s in p.pivot_supports[:3]:
        zone = s - atr * 0.5
        lines.append(f"  Ниже {fp(s, p.cur, p.dec)}: зона стопов ~{fp(zone, p.cur, p.dec)}")
        goals.append((zone, 11))
    if not p.pivot_resistances and not p.pivot_supports:
        lines.append("  Нет данных о S/R для расчёта зон стопов.")
    return "\n".join(lines), goals


def section_12(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ТЕМП РЫНКА"""
    atr_val = p.atr_last
    atr_pct = atr_val / p.cur * 100 if p.cur else 0
    # daily ATR approximation
    if p.tfh < 24:
        bars_per_day = 24 / p.tfh
        daily_bars = min(int(bars_per_day), p.n)
        if daily_bars > 0:
            daily_range = float(np.max(p.high[-daily_bars:]) - np.min(p.low[-daily_bars:]))
            atr_daily_pct = daily_range / p.cur * 100 if p.cur else 0
        else:
            atr_daily_pct = atr_pct
    else:
        atr_daily_pct = atr_pct
    # K-tempo
    tr20 = np.mean(np.maximum(p.high[-20:] - p.low[-20:],
                               np.maximum(np.abs(p.high[-20:] - np.roll(p.close, 1)[-20:]),
                                          np.abs(p.low[-20:] - np.roll(p.close, 1)[-20:])))) if p.n >= 20 else atr_val
    k_tempo = tr20 / atr_val if atr_val > 0 else 1.0
    # Timing factor
    F = 0.8
    lines = [
        f"ATR({p.tfl}): {atr_val:.{p.dec}f} ({atr_pct:.2f}%)",
        f"ATR(daily) ≈ {atr_daily_pct:.2f}%",
        f"K-tempo: {k_tempo:.2f}",
    ]
    # Timing for nearest goals
    if p.pivot_resistances:
        dist = abs(p.pivot_resistances[0] - p.cur)
        dist_pct = dist / p.cur * 100
        if atr_daily_pct > 0:
            days = math.ceil(dist_pct / (k_tempo * atr_daily_pct * F))
            lines.append(f"До R1: ~{days} дн (при ATR daily {atr_daily_pct:.2f}%)")
    if p.pivot_supports:
        dist = abs(p.pivot_supports[0] - p.cur)
        dist_pct = dist / p.cur * 100
        if atr_daily_pct > 0:
            days = math.ceil(dist_pct / (k_tempo * atr_daily_pct * F))
            lines.append(f"До S1: ~{days} дн")
    return "\n".join(lines), []


def section_13(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ИМБАЛАНСЫ / FVG"""
    goals = []
    bull_fvg = 0; bear_fvg = 0; open_bull = []; open_bear = []
    fvg_start = max(2, p.n - 200)
    for i in range(fvg_start, p.n):
        # Bullish FVG: low[i] > high[i-2]
        if p.low[i] > p.high[i - 2]:
            bull_fvg += 1
            # check if filled
            filled = any(p.low[j] <= p.high[i - 2] for j in range(i + 1, min(i + 20, p.n)))
            if not filled:
                open_bull.append({"low": p.low[i], "high_prev2": p.high[i-2], "bar": i})
        # Bearish FVG: high[i] < low[i-2]
        if p.high[i] < p.low[i - 2]:
            bear_fvg += 1
            filled = any(p.high[j] >= p.low[i - 2] for j in range(i + 1, min(i + 20, p.n)))
            if not filled:
                open_bear.append({"high": p.high[i], "low_prev2": p.low[i-2], "bar": i})
    # first step direction
    if open_bull and open_bear:
        first_dir = "бычий" if open_bull[-1]["bar"] > open_bear[-1]["bar"] else "медвежий"
    elif open_bull:
        first_dir = "бычий"
    elif open_bear:
        first_dir = "медвежий"
    else:
        first_dir = "нейтральный"
    lines = [
        f"Bullish FVG: всего {bull_fvg}, открытых {len(open_bull)}",
        f"Bearish FVG: всего {bear_fvg}, открытых {len(open_bear)}",
        f"Первый шаг: {first_dir}",
    ]
    # Last 3 open FVGs as goals
    for fvg in open_bull[-3:]:
        mid = (fvg["low"] + fvg["high_prev2"]) / 2
        goals.append((mid, 13))
        lines.append(f"  Bull FVG: {fp(mid, p.cur, p.dec)}")
    for fvg in open_bear[-3:]:
        mid = (fvg["high"] + fvg["low_prev2"]) / 2
        goals.append((mid, 13))
        lines.append(f"  Bear FVG: {fp(mid, p.cur, p.dec)}")
    return "\n".join(lines), goals


def section_14(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """BOLLINGER / KELTNER / SQUEEZE"""
    goals = []
    # Bollinger
    if "bb" in p.avail:
        bb_u = safe_last(p.df["Upper"].values.astype(float))
        bb_b = safe_last(p.df["Basis"].values.astype(float))
        bb_l = safe_last(p.df["Lower"].values.astype(float))
    else:
        sma20 = calc_sma(p.close, BB_PERIOD)
        std20 = np.full_like(p.close, np.nan)
        for i in range(BB_PERIOD - 1, p.n):
            std20[i] = np.std(p.close[i - BB_PERIOD + 1:i + 1])
        bb_b = safe_last(sma20); bb_u = bb_b + BB_STD * safe_last(std20); bb_l = bb_b - BB_STD * safe_last(std20)

    bb_width = bb_u - bb_l if not (np.isnan(bb_u) or np.isnan(bb_l)) else 0
    pct_b = (p.cur - bb_l) / bb_width * 100 if bb_width > 0 else 50

    # BB width percentile
    if p.n >= 100:
        widths = []
        sma_arr = calc_sma(p.close, BB_PERIOD)
        for i in range(BB_PERIOD - 1, p.n):
            s = np.std(p.close[i - BB_PERIOD + 1:i + 1])
            widths.append(4 * s)
        if widths:
            bb_pctl = float(np.sum(np.array(widths) < bb_width) / len(widths) * 100)
        else:
            bb_pctl = 50
    else:
        bb_pctl = 50

    # Keltner
    ema20 = calc_ema(p.close, KC_EMA_PERIOD)
    atr10 = calc_atr(p.high, p.low, p.close, KC_ATR_PERIOD)
    kc_mid = safe_last(ema20); kc_atr = safe_last(atr10)
    kc_u = kc_mid + KC_MULT * kc_atr if not np.isnan(kc_atr) else np.nan
    kc_l = kc_mid - KC_MULT * kc_atr if not np.isnan(kc_atr) else np.nan

    # Squeeze
    squeeze = False
    if not (np.isnan(bb_u) or np.isnan(kc_u)):
        squeeze = bb_u < kc_u and bb_l > kc_l

    goals.append((bb_u, 14)); goals.append((bb_l, 14))

    lines = [
        f"BB: Upper={fp(bb_u, p.cur, p.dec)}, Basis={fp(bb_b, p.cur, p.dec)}, Lower={fp(bb_l, p.cur, p.dec)}",
        f"%B = {pct_b:.1f}%, Width percentile = {bb_pctl:.0f}%",
        f"Keltner: Upper={fp(kc_u, p.cur, p.dec)}, Lower={fp(kc_l, p.cur, p.dec)}" if not np.isnan(kc_u) else "Keltner: N/A",
        f"Squeeze: {'ДА — сжатие волатильности' if squeeze else 'НЕТ'}",
    ]
    return "\n".join(lines), goals


def section_15(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ЭФФЕКТИВНОСТЬ"""
    lines = ["Efficiency Ratio:"]
    for w in [10, 20, 50, 100]:
        if p.n < w + 1:
            continue
        net = abs(p.close[-1] - p.close[-w - 1])
        path = sum(abs(p.close[i] - p.close[i - 1]) for i in range(-w, 0))
        er = net / path if path > 0 else 0
        lines.append(f"  ER({w}) = {er:.3f}")
    # Profit Factor & Win Rate
    if p.n >= 20:
        gains = losses = 0; wins = total = 0
        for i in range(-20, 0):
            d = p.close[i] - p.close[i - 1]
            total += 1
            if d > 0: gains += d; wins += 1
            else: losses += abs(d)
        pf = gains / losses if losses > 0 else float("inf")
        wr = wins / total * 100 if total > 0 else 0
        lines.append(f"  Profit Factor (20): {pf:.2f}")
        lines.append(f"  Win Rate (20): {wr:.0f}%")
    return "\n".join(lines), []


def section_16(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """ПОТОК"""
    mfi = calc_mfi(p.high, p.low, p.close, p.volume)
    mfi_val = safe_last(mfi)
    if np.isnan(mfi_val): mfi_val = 50
    if mfi_val > 80: mfi_zone = "перекупленность"
    elif mfi_val < 20: mfi_zone = "перепроданность"
    else: mfi_zone = "нейтральная"
    cmf = calc_cmf(p.high, p.low, p.close, p.volume)
    cmf_val = safe_last(cmf)
    cmf_dir = "положительный (приток)" if (not np.isnan(cmf_val) and cmf_val > 0) else "отрицательный (отток)"
    cvd_dir = "N/A"
    if "cvd" in p.avail and "CVD (Close)" in p.df.columns:
        cvd = p.df["CVD (Close)"].values.astype(float)
        cvd_dir = "растёт" if _slope_dir(cvd, 10) > 0 else "падает"
    lines = [
        f"MFI: {mfi_val:.1f} ({mfi_zone})",
        f"CMF: {cmf_val:.3f} — {cmf_dir}" if not np.isnan(cmf_val) else "CMF: N/A",
        f"CVD: {cvd_dir}",
    ]
    return "\n".join(lines), []


def section_17(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """МИКРОСТРУКТУРА"""
    # VWAP
    if "vwap" in p.avail and "VWAP" in p.df.columns:
        vwap = safe_last(p.df["VWAP"].values.astype(float))
    else:
        vwap = safe_last(calc_vwap(p.high, p.low, p.close, p.volume))
    vwap_dev = (p.cur - vwap) / p.atr_last if p.atr_last > 0 and not np.isnan(vwap) else 0
    # Volume anomalies
    if p.n >= 20:
        vol_arr = p.volume[-20:]
        vm = float(np.mean(vol_arr)); vs = float(np.std(vol_arr))
        anomalies = []
        for i in range(max(0, p.n - 5), p.n):
            z = (p.volume[i] - vm) / vs if vs > 0 else 0
            if abs(z) > 2:
                d = "бычий" if p.close[i] > p.open_[i] else "медвежий"
                anomalies.append(f"бар {i} z={z:.1f} ({d})")
    else:
        anomalies = []
    lines = [
        f"VWAP: {fp(vwap, p.cur, p.dec)}" if not np.isnan(vwap) else "VWAP: N/A",
        f"VWAP deviation: {vwap_dev:.2f} ATR",
        f"Аномалии объёма: {', '.join(anomalies) if anomalies else 'нет'}",
    ]
    return "\n".join(lines), []


def section_18(p: TFPack) -> Tuple[str, List[Tuple[float, int]]]:
    """КОРРЕЛЯЦИИ"""
    # Close-Volume correlation
    if p.n >= 50:
        cv_corr = float(np.corrcoef(p.close[-50:], p.volume[-50:])[0, 1])
    else:
        cv_corr = 0
    # ALMA fan spreads
    a20 = safe_last(calc_alma(p.close, 20))
    a50 = safe_last(calc_alma(p.close, 50))
    a200_val = safe_last(calc_alma(p.close, 200)) if p.n >= 200 else np.nan
    spreads = []
    if not np.isnan(a20) and not np.isnan(a50):
        spreads.append(f"ALMA 20-50: {(a20-a50)/p.cur*100:.2f}%")
    if not np.isnan(a50) and not np.isnan(a200_val):
        spreads.append(f"ALMA 50-200: {(a50-a200_val)/p.cur*100:.2f}%")
    # Autocorrelation lag-1
    if p.n >= 20:
        ret = np.diff(np.log(p.close[-21:]))
        if len(ret) >= 2:
            ac1 = float(np.corrcoef(ret[:-1], ret[1:])[0, 1])
        else:
            ac1 = 0
    else:
        ac1 = 0
    lines = [
        f"Корреляция Close-Volume: {cv_corr:.3f}",
        f"Спреды ALMA: {', '.join(spreads) if spreads else 'N/A'}",
        f"Автокорреляция lag-1: {ac1:.3f}",
    ]
    return "\n".join(lines), []


# ════════════════════════════════════════════════════════════════
#  SECTION 19: ВЫВОД
# ════════════════════════════════════════════════════════════════

def section_19(p: TFPack, all_goals: List[Tuple[float, int]]) -> str:
    """Collect all goals, cluster, build route."""
    cur = p.cur
    if not all_goals:
        return "Недостаточно данных для построения маршрута."

    # Filter NaN / inf
    all_goals = [(g, s) for g, s in all_goals if np.isfinite(g) and g > 0]
    if not all_goals:
        return "Нет валидных целей."

    # Filter goals too far from current price (>20%) or too close (<0.3%)
    all_goals = [(p_, s) for p_, s in all_goals if abs(pct_from(p_, cur)) <= 20.0]
    all_goals = [(p_, s) for p_, s in all_goals if abs(pct_from(p_, cur)) > 0.3]
    if not all_goals:
        return "Нет валидных целей после фильтрации."

    # Cluster goals within 0.3%
    sorted_goals = sorted(all_goals, key=lambda x: x[0])
    clusters = []  # each cluster: {"price": float, "sections": set, "count": int}
    for price, sec in sorted_goals:
        merged = False
        for cl in clusters:
            if abs(price - cl["price"]) / cl["price"] < 0.003:
                cl["prices"].append(price)
                cl["price"] = np.mean(cl["prices"])
                cl["sections"].add(sec)
                cl["count"] += 1
                merged = True
                break
        if not merged:
            clusters.append({"price": price, "prices": [price], "sections": {sec}, "count": 1})

    # Classify
    key_targets = [c for c in clusters if len(c["sections"]) >= 2]
    secondary = [c for c in clusters if len(c["sections"]) == 1]

    # Sort by distance from current price
    key_targets.sort(key=lambda c: abs(c["price"] - cur))
    secondary.sort(key=lambda c: abs(c["price"] - cur))

    lines = []

    # Key targets
    lines.append("📌 Ключевые цели (подтверждены 2+ разделами):")
    if key_targets:
        for c in key_targets[:7]:
            secs = ",".join(str(s) for s in sorted(c["sections"]))
            lines.append(f"  {fp(c['price'], cur, p.dec)} — разделы [{secs}], подтверждений: {c['count']}")
    else:
        lines.append("  нет")

    # Secondary targets
    lines.append("📌 Второстепенные цели:")
    if secondary:
        for c in secondary[:5]:
            secs = ",".join(str(s) for s in sorted(c["sections"]))
            lines.append(f"  {fp(c['price'], cur, p.dec)} — раздел [{secs}]")
    else:
        lines.append("  нет")

    # Route: build zigzag through targets
    above = sorted([c for c in clusters if c["price"] > cur], key=lambda c: c["price"])
    below = sorted([c for c in clusters if c["price"] < cur], key=lambda c: c["price"], reverse=True)
    route = []
    # Liquidity first (stop zones), then POC, then key targets
    # Alternate: nearest above, nearest below, etc.
    ia = ib = 0
    # Determine primary direction
    if p.trend_dir == "восходящий" or p.local_dir == "восходящий":
        # Go up first
        while ia < len(above) or ib < len(below):
            if ia < len(above):
                route.append(above[ia]); ia += 1
            if ib < len(below):
                route.append(below[ib]); ib += 1
    else:
        # Go down first
        while ib < len(below) or ia < len(above):
            if ib < len(below):
                route.append(below[ib]); ib += 1
            if ia < len(above):
                route.append(above[ia]); ia += 1

    lines.append("📌 Вероятный маршрут:")
    if route:
        route_str = " → ".join(fp(c["price"], cur, p.dec) for c in route[:8])
        lines.append(f"  {route_str}")
    else:
        lines.append("  не определён")

    # Timing
    atr_pct = p.atr_last / cur * 100 if cur else 0
    lines.append("📌 Вероятные сроки:")
    if route and atr_pct > 0:
        for c in route[:3]:
            dist_pct = abs(c["price"] - cur) / cur * 100
            days = max(1, math.ceil(dist_pct / (atr_pct * 0.8)))
            lines.append(f"  {fp(c['price'], cur, p.dec)}: ~{days} дн")

    # Invalidation level (capped within 5% of price)
    lines.append("📌 Уровень слома:")
    if p.swings:
        if p.local_dir == "восходящий":
            lows = [s["price"] for s in p.swings if s["type"] == "low"]
            inv = lows[-1] if lows else cur * 0.95
            inv = max(inv, cur * 0.95)  # cap at 5% below
            lines.append(f"  Закрепление ниже {fp(inv, cur, p.dec)} отменяет восходящий сценарий")
        elif p.local_dir == "нисходящий":
            highs = [s["price"] for s in p.swings if s["type"] == "high"]
            inv = highs[-1] if highs else cur * 1.05
            inv = min(inv, cur * 1.05)  # cap at 5% above
            lines.append(f"  Закрепление выше {fp(inv, cur, p.dec)} отменяет нисходящий сценарий")
        else:
            lines.append(f"  Выход из диапазона {fp(cur * 0.97, cur, p.dec)} — {fp(cur * 1.03, cur, p.dec)}")
    else:
        lines.append("  не определён (нет swing points)")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  REPORT RENDERING
# ════════════════════════════════════════════════════════════════

def render_report(p: TFPack) -> str:
    dt = datetime.now(timezone(timedelta(hours=3)))

    header = [
        f"📘ТИКЕР: #{p.ticker} (#{p.exchange})",
        f"БИРЖА: #{p.exchange}",
        f"ТАЙМФРЕЙМ: {p.tfl}",
        f"ЦЕНА: {p.cur:.{p.dec}f}",
        f"ДАТА И ВРЕМЯ: {dt:%Y-%m-%d %H:%M} UTC+3",
        "❗ НЕ ЯВЛЯЕТСЯ ИИР ❗",
        "",
    ]

    sections = [
        (1, section_1), (2, section_2), (3, section_3), (4, section_4),
        (5, section_5), (6, section_6), (7, section_7), (8, section_8),
        (9, section_9), (10, section_10), (11, section_11), (12, section_12),
        (13, section_13), (14, section_14), (15, section_15), (16, section_16),
        (17, section_17), (18, section_18),
    ]

    all_goals: List[Tuple[float, int]] = []
    parts = list(header)

    for sid, fn in sections:
        emoji = SECTION_EMOJIS[sid]
        title = SECTION_TITLES[sid]
        try:
            text, goals = fn(p)
            all_goals.extend(goals)
        except Exception as e:
            text = f"Ошибка: {e}"
        parts.append(f"{emoji} {sid}. {title}")
        parts.append(text)
        parts.append("")

    # Section 19
    emoji19 = SECTION_EMOJIS[19]
    title19 = SECTION_TITLES[19]
    parts.append(f"{emoji19} 19. {title19}")
    try:
        text19 = section_19(p, all_goals)
    except Exception as e:
        text19 = f"Ошибка: {e}"
    parts.append(text19)
    parts.append("")

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════
#  FLASK INTEGRATION
# ════════════════════════════════════════════════════════════════

def run_v2(csv_path: str, on_progress=None) -> dict:
    """Flask-compatible entry point.

    Returns {"meta": {...}, "report_text": str}
    """
    p = build_pack(csv_path)
    report = render_report(p)
    meta = {
        "ticker": p.ticker,
        "exchange": p.exchange,
        "timeframe": p.tfl,
        "current_price": p.cur,
        "n_bars": p.n,
        "period_start": str(p.df["time"].iloc[0]),
        "period_end": str(p.df["time"].iloc[-1]),
    }
    return {"meta": meta, "report_text": report}


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="birzha_v2 — standalone TA report")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {csv_path} ...", file=sys.stderr)
    p = build_pack(str(csv_path))
    print(f"  Ticker: {p.ticker}, Exchange: {p.exchange}, TF: {p.tfl}, "
          f"Price: {p.cur:.{p.dec}f}, Bars: {p.n}", file=sys.stderr)

    print("Generating report ...", file=sys.stderr)
    report = render_report(p)

    if args.output:
        out = Path(args.output)
        out.write_text(report, encoding="utf-8")
        print(f"Report written to {out}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(report)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
