"""
Ретроактивный тест ATR-based cancel level vs старого % cancel.
Сравниваем win rate при разных мультипликаторах ATR.
"""
import sqlite3, csv, json, re, numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

DB = Path(r"Z:\birza_digital\birza_digital.db")
PRICES = {
    "NG1!":     Path(r"Z:\birza_digital\prices\NG1_1H.csv"),
    "PLATINUM": Path(r"Z:\birza_digital\prices\PLATINUM_1H.csv"),
    "XPTUSD":   Path(r"Z:\birza_digital\prices\PLATINUM_1H.csv"),
}

price_cache = {}
for tk, path in PRICES.items():
    bars = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["time"])
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                bars.append({"ts": ts, "h": float(row.get("high", 0)),
                             "l": float(row.get("low", 0)), "c": float(row.get("close", 0))})
            except: pass
    price_cache[tk] = sorted(bars, key=lambda b: b["ts"])

def future_bars(tk, dt, days=10):
    if tk not in price_cache: return []
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    end = dt + timedelta(days=days)
    return [b for b in price_cache[tk] if dt <= b["ts"] <= end]

conn = sqlite3.connect(str(DB))
conn.row_factory = sqlite3.Row
rows = conn.execute("""SELECT ticker, created_at, price, report_text, report_json
    FROM reports WHERE report_text IS NOT NULL AND price > 0
    AND ticker IN ('NG1!','PLATINUM','XPTUSD')""").fetchall()
conn.close()

TF_MULT = {0.25: 2.5, 1.0: 2.0, 4.0: 1.8, 24.0: 1.5, 168.0: 1.2}

def get_dir(text):
    m = re.search(r"вероятный маршрут:\s*(.+?)(?:\n|$)", text, re.I)
    if not m: return None
    t = m.group(1)
    g = len(re.findall(r"\U0001f7e2", t[:300]))
    r = len(re.findall(r"\U0001f534", t[:300]))
    if g > r: return "bull"
    if r > g: return "bear"
    return None

def get_first_target(text, price, direction):
    m = re.search(r"вероятный маршрут:\s*(.+?)(?:\n|$)", text, re.I)
    if not m: return None
    nodes = [float(p) for p in re.findall(r"([\d.]+)\s*\([+-]?[\d.]+%\)", m.group(1))]
    return nodes[1] if len(nodes) >= 2 else None

def simulate(price, direction, first_target, cancel, bars):
    if not bars or not first_target: return None
    for bar in bars:
        if cancel:
            if direction == "bull" and bar["l"] <= cancel: return "cancel"
            if direction == "bear" and bar["h"] >= cancel: return "cancel"
        if direction == "bull" and bar["h"] >= first_target: return "win"
        if direction == "bear" and bar["l"] <= first_target: return "win"
    return "open"

# Test different ATR multipliers
multipliers = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, "original"]

results = {m: {"n": 0, "win": 0, "cancel": 0} for m in multipliers}

for r in rows:
    direction = get_dir(r["report_text"])
    if not direction: continue
    price = r["price"]
    try: dt = datetime.strptime(r["created_at"][:19], "%Y-%m-%d %H:%M:%S")
    except: continue
    bars = future_bars(r["ticker"], dt, 10)
    if not bars: continue
    ft = get_first_target(r["report_text"], price, direction)
    if not ft: continue

    # Get ATR and TF from report_json
    rj = None
    if r["report_json"]:
        try: rj = json.loads(r["report_json"])
        except: pass

    atr = None
    tfh = 1.0
    if rj:
        meta = rj.get("meta") or {}
        atr = meta.get("atr_last")
        tfh = float(meta.get("tf_hours") or 1.0)

    if not atr:
        # Compute ATR from price bars before analysis date
        prev = [b for b in price_cache.get(r["ticker"], []) if b["ts"] < (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt)]
        if len(prev) >= 14:
            highs = [b["h"] for b in prev[-14:]]
            lows = [b["l"] for b in prev[-14:]]
            atr = float(np.mean([h - l for h, l in zip(highs, lows)]))
        else:
            continue

    # Original cancel
    m_orig = re.search(r"отмена маршрута:\s*([\d.]+)", r["report_text"], re.I)
    orig_cancel = float(m_orig.group(1)) if m_orig else None

    best_tf = min(TF_MULT.keys(), key=lambda t: abs(t - tfh))

    for mult in multipliers:
        if mult == "original":
            cancel = orig_cancel
        else:
            if direction == "bull":
                cancel = price - atr * mult
            else:
                cancel = price + atr * mult

        sim = simulate(price, direction, ft, cancel, bars)
        if sim:
            results[mult]["n"] += 1
            if sim == "win": results[mult]["win"] += 1
            elif sim == "cancel": results[mult]["cancel"] += 1

print("=" * 60)
print("ATR MULTIPLIER SWEEP — Win Rate & Cancel Rate")
print("=" * 60)
print(f"  {'Multiplier':>12}  {'N':>5}  {'Win':>5}  {'Cancel':>7}  {'WR':>7}  {'CR':>7}")
print("-" * 60)
for mult in multipliers:
    s = results[mult]
    wr = s["win"] / s["n"] * 100 if s["n"] else 0
    cr = s["cancel"] / s["n"] * 100 if s["n"] else 0
    marker = " <-- BEST" if wr >= 85 else (" <-- TARGET" if wr >= 80 else "")
    print(f"  {str(mult):>12}  {s['n']:>5}  {s['win']:>5}  {s['cancel']:>7}  {wr:>6.1f}%  {cr:>6.1f}%{marker}")
