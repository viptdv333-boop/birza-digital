"""
Финальный тест: ATR cancel (3.0x) + умные фильтры → цель 90% WR.
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
print(f"Prices: {', '.join(f'{k}:{len(v)}' for k,v in price_cache.items())}")

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
print(f"Reports: {len(rows)}\n")

TF_MULT = {0.25: 2.5, 1.0: 2.0, 4.0: 1.8, 24.0: 1.5, 168.0: 1.2}
ATR_CANCEL_MULT = 3.0  # оптимум по бэктесту


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


def get_atr(ticker, dt):
    """ATR из последних 14 баров перед датой анализа."""
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    prev = [b for b in price_cache.get(ticker, []) if b["ts"] < dt]
    if len(prev) < 14: return None
    return float(np.mean([b["h"] - b["l"] for b in prev[-14:]]))


def smart_score(rj, direction):
    """Балльная система фильтров (из backtest.py). 0 = чисто, >=2 = риск."""
    if not rj: return 0
    secs = rj.get("sections", {})

    def get(sec_id, *keys, default=None):
        s = secs.get(str(sec_id), {})
        d = s.get("data", {}) if isinstance(s, dict) else {}
        v = d
        for k in keys:
            if not isinstance(v, dict): return default
            v = v.get(k, default)
            if v is None: return default
        return v

    score = 0
    if direction == "bull":
        wyck = (get(10, "structure_type") or "").lower()
        if "распределение" in wyck: score += 2
        dist_sc = float(get(10, "evidence", "distribution_score") or 0)
        if dist_sc >= 4: score += 2
        alma_pos = (get(1, "alma_200_position") or "").lower()
        if "ниже" in alma_pos: score += 1
        cmf_sig = (get(16, "cmf", "signal") or "").lower()
        if "продавц" in cmf_sig: score += 2
        rsi_zone = (get(5, "rsi_zone") or "").lower()
        if "перекупленност" in rsi_zone: score += 1
    return score


def simulate(price, direction, first_target, cancel, bars):
    if not bars or not first_target: return None
    for bar in bars:
        if cancel:
            if direction == "bull" and bar["l"] <= cancel: return "cancel"
            if direction == "bear" and bar["h"] >= cancel: return "cancel"
        if direction == "bull" and bar["h"] >= first_target: return "win"
        if direction == "bear" and bar["l"] <= first_target: return "win"
    return "open"


# Test scenarios
scenarios = {
    "baseline (orig cancel, no filter)": {"use_atr": False, "filter": False},
    "ATR cancel only":                   {"use_atr": True,  "filter": False},
    "Filters only (score<2)":            {"use_atr": False, "filter": True},
    "ATR + Filters (FINAL)":             {"use_atr": True,  "filter": True},
}
results = {k: {"n": 0, "win": 0, "cancel": 0, "skip": 0} for k in scenarios}
tf_results = defaultdict(lambda: {"n": 0, "win": 0})
ticker_results = defaultdict(lambda: {"n": 0, "win": 0})

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

    rj = None
    if r["report_json"]:
        try: rj = json.loads(r["report_json"])
        except: pass

    # Original cancel
    mc = re.search(r"отмена маршрута:\s*([\d.]+)", r["report_text"], re.I)
    orig_cancel = float(mc.group(1)) if mc else None

    # ATR cancel
    atr = get_atr(r["ticker"], dt)
    if atr:
        atr_cancel = price - atr * ATR_CANCEL_MULT if direction == "bull" else price + atr * ATR_CANCEL_MULT
    else:
        atr_cancel = orig_cancel

    score = smart_score(rj, direction)
    tf = (rj.get("meta") or {}).get("timeframe", "?") if rj else "?"
    tk = r["ticker"]

    for name, cfg in scenarios.items():
        cancel = atr_cancel if cfg["use_atr"] else orig_cancel
        if cfg["filter"] and score >= 2:
            results[name]["skip"] += 1
            continue
        sim = simulate(price, direction, ft, cancel, bars)
        if sim:
            results[name]["n"] += 1
            if sim == "win": results[name]["win"] += 1
            elif sim == "cancel": results[name]["cancel"] += 1

        # For final scenario: track by TF and ticker
        if name == "ATR + Filters (FINAL)" and sim:
            tf_results[tf]["n"] += 1
            ticker_results[tk]["n"] += 1
            if sim == "win":
                tf_results[tf]["win"] += 1
                ticker_results[tk]["win"] += 1

print("=" * 65)
print(f"ФИНАЛЬНЫЙ ТЕСТ: ATR {ATR_CANCEL_MULT}x + Фильтры (score<2)")
print("=" * 65)
print(f"\n  {'Сценарий':<40} {'N':>5}  {'Win':>5}  {'Cancel':>7}  {'WR':>7}  {'Skip':>6}")
print("-" * 65)
for name, s in results.items():
    wr = s["win"] / s["n"] * 100 if s["n"] else 0
    cr = s["cancel"] / s["n"] * 100 if s["n"] else 0
    marker = " <-- 90%!" if wr >= 90 else (" <-- TARGET" if wr >= 85 else "")
    print(f"  {name:<40} {s['n']:>5}  {s['win']:>5}  {s['cancel']:>7}  {wr:>6.1f}%  {s['skip']:>6}{marker}")

print("\n  По тикерам (ATR + Filters):")
for tk, s in sorted(ticker_results.items(), key=lambda x: -x[1]["n"]):
    wr = s["win"] / s["n"] * 100 if s["n"] else 0
    print(f"    {tk:<15} N={s['n']:>4}  Win={s['win']:>4}  WR={wr:.1f}%")

print("\n  По ТФ (ATR + Filters):")
for tf, s in sorted(tf_results.items(), key=lambda x: -x[1]["n"]):
    wr = s["win"] / s["n"] * 100 if s["n"] else 0
    print(f"    {tf:<12} N={s['n']:>4}  Win={s['win']:>4}  WR={wr:.1f}%")
