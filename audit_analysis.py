import sqlite3, json, re, csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

DB = Path(r"Z:\birza_digital\birza_digital.db")
PRICES = {
    "NG1!": Path(r"Z:\birza_digital\prices\NG1_1H.csv"),
    "PLATINUM": Path(r"Z:\birza_digital\prices\PLATINUM_1H.csv"),
    "XPTUSD": Path(r"Z:\birza_digital\prices\PLATINUM_1H.csv"),
}

# Load prices
def load_p(path):
    bars = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["time"])
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                bars.append({"ts": ts, "h": float(row.get("high",0)), "l": float(row.get("low",0)), "c": float(row.get("close",0))})
            except: pass
    return sorted(bars, key=lambda b: b["ts"])

price_cache = {k: load_p(v) for k,v in PRICES.items()}

def future(tk, dt, days):
    if tk not in price_cache: return []
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    end = dt + timedelta(days=days)
    return [b for b in price_cache[tk] if dt <= b["ts"] <= end]

conn = sqlite3.connect(str(DB))
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT ticker, created_at, price, report_text, report_json FROM reports WHERE report_text IS NOT NULL AND price > 0 AND ticker IN ('NG1!','PLATINUM','XPTUSD')").fetchall()
conn.close()

# For each report: get direction, first target, cancel, simulate
def get_dir(text):
    m = re.search(r"вероятный маршрут:\s*(.+?)(?:\n|$)", text, re.I)
    if not m: return None
    t = m.group(1)
    g = len(re.findall(r"\U0001f7e2", t[:300]))
    r = len(re.findall(r"\U0001f534", t[:300]))
    return "bull" if g > r else ("bear" if r > g else None)

def get_first_target(text, price_at, direction):
    m = re.search(r"вероятный маршрут:\s*(.+?)(?:\n|$)", text, re.I)
    if not m: return None
    nodes = [float(p) for p in re.findall(r"([\d.]+)\s*\([+-]?[\d.]+%\)", m.group(1))]
    if len(nodes) >= 2: return nodes[1]
    return None

def get_cancel(text):
    m = re.search(r"отмена маршрута:\s*([\d.]+)", text, re.I)
    return float(m.group(1)) if m else None

# Analyze sections in report_json
section_win = defaultdict(lambda: defaultdict(list))  # section_id -> field -> [win_rate]
wyckoff_stats = defaultdict(lambda: {"wins": 0, "total": 0})
cmf_stats = defaultdict(lambda: {"wins": 0, "total": 0})
alma_stats = defaultdict(lambda: {"wins": 0, "total": 0})
cancel_pct_list = {"win": [], "loss": []}

for r in rows:
    direction = get_dir(r["report_text"])
    if not direction: continue
    price = r["price"]
    try: dt = datetime.strptime(r["created_at"][:19], "%Y-%m-%d %H:%M:%S")
    except: continue
    
    bars = future(r["ticker"], dt, 10)
    if not bars: continue
    
    first_t = get_first_target(r["report_text"], price, direction)
    cancel = get_cancel(r["report_text"])
    if not first_t: continue
    
    win = False
    cancel_hit = False
    for bar in bars:
        if cancel:
            if direction == "bull" and bar["l"] <= cancel: cancel_hit = True; break
            if direction == "bear" and bar["h"] >= cancel: cancel_hit = True; break
        if direction == "bull" and bar["h"] >= first_t: win = True; break
        if direction == "bear" and bar["l"] <= first_t: win = True; break
    
    # Cancel level distance
    if cancel:
        cancel_dist = abs(cancel - price) / price * 100
        if win: cancel_pct_list["win"].append(cancel_dist)
        elif cancel_hit: cancel_pct_list["loss"].append(cancel_dist)
    
    # Wyckoff
    rj = None
    if r["report_json"]:
        try: rj = json.loads(r["report_json"])
        except: pass
    
    if rj:
        secs = rj.get("sections", {})
        # Wyckoff
        s10 = (secs.get("10") or {}).get("data", {})
        wyck_type = (s10.get("structure_type") or "").lower()
        if wyck_type:
            key = f"{wyck_type}_{direction}"
            wyckoff_stats[key]["total"] += 1
            if win: wyckoff_stats[key]["wins"] += 1
        
        # CMF
        s16 = (secs.get("16") or {}).get("data", {})
        cmf_sig = ((s16.get("cmf") or {}).get("signal") or "").lower()
        if cmf_sig:
            key = f"{cmf_sig}_{direction}"
            cmf_stats[key]["total"] += 1
            if win: cmf_stats[key]["wins"] += 1
        
        # ALMA
        s1 = (secs.get("1") or {}).get("data", {})
        alma_pos = (s1.get("alma_200_position") or "").lower()
        if alma_pos:
            key = f"{alma_pos}_{direction}"
            alma_stats[key]["total"] += 1
            if win: alma_stats[key]["wins"] += 1

print("=== WYCKOFF vs WIN RATE ===")
for k, v in sorted(wyckoff_stats.items()):
    wr = v["wins"]/v["total"]*100 if v["total"] else 0
    print(f"  {k:<35} N={v['total']:>3}  WR={wr:.1f}%")

print("\n=== CMF vs WIN RATE ===")
for k, v in sorted(cmf_stats.items()):
    wr = v["wins"]/v["total"]*100 if v["total"] else 0
    print(f"  {k:<35} N={v['total']:>3}  WR={wr:.1f}%")

print("\n=== ALMA 200 POSITION vs WIN RATE ===")
for k, v in sorted(alma_stats.items()):
    wr = v["wins"]/v["total"]*100 if v["total"] else 0
    print(f"  {k:<35} N={v['total']:>3}  WR={wr:.1f}%")

print("\n=== CANCEL LEVEL DISTANCE ===")
import numpy as np
if cancel_pct_list["win"]:
    print(f"  Winners cancel dist: avg={np.mean(cancel_pct_list['win']):.2f}%  median={np.median(cancel_pct_list['win']):.2f}%")
if cancel_pct_list["loss"]:
    print(f"  Losers cancel dist:  avg={np.mean(cancel_pct_list['loss']):.2f}%  median={np.median(cancel_pct_list['loss']):.2f}%")
