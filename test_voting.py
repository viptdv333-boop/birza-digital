"""
test_voting.py — Ретроактивный тест нового голосования направления.

Берёт старые отчёты из БД, применяет новую логику голосования (из report_json),
сравнивает старое направление vs новое, проверяет на реальных ценах.
"""
import sqlite3, csv, json, re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

DB = Path(r"Z:\birza_digital\birza_digital.db")
PRICES = {
    "NG1!":     Path(r"Z:\birza_digital\prices\NG1_1H.csv"),
    "PLATINUM": Path(r"Z:\birza_digital\prices\PLATINUM_1H.csv"),
    "XPTUSD":   Path(r"Z:\birza_digital\prices\PLATINUM_1H.csv"),
}

# Load prices
price_cache = {}
for tk, path in PRICES.items():
    bars = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["time"])
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                bars.append({"ts": ts, "h": float(row.get("high",0)),
                             "l": float(row.get("low",0)), "c": float(row.get("close",0))})
            except: pass
    price_cache[tk] = sorted(bars, key=lambda b: b["ts"])
print(f"Prices loaded: {', '.join(f'{k}:{len(v)}' for k,v in price_cache.items())}")

def future_bars(tk, dt, days=10):
    if tk not in price_cache: return []
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    end = dt + timedelta(days=days)
    return [b for b in price_cache[tk] if dt <= b["ts"] <= end]


def get_old_dir(text):
    """Старое направление из отчёта (по эмодзи маршрута)."""
    m = re.search(r"вероятный маршрут:\s*(.+?)(?:\n|$)", text, re.I)
    if not m: return None
    t = m.group(1)
    g = len(re.findall(r"\U0001f7e2", t[:300]))
    r = len(re.findall(r"\U0001f534", t[:300]))
    if g > r: return "bull"
    if r > g: return "bear"
    return None


def voting_direction(rj):
    """Новое направление через голосование 5 индикаторов."""
    if not rj: return None
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

    votes = []

    # 1. Тренд старший (вес 2)
    s1 = (get(1, "direction") or "").lower()
    if "восходящ" in s1: votes.extend([1, 1])
    elif "нисходящ" in s1: votes.extend([-1, -1])

    # 2. Тренд локальный (вес 1)
    # Из текста раздела 1 — "локальный"
    local = (get(1, "local_direction") or "").lower()
    if "восходящ" in local: votes.append(1)
    elif "нисходящ" in local: votes.append(-1)

    # 3. Вайкофф (вес 2)
    wyck = (get(10, "structure_type") or "").lower()
    if "накопление" in wyck: votes.extend([1, 1])
    elif "распределение" in wyck: votes.extend([-1, -1])

    # 4. CMF (вес 1)
    cmf_sig = ((get(16, "cmf", "signal") or "")).lower()
    if "покупател" in cmf_sig: votes.append(1)
    elif "продавц" in cmf_sig: votes.append(-1)

    # 5. ALMA 200 (вес 1)
    alma = (get(1, "alma_200_position") or "").lower()
    if "выше" in alma: votes.append(1)
    elif "ниже" in alma: votes.append(-1)

    # 6. RSI (вес 1)
    rsi_zone = (get(5, "rsi_zone") or "").lower()
    if "перекупленност" in rsi_zone: votes.append(-1)
    elif "перепроданност" in rsi_zone: votes.append(1)

    bull = sum(1 for v in votes if v > 0)
    bear = sum(1 for v in votes if v < 0)
    if bull > bear: return "bull"
    if bear > bull: return "bear"
    return None


def get_first_target(text, price_at, direction):
    m = re.search(r"вероятный маршрут:\s*(.+?)(?:\n|$)", text, re.I)
    if not m: return None
    nodes = [float(p) for p in re.findall(r"([\d.]+)\s*\([+-]?[\d.]+%\)", m.group(1))]
    return nodes[1] if len(nodes) >= 2 else None


def get_cancel(text):
    m = re.search(r"отмена маршрута:\s*([\d.]+)", text, re.I)
    return float(m.group(1)) if m else None


def simulate(price, direction, first_target, cancel, bars):
    if not bars or not first_target: return None
    for bar in bars:
        if cancel:
            if direction == "bull" and bar["l"] <= cancel: return "cancel"
            if direction == "bear" and bar["h"] >= cancel: return "cancel"
        if direction == "bull" and bar["h"] >= first_target: return "win"
        if direction == "bear" and bar["l"] <= first_target: return "win"
    return "open"


# Load reports
conn = sqlite3.connect(str(DB))
conn.row_factory = sqlite3.Row
rows = conn.execute("""SELECT ticker, created_at, price, report_text, report_json
    FROM reports WHERE report_text IS NOT NULL AND price > 0
    AND ticker IN ('NG1!','PLATINUM','XPTUSD')""").fetchall()
conn.close()
print(f"Reports: {len(rows)}\n")

stats = {
    "old": {"n": 0, "win": 0, "cancel": 0},
    "new": {"n": 0, "win": 0, "cancel": 0},
    "changed": {"total": 0, "new_better": 0, "old_better": 0},
    "same_dir": {"n": 0, "win": 0},
}

by_change = defaultdict(lambda: {"n": 0, "win_old": 0, "win_new": 0})

for r in rows:
    price = r["price"]
    try: dt = datetime.strptime(r["created_at"][:19], "%Y-%m-%d %H:%M:%S")
    except: continue

    old_dir = get_old_dir(r["report_text"])
    if not old_dir: continue

    rj = None
    if r["report_json"]:
        try: rj = json.loads(r["report_json"])
        except: pass

    new_dir = voting_direction(rj)
    if not new_dir: new_dir = old_dir  # нет данных — оставляем старое

    bars = future_bars(r["ticker"], dt, 10)
    if not bars: continue

    cancel = get_cancel(r["report_text"])

    # Тест старого направления
    ft_old = get_first_target(r["report_text"], price, old_dir)
    old_result = simulate(price, old_dir, ft_old, cancel, bars)
    if old_result:
        stats["old"]["n"] += 1
        if old_result == "win": stats["old"]["win"] += 1
        elif old_result == "cancel": stats["old"]["cancel"] += 1

    # Тест нового направления
    # Если направление изменилось — нужно найти первую цель в новом направлении
    if new_dir != old_dir:
        # Для нового направления берём ближайшую ключевую цель в нужную сторону
        m = re.search(r"ключевые цели:\s*(.+?)(?:\n|$)", r["report_text"], re.I)
        targets = [float(p) for p in re.findall(r"([\d.]+)\s*\([+-]?[\d.]+%\)", m.group(1))] if m else []
        if new_dir == "bull":
            cands = [t for t in targets if t > price]
        else:
            cands = [t for t in targets if t < price]
        ft_new = min(cands, key=lambda t: abs(t - price)) if cands else None
        new_result = simulate(price, new_dir, ft_new, cancel, bars)
        stats["changed"]["total"] += 1
        change_key = f"{old_dir}->{new_dir}"
        by_change[change_key]["n"] += 1
        if old_result == "win": by_change[change_key]["win_old"] += 1
        if new_result == "win": by_change[change_key]["win_new"] += 1
        if new_result and old_result:
            if new_result == "win" and old_result != "win":
                stats["changed"]["new_better"] += 1
            elif old_result == "win" and new_result != "win":
                stats["changed"]["old_better"] += 1
    else:
        stats["same_dir"]["n"] += 1
        if old_result == "win": stats["same_dir"]["win"] += 1

    # New total
    if new_dir != old_dir and ft_new:
        new_result2 = simulate(price, new_dir, ft_new, cancel, bars)
    else:
        new_result2 = old_result
    if new_result2:
        stats["new"]["n"] += 1
        if new_result2 == "win": stats["new"]["win"] += 1
        elif new_result2 == "cancel": stats["new"]["cancel"] += 1


print("=" * 55)
print("РЕТРОАКТИВНЫЙ ТЕСТ НОВОГО ГОЛОСОВАНИЯ")
print("=" * 55)
o_wr = stats["old"]["win"] / stats["old"]["n"] * 100 if stats["old"]["n"] else 0
n_wr = stats["new"]["win"] / stats["new"]["n"] * 100 if stats["new"]["n"] else 0
print(f"\nСТАРОЕ направление:  {stats['old']['win']}/{stats['old']['n']} = {o_wr:.1f}% WR")
print(f"НОВОЕ направление:   {stats['new']['win']}/{stats['new']['n']} = {n_wr:.1f}% WR")
print(f"Улучшение:           +{n_wr-o_wr:.1f}%")

s = stats["changed"]
print(f"\nСмена направления: {s['total']} отчётов")
print(f"  Новое лучше:  {s['new_better']}")
print(f"  Старое лучше: {s['old_better']}")

print(f"\nПо типам смены:")
for k, v in sorted(by_change.items()):
    o_wr2 = v["win_old"] / v["n"] * 100 if v["n"] else 0
    n_wr2 = v["win_new"] / v["n"] * 100 if v["n"] else 0
    print(f"  {k:<15} N={v['n']:>4}  OLD={o_wr2:.0f}%  NEW={n_wr2:.0f}%")

s2 = stats["same_dir"]
same_wr = s2["win"] / s2["n"] * 100 if s2["n"] else 0
print(f"\nБез смены: {s2['n']} отчётов, WR={same_wr:.1f}%")
