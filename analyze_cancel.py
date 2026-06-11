"""Анализ уровня отмены маршрута vs ATR."""
import sqlite3, json, re, numpy as np
from pathlib import Path

DB = Path(r"Z:\birza_digital\birza_digital.db")
conn = sqlite3.connect(str(DB))
conn.row_factory = sqlite3.Row
rows = conn.execute("""SELECT ticker, created_at, price, report_text, report_json
    FROM reports WHERE report_text IS NOT NULL AND price > 0
    AND ticker IN ('NG1!','PLATINUM','XPTUSD')""").fetchall()
conn.close()

atr_pcts = []
cancel_pcts = []

for r in rows:
    m = re.search(r"отмена маршрута:\s*([\d.]+)", r["report_text"], re.I)
    if not m: continue
    cancel = float(m.group(1))
    price = r["price"]
    cancel_pct = abs(cancel - price) / price * 100
    cancel_pcts.append(cancel_pct)

    rj = None
    if r["report_json"]:
        try: rj = json.loads(r["report_json"])
        except: pass
    if rj:
        atr = (rj.get("meta") or {}).get("atr_last")
        if atr and float(atr) > 0:
            atr_pct = float(atr) / price * 100
            ratio = cancel_pct / atr_pct
            atr_pcts.append({"cancel_pct": cancel_pct, "atr_pct": atr_pct,
                             "ratio": ratio, "ticker": r["ticker"]})

if atr_pcts:
    ratios = [x["ratio"] for x in atr_pcts]
    print(f"Cancel / ATR ratio (N={len(ratios)}):")
    print(f"  Mean:   {np.mean(ratios):.2f}x ATR")
    print(f"  Median: {np.median(ratios):.2f}x ATR")
    print(f"  Min:    {np.min(ratios):.2f}x  Max: {np.max(ratios):.2f}x")
    print(f"  <1x ATR:  {sum(1 for r in ratios if r < 1):>4}  (слишком близко)")
    print(f"  1-2x ATR: {sum(1 for r in ratios if 1 <= r < 2):>4}  (норма)")
    print(f"  >2x ATR:  {sum(1 for r in ratios if r >= 2):>4}  (далеко)")

print(f"\nCancel level от цены (N={len(cancel_pcts)}):")
print(f"  Mean:   {np.mean(cancel_pcts):.2f}%")
print(f"  Median: {np.median(cancel_pcts):.2f}%")
print(f"  <2%:  {sum(1 for c in cancel_pcts if c < 2):>4}  (очень близко)")
print(f"  2-4%: {sum(1 for c in cancel_pcts if 2 <= c < 4):>4}")
print(f"  4-6%: {sum(1 for c in cancel_pcts if 4 <= c < 6):>4}")
print(f"  >6%:  {sum(1 for c in cancel_pcts if c >= 6):>4}")

# По тикерам
print("\nПо тикерам:")
tk_data = {}
for x in atr_pcts:
    tk = x["ticker"]
    if tk not in tk_data: tk_data[tk] = []
    tk_data[tk].append(x["ratio"])
for tk, ratios in sorted(tk_data.items()):
    print(f"  {tk:<15} median={np.median(ratios):.2f}x ATR  mean={np.mean(ratios):.2f}x")
