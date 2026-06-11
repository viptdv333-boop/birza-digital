"""
backtest_new.py — Бэктест НОВОГО скрипта.

Нарезает CSV на периоды, прогоняет birzha_v2, проверяет по реальным ценам.
Win rate = первая цель достигнута ДО уровня отмены.

Запуск:
  python backtest_new.py prices/NG1_1H.csv --step 200
  python backtest_new.py --all    # все 9 комбинаций
"""
import sys, os, re, csv, argparse, tempfile, shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Загрузить birzha_v2 ОДИН раз
import birzha_v2

PRICES_DIR = Path(__file__).parent / "prices"
ALL_CONFIGS = [
    ("NG1_15m.csv",      300, 100, 480),   # 15m: 300 min, step 100, future 480 (5 дней)
    ("NG1_1H.csv",       500, 200, 120),   # 1H:  500 min, step 200, future 120 (5 дней)
    ("NG1_4H.csv",       200,  50,  30),   # 4H:  200 min, step 50,  future 30 (5 дней)
    ("CC1_15m.csv",      300, 100, 480),
    ("CC1_1H.csv",       500, 200, 120),
    ("CC1_4H.csv",       200,  50,  30),
    ("PLATINUM_15m.csv", 300, 100, 480),
    ("PLATINUM_1H.csv",  500, 200, 120),
    ("PLATINUM_4H.csv",  200,  50,  30),
]


def load_bars(path):
    bars = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            bars.append(row)
    return bars, headers


def write_chunk(bars, headers, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(bars)


def parse_report(report: str) -> dict:
    p = {"direction": None, "first_target": None, "cancel_level": None,
         "quality": None, "route": []}

    m = re.search(r"вероятный маршрут:\s*(.+?)(?:\n|$)", report, re.I)
    if m:
        rt = m.group(1)
        nodes = [float(x) for x in re.findall(r"([\d.]+)\s*\([+-]?[\d.]+%\)", rt)]
        p["route"] = nodes
        if len(nodes) >= 2:
            p["first_target"] = nodes[1]
        greens = len(re.findall(r"\U0001f7e2", rt[:500]))
        reds = len(re.findall(r"\U0001f534", rt[:500]))
        p["direction"] = "bull" if greens > reds else ("bear" if reds > greens else None)

    m = re.search(r"(?:отмена маршрута|Закрепление (?:ниже|выше))\s*:?\s*([\d.]+)", report, re.I)
    if m:
        p["cancel_level"] = float(m.group(1))

    m = re.search(r"score=(\d+)", report, re.I)
    if m:
        p["quality"] = int(m.group(1))

    return p


def simulate(price, direction, target, cancel, future_bars):
    if not future_bars or not target:
        return "no_data"
    for bar in future_bars:
        h, l = float(bar.get("high") or 0), float(bar.get("low") or 0)
        if cancel:
            if direction == "bull" and l <= cancel: return "cancel"
            if direction == "bear" and h >= cancel: return "cancel"
        if direction == "bull" and h >= target: return "win"
        if direction == "bear" and l <= target: return "win"
    return "open"


def run_one(csv_name, min_bars, step, future_n):
    csv_path = PRICES_DIR / csv_name
    if not csv_path.exists():
        print(f"  SKIP {csv_name} -- file not found")
        return None

    all_bars, headers = load_bars(str(csv_path))
    n = len(all_bars)
    tmp = tempfile.mktemp(suffix=".csv")

    cuts = list(range(min_bars, n - future_n, step))
    if not cuts:
        print(f"  SKIP {csv_name} -- too few bars ({n})")
        return None

    stats = {"n": 0, "win": 0, "cancel": 0, "open": 0, "error": 0}
    by_q = defaultdict(lambda: {"n": 0, "win": 0})

    for i, cut in enumerate(cuts):
        # Даём скрипту только последние min_bars (не весь файл!)
        chunk_start = max(0, cut - min_bars)
        chunk = all_bars[chunk_start:cut]
        future = all_bars[cut:cut + future_n]
        last_close = float(chunk[-1].get("close") or 0)
        last_time = chunk[-1].get("time", "?")[:16]

        write_chunk(chunk, headers, tmp)
        try:
            result = birzha_v2.run_v2(tmp)
            report = result.get("report_text", "")
        except Exception as e:
            stats["error"] += 1
            continue

        p = parse_report(report)
        if not p["direction"] or not p["first_target"]:
            stats["error"] += 1
            continue

        r = simulate(last_close, p["direction"], p["first_target"], p["cancel_level"], future)
        stats["n"] += 1
        if r in stats:
            stats[r] += 1

        q = p.get("quality") or 0
        by_q[q]["n"] += 1
        if r == "win":
            by_q[q]["win"] += 1

        marker = "W" if r == "win" else ("X" if r == "cancel" else "-")
        td = abs(p["first_target"] - last_close) / last_close * 100 if last_close else 0
        print(f"    [{i+1}/{len(cuts)}] {last_time} p={last_close:.4g} "
              f"dir={p['direction']:5s} t={p['first_target']:.4g}({td:.1f}%) "
              f"q={q} [{marker}]", flush=True)

    try: os.unlink(tmp)
    except: pass

    wr = stats["win"] / stats["n"] * 100 if stats["n"] else 0
    cr = stats["cancel"] / stats["n"] * 100 if stats["n"] else 0
    return {"file": csv_name, "n": stats["n"], "win": stats["win"],
            "cancel": stats["cancel"], "wr": wr, "cr": cr,
            "error": stats["error"], "by_quality": dict(by_q)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", nargs="?", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--min-bars", type=int, default=500)
    parser.add_argument("--future-bars", type=int, default=120)
    args = parser.parse_args()

    if args.all:
        configs = ALL_CONFIGS
    elif args.csv_file:
        configs = [(args.csv_file, args.min_bars, args.step, args.future_bars)]
    else:
        print("Укажи CSV файл или --all"); return

    print(f"Бэктест нового скрипта — {len(configs)} конфигураций\n", flush=True)

    all_results = []
    for csv_name, mb, st, fb in configs:
        print(f"=== {csv_name} (min={mb}, step={st}, future={fb}) ===", flush=True)
        r = run_one(csv_name, mb, st, fb)
        if r:
            all_results.append(r)
            print(f"  >> WR={r['wr']:.1f}% ({r['win']}/{r['n']}), Cancel={r['cr']:.1f}%\n", flush=True)

    if not all_results:
        print("Нет результатов."); return

    # Сводка
    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 70)
    print(f"  {'Файл':<25} {'N':>5} {'Win':>5} {'Cancel':>7} {'WR':>8} {'CR':>8}")
    print("  " + "-" * 62)
    total_n = total_w = total_c = 0
    for r in all_results:
        print(f"  {r['file']:<25} {r['n']:>5} {r['win']:>5} {r['cancel']:>7} {r['wr']:>7.1f}% {r['cr']:>7.1f}%")
        total_n += r["n"]; total_w += r["win"]; total_c += r["cancel"]
    total_wr = total_w / total_n * 100 if total_n else 0
    total_cr = total_c / total_n * 100 if total_n else 0
    print("  " + "-" * 62)
    print(f"  {'TOTAL':<25} {total_n:>5} {total_w:>5} {total_c:>7} {total_wr:>7.1f}% {total_cr:>7.1f}%")

    # По quality
    print(f"\n  По качеству (score):")
    merged_q = defaultdict(lambda: {"n": 0, "win": 0})
    for r in all_results:
        for q, s in r["by_quality"].items():
            merged_q[q]["n"] += s["n"]
            merged_q[q]["win"] += s["win"]
    for q in sorted(merged_q.keys()):
        s = merged_q[q]
        qwr = s["win"] / s["n"] * 100 if s["n"] else 0
        print(f"    score={q}: N={s['n']:>3}  Win={s['win']:>3}  WR={qwr:.1f}%")


if __name__ == "__main__":
    main()
