"""
backtest_v8.py — Массовый walk-forward бэктест НОВОГО v8-пайплайна
(preprocessor + route_engine по Регламенту v4).

Логика: нарезаем историю на окна, на каждом окне строим маршрут,
смотрим в будущие бары — достигнута ли первая цель маршрута ДО слома.

Win  = первая цель достигнута раньше слома
Cancel = слом пробит раньше цели
Open = за горизонт ни то ни другое

Запуск:
  python backtest_v8.py prices/NG1_1H.csv
  python backtest_v8.py --all
  python backtest_v8.py --all --fast   (реже срезы — быстрая прикидка)
"""
import sys
import os
import csv
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.preprocessor import run_preprocessor
from pipeline.route_engine import build_route

PRICES_DIR = Path(__file__).parent / "prices"

# (файл, мин.баров в окне, шаг среза, будущих баров для проверки)
ALL_CONFIGS = [
    ("NG1_15m.csv",      500, 200, 480),   # 15m: future 480 = 5 дней
    ("NG1_1H.csv",       500, 200, 120),   # 1H:  future 120 = 5 дней
    ("NG1_4H.csv",       300,  60,  30),   # 4H:  future 30  = 5 дней
    ("CC1_15m.csv",      500, 200, 480),
    ("CC1_1H.csv",       500, 200, 120),
    ("CC1_4H.csv",       300,  60,  30),
    ("PLATINUM_15m.csv", 500, 200, 480),
    ("PLATINUM_1H.csv",  500, 200, 120),
    ("PLATINUM_4H.csv",  300,  60,  30),
    ("BRENT_1H.csv",     500, 200, 120),
]


def load_bars(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def write_chunk(bars, headers, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(bars)


def forecast_from_chunk(tmp_csv, fname):
    """Прогон v8-пайплайна на окне → структурный прогноз (без парсинга текста)."""
    pre = run_preprocessor(tmp_csv, original_filename=fname)
    smap = {s["section_id"]: s for s in pre["sections_data"]}
    s1 = smap.get(1, {}).get("data", {})
    price = s1.get("current_price")
    direction = (s1.get("senior_trend") or {}).get("direction", "—")
    tf_hours = pre["meta"].get("tf_hours", 1.0)

    rr = build_route(smap, price, direction, tf_hours=tf_hours)
    route = rr.get("route") or []
    if len(route) < 2 or not price:
        return None

    first_target = route[1]["price"]
    slam = rr.get("slam_price")
    final_up = route[-1]["price"] >= price
    return {
        "price": price,
        "first_target": first_target,
        "first_up": first_target > price,
        "final_up": final_up,
        "slam": slam,
        "manip": bool(rr.get("manipulation", {}).get("is_manipulation")),
        "days": rr.get("days"),
    }


def simulate(fc, future_bars):
    """Win = первая цель до слома; Cancel = слом раньше; Open = ничего."""
    target = fc["first_target"]
    slam = fc["slam"]
    up = fc["first_up"]
    for bar in future_bars:
        try:
            h = float(bar.get("high") or 0)
            l = float(bar.get("low") or 0)
        except ValueError:
            continue
        if slam:
            # слом против ФИНАЛЬНОГО направления маршрута
            if fc["final_up"] and l <= slam:
                return "cancel"
            if not fc["final_up"] and h >= slam:
                return "cancel"
        if up and h >= target:
            return "win"
        if not up and l <= target:
            return "win"
    return "open"


def run_one(csv_name, min_bars, step, future_n, fast=False):
    csv_path = PRICES_DIR / csv_name
    if not csv_path.exists():
        print(f"  SKIP {csv_name} — файла нет")
        return None

    all_bars, headers = load_bars(str(csv_path))
    n = len(all_bars)
    if fast:
        step *= 3
    cuts = list(range(min_bars, n - future_n, step))
    if not cuts:
        print(f"  SKIP {csv_name} — мало баров ({n})")
        return None

    tmp = tempfile.mktemp(suffix=".csv")
    stats = {"n": 0, "win": 0, "cancel": 0, "open": 0, "error": 0}
    by_manip = defaultdict(lambda: {"n": 0, "win": 0})

    for i, cut in enumerate(cuts):
        chunk = all_bars[max(0, cut - min_bars):cut]
        future = all_bars[cut:cut + future_n]
        write_chunk(chunk, headers, tmp)
        try:
            fc = forecast_from_chunk(tmp, csv_name)
        except Exception:
            stats["error"] += 1
            continue
        if not fc:
            stats["error"] += 1
            continue

        r = simulate(fc, future)
        stats["n"] += 1
        stats[r] += 1
        mk = "manip" if fc["manip"] else "clean"
        by_manip[mk]["n"] += 1
        if r == "win":
            by_manip[mk]["win"] += 1

        marker = {"win": "W", "cancel": "X", "open": "-"}[r]
        t = chunk[-1].get("time", "?")[:16]
        td = abs(fc["first_target"] - fc["price"]) / fc["price"] * 100
        print(f"    [{i+1}/{len(cuts)}] {t} p={fc['price']:.4g} "
              f"{'UP ' if fc['first_up'] else 'DN '}t={fc['first_target']:.4g}({td:.1f}%) "
              f"{'M' if fc['manip'] else ' '} [{marker}]", flush=True)

    try:
        os.unlink(tmp)
    except OSError:
        pass

    wr = stats["win"] / stats["n"] * 100 if stats["n"] else 0
    cr = stats["cancel"] / stats["n"] * 100 if stats["n"] else 0
    return {"file": csv_name, **stats, "wr": wr, "cr": cr,
            "by_manip": {k: dict(v) for k, v in by_manip.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file", nargs="?", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fast", action="store_true", help="шаг ×3 — быстрая прикидка")
    ap.add_argument("--step", type=int, default=200)
    ap.add_argument("--min-bars", type=int, default=500)
    ap.add_argument("--future-bars", type=int, default=120)
    args = ap.parse_args()

    if args.all:
        configs = ALL_CONFIGS
    elif args.csv_file:
        configs = [(Path(args.csv_file).name, args.min_bars, args.step, args.future_bars)]
    else:
        print("Укажи CSV или --all")
        return

    print(f"Бэктест v8-пайплайна — {len(configs)} конфигураций"
          f"{' (fast)' if args.fast else ''}\n", flush=True)

    all_results = []
    for csv_name, mb, st, fb in configs:
        print(f"=== {csv_name} (окно={mb}, шаг={st}, будущее={fb}) ===", flush=True)
        r = run_one(csv_name, mb, st, fb, fast=args.fast)
        if r:
            all_results.append(r)
            print(f"  >> WR={r['wr']:.1f}% ({r['win']}/{r['n']}), "
                  f"Cancel={r['cr']:.1f}%, errors={r['error']}\n", flush=True)

    if not all_results:
        print("Нет результатов.")
        return

    print("\n" + "=" * 72)
    print("СВОДНАЯ ТАБЛИЦА — v8 route_engine")
    print("=" * 72)
    print(f"  {'Файл':<22} {'N':>4} {'Win':>4} {'Cancel':>6} {'Open':>5} {'WR':>7} {'CR':>7}")
    print("  " + "-" * 64)
    tn = tw = tc = 0
    for r in all_results:
        print(f"  {r['file']:<22} {r['n']:>4} {r['win']:>4} {r['cancel']:>6} "
              f"{r['open']:>5} {r['wr']:>6.1f}% {r['cr']:>6.1f}%")
        tn += r["n"]; tw += r["win"]; tc += r["cancel"]
    print("  " + "-" * 64)
    twr = tw / tn * 100 if tn else 0
    tcr = tc / tn * 100 if tn else 0
    print(f"  {'TOTAL':<22} {tn:>4} {tw:>4} {tc:>6} {'':>5} {twr:>6.1f}% {tcr:>6.1f}%")

    # Разрез по манипуляции
    mn = defaultdict(lambda: {"n": 0, "win": 0})
    for r in all_results:
        for k, v in r["by_manip"].items():
            mn[k]["n"] += v["n"]
            mn[k]["win"] += v["win"]
    print("\n  По чек-листу манипуляции:")
    for k, v in sorted(mn.items()):
        wr = v["win"] / v["n"] * 100 if v["n"] else 0
        print(f"    {k:>6}: WR={wr:.1f}% ({v['win']}/{v['n']})")


if __name__ == "__main__":
    main()
