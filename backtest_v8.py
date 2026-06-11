"""
backtest_v8.py — Массовый walk-forward бэктест v8-пайплайна
(preprocessor + route_engine по Регламенту v4).

Метрики:
  • WR первой цели (достигнута до слома, горизонт 5 торговых дней)
  • Глубина маршрута: сколько целей цепочки отработало до слома (1/2/3+)
  • Сроки: доля сделок, уложившихся в прогнозный диапазон дней
  • Экспектансия: WR×ср.путь_к_цели% − CR×ср.путь_к_слому% (R:R-взвешенная)

Запуск:
  python backtest_v8.py prices/NG1_1H.csv
  python backtest_v8.py --all
  python backtest_v8.py --all --fast          (шаг ×3 — прикидка)
  python backtest_v8.py --calibrate           (подбор запаса слома на 4H)
  python backtest_v8.py NG1_4H.csv --slam-buf 0.5   (слом шире на 0.5×ATR_дн)
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
    ("NG1_15m.csv",      500, 200, 480),
    ("NG1_1H.csv",       500, 200, 120),
    ("NG1_4H.csv",       300,  60,  30),
    ("CC1_15m.csv",      500, 200, 480),
    ("CC1_1H.csv",       500, 200, 120),
    ("CC1_4H.csv",       300,  60,  30),
    ("PLATINUM_15m.csv", 500, 200, 480),
    ("PLATINUM_1H.csv",  500, 200, 120),
    ("PLATINUM_4H.csv",  300,  60,  30),
    ("BRENT_1H.csv",     500, 200, 120),
]
CONFIGS_4H = [c for c in ALL_CONFIGS if "_4H" in c[0]]


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
    """Прогон v8-пайплайна на окне → структурный прогноз."""
    pre = run_preprocessor(tmp_csv, original_filename=fname)
    smap = {s["section_id"]: s for s in pre["sections_data"]}
    s1 = smap.get(1, {}).get("data", {})
    s12 = smap.get(12, {}).get("data", {})
    price = s1.get("current_price")
    direction = (s1.get("senior_trend") or {}).get("direction", "—")
    tf_hours = pre["meta"].get("tf_hours", 1.0)

    rr = build_route(smap, price, direction, tf_hours=tf_hours)
    route = rr.get("route") or []
    if len(route) < 2 or not price:
        return None

    final_up = route[-1]["price"] >= price
    # Цели импульса в порядке маршрута (без P0/откатов/манип-точек)
    impulse, seen = [], set()
    for rp in route[1:]:
        st = rp.get("status", "")
        if st in ("start", "pullback", "manip", "manip_return", "lpsy"):
            continue
        p = rp["price"]
        if ((p > price) == final_up) and round(p, 6) not in seen:
            impulse.append(p)
            seen.add(round(p, 6))
    if not impulse:
        return None

    return {
        "price": price,
        "tf_hours": tf_hours,
        "targets": impulse,                 # цели по глубине
        "first_target": impulse[0],
        "final_up": final_up,
        "slam": rr.get("slam_price"),
        "atr_daily": s12.get("atr_daily") or 0.0,
        "manip": bool(rr.get("manipulation", {}).get("is_manipulation")),
        "days": rr.get("days") or 0,
        "days_max": rr.get("days_max") or rr.get("days") or 0,
    }


def simulate(fc, future_bars, slam_buf_atr=0.0):
    """Прогон будущих баров.

    Returns dict: result (win/cancel/open по 1-й цели), depth (целей достигнуто
    до слома), bars_to_depth (баров до самой глубокой цели).
    """
    up = fc["final_up"]
    slam = fc["slam"]
    if slam and slam_buf_atr and fc["atr_daily"]:
        # Расширяем слом ОТ цены на buf×ATR_дн
        slam = slam - slam_buf_atr * fc["atr_daily"] if up else slam + slam_buf_atr * fc["atr_daily"]

    targets = fc["targets"]
    depth = 0
    bars_to_depth = None
    result = None

    for i, bar in enumerate(future_bars):
        try:
            h = float(bar.get("high") or 0)
            l = float(bar.get("low") or 0)
        except ValueError:
            continue
        if slam:
            if up and l <= slam:
                result = result or ("cancel" if depth == 0 else None)
                break
            if not up and h >= slam:
                result = result or ("cancel" if depth == 0 else None)
                break
        while depth < len(targets):
            t = targets[depth]
            hit = (h >= t) if up else (l <= t)
            if not hit:
                break
            depth += 1
            bars_to_depth = i + 1
        if depth >= len(targets):
            break

    if result is None:
        result = "win" if depth >= 1 else "open"
    return {"result": result, "depth": depth, "bars_to_depth": bars_to_depth}


def run_one(csv_name, min_bars, step, future_n, fast=False, slam_buf=0.0,
            quiet=False, sweep_bufs=None):
    """Если sweep_bufs задан — на каждом срезе симулируем для всех буферов."""
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

    bufs = sweep_bufs if sweep_bufs is not None else [slam_buf]
    tmp = tempfile.mktemp(suffix=".csv")

    # stats[buf] = {...}
    def new_stats():
        return {"n": 0, "win": 0, "cancel": 0, "open": 0, "error": 0,
                "depth_sum": 0, "depth_counts": defaultdict(int),
                "on_time": 0, "timed": 0,
                "gain_pct_sum": 0.0, "loss_pct_sum": 0.0,
                "by_manip": defaultdict(lambda: {"n": 0, "win": 0})}
    stats = {b: new_stats() for b in bufs}

    for i, cut in enumerate(cuts):
        chunk = all_bars[max(0, cut - min_bars):cut]
        future = all_bars[cut:cut + future_n]
        write_chunk(chunk, headers, tmp)
        try:
            fc = forecast_from_chunk(tmp, csv_name)
        except Exception:
            for b in bufs:
                stats[b]["error"] += 1
            continue
        if not fc:
            for b in bufs:
                stats[b]["error"] += 1
            continue

        price = fc["price"]
        t1_dist = abs(fc["first_target"] - price) / price * 100

        for b in bufs:
            st = stats[b]
            sim = simulate(fc, future, slam_buf_atr=b)
            r = sim["result"]
            st["n"] += 1
            st[r] += 1
            st["depth_sum"] += sim["depth"]
            st["depth_counts"][min(sim["depth"], 4)] += 1

            # Экспектансия: win → +путь до 1-й цели; cancel → −путь до слома
            if r == "win":
                st["gain_pct_sum"] += t1_dist
            elif r == "cancel":
                slam_eff = fc["slam"]
                if slam_eff and b and fc["atr_daily"]:
                    slam_eff = (slam_eff - b * fc["atr_daily"]) if fc["final_up"] \
                        else (slam_eff + b * fc["atr_daily"])
                if slam_eff:
                    st["loss_pct_sum"] += abs(slam_eff - price) / price * 100

            # Сроки: добрались до 1-й цели — уложились ли в прогноз days_max
            if sim["depth"] >= 1 and sim["bars_to_depth"] and fc["days_max"]:
                actual_days = sim["bars_to_depth"] * fc["tf_hours"] / 24.0
                st["timed"] += 1
                if actual_days <= fc["days_max"] + 1e-9:
                    st["on_time"] += 1

            mk = "manip" if fc["manip"] else "clean"
            st["by_manip"][mk]["n"] += 1
            if r == "win":
                st["by_manip"][mk]["win"] += 1

        if not quiet and sweep_bufs is None:
            sim0 = simulate(fc, future, slam_buf_atr=bufs[0])
            marker = {"win": "W", "cancel": "X", "open": "-"}[sim0["result"]]
            t = chunk[-1].get("time", "?")[:16]
            print(f"    [{i+1}/{len(cuts)}] {t} p={price:.4g} "
                  f"{'UP ' if fc['final_up'] else 'DN '}t1={fc['first_target']:.4g}({t1_dist:.1f}%) "
                  f"d={sim0['depth']} {'M' if fc['manip'] else ' '} [{marker}]", flush=True)

    try:
        os.unlink(tmp)
    except OSError:
        pass

    out = {}
    for b in bufs:
        st = stats[b]
        nn = st["n"]
        wr = st["win"] / nn * 100 if nn else 0
        cr = st["cancel"] / nn * 100 if nn else 0
        avg_gain = st["gain_pct_sum"] / st["win"] if st["win"] else 0
        avg_loss = st["loss_pct_sum"] / st["cancel"] if st["cancel"] else 0
        expect = (wr / 100) * avg_gain - (cr / 100) * avg_loss
        out[b] = {"file": csv_name, **{k: st[k] for k in ("n", "win", "cancel", "open", "error")},
                  "wr": wr, "cr": cr,
                  "avg_depth": st["depth_sum"] / nn if nn else 0,
                  "depth_counts": dict(st["depth_counts"]),
                  "on_time": st["on_time"], "timed": st["timed"],
                  "avg_gain": avg_gain, "avg_loss": avg_loss, "expect": expect,
                  "by_manip": {k: dict(v) for k, v in st["by_manip"].items()}}
    return out if sweep_bufs is not None else out[bufs[0]]


def print_summary(all_results):
    print("\n" + "=" * 100)
    print("СВОДНАЯ ТАБЛИЦА — v8 route_engine")
    print("=" * 100)
    print(f"  {'Файл':<20} {'N':>4} {'WR':>6} {'CR':>6} {'Глуб':>5} "
          f"{'d1+':>5} {'d2+':>5} {'d3+':>5} {'Срок':>6} {'+%':>5} {'-%':>5} {'Ожид':>6}")
    print("  " + "-" * 96)
    agg = defaultdict(float)
    agg_depth = defaultdict(int)
    mn = defaultdict(lambda: {"n": 0, "win": 0})
    for r in all_results:
        nn = r["n"] or 1
        d = r["depth_counts"]
        d1 = sum(v for k, v in d.items() if k >= 1) / nn * 100
        d2 = sum(v for k, v in d.items() if k >= 2) / nn * 100
        d3 = sum(v for k, v in d.items() if k >= 3) / nn * 100
        ontime = r["on_time"] / r["timed"] * 100 if r["timed"] else 0
        print(f"  {r['file']:<20} {r['n']:>4} {r['wr']:>5.1f}% {r['cr']:>5.1f}% "
              f"{r['avg_depth']:>5.2f} {d1:>4.0f}% {d2:>4.0f}% {d3:>4.0f}% "
              f"{ontime:>5.0f}% {r['avg_gain']:>5.2f} {r['avg_loss']:>5.2f} "
              f"{r['expect']:>+5.2f}%")
        agg["n"] += r["n"]; agg["win"] += r["win"]; agg["cancel"] += r["cancel"]
        agg["gain"] += r["avg_gain"] * r["win"]; agg["loss"] += r["avg_loss"] * r["cancel"]
        agg["on_time"] += r["on_time"]; agg["timed"] += r["timed"]
        agg["depth_sum"] += r["avg_depth"] * r["n"]
        for k, v in r["depth_counts"].items():
            agg_depth[k] += v
        for k, v in r["by_manip"].items():
            mn[k]["n"] += v["n"]; mn[k]["win"] += v["win"]
    n = agg["n"] or 1
    twr = agg["win"] / n * 100
    tcr = agg["cancel"] / n * 100
    tg = agg["gain"] / agg["win"] if agg["win"] else 0
    tl = agg["loss"] / agg["cancel"] if agg["cancel"] else 0
    texp = twr / 100 * tg - tcr / 100 * tl
    d1 = sum(v for k, v in agg_depth.items() if k >= 1) / n * 100
    d2 = sum(v for k, v in agg_depth.items() if k >= 2) / n * 100
    d3 = sum(v for k, v in agg_depth.items() if k >= 3) / n * 100
    ontime = agg["on_time"] / agg["timed"] * 100 if agg["timed"] else 0
    print("  " + "-" * 96)
    print(f"  {'TOTAL':<20} {int(n):>4} {twr:>5.1f}% {tcr:>5.1f}% "
          f"{agg['depth_sum']/n:>5.2f} {d1:>4.0f}% {d2:>4.0f}% {d3:>4.0f}% "
          f"{ontime:>5.0f}% {tg:>5.2f} {tl:>5.2f} {texp:>+5.2f}%")
    print("\n  По чек-листу манипуляции:")
    for k, v in sorted(mn.items()):
        wr = v["win"] / v["n"] * 100 if v["n"] else 0
        print(f"    {k:>6}: WR={wr:.1f}% ({v['win']}/{v['n']})")
    print("\n  Легенда: Глуб=ср. число целей до слома; d1+/d2+/d3+ = доля сделок,")
    print("  где достигнута 1-я/2-я/3-я цель; Срок=доля сделок в прогнозном времени;")
    print("  Ожид = WR×ср.путь_к_цели − CR×ср.путь_к_слому (% на сделку).")


def run_calibration(fast=False):
    """Подбор запаса слома (×ATR_дн) на 4H наборах."""
    bufs = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"Калибровка слома на 4H: buf ∈ {bufs} ×ATR_дн\n", flush=True)
    per_buf = {b: [] for b in bufs}
    for csv_name, mb, st, fb in CONFIGS_4H:
        print(f"=== {csv_name} ===", flush=True)
        res = run_one(csv_name, mb, st, fb, fast=fast, quiet=True, sweep_bufs=bufs)
        if res:
            for b in bufs:
                per_buf[b].append(res[b])
            for b in bufs:
                r = res[b]
                print(f"  buf={b:<4} WR={r['wr']:5.1f}% CR={r['cr']:5.1f}% "
                      f"ожид={r['expect']:+5.2f}%", flush=True)
    print("\n" + "=" * 60)
    print("КАЛИБРОВКА СЛОМА 4H — агрегат по всем наборам")
    print("=" * 60)
    print(f"  {'buf×ATR':>8} {'N':>5} {'WR':>7} {'CR':>7} {'ср.+%':>6} {'ср.-%':>6} {'Ожид':>7}")
    for b in bufs:
        rs = per_buf[b]
        n = sum(r["n"] for r in rs) or 1
        w = sum(r["win"] for r in rs)
        c = sum(r["cancel"] for r in rs)
        g = sum(r["avg_gain"] * r["win"] for r in rs) / w if w else 0
        l = sum(r["avg_loss"] * r["cancel"] for r in rs) / c if c else 0
        wr, cr = w / n * 100, c / n * 100
        exp = wr / 100 * g - cr / 100 * l
        print(f"  {b:>8} {n:>5} {wr:>6.1f}% {cr:>6.1f}% {g:>6.2f} {l:>6.2f} {exp:>+6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file", nargs="?", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="подбор запаса слома на 4H")
    ap.add_argument("--slam-buf", type=float, default=0.0, help="запас слома ×ATR_дн")
    ap.add_argument("--step", type=int, default=200)
    ap.add_argument("--min-bars", type=int, default=500)
    ap.add_argument("--future-bars", type=int, default=120)
    args = ap.parse_args()

    if args.calibrate:
        run_calibration(fast=args.fast)
        return

    if args.all:
        configs = ALL_CONFIGS
    elif args.csv_file:
        configs = [(Path(args.csv_file).name, args.min_bars, args.step, args.future_bars)]
    else:
        print("Укажи CSV, --all или --calibrate")
        return

    print(f"Бэктест v8-пайплайна — {len(configs)} конфигураций"
          f"{' (fast)' if args.fast else ''}, slam_buf={args.slam_buf}\n", flush=True)

    all_results = []
    for csv_name, mb, st, fb in configs:
        print(f"=== {csv_name} (окно={mb}, шаг={st}, будущее={fb}) ===", flush=True)
        r = run_one(csv_name, mb, st, fb, fast=args.fast, slam_buf=args.slam_buf)
        if r:
            all_results.append(r)
            print(f"  >> WR={r['wr']:.1f}% ({r['win']}/{r['n']}), CR={r['cr']:.1f}%, "
                  f"глубина={r['avg_depth']:.2f}, ожид={r['expect']:+.2f}%\n", flush=True)

    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    main()
