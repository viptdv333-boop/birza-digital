"""
backtest.py — Бэктест прогнозов birza_digital v2.

Win rate = первая цель достигнута ДО уровня отмены маршрута.
Гибкая система фильтрации: балльная оценка вместо жёстких отсечений.

Запуск:
  python backtest.py                     # все тикеры, балльный фильтр
  python backtest.py --no-filter         # без фильтров (базовая линия)
  python backtest.py --ticker NG1!       # только газ
  python backtest.py --tf 4H             # только 4H
  python backtest.py --score-cutoff 2    # порог баллов (0=выкл, 1/2/3...)
  python backtest.py --analyze-filters   # анализ вклада каждого фильтра
  python backtest.py --sweep             # перебор порогов (найти оптимум)
"""
import sqlite3
import csv as csvlib
import json
import re
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

DB_PATH    = Path(__file__).parent / "birza_digital.db"
PRICES_DIR = Path(__file__).parent / "prices"

PRICE_FILES = {
    "NG1!":     PRICES_DIR / "NG1_1H.csv",
    "NG1":      PRICES_DIR / "NG1_1H.csv",
    "PLATINUM": PRICES_DIR / "PLATINUM_1H.csv",
    "XPTUSD":   PRICES_DIR / "PLATINUM_1H.csv",
}

_price_cache = {}


# ── Загрузка цен ───────────────────────────────────────────────────────────

def _load_prices(ticker: str) -> list:
    if ticker in _price_cache:
        return _price_cache[ticker]
    path = PRICE_FILES.get(ticker)
    if not path or not path.exists():
        _price_cache[ticker] = []
        return []
    bars = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["time"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                bars.append({
                    "ts": ts,
                    "high":  float(row.get("high") or 0),
                    "low":   float(row.get("low") or 0),
                    "close": float(row.get("close") or 0),
                })
            except Exception:
                continue
    bars.sort(key=lambda b: b["ts"])
    _price_cache[ticker] = bars
    print(f"  [prices] {ticker}: {len(bars)} bars", flush=True)
    return bars


def get_future_bars(ticker: str, from_dt: datetime, days: int) -> list:
    bars = _load_prices(ticker)
    if not bars:
        return []
    if from_dt.tzinfo is None:
        from_dt = from_dt.replace(tzinfo=timezone.utc)
    end_dt = from_dt + timedelta(days=days)
    return [b for b in bars if from_dt <= b["ts"] <= end_dt]


# ── Парсинг report_text ────────────────────────────────────────────────────

def parse_targets(text: str) -> dict:
    result = {"key_targets": [], "route": [], "direction": None,
              "cancel_level": None, "timeframe_days": None}

    m = re.search(r"ключевые цели:\s*(.+?)(?:\n|$)", text, re.I)
    if m:
        result["key_targets"] = [float(p) for p in
            re.findall(r"([\d.]+)\s*\([+-]?[\d.]+%\)", m.group(1))]

    m = re.search(r"вероятный маршрут:\s*(.+?)(?:\n|$)", text, re.I)
    if m:
        route_text = m.group(1)
        result["route"] = [float(p) for p in
            re.findall(r"([\d.]+)\s*\([+-]?[\d.]+%\)", route_text)]

        # Направление — по финальному узлу ⚫ (цель всего маршрута)
        # Если нет ⚫ — смотрим соотношение зелёных/красных
        price_at_idx = None
        nodes_raw = re.findall(r"([🟢🔴⚫])\s*([\d.]+)\s*\([+-]?[\d.]+%\)", route_text)
        if nodes_raw:
            colors = [n[0] for n in nodes_raw]
            prices = [float(n[1]) for n in nodes_raw]
            # Финальный узел определяет итоговое направление
            if "⚫" in colors:
                final_idx = len(colors) - 1 - colors[::-1].index("⚫")
                first_price = result["route"][0] if result["route"] else 0
                result["direction"] = "bull" if prices[final_idx] > first_price else "bear"
            else:
                greens = colors.count("🟢")
                reds   = colors.count("🔴")
                result["direction"] = "bull" if greens > reds else ("bear" if reds > greens else None)

    m = re.search(r"отмена маршрута:\s*([\d.]+)", text, re.I)
    if m:
        result["cancel_level"] = float(m.group(1))

    m = re.search(r"вероятные сроки:\s*(.+?)(?:\n|$)", text, re.I)
    if m:
        dm = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*(?:дн|торг|сесс)", m.group(1))
        result["timeframe_days"] = int(dm.group(2)) if dm else None
        if not result["timeframe_days"]:
            dm = re.search(r"(\d+)\s*(?:дн|торг|сесс)", m.group(1))
            result["timeframe_days"] = int(dm.group(1)) if dm else None

    return result


# ── Балльная система фильтров ──────────────────────────────────────────────
# Каждый фильтр добавляет 1 балл (риск-фактор).
# score_cutoff = порог: score >= cutoff → отчёт отклонён.
# cutoff=0 → без фильтров, cutoff=1 → исключаем при 1+ баллах.

FILTERS = {
    # Структурные красные флаги (вес 2 — критичны, проверены бэктестом)
    "distribution_bull":  (2, "Вайкофф распределение на bull"),   # +4.1% WR
    "dist_score_high":    (2, "Distribution score >= 4 на bull"),  # +2.5% WR
    "cmf_sellers_bull":   (2, "CMF продавцы на bull"),             # +2.6% WR
    "alma_bear_on_bull":  (1, "Цена ниже ALMA 200 на bull"),       # +1.2% WR

    # Технические (вес 1, слабый эффект но не вредят)
    "rsi_extreme_bull":   (1, "RSI перекуплен на bull"),

    # УДАЛЕНЫ как вредящие или нейтральные:
    # "cvd_fork"      → снижает WR! (-5.3%)
    # "vol_low"       → нейтральный (-0.2%)
    # "squeeze_active"→ 0 отчётов в данных
    # "no_expansion"  → 0 отчётов в данных
    # "tempo_hot"     → только 2 отчёта, нейтральный
}


def score_report(rj: dict, direction: str) -> dict:
    """
    Вернуть {total_score, flags: {filter_name: weight}, details: {...}}
    """
    if not rj:
        return {"total_score": 0, "flags": {}, "details": {}}

    secs = rj.get("sections", {})

    def get(sec_id, *keys, default=None):
        s = secs.get(str(sec_id), {})
        d = s.get("data", {}) if isinstance(s, dict) else {}
        v = d
        for k in keys:
            if not isinstance(v, dict):
                return default
            v = v.get(k, default)
            if v is None:
                return default
        return v

    flags = {}

    # -- Вайкофф распределение на bull
    if direction == "bull":
        wyck = (get(10, "structure_type") or "").lower()
        if "распределение" in wyck:
            flags["distribution_bull"] = FILTERS["distribution_bull"][0]

        dist_sc = float(get(10, "evidence", "distribution_score") or 0)
        if dist_sc >= 4:
            flags["dist_score_high"] = FILTERS["dist_score_high"][0]

        alma_pos = (get(1, "alma_200_position") or "").lower()
        if "ниже" in alma_pos:
            flags["alma_bear_on_bull"] = FILTERS["alma_bear_on_bull"][0]

        cmf_sig = (get(16, "cmf", "signal") or "").lower()
        if "продавц" in cmf_sig:
            flags["cmf_sellers_bull"] = FILTERS["cmf_sellers_bull"][0]

    # -- Squeeze
    if get(14, "squeeze_active") is True:
        flags["squeeze_active"] = FILTERS["squeeze_active"][0]

    # -- Волатильность не расширяется
    vol_phase = (get(14, "vol_phase") or "").lower()
    if vol_phase and "расширение" not in vol_phase and "нормальн" not in vol_phase:
        flags["no_expansion"] = FILTERS["no_expansion"][0]

    # -- RSI перекуплен на bull
    if direction == "bull":
        rsi_zone = (get(5, "rsi_zone") or "").lower()
        if "перекупленност" in rsi_zone:
            flags["rsi_extreme_bull"] = FILTERS["rsi_extreme_bull"][0]

    total = sum(flags.values())
    return {"total_score": total, "flags": flags}


# ── Симуляция сделки ──────────────────────────────────────────────────────

def simulate_trade(price_at: float, direction: str, targets: dict, future_bars: list) -> dict:
    """
    Win = первая цель достигнута ДО уровня отмены.
    Первая цель = второй узел маршрута (или ближайшая key_target).
    """
    cancel = targets.get("cancel_level")
    route  = targets.get("route", [])
    key_targets = targets.get("key_targets", [])

    # Первая цель маршрута
    first_target = None
    if len(route) >= 2:
        first_target = route[1]
    if first_target is None and key_targets:
        cands = [t for t in key_targets if (t > price_at if direction == "bull" else t < price_at)]
        if cands:
            first_target = min(cands, key=lambda t: abs(t - price_at))

    result = {"win": False, "cancel_hit": False, "bars_to_win": None, "max_move_pct": 0.0,
              "targets_hit": 0, "targets_total": len(key_targets)}

    if not future_bars or first_target is None:
        return result

    max_p = min_p = None

    for i, bar in enumerate(future_bars):
        h, l = bar["high"], bar["low"]

        # Cancel level
        if cancel:
            if direction == "bull" and l <= cancel:
                result["cancel_hit"] = True; break
            if direction == "bear" and h >= cancel:
                result["cancel_hit"] = True; break

        # Win check
        if not result["win"]:
            if (direction == "bull" and h >= first_target) or \
               (direction == "bear" and l <= first_target):
                result["win"] = True
                result["bars_to_win"] = i + 1

        # Max move in our direction
        if direction == "bull":
            move = (h - price_at) / price_at * 100
        else:
            move = (price_at - l) / price_at * 100
        result["max_move_pct"] = max(result["max_move_pct"], move)

        if max_p is None:
            max_p, min_p = h, l
        else:
            max_p = max(max_p, h)
            min_p = min(min_p, l)

    # All targets hit
    if max_p is not None:
        for t in key_targets:
            if direction == "bull" and max_p >= t > price_at:
                result["targets_hit"] += 1
            elif direction == "bear" and min_p <= t < price_at:
                result["targets_hit"] += 1

    return result


# ── Основной бэктест ──────────────────────────────────────────────────────

def run_backtest(ticker_filter=None, tf_filter=None, score_cutoff=3,
                 analyze_filters=False, sweep=False):

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    q = "SELECT id, created_at, ticker, timeframe, price, report_text, report_json FROM reports WHERE report_text IS NOT NULL AND price > 0"
    params = []
    if ticker_filter:
        q += " AND (ticker=? OR ticker LIKE ?)"; params += [ticker_filter, f"%{ticker_filter}%"]
    if tf_filter:
        q += " AND timeframe=?"; params.append(tf_filter)
    q += " ORDER BY created_at ASC"
    rows = conn.execute(q, params).fetchall()
    conn.close()

    valid = [r for r in rows if r["ticker"] in PRICE_FILES]
    print(f"\nОтчётов в БД: {len(rows)}, с реальными ценами: {len(valid)}", flush=True)

    records = []
    for r in valid:
        targets = parse_targets(r["report_text"])
        if not targets["route"] or not targets["key_targets"]:
            continue
        direction = targets["direction"]
        if direction not in ("bull", "bear"):
            continue

        try:
            dt = datetime.strptime(r["created_at"][:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

        days = (targets["timeframe_days"] or 5) * 2
        future_bars = get_future_bars(r["ticker"], dt, days)

        rj = None
        if r["report_json"]:
            try: rj = json.loads(r["report_json"])
            except: pass

        scoring = score_report(rj, direction)
        sim = simulate_trade(r["price"], direction, targets, future_bars)

        records.append({
            "id": r["id"], "date": r["created_at"][:10],
            "ticker": r["ticker"], "tf": r["timeframe"],
            "price": r["price"], "direction": direction,
            "win": sim["win"], "cancel": sim["cancel_hit"],
            "no_data": not bool(future_bars),
            "targets_hit": sim["targets_hit"],
            "targets_total": sim["targets_total"],
            "max_move": round(sim["max_move_pct"], 2),
            "bars_to_win": sim["bars_to_win"],
            "score": scoring["total_score"],
            "flags": scoring["flags"],
        })

    if not records:
        print("Нет данных для анализа."); return

    # -- Анализ вклада каждого фильтра по отдельности
    if analyze_filters:
        print("\n" + "="*65)
        print("ВКЛАД КАЖДОГО ФИЛЬТРА (исключаем только его одного)")
        print("="*65)
        print(f"  {'Фильтр':<35} {'Режет':>6} {'WR без него':>12} {'WR после':>10}")
        print("-"*65)
        base_recs = [r for r in records if not r["no_data"]]
        base_wr = _wr(base_recs)
        for fname, (weight, desc) in FILTERS.items():
            flagged = [r for r in base_recs if fname in r["flags"]]
            remaining = [r for r in base_recs if fname not in r["flags"]]
            wr_after = _wr(remaining)
            pct_cut = len(flagged) / len(base_recs) * 100 if base_recs else 0
            marker = " <--" if wr_after >= base_wr + 2 else ""
            print(f"  {desc:<35} {len(flagged):>4} ({pct_cut:4.0f}%)  {base_wr:>7.1f}%  {wr_after:>8.1f}%{marker}")
        print(f"\n  Базовая линия (без фильтров): {base_wr:.1f}%  ({len(base_recs)} отчётов)")

    # -- Перебор порогов
    if sweep:
        print("\n" + "="*55)
        print("ПЕРЕБОР ПОРОГОВ score_cutoff")
        print("="*55)
        print(f"  {'Cutoff':>8} {'Прошло':>8} {'% от всех':>10} {'WinRate':>9}")
        print("-"*45)
        base_recs = [r for r in records if not r["no_data"]]
        for cutoff in range(0, 9):
            passed = [r for r in base_recs if r["score"] < cutoff or cutoff == 0]
            wr = _wr(passed)
            pct = len(passed) / len(base_recs) * 100 if base_recs else 0
            marker = " <-- TARGET" if wr >= 80 and pct >= 30 else ""
            print(f"  {cutoff:>8} {len(passed):>8} {pct:>9.0f}% {wr:>8.1f}%{marker}")

    # -- Основные результаты
    _print_results(records, score_cutoff, "С ФИЛЬТРАМИ (score_cutoff=" + str(score_cutoff) + ")")


def _wr(recs: list) -> float:
    decided = [r for r in recs if not r["no_data"]]
    if not decided: return 0.0
    return sum(1 for r in decided if r["win"]) / len(decided) * 100


def _print_results(records: list, cutoff: int, label: str):
    base = [r for r in records if not r["no_data"]]
    filtered = [r for r in base if r["score"] < cutoff] if cutoff > 0 else base

    print("\n" + "="*65)
    print("БЕЗ ФИЛЬТРОВ (базовая линия)")
    print("="*65)
    _print_tf_table(base)

    print("\n" + "="*65)
    print(label)
    print("="*65)
    _print_tf_table(filtered)

    cut_cnt = len(base) - len(filtered)
    base_wr = _wr(base)
    filt_wr = _wr(filtered)
    pct_passed = len(filtered) / len(base) * 100 if base else 0
    print(f"\n  Отфильтровано: {cut_cnt}/{len(base)} ({100-pct_passed:.0f}%)")
    print(f"  Прошло: {len(filtered)} ({pct_passed:.0f}%)")
    print(f"  Win rate: {base_wr:.1f}% -> {filt_wr:.1f}%  (+{filt_wr-base_wr:.1f}%)")

    # По тикерам с фильтрами
    print(f"\n  Win rate по тикерам (с фильтрами):")
    tk_st = defaultdict(lambda: {"n": 0, "win": 0})
    for r in filtered:
        tk_st[r["ticker"]]["n"] += 1
        if r["win"]: tk_st[r["ticker"]]["win"] += 1
    for tk, s in sorted(tk_st.items(), key=lambda x: -x[1]["n"]):
        wr = s["win"] / s["n"] * 100 if s["n"] else 0
        print(f"    {tk:<15} N={s['n']:>4}  Win={s['win']:>4}  WR={wr:.1f}%")

    # По ТФ с фильтрами
    print(f"\n  Win rate по таймфреймам (с фильтрами):")
    tf_st = defaultdict(lambda: {"n": 0, "win": 0})
    for r in filtered:
        tf_st[r["tf"]]["n"] += 1
        if r["win"]: tf_st[r["tf"]]["win"] += 1
    for tf, s in sorted(tf_st.items(), key=lambda x: -x[1]["n"]):
        wr = s["win"] / s["n"] * 100 if s["n"] else 0
        print(f"    {tf:<12} N={s['n']:>4}  Win={s['win']:>4}  WR={wr:.1f}%")

    # Самые частые флаги у проигравших
    losers = [r for r in filtered if not r["win"]]
    if losers:
        print(f"\n  Топ флагов у ПРОИГРАВШИХ ({len(losers)} сделок):")
        flag_cnt = defaultdict(int)
        for r in losers:
            for f in r["flags"]:
                flag_cnt[f] += 1
        for fname, cnt in sorted(flag_cnt.items(), key=lambda x: -x[1])[:5]:
            desc = FILTERS.get(fname, (0, fname))[1]
            print(f"    {cnt:>3}x  {desc}")


def _print_tf_table(recs: list):
    if not recs:
        print("  Нет данных."); return
    total_win = sum(1 for r in recs if r["win"])
    total_cancel = sum(1 for r in recs if r["cancel"])
    wr = total_win / len(recs) * 100 if recs else 0
    print(f"  {'TF':<12} {'N':>5} {'Win':>5} {'Cancel':>7} {'WR':>8}")
    print("  " + "-"*42)
    tf_grp = defaultdict(lambda: {"n": 0, "win": 0, "cancel": 0})
    for r in recs:
        tf_grp[r["tf"]]["n"] += 1
        if r["win"]: tf_grp[r["tf"]]["win"] += 1
        if r["cancel"]: tf_grp[r["tf"]]["cancel"] += 1
    for tf, s in sorted(tf_grp.items(), key=lambda x: -x[1]["n"]):
        rate = s["win"] / s["n"] * 100 if s["n"] else 0
        print(f"  {tf:<12} {s['n']:>5} {s['win']:>5} {s['cancel']:>7} {rate:>7.1f}%")
    print("  " + "-"*42)
    print(f"  {'TOTAL':<12} {len(recs):>5} {total_win:>5} {total_cancel:>7} {wr:>7.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",          default=None)
    parser.add_argument("--tf",              default=None)
    parser.add_argument("--score-cutoff",    type=int, default=3)
    parser.add_argument("--no-filter",       action="store_true")
    parser.add_argument("--analyze-filters", action="store_true")
    parser.add_argument("--sweep",           action="store_true")
    args = parser.parse_args()

    run_backtest(
        ticker_filter=args.ticker,
        tf_filter=args.tf,
        score_cutoff=0 if args.no_filter else args.score_cutoff,
        analyze_filters=args.analyze_filters,
        sweep=args.sweep,
    )
