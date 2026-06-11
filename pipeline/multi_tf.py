"""
Биржа-цифровой — Сводный мульти-ТФ анализ (ЯДРО v8.1).

Принимает список результатов препроцессора для разных ТФ одного тикера
и формирует отчёт:
- Шапка тикера + список ТФ и ролей
- 18 разделов, в каждом — ПОЛНОЕ содержимое раздела для каждого ТФ
  (через ai.formatter.SECTION_FORMATTERS) + содержательный cross-TF вывод
- Раздел 19 — ВЫВОД с вердиктом Элдера и маршрутом через route_engine (v8.1)

Веса Элдера: 1 ТФ→3, 2→[1,3], 3→[1,2,3], 4→[1,2,3,4], 5→[1..5].
"""
from __future__ import annotations
from typing import List, Dict, Any

from core.utils import tf_label
from config import TF_HOURS


# ──────────────────────────────────────────────────────────────
# Извлечение сигналов одного ТФ (для Elder-скоринга)
# ──────────────────────────────────────────────────────────────

def _extract_tf_signals(sections_data: list, meta: dict) -> dict:
    by_id = {s["section_id"]: s.get("data", {}) for s in sections_data}
    s1 = by_id.get(1, {}) or {}
    s5 = by_id.get(5, {}) or {}
    s14 = by_id.get(14, {}) or {}
    s15 = by_id.get(15, {}) or {}
    s16 = by_id.get(16, {}) or {}

    macd = s5.get("macd") or {}
    cvd = s16.get("cvd") or {}
    cmf = s16.get("cmf") or {}
    mfi = s16.get("mfi") or {}

    return {
        "tf": meta.get("timeframe", "?"),
        "price": meta.get("current_price"),
        "direction": s1.get("direction", "неопределено"),
        "state": s1.get("state", "—"),
        "stage": s1.get("stage", "—"),
        "alma_order": s1.get("alma_order", "нейтральный"),
        "structure_break": bool(s1.get("structure_break", False)),
        "rsi": s5.get("rsi_current"),
        "rsi_zone": s5.get("rsi_zone", "—"),
        "div_signal": s5.get("signal", "—"),
        "macd_cross": macd.get("cross"),
        "macd_hist_trend": macd.get("histogram_trend"),
        "macd_hist_reversal": macd.get("histogram_reversal"),
        "squeeze_active": bool(s14.get("squeeze_active", False)),
        "er_class": s15.get("er_classification", "—"),
        "cmf_signal": cmf.get("signal", "—"),
        "cvd_dir_50": cvd.get("dir_50"),
        "cvd_conflict": bool(cvd.get("conflict", False)),
        "mfi_zone": mfi.get("zone", "—"),
    }


def _vote_tf(sig: dict) -> tuple[float, list[str]]:
    """Голосование Элдера по сигналам одного ТФ."""
    score = 0.0
    reasons: list[str] = []

    def add(pts, label):
        nonlocal score
        if pts != 0:
            score += pts
            reasons.append(f"{'+' if pts > 0 else '−'}{abs(pts):.1f} {label}")

    d = sig.get("direction", "")
    if d == "восходящий":
        add(+2.0, f"тренд {d}")
    elif d == "нисходящий":
        add(-2.0, f"тренд {d}")

    stage = sig.get("stage", "")
    if stage == "Развитие" and d != "боковик":
        add(+0.5 if d == "восходящий" else -0.5, f"стадия {stage}")
    elif stage == "Затухание" and d != "боковик":
        add(-0.3 if d == "восходящий" else +0.3, f"стадия {stage}")

    alma = sig.get("alma_order", "")
    if "бычий" in alma and "смешанный" not in alma:
        add(+1.0, f"ALMA {alma}")
    elif "медвежий" in alma and "смешанный" not in alma:
        add(-1.0, f"ALMA {alma}")

    if sig.get("structure_break"):
        mult = +0.5 if d == "восходящий" else (-0.5 if d == "нисходящий" else 0)
        if mult != 0:
            add(mult, "пробой структуры")

    rsi_zone = sig.get("rsi_zone", "")
    if rsi_zone == "перекупленность":
        add(-0.8, "RSI перекупленность")
    elif rsi_zone == "перепроданность":
        add(+0.8, "RSI перепроданность")
    elif rsi_zone == "близко к перекупленности":
        add(-0.3, "RSI близко к перекупленности")
    elif rsi_zone == "близко к перепроданности":
        add(+0.3, "RSI близко к перепроданности")

    cross = sig.get("macd_cross") or ""
    if "бычий" in cross:
        add(+0.5, "MACD бычий кросс")
    elif "медвежий" in cross:
        add(-0.5, "MACD медвежий кросс")

    hist = sig.get("macd_hist_trend") or ""
    if hist == "растёт":
        add(+0.3, "MACD гистограмма растёт")
    elif hist == "падает":
        add(-0.3, "MACD гистограмма падает")

    rev = sig.get("macd_hist_reversal") or ""
    if "бычий разворот" in rev:
        add(+0.5, "MACD разворот вверх")
    elif "медвежий разворот" in rev:
        add(-0.5, "MACD разворот вниз")

    dsig = sig.get("div_signal", "")
    if dsig == "бычий разворот":
        add(+0.7, "дивергенция бычья")
    elif dsig == "медвежий разворот":
        add(-0.7, "дивергенция медвежья")

    cmf_s = sig.get("cmf_signal") or ""
    if cmf_s == "покупатели":
        add(+0.3, "CMF покупатели")
    elif cmf_s == "продавцы":
        add(-0.3, "CMF продавцы")

    cvd50 = sig.get("cvd_dir_50") or ""
    if cvd50 == "покупатели":
        add(+0.4, "CVD(50) покупатели")
    elif cvd50 == "продавцы":
        add(-0.4, "CVD(50) продавцы")

    mfi_zone = sig.get("mfi_zone") or ""
    if mfi_zone == "перекупленность":
        add(-0.3, "MFI перекупленность")
    elif mfi_zone == "перепроданность":
        add(+0.3, "MFI перепроданность")

    er_class = sig.get("er_class") or ""
    if er_class == "сильный тренд" and d != "боковик":
        add(+0.3 if d == "восходящий" else -0.3, "ER сильный тренд")
    elif er_class == "шум":
        add(-0.2, "ER шум")

    return round(score, 2), reasons


def _elder_verdict(tf_entries: list) -> dict:
    """Свернуть голоса всех ТФ в финальный вердикт."""
    if not tf_entries:
        return {"total_score": 0.0, "verdict": "no_data",
                "verdict_text": "Нет данных", "agreement": "none",
                "entry_zone_alert": False, "warnings": []}

    total_weighted = sum(e["weighted_vote"] for e in tf_entries)
    total_weight = sum(e["weight"] for e in tf_entries)
    score = total_weighted / total_weight if total_weight else 0.0

    senior = tf_entries[-1]
    junior = tf_entries[0]
    mid = tf_entries[len(tf_entries) // 2] if len(tf_entries) >= 3 else None

    def side(v):
        if v > 1.0: return "bull"
        if v < -1.0: return "bear"
        return "neutral"

    sides = [side(e["vote"]) for e in tf_entries]
    unique_sides = set(sides)

    if len(unique_sides) == 1 and "neutral" not in unique_sides:
        agreement = "full"
    elif "bull" in unique_sides and "bear" in unique_sides:
        agreement = "conflict"
    elif unique_sides == {"neutral"}:
        agreement = "sideways"
    else:
        agreement = "partial"

    entry_zone_alert = False
    entry_direction = None
    if mid and len(tf_entries) >= 3:
        s_side = side(senior["vote"])
        m_side = side(mid["vote"])
        j_side = side(junior["vote"])
        if s_side == m_side and s_side != "neutral" and j_side != s_side:
            entry_zone_alert = True
            entry_direction = s_side

    warnings = []
    senior_dir = (senior.get("signals") or {}).get("direction", "—")
    if senior_dir == "боковик":
        warnings.append("Старший ТФ в боковике — нет глобального тренда, торговля без тренда")
    if agreement == "conflict":
        warnings.append("Конфликт направлений между ТФ — ждать согласования")

    if score >= 2.5:
        verdict = "strong_bull"
        text = "СИЛЬНЫЙ БЫЧИЙ сигнал — все ТФ указывают вверх"
    elif score >= 1.2:
        verdict = "bull"
        text = "Бычий сигнал — преобладает движение вверх"
    elif score >= 0.4:
        verdict = "weak_bull"
        text = "Слабый бычий уклон"
    elif score <= -2.5:
        verdict = "strong_bear"
        text = "СИЛЬНЫЙ МЕДВЕЖИЙ сигнал — все ТФ указывают вниз"
    elif score <= -1.2:
        verdict = "bear"
        text = "Медвежий сигнал — преобладает движение вниз"
    elif score <= -0.4:
        verdict = "weak_bear"
        text = "Слабый медвежий уклон"
    else:
        verdict = "neutral"
        text = "Нейтрально — нет явного преимущества"

    return {
        "total_score": round(float(score), 2),
        "verdict": verdict,
        "verdict_text": text,
        "agreement": agreement,
        "entry_zone_alert": entry_zone_alert,
        "entry_direction": entry_direction,
        "warnings": warnings,
    }


# ──────────────────────────────────────────────────────────────
# Главная функция — consolidate()
# ──────────────────────────────────────────────────────────────

def consolidate(results: List[Dict[str, Any]], horizon: str = "—",
                analysis_type_name: str = "—") -> dict:
    """Свести данные нескольких ТФ в единую структуру."""
    if not results:
        return {"error": "Нет данных"}

    def _tf_hours(s):
        return TF_HOURS.get(s, 0.0)

    sorted_results = sorted(
        results, key=lambda r: _tf_hours(r["meta"].get("timeframe", ""))
    )
    n_tfs = len(sorted_results)

    weights_map = {
        1: [3],
        2: [1, 3],
        3: [1, 2, 3],
        4: [1, 2, 3, 4],
        5: [1, 2, 3, 4, 5],
    }
    weights = weights_map.get(n_tfs, list(range(1, n_tfs + 1)))

    if n_tfs == 1:
        roles = ["single"]
        role_labels = ["единственный"]
    elif n_tfs == 2:
        roles = ["junior", "senior"]
        role_labels = ["младший", "старший"]
    elif n_tfs == 3:
        roles = ["junior", "mid", "senior"]
        role_labels = ["младший", "средний", "старший"]
    elif n_tfs == 4:
        roles = ["junior", "mid_low", "mid_high", "senior"]
        role_labels = ["младший", "мл-средний", "ст-средний", "старший"]
    else:
        roles = ["junior"] + [f"mid{i}" for i in range(1, n_tfs - 1)] + ["senior"]
        role_labels = (["младший"]
                       + [f"средний-{i}" for i in range(1, n_tfs - 1)]
                       + ["старший"])

    tf_entries = []
    for i, res in enumerate(sorted_results):
        meta = res["meta"]
        sections_data = res["sections_data"]
        sig = _extract_tf_signals(sections_data, meta)
        vote, reasons = _vote_tf(sig)
        weight = weights[i]

        tf_entries.append({
            "tf": meta.get("timeframe", "?"),
            "tf_hours": _tf_hours(meta.get("timeframe", "")),
            "role": roles[i],
            "role_label": role_labels[i],
            "weight": weight,
            "meta": meta,
            "sections_data": sections_data,
            "signals": sig,
            "vote": vote,
            "weighted_vote": round(vote * weight, 2),
            "reasons": reasons,
            "report_id": res.get("report_id"),
        })

    elder = _elder_verdict(tf_entries)

    senior_entry = tf_entries[-1]
    current_price = senior_entry["meta"].get("current_price") or 0

    return {
        "ticker": sorted_results[0]["meta"].get("ticker", "—"),
        "exchange": sorted_results[0]["meta"].get("exchange", "—"),
        "horizon": horizon,
        "analysis_type_name": analysis_type_name,
        "tf_list": [e["tf"] for e in tf_entries],
        "current_price": current_price,
        "per_tf": tf_entries,
        "elder": elder,
    }


# ──────────────────────────────────────────────────────────────
# Метаданные 18 разделов
# ──────────────────────────────────────────────────────────────

SECTION_META = [
    (1,  "📈", "ТРЕНДЫ"),
    (2,  "🌊", "ВОЛНОВОЙ АНАЛИЗ"),
    (3,  "🔺", "ГРАФИЧЕСКИЕ ПАТТЕРНЫ"),
    (4,  "🕯", "СВЕЧНЫЕ ПАТТЕРНЫ"),
    (5,  "📉", "ДИВЕРГЕНЦИИ"),
    (6,  "📊", "УРОВНИ"),
    (7,  "📐", "ФИБОНАЧЧИ"),
    (8,  "🔊", "VSA"),
    (9,  "📊", "ОБЪЁМНЫЕ ЗОНЫ"),
    (10, "⚖",  "ВАЙКОФФ"),
    (11, "📍", "ЗОНЫ СБОРА СТОПОВ"),
    (12, "🔄", "ТЕМП РЫНКА"),
    (13, "💧", "ИМБАЛАНСЫ / ЛИКВИДНОСТЬ / ГЭПЫ / FVG"),
    (14, "📊", "BOLLINGER / KELTNER / SQUEEZE"),
    (15, "⚡", "ЭФФЕКТИВНОСТЬ ДВИЖЕНИЙ"),
    (16, "💰", "ПОТОКОВЫЕ ИНДИКАТОРЫ"),
    (17, "🔬", "МИКРОСТРУКТУРА"),
    (18, "🔗", "КОРРЕЛЯЦИИ И КОНВЕРГЕНЦИЯ"),
]


# ──────────────────────────────────────────────────────────────
# Содержательные cross-TF выводы по каждому разделу
# ──────────────────────────────────────────────────────────────

def _pct(val, current):
    if val is None:
        return "—"
    if current and current > 0:
        pct = (val - current) / current * 100
        sign = "+" if pct >= 0 else ""
        v_str = f"{val:.2f}" if val >= 1000 else f"{val:.4g}"
        return f"{v_str} ({sign}{pct:.2f}%)"
    return f"{val:.2f}" if val >= 1000 else f"{val:.4g}"


def _section_conclusion(sid: int, per_tf_data_by_label: list, cons: dict, price: float) -> str:
    """Содержательный общий вывод раздела.

    per_tf_data_by_label: [(tf_label, role_label, data_dict), ...] от senior к junior.
    """
    items = per_tf_data_by_label
    labels = [t[0] for t in items]
    n = len(items)

    def get(attr, default=None):
        return [(t[0], (t[2] or {}).get(attr, default)) for t in items]

    # ─── S01 ТРЕНДЫ ───
    if sid == 1:
        dirs = [(t[0], (t[2] or {}).get("direction", "—")) for t in items]
        unique_dirs = {d for _, d in dirs}
        stages = [(t[0], (t[2] or {}).get("stage", "—")) for t in items]
        almas = [(t[0], (t[2] or {}).get("alma_order", "—")) for t in items]
        slopes = []
        for tf, _, d in items:
            lr = (d or {}).get("linreg") or {}
            slope = lr.get("slope_pct_per_bar")
            if slope is not None:
                slopes.append((tf, slope))

        lines = []
        if len(unique_dirs) == 1 and list(unique_dirs)[0] != "—":
            dir_ = list(unique_dirs)[0]
            if dir_ != "боковик":
                # Склонение: «восходящий → восходящем», «нисходящий → нисходящем»
                dir_prep = dir_.replace("ий", "ем") if dir_.endswith("ий") else dir_
                lines.append(f"Все {n} ТФ согласованы в едином {dir_prep} тренде — senior подтверждает направление, торговля только вдоль тренда.")
            else:
                lines.append(f"Все ТФ в боковике — направленной торговли нет, ждать пробоя.")
        else:
            dirs_str = ", ".join(f"{tf}: {d}" for tf, d in dirs)
            lines.append(f"Разные направления на ТФ ({dirs_str}) — приоритет у senior, остальные подтверждают откатами.")

        if slopes:
            slope_str = ", ".join(f"{tf}={s:+.3f}%/бар" for tf, s in slopes)
            senior_slope = slopes[0][1]
            junior_slope = slopes[-1][1] if len(slopes) > 1 else senior_slope
            if senior_slope and junior_slope and (senior_slope * junior_slope < 0):
                lines.append(f"⚠ Наклон канала различается по знаку между senior и junior ({slope_str}) — возможна коррекция.")
            else:
                lines.append(f"Скорости трендов по ТФ: {slope_str}.")

        unique_alma = {a for _, a in almas if a and a != "—"}
        if len(unique_alma) == 1:
            lines.append(f"ALMA-порядок согласован на всех ТФ: {list(unique_alma)[0]}.")
        else:
            alma_str = "; ".join(f"{tf}: {a}" for tf, a in almas)
            lines.append(f"ALMA-порядок различается: {alma_str}.")

        return " ".join(lines)

    # ─── S02 ВОЛНОВОЙ АНАЛИЗ ───
    if sid == 2:
        waves_info = []
        for tf, role, d in items:
            if not d:
                continue
            pattern = d.get("pattern", "?")
            cur_w = d.get("current_wave", "?")
            is_up = d.get("is_uptrend", True)
            waves_info.append((tf, role, pattern, cur_w, is_up))

        if not waves_info:
            return "Волновая разметка не посчитана."

        patterns = {w[2] for w in waves_info}
        current_waves = {w[3] for w in waves_info}
        directions = {w[4] for w in waves_info}

        lines = []
        if len(patterns) == 1 and len(directions) == 1:
            lines.append(f"Волновая структура единая на всех ТФ: {list(patterns)[0]}, "
                         f"направление {'восходящее' if list(directions)[0] else 'нисходящее'}.")
        else:
            tfs_w = ", ".join(f"{tf}: {p} волна {cw}" for tf, _, p, cw, _ in waves_info)
            lines.append(f"Волновая разметка по ТФ: {tfs_w}.")

        # Senior wave = приоритет
        senior_w = waves_info[0]
        lines.append(f"Главный отсчёт — по senior ({senior_w[0]}): {senior_w[2]}, в развитии волна {senior_w[3]}.")

        # Если junior в волне 2/4/B — это потенциальная зона входа
        junior_w = waves_info[-1]
        if junior_w[3] in ("2", "4", "B", "(ii)", "(iv)"):
            lines.append(f"Младший ТФ ({junior_w[0]}) в коррекционной волне {junior_w[3]} — классическая зона входа по Элдеру вдоль senior.")

        return " ".join(lines)

    # ─── S03 ПАТТЕРНЫ (важен конфликт!) ───
    if sid == 3:
        by_tf = []
        for tf, role, d in items:
            pats = (d or {}).get("patterns") or []
            if pats:
                top = pats[0]
                name = top.get("name", "?")
                target = top.get("target")
                # Направление паттерна по знаку цели
                if target and price:
                    pdir = "вверх" if target > price else "вниз"
                else:
                    pdir = "?"
                by_tf.append((tf, role, name, target, pdir, len(pats)))

        if not by_tf:
            return "Классических паттернов не выявлено ни на одном ТФ."

        lines = []
        lines.append(f"Паттерны найдены на {len(by_tf)} из {n} ТФ.")

        # Конфликт направлений?
        dirs = {x[4] for x in by_tf if x[4] != "?"}
        if len(dirs) > 1:
            dir_lines = "; ".join(f"{tf}: {name} ({pdir}, цель {_pct(target, price)})"
                                   for tf, _, name, target, pdir, _ in by_tf)
            lines.append(f"⚠ КОНФЛИКТ ПАТТЕРНОВ между ТФ: {dir_lines}. "
                         f"Приоритет — senior ({by_tf[0][0]}: {by_tf[0][2]}, {by_tf[0][4]}).")
        else:
            dir_ = list(dirs)[0] if dirs else "?"
            pat_list = ", ".join(f"{tf}: {name}" for tf, _, name, _, _, _ in by_tf)
            lines.append(f"Все паттерны однонаправленные ({dir_}): {pat_list}.")

        return " ".join(lines)

    # ─── S04 СВЕЧИ ───
    if sid == 4:
        chars = []
        for tf, role, d in items:
            stats = (d or {}).get("stats") or {}
            ch = (d or {}).get("character") or stats.get("character") or "—"
            bull = stats.get("bull_count", 0)
            bear = stats.get("bear_count", 0)
            chars.append((tf, bull, bear, ch))
        dominance = ["бычьи" if b > s else "медвежьи" if s > b else "равновесие"
                     for _, b, s, _ in chars]
        unique = set(dominance)
        if len(unique) == 1:
            return f"На всех ТФ одинаковое свечное доминирование: {dominance[0]}. Характер: {chars[0][3]}."
        return "Свечное доминирование различается по ТФ: " + ", ".join(f"{c[0]}: {d}" for c, d in zip(chars, dominance)) + "."

    # ─── S05 ДИВЕРГЕНЦИИ ───
    if sid == 5:
        sigs = []
        rsis = []
        for tf, role, d in items:
            sigs.append((tf, (d or {}).get("signal", "нейтрально")))
            r = (d or {}).get("rsi_current")
            if r is not None:
                rsis.append((tf, r, (d or {}).get("rsi_zone", "—")))

        lines = []
        # Проверяем иерархию RSI: если junior в зоне перепроданности, а senior в тренде — разворот
        if rsis:
            rsi_str = ", ".join(f"{tf} {r:.1f} ({z})" for tf, r, z in rsis)
            lines.append(f"RSI по ТФ: {rsi_str}.")

        # Собираем сигналы
        sig_types = {s for _, s in sigs}
        if "бычий разворот" in sig_types and "медвежий разворот" in sig_types:
            lines.append("⚠ КОНФЛИКТ ДИВЕРГЕНЦИЙ: на одних ТФ бычий разворот, на других — медвежий. Приоритет senior.")
        elif "бычий разворот" in sig_types:
            tfs_b = [tf for tf, s in sigs if s == "бычий разворот"]
            lines.append(f"Сигналы бычьего разворота на {', '.join(tfs_b)} — потенциальный отскок.")
        elif "медвежий разворот" in sig_types:
            tfs_b = [tf for tf, s in sigs if s == "медвежий разворот"]
            lines.append(f"Сигналы медвежьего разворота на {', '.join(tfs_b)} — давление на цену.")
        else:
            lines.append("Явных дивергенций нет, импульс сохраняется.")

        return " ".join(lines)

    # ─── S06 УРОВНИ ───
    if sid == 6:
        # Собираем все уровни и ищем конфлюенс (близкие уровни с разных ТФ)
        all_res = []
        all_sup = []
        for tf, role, d in items:
            for r in ((d or {}).get("resistances") or []):
                p = r.get("price")
                if p and p > price:
                    all_res.append((tf, role, r.get("label", ""), p))
            for s in ((d or {}).get("supports") or []):
                p = s.get("price")
                if p and p < price:
                    all_sup.append((tf, role, s.get("label", ""), p))
        nearest_r = sorted(all_res, key=lambda x: x[3])[:3]
        nearest_s = sorted(all_sup, key=lambda x: -x[3])[:3]

        lines = []
        if nearest_r:
            rs = ", ".join(f"{_pct(p, price)} [{lbl}, {tf}]" for tf, _, lbl, p in nearest_r)
            lines.append(f"Ближайшие сопротивления по всем ТФ: {rs}.")
        if nearest_s:
            ss = ", ".join(f"{_pct(p, price)} [{lbl}, {tf}]" for tf, _, lbl, p in nearest_s)
            lines.append(f"Ближайшие поддержки по всем ТФ: {ss}.")
        if not lines:
            lines.append("Свободных уровней вокруг цены нет.")
        return " ".join(lines)

    # ─── S07 ФИБОНАЧЧИ ───
    if sid == 7:
        halves = []
        for tf, _, d in items:
            senior_block = (d or {}).get("senior_trend") or {}
            rets = senior_block.get("retracements") or []
            for r in rets:
                if str(r.get("level", "")).startswith("0.5"):
                    halves.append((tf, r.get("price")))
                    break
        if halves and len({round(p, 4) for _, p in halves if p}) == 1:
            return f"Fibo 0.5 старшего тренда совпадает на всех ТФ: {_pct(halves[0][1], price)}."
        if halves:
            s = ", ".join(f"{tf} 0.5 {_pct(p, price)}" for tf, p in halves)
            return f"Fibo 0.5 различается по ТФ: {s}."
        return "Fibo-разметка посчитана на каждом ТФ отдельно."

    # ─── S08 VSA ───
    if sid == 8:
        obvs = [(tf, (d or {}).get("obv_direction", "—")) for tf, _, d in items]
        vls = [(tf, (d or {}).get("vl_relative")) for tf, _, d in items]
        obv_vals = {o for _, o in obvs if o and o != "—"}
        if len(obv_vals) == 1:
            return f"OBV на всех ТФ {list(obv_vals)[0]}, объёмные усилия согласованы."
        s = ", ".join(f"{tf}: {o}" for tf, o in obvs)
        return f"OBV по ТФ различается ({s}) — разные объёмные режимы."

    # ─── S09 VP ───
    if sid == 9:
        # Данные в profile_a / profile_b
        pocs_a = []
        positions = []
        for tf, _, d in items:
            pa = (d or {}).get("profile_a") or {}
            poc = (pa.get("POC") or {}).get("price")
            pos = pa.get("position", "—")
            if poc:
                pocs_a.append((tf, poc))
            if pos and pos != "—":
                positions.append((tf, pos))

        lines = []
        if pocs_a:
            poc_str = ", ".join(f"{tf}: {_pct(p, price)}" for tf, p in pocs_a)
            lines.append(f"POC (Profile A) по ТФ: {poc_str}.")
        if positions:
            pos_str = "; ".join(f"{tf}: {p}" for tf, p in positions)
            lines.append(f"Положение цены относительно зоны стоимости: {pos_str}.")
        return " ".join(lines) if lines else "Volume Profile посчитан отдельно по каждому ТФ."

    # ─── S10 ВАЙКОФФ ───
    if sid == 10:
        phases = [(tf, (d or {}).get("structure_type", "—"), (d or {}).get("phase", "—"))
                  for tf, _, d in items]
        unique_types = {t[1] for t in phases if t[1] != "—"}
        if len(unique_types) == 1:
            types_str = ", ".join(f"{tf} фаза {ph}" for tf, _, ph in phases)
            return f"Структура Вайкоффа согласована: {list(unique_types)[0]}. Фазы: {types_str}."
        s = "; ".join(f"{tf}: {st} (фаза {ph})" for tf, st, ph in phases)
        return f"⚠ Структуры Вайкоффа различаются: {s}."

    # ─── S11 СТОПЫ ───
    if sid == 11:
        clusters_above = []
        clusters_below = []
        for tf, _, d in items:
            for s in ((d or {}).get("stops_above_resistances") or []):
                clusters_above.append((tf, s.get("cluster_count", 1), s.get("stop_zone")))
            for s in ((d or {}).get("stops_below_supports") or []):
                clusters_below.append((tf, s.get("cluster_count", 1), s.get("stop_zone")))
        lines = []
        if clusters_above:
            top_a = sorted(clusters_above, key=lambda x: -x[1])[:2]
            lines.append("Крупные ликвидные пулы сверху: "
                         + ", ".join(f"{_pct(p, price)} [{tf}, {cc} кластеров]" for tf, cc, p in top_a) + ".")
        if clusters_below:
            top_b = sorted(clusters_below, key=lambda x: -x[1])[:2]
            lines.append("Крупные ликвидные пулы снизу: "
                         + ", ".join(f"{_pct(p, price)} [{tf}, {cc} кластеров]" for tf, cc, p in top_b) + ".")
        return " ".join(lines) if lines else "Значимых кластеров стопов не обнаружено."

    # ─── S12 ТЕМП ───
    if sid == 12:
        ks = [(tf, (d or {}).get("k_tempo"), (d or {}).get("tempo_class", "—")) for tf, _, d in items]
        atrs = [(tf, (d or {}).get("atr_tf_pct")) for tf, _, d in items]
        k_str = ", ".join(f"{tf} K={k} ({cls})" for tf, k, cls in ks if k is not None)
        atr_str = ", ".join(f"{tf} {a}%" for tf, a in atrs if a is not None)
        senior_k = ks[0][1] if ks else None
        lines = [f"Темп по ТФ: {k_str}."]
        if atr_str:
            lines.append(f"ATR%: {atr_str}.")
        if senior_k is not None:
            lines.append(f"Senior ({ks[0][0]}) задаёт базовый темп K={senior_k}.")
        return " ".join(lines)

    # ─── S13 FVG ───
    if sid == 13:
        magnets = []
        first_steps = []
        for tf, _, d in items:
            m = (d or {}).get("nearest_magnet")
            if m and isinstance(m, dict) and m.get("price"):
                magnets.append((tf, m.get("type", "?"), m.get("price")))
            fs = (d or {}).get("first_step")
            if fs:
                first_steps.append((tf, fs.get("direction", "—"), fs.get("target")))

        lines = []
        if magnets:
            m_str = ", ".join(f"{tf}: {t} {_pct(p, price)}" for tf, t, p in magnets)
            lines.append(f"Ближайшие магниты: {m_str}.")
        if first_steps:
            fs_str = ", ".join(f"{tf}: {dir_} к {_pct(p, price)}" for tf, dir_, p in first_steps)
            lines.append(f"Первый шаг по ТФ: {fs_str}.")
            dirs = {fs[1] for fs in first_steps}
            if len(dirs) > 1:
                lines.append("⚠ Направление первого шага различается между ТФ.")
        return " ".join(lines) if lines else "Активных FVG/гэпов не обнаружено."

    # ─── S14 SQUEEZE ───
    if sid == 14:
        squeezes = [(tf, bool((d or {}).get("squeeze_active"))) for tf, _, d in items]
        active = [tf for tf, a in squeezes if a]
        phases = [(tf, (d or {}).get("vol_phase", "—")) for tf, _, d in items]
        if active:
            return (f"🔒 Squeeze АКТИВЕН на {', '.join(active)} — готовится импульсный выход. "
                    f"Фазы волатильности: " + ", ".join(f"{tf}: {p}" for tf, p in phases) + ".")
        return "Squeeze неактивен ни на одном ТФ, волатильность в норме. " \
               + "Фазы: " + ", ".join(f"{tf}: {p}" for tf, p in phases) + "."

    # ─── S15 ЭФФЕКТИВНОСТЬ ───
    if sid == 15:
        ers = [(tf, (d or {}).get("efficiency_ratio"), (d or {}).get("er_classification", "—"))
               for tf, _, d in items]
        er_str = ", ".join(f"{tf}: ER {er} ({cls})" for tf, er, cls in ers if er is not None)
        classes = [cls for _, _, cls in ers]
        if all(c in ("сильный тренд", "тренд") for c in classes):
            return f"Все ТФ трендовые: {er_str}."
        if all(c in ("шум", "боковик") for c in classes):
            return f"Все ТФ в шуме/боковике: {er_str}."
        return f"Эффективность смешанная: {er_str} — senior задаёт приоритет."

    # ─── S16 ПОТОКИ ───
    if sid == 16:
        cmfs = [(tf, (d or {}).get("cmf", {}).get("signal"), (d or {}).get("cmf", {}).get("value"))
                for tf, _, d in items]
        cvds = [(tf, (d or {}).get("cvd", {}).get("dir_50")) for tf, _, d in items]
        cmf_sigs = {c[1] for c in cmfs if c[1]}
        lines = []
        if len(cmf_sigs) == 1:
            lines.append(f"CMF на всех ТФ — {list(cmf_sigs)[0]}.")
        else:
            lines.append("CMF различается: " + ", ".join(f"{tf}: {sig} ({val})" for tf, sig, val in cmfs if sig) + ".")
        cvd_sigs = {c[1] for c in cvds if c[1]}
        if len(cvd_sigs) == 1:
            lines.append(f"CVD(50): {list(cvd_sigs)[0]} на всех ТФ.")
        else:
            lines.append("CVD(50): " + ", ".join(f"{tf}: {d}" for tf, d in cvds if d) + ".")
        return " ".join(lines)

    # ─── S17 МИКРОСТРУКТУРА ───
    if sid == 17:
        sigs = [(tf, (d or {}).get("institutional_signal", "—")) for tf, _, d in items]
        bals = [(tf, (d or {}).get("supply_demand_balance", "—")) for tf, _, d in items]
        sig_vals = {s for _, s in sigs if s and s != "—" and s != "отсутствует"}
        if sig_vals:
            s_str = ", ".join(f"{tf}: {s}" for tf, s in sigs if s != "—")
            b_str = ", ".join(f"{tf}: {b}" for tf, b in bals)
            return f"Институциональные сигналы: {s_str}. Баланс спрос/предложение: {b_str}."
        return "Институциональных следов нет, баланс нейтральный на всех ТФ."

    # ─── S18 КОРРЕЛЯЦИИ ───
    if sid == 18:
        regimes = [(tf, (d or {}).get("market_regime", "—")) for tf, _, d in items]
        fans = [(tf, (d or {}).get("alma_fan_state", "—")) for tf, _, d in items]
        unique_r = {r for _, r in regimes if r != "—"}
        if len(unique_r) == 1:
            r_text = f"Все ТФ в одном режиме: {list(unique_r)[0]}."
        else:
            r_text = "Режимы ТФ: " + ", ".join(f"{tf}: {r}" for tf, r in regimes) + "."
        unique_fans = {f for _, f in fans if f != "—"}
        if len(unique_fans) == 1:
            f_text = f" ALMA-веер: {list(unique_fans)[0]} (единый по всем ТФ)."
        else:
            f_text = " ALMA-веер различается: " + ", ".join(f"{tf}: {f}" for tf, f in fans) + "."
        return r_text + f_text

    return "Параметры раздела сведены по всем ТФ."


# ──────────────────────────────────────────────────────────────
# Раздел 19 — ВЫВОД с Elder-вердиктом и сводными уровнями
# ──────────────────────────────────────────────────────────────

def _dedup_levels(levels: list, price: float, threshold_pct: float = 0.25) -> list:
    """Склеить уровни, отстоящие друг от друга менее чем на threshold_pct%."""
    if not levels:
        return []
    # levels: [{"price": p, "label": str, "tf": str, "pct": float}, ...]
    sorted_by_dist = sorted(levels, key=lambda x: abs(x["pct"]))
    merged = []
    for lvl in sorted_by_dist:
        matched = False
        for m in merged:
            if abs(lvl["pct"] - m["pct"]) < threshold_pct:
                # Добавить ТФ и метку к существующему кластеру
                m["tfs"].add(lvl["tf"])
                m["labels"].add(lvl["label"])
                matched = True
                break
        if not matched:
            merged.append({
                "price": lvl["price"],
                "pct": lvl["pct"],
                "tfs": {lvl["tf"]},
                "labels": {lvl["label"]},
            })
    return merged


def _format_level_cluster(cluster: dict) -> str:
    tfs = "/".join(sorted(cluster["tfs"]))
    labels = ", ".join(sorted(l for l in cluster["labels"] if l))
    label_part = f" [{labels}]" if labels else ""
    n_tfs = len(cluster["tfs"])
    conf = " 🔥конфлюенс" if n_tfs >= 2 else ""
    cp = cluster['price']
    p_str = f"{cp:.2f}" if cp >= 1000 else f"{cp:.4g}"
    return f"{p_str} ({cluster['pct']:+.2f}%, {tfs}){label_part}{conf}"


def _format_section_19(cons: dict) -> str:
    """Финальный вывод по всем ТФ (формат v8.1).

    Использует route_engine для построения маршрута по данным SENIOR ТФ,
    обогащённым кросс-ТФ конфлюенсом. Буллиты в формате v8.1.
    """
    from pipeline.route_engine import build_route

    lines = []
    elder = cons.get("elder", {})
    ticker = cons.get("ticker", "—")
    price = cons.get("current_price", 0)
    tfs = cons.get("tf_list", [])
    horizon = cons.get("horizon", "—")

    verdict_text = elder.get("verdict_text", "—")
    verdict = elder.get("verdict", "neutral")
    score = elder.get("total_score", 0)
    agreement = elder.get("agreement", "—")

    per_tf = cons.get("per_tf", [])
    senior = per_tf[-1] if per_tf else None
    senior_dir = senior["signals"].get("direction", "—") if senior else "—"
    senior_tf = senior["tf"] if senior else "—"

    if verdict.startswith("strong"):
        direction_word = "явно восходящее" if "bull" in verdict else "явно нисходящее"
    elif verdict in ("bull", "weak_bull"):
        direction_word = "восходящее с умеренной силой"
    elif verdict in ("bear", "weak_bear"):
        direction_word = "нисходящее с умеренной силой"
    else:
        direction_word = "неопределённое"

    lines.append(
        f"{ticker} торгуется по {price:.6g}. Сводный анализ по {len(tfs)} ТФ "
        f"({' / '.join(tfs)}) показывает {direction_word} движение. "
        f"Итоговый счёт Элдера: {score:+.2f} ({verdict_text}). "
        f"Согласованность ТФ: {agreement}. Горизонт прогноза — {horizon}."
    )

    lines.append("")
    lines.append("Разбор по ТФ (от старшего к младшему):")
    for e in reversed(per_tf):
        sig = e["signals"]
        lines.append(
            f"  • {e.get('role_label', e['role'])} ТФ {e['tf']} (вес ×{e['weight']}): "
            f"{sig.get('direction', '—')}, стадия {sig.get('stage', '—')}, "
            f"ALMA {sig.get('alma_order', '—')}, голос {e['vote']:+.2f} "
            f"→ взвешенный {e['weighted_vote']:+.2f}."
        )

    warnings = elder.get("warnings", [])
    for w in warnings:
        lines.append(f"⚠ {w}")

    if elder.get("entry_zone_alert"):
        dir_word = "ЛОНГ" if elder.get("entry_direction") == "bull" else "ШОРТ"
        lines.append(
            f"⚡ Классическая зона входа по Элдеру: младший ТФ корректируется против "
            f"старшего тренда — потенциальный {dir_word} в направлении senior."
        )

    # ─── Направление сценария: Регламент v4 — СТАРШИЙ ТФ задаёт направление,
    # младшие уточняют структуру, но не переписывают сценарий старшего.
    # Голос Элдера используется только если старший тренд не определён.
    if senior_dir in ("восходящий", "нисходящий"):
        corrected_dir = senior_dir
    elif verdict in ("bear", "weak_bear", "strong_bear"):
        corrected_dir = "нисходящий"
    elif verdict in ("bull", "weak_bull", "strong_bull"):
        corrected_dir = "восходящий"
    else:
        corrected_dir = senior_dir

    # ─── Маршрут через route_engine: цели старшего ТФ + уточнение младшими ───
    route_result = None
    if senior:
        try:
            senior_sections_map = {
                s["section_id"]: s for s in senior.get("sections_data", [])
            }
            senior_tf_hours = senior.get("tf_hours", 4.0)

            # Младшие/средние ТФ уточняют структуру маршрута (Регламент v4)
            extra_maps = []
            for e in per_tf:
                if e is senior:
                    continue
                extra_maps.append({
                    s["section_id"]: s for s in e.get("sections_data", [])
                })

            # k_эффект = взвешенное k по ТФ: старший 0.5, рабочий 0.3, младший 0.2
            # (Регламент v4, раздел IV). Нормируем по присутствующим ТФ.
            k_effect = None
            try:
                ordered = sorted(per_tf, key=lambda e: e.get("tf_hours", 0))  # junior → senior
                spec_w = {1: [1.0], 2: [0.4, 0.6], 3: [0.2, 0.3, 0.5]}
                n_tf = min(len(ordered), 3)
                # При >3 ТФ берём младший, средний и старший
                if len(ordered) > 3:
                    picked = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
                else:
                    picked = ordered
                ws = spec_w.get(len(picked), [1.0])
                num, den = 0.0, 0.0
                for e_tf, w in zip(picked, ws):
                    s12_tf = next(
                        (s.get("data", {}) for s in e_tf.get("sections_data", [])
                         if s.get("section_id") == 12), {})
                    k_val = s12_tf.get("k_tempo")
                    if k_val and k_val > 0:
                        num += k_val * w
                        den += w
                if den > 0:
                    k_effect = num / den
            except Exception:
                k_effect = None

            route_result = build_route(
                senior_sections_map, price, corrected_dir, tf_hours=senior_tf_hours,
                k_effect=k_effect, extra_sections_maps=extra_maps,
            )
        except Exception:
            route_result = None

    lines.append("")

    if route_result and (route_result.get("key_targets") or route_result.get("other_targets")):
        # v8.1 буллиты из route_engine
        key_str = ", ".join(route_result.get("key_targets", []))
        other_str = ", ".join(route_result.get("other_targets", []))
        lines.append(f"📌 ключевые цели: {key_str or '—'}")
        lines.append(f"📌 остальные цели: {other_str or '—'}")
        lines.append(f"📌 вероятный маршрут: {route_result.get('route_str', '—')}")

        # Форматирование сроков (как в single-TF)
        hl = route_result.get("horizon_label", "")
        d_min = route_result.get("days_min") or route_result.get("days", 0)
        d_max = route_result.get("days_max") or route_result.get("days", 0)
        if hl == "интрадей":
            dur_range = "интрадей"
        elif d_min and d_max and d_min != d_max:
            def _dw(n):
                if n % 10 == 1 and n % 100 != 11:
                    return "день"
                if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
                    return "дня"
                return "дней"
            dur_range = f"{d_min}–{d_max} {_dw(d_max)}"
        elif d_min:
            dur_range = f"{d_min} дней"
        else:
            dur_range = horizon
        lines.append(f"📌 вероятные сроки: {dur_range}")
        lines.append(f"📌 отмена маршрута: {route_result.get('slam_str', '—')}")
    else:
        # Fallback: сводные уровни из всех ТФ с дедупликацией
        all_res = []
        all_sup = []
        for e in per_tf:
            by_id = {s["section_id"]: s.get("data", {}) for s in e["sections_data"]}
            s6 = by_id.get(6, {}) or {}
            for r in (s6.get("resistances") or s6.get("resistances_5") or []):
                p = r.get("price")
                if p and p > price:
                    all_res.append({
                        "price": p, "label": r.get("label", ""),
                        "tf": e["tf"],
                        "pct": (p - price) / price * 100 if price else 0,
                    })
            for s in (s6.get("supports") or s6.get("supports_5") or []):
                p = s.get("price")
                if p and p < price:
                    all_sup.append({
                        "price": p, "label": s.get("label", ""),
                        "tf": e["tf"],
                        "pct": (p - price) / price * 100 if price else 0,
                    })

        res_clusters = _dedup_levels(all_res, price, threshold_pct=0.25)
        sup_clusters = _dedup_levels(all_sup, price, threshold_pct=0.25)
        res_clusters.sort(key=lambda x: x["pct"])
        sup_clusters.sort(key=lambda x: -x["pct"])

        key_targets = []
        other_targets = []
        slam_cluster = None
        if verdict in ("strong_bull", "bull", "weak_bull"):
            key_targets = res_clusters[:3]
            other_targets = res_clusters[3:6]
            slam_cluster = sup_clusters[0] if sup_clusters else None
        elif verdict in ("strong_bear", "bear", "weak_bear"):
            key_targets = sup_clusters[:3]
            other_targets = sup_clusters[3:6]
            slam_cluster = res_clusters[0] if res_clusters else None
        else:
            # Нейтрально — ближайшие с обеих сторон
            key_targets = (res_clusters[:2] + sup_clusters[:2])[:4]
            other_targets = (res_clusters[2:4] + sup_clusters[2:4])[:4]

        _fp = lambda v: f"{v:.2f}" if v >= 1000 else f"{v:.4g}"

        if key_targets:
            lines.append(f"📌 ключевые цели: {', '.join(_format_level_cluster(x) for x in key_targets)}")
        else:
            lines.append("📌 ключевые цели: —")

        if other_targets:
            lines.append(f"📌 остальные цели: {', '.join(_format_level_cluster(x) for x in other_targets)}")
        else:
            lines.append("📌 остальные цели: —")

        all_tg = key_targets + other_targets
        if all_tg:
            route = f"{_fp(price)} → " + " → ".join(_fp(x['price']) for x in all_tg)
            lines.append(f"📌 вероятный маршрут: {route}")
        else:
            lines.append("📌 вероятный маршрут: —")

        lines.append(f"📌 вероятные сроки: {horizon}")

        if slam_cluster:
            lines.append(f"📌 отмена маршрута: {_format_level_cluster(slam_cluster)}")
        else:
            lines.append("📌 отмена маршрута: —")

    # ── Оценка качества сигнала ──
    score_val = 0
    flags = []
    is_bull = verdict in ("bull", "weak_bull", "strong_bull")

    if is_bull:
        for e in per_tf:
            sig = e.get("signals", {})
            if sig.get("alma_order", "").startswith("медвежий"):
                score_val += 1
                flags.append(f"ALMA медвежий на {e['tf']}")
                break
        wyck_type = ""
        for e in per_tf:
            for s in e.get("sections_data", []):
                if s.get("section_id") == 10:
                    text10 = s.get("text", "").lower()
                    if "распределение" in text10:
                        wyck_type = "распределение"
                    elif "накопление" in text10:
                        wyck_type = "накопление"
        if wyck_type == "распределение":
            score_val += 2
            flags.append("Вайкофф распределение на bull")

        for e in per_tf:
            for s in e.get("sections_data", []):
                if s.get("section_id") == 16:
                    text16 = s.get("text", "").lower()
                    if "отрицательный" in text16 or "отток" in text16:
                        score_val += 2
                        flags.append("CMF продавцы на bull")
                        break
            if score_val >= 2:
                break

    if score_val == 0:
        q_verdict = "ВЫСОКОЕ (score=0)"
        q_emoji = "V"
    elif score_val == 1:
        q_verdict = "СРЕДНЕЕ (score=1)"
        q_emoji = "!"
    else:
        q_verdict = f"НИЗКОЕ (score={score_val}) -- рекомендуется пропустить"
        q_emoji = "X"

    lines.append("")
    lines.append("=" * 55)
    lines.append(f"[{q_emoji}] КАЧЕСТВО СИГНАЛА: {q_verdict}")
    lines.append("=" * 55)
    if flags:
        lines.append("Факторы риска:")
        for f in flags:
            lines.append(f"  - {f}")
    lines.append(f"Elder score: {score:+.2f} | Agreement: {agreement}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Главная функция — форматирование отчёта
# ──────────────────────────────────────────────────────────────

def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + ln if ln else ln for ln in text.split("\n"))


def format_multi_tf_report(consolidated: dict) -> str:
    """Сформировать мульти-ТФ отчёт (формат v8.1).

    Для каждого из 18 разделов выводит:
    - Заголовок раздела (один раз)
    - Полное содержимое раздела для каждого ТФ (senior → junior) через ai.formatter
    - Содержательный cross-TF вывод (📍 Вывод раздела)

    Затем раздел 19 — сводный вердикт Элдера с маршрутом через route_engine.
    """
    if "error" in consolidated:
        return f"Ошибка: {consolidated['error']}"

    # Ленивый импорт, чтобы избежать циклических зависимостей
    from ai.formatter import SECTION_FORMATTERS, _format_generic
    from datetime import datetime

    parts = []
    ticker = consolidated.get("ticker", "—")
    exchange = consolidated.get("exchange", "—")
    tfs = consolidated.get("tf_list", [])
    price = consolidated.get("current_price", 0)
    horizon = consolidated.get("horizon", "—")
    at_name = consolidated.get("analysis_type_name", "—")
    per_tf = consolidated.get("per_tf", [])

    # Шапка (формат v8.1)
    parts.append(f"📘ТИКЕР: #{ticker} (#{exchange})")
    parts.append(f"БИРЖА: #{exchange}")
    parts.append(f"ТИП АНАЛИЗА: Мульти-ТФ {at_name}")
    parts.append(f"ТАЙМФРЕЙМЫ: {' / '.join(tfs)}")
    parts.append(f"ЦЕНА: {price}")
    parts.append(f"ДАТА И ВРЕМЯ: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC+3")
    parts.append("❗ НЕ ЯВЛЯЕТСЯ ИИР ❗")
    parts.append("")
    roles_str = ", ".join(
        f"{e['tf']}={e.get('role_label', e['role'])} (×{e['weight']})"
        for e in per_tf
    )
    parts.append(f"РОЛИ ТФ: {roles_str}")
    parts.append("")
    parts.append("═" * 72)
    parts.append("")

    # 18 разделов
    for sid, emoji, title in SECTION_META:
        parts.append(f"{emoji} {sid}. {title}")
        parts.append("")

        # Данные этого раздела по каждому ТФ (senior → junior)
        per_tf_data_by_label = []  # [(tf_label, role_label, data_dict), ...]
        for e in reversed(per_tf):
            by_id = {s["section_id"]: s.get("data", {}) for s in e["sections_data"]}
            d = by_id.get(sid, {}) or {}
            per_tf_data_by_label.append((e["tf"], e.get("role_label", e["role"]), d))

            # Полное содержимое раздела для этого ТФ
            role_label = e.get("role_label", e["role"])
            parts.append(f"── {e['tf']} ({role_label}, ×{e['weight']}) ──")

            formatter = SECTION_FORMATTERS.get(sid)
            try:
                if formatter:
                    section_text = formatter(d, price, include_conclusion=False)
                else:
                    section_text = _format_generic(d, price)
            except Exception as ex:
                section_text = f"(ошибка форматирования: {ex})"

            parts.append(_indent(section_text, "    "))
            parts.append("")

        # Cross-TF вывод раздела
        try:
            conclusion = _section_conclusion(sid, per_tf_data_by_label, consolidated, price)
        except Exception as ex:
            conclusion = f"(ошибка вывода: {ex})"
        parts.append(f"📍 Вывод раздела: {conclusion}")
        parts.append("")
        parts.append("─" * 72)
        parts.append("")

    # Раздел 19 — ВЫВОД
    parts.append("🧠 Раздел 19 ВЫВОД")
    parts.append("")
    parts.append(_format_section_19(consolidated))
    parts.append("")
    parts.append("━" * 24)
    parts.append("🟥 Платная аналитика по ГАЗУ и ПЛАТИНЕ: @Siroezhkin_bot")
    parts.append("💰 Для донатов:")
    parts.append("💳 https://pay.cloudtips.ru/p/562cbedb")
    parts.append("💳 2200 7006 2350 2977 (Т-Банк)")
    parts.append("🔥 Больше инструментов в профиле")
    parts.append("")
    parts.append("Нужен ФОМО?")
    parts.append("")

    return "\n".join(parts)
