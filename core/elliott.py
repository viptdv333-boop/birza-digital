"""
Биржа-цифровой — Алгоритмическая разволновка Эллиотта.

По Пректеру/Фросту ("Elliott Wave Principle"):

АБСОЛЮТНЫЕ ПРАВИЛА (нарушение = невалидный счёт):
1. Волна 2 НЕ может ретрейсить > 100% волны 1
2. Волна 3 НЕ может быть самой короткой из 1, 3, 5
3. Волна 4 НЕ заходит на территорию волны 1 (кроме диагоналей)

РУКОВОДЯЩИЕ ПРИНЦИПЫ (Пректер):
- Волна 2 обычно ретрейсит 50-61.8% волны 1 (часто зигзаг)
- Волна 3 обычно самая длинная, часто 1.618× волны 1
- Волна 4 обычно ретрейсит 23.6-38.2% волны 3
- Волна 5 часто = волне 1 или 0.618× волны 1
- АЛЬТЕРНАЦИЯ: если W2 резкая → W4 плоская, и наоборот
- РАВЕНСТВО: не-удлинённые волны стремятся к равенству
- КАНАЛИРОВАНИЕ: линия через окончания 1-3, параллель через 2
- РАСШИРЕНИЕ: одна из 1/3/5 расширена (обычно 3)
- Усечение (truncation): W5 не пробивает конец W3

КОРРЕКЦИИ:
- Зигзаг (A-B-C): B ретрейсит 38.2-78.6% A; C ≈ 100% A или 1.618× A
- Плоская (A-B-C): B ретрейсит >90% A; C ≈ A
- Треугольник: каждая нога короче предыдущей
- Расширенная плоская: B > 100% A; C > 100% A
"""
import numpy as np


# ──────────────────────────────────────────────
# Fibonacci ratios
# ──────────────────────────────────────────────
FIB = {
    "0.236": 0.236, "0.382": 0.382, "0.500": 0.500,
    "0.618": 0.618, "0.786": 0.786, "1.000": 1.000,
    "1.272": 1.272, "1.618": 1.618, "2.000": 2.000, "2.618": 2.618,
}


def find_optimal_zigzag(high, low, times=None):
    """Подобрать ZigZag deviation для получения 8-15 крупных волн.

    Начинаем с 5%, если точек слишком много — увеличиваем.
    """
    from core.zigzag import zigzag as zz_func
    for dev in [5, 8, 10, 12, 15, 20, 25]:
        pts = zz_func(high, low, float(dev), times)
        if 6 <= len(pts) <= 20:
            return pts, dev
    # Fallback — крупный ZZ
    pts = zz_func(high, low, 20.0, times)
    return pts, 20


def label_elliott_waves(zigzag_points: list[dict], trend_start_price: float,
                        current_price: float,
                        high=None, low=None, times=None) -> dict:
    """Двухуровневая разволновка Эллиотта.

    Уровень 1 (старший): крупные волны I, II, III... по ZZ 20-25%
    Уровень 2 (младший): подволны (i), (ii), (iii)... текущей крупной волны по ZZ 5%
    """
    from core.zigzag import zigzag as zz_func

    if len(zigzag_points) < 4:
        return {"waves": [], "current_wave": "—", "wave_targets": [],
                "pattern": "unknown", "is_uptrend": True,
                "validation": {}, "guidelines": [], "subwaves": []}

    is_uptrend = current_price > trend_start_price

    # ═══════════════════════════════════════════
    # УРОВЕНЬ 1: СТАРШИЙ СЧЁТ (крупные волны)
    # ═══════════════════════════════════════════
    # Подбираем крупный ZZ чтобы получить 4-8 точек
    senior_points = zigzag_points
    senior_dev = 5
    if high is not None:
        for dev in [20, 25, 30, 15, 10]:
            pts = zz_func(high, low, float(dev), times)
            if 4 <= len(pts) <= 10:
                senior_points = pts
                senior_dev = dev
                break

    # Найти якорь в старших точках
    anchor_idx = _find_anchor(senior_points, trend_start_price)
    senior = senior_points[anchor_idx:]
    if len(senior) < 3:
        senior = senior_points

    alternating = _extract_alternating(senior, is_uptrend)

    # Разметить старшие волны
    impulse = _fit_impulse(alternating, is_uptrend)
    correction = None

    waves = []
    validation = {}
    guidelines = []
    pattern = "unknown"
    current_wave = "—"

    if impulse and impulse["waves"]:
        # Переименовать в римские: 0→0, 1→I, 2→II, 3→III, 4→IV, 5→V
        roman = {"0": "0", "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V"}
        for w in impulse["waves"]:
            w["label"] = roman.get(w["label"], w["label"])
        waves = impulse["waves"]
        validation = impulse["validation"]
        guidelines = impulse["guidelines"]
        pattern = "impulse"
        current_wave = roman.get(impulse["current"], impulse["current"])

        # Проверить: если импульс не complete, но цена уже развернулась
        # от последней волны в сторону тренда → следующая волна в развитии
        if not impulse["complete"] and len(impulse["waves"]) >= 5:
            last_wave = impulse["waves"][-1]
            # Для нисходящего: если последняя волна = коррекция вверх (чётная)
            # и цена уже НИЖЕ неё → следующая (нечётная) в развитии
            if not is_uptrend and last_wave["type"] == "high" and current_price < last_wave["price"]:
                current_wave = roman.get(str(int(impulse["current"]) + 1), "V")
            elif is_uptrend and last_wave["type"] == "low" and current_price > last_wave["price"]:
                current_wave = roman.get(str(int(impulse["current"]) + 1), "V")

        if impulse["complete"]:
            remaining = alternating[impulse["used_points"]:]
            if len(remaining) >= 3:
                correction = _fit_correction(remaining, is_uptrend)
                if correction and correction["waves"]:
                    corr_roman = {"start_corr": "start_corr", "A": "A", "B": "B", "C": "C"}
                    for w in correction["waves"]:
                        w["label"] = corr_roman.get(w["label"], w["label"])
                    waves.extend(correction["waves"])
                    current_wave = correction["current"]
                    pattern = "impulse+correction"
                    guidelines.extend(correction.get("guidelines", []))
    else:
        # Попробовать как коррекцию
        correction = _fit_correction(alternating, is_uptrend)
        if correction and correction["waves"]:
            waves = correction["waves"]
            current_wave = correction["current"]
            pattern = "correction"
            guidelines = correction.get("guidelines", [])

    # Определить текущую крупную волну — последняя в списке
    current_senior_wave = waves[-1] if waves else None

    # ═══════════════════════════════════════════
    # УРОВЕНЬ 2: МЛАДШИЙ СЧЁТ (подволны текущей)
    # ═══════════════════════════════════════════
    subwaves = []
    sub_targets = []
    sub_current = "—"
    sub_validation = {}

    if current_senior_wave and len(zigzag_points) >= 4:
        # Найти ZZ5% точки ПОСЛЕ начала текущей крупной волны
        senior_start_price = current_senior_wave["price"]
        sub_anchor = _find_anchor(zigzag_points, senior_start_price)
        sub_points = zigzag_points[sub_anchor:]

        if len(sub_points) >= 4:
            sub_alternating = _extract_alternating(sub_points, is_uptrend)

            # Попробовать импульс на подволнах
            sub_impulse = _fit_impulse(sub_alternating, is_uptrend)
            if sub_impulse and sub_impulse["waves"]:
                # Подволны: (i), (ii), (iii), (iv), (v)
                sub_label = {"0": "(0)", "1": "(i)", "2": "(ii)",
                             "3": "(iii)", "4": "(iv)", "5": "(v)"}
                for w in sub_impulse["waves"]:
                    w["label"] = sub_label.get(w["label"], w["label"])
                subwaves = sub_impulse["waves"]
                sub_current = sub_label.get(sub_impulse["current"],
                                            sub_impulse["current"])
                sub_validation = sub_impulse.get("validation", {})

                # Подкоррекция
                if sub_impulse["complete"]:
                    sub_rem = sub_alternating[sub_impulse["used_points"]:]
                    if len(sub_rem) >= 3:
                        sub_corr = _fit_correction(sub_rem, is_uptrend)
                        if sub_corr and sub_corr["waves"]:
                            sub_corr_label = {"start_corr": "(start_corr)",
                                              "A": "(a)", "B": "(b)", "C": "(c)"}
                            for w in sub_corr["waves"]:
                                w["label"] = sub_corr_label.get(w["label"], w["label"])
                            subwaves.extend(sub_corr["waves"])
                            sub_current = sub_corr_label.get(
                                sub_corr["current"], sub_corr["current"])
            else:
                # Partial импульс
                sub_partial = _fit_partial_impulse(sub_alternating, is_uptrend)
                if sub_partial and sub_partial["waves"]:
                    sub_label = {"0": "(0)", "1": "(i)", "2": "(ii)",
                                 "3": "(iii)", "4": "(iv)", "5": "(v)"}
                    for w in sub_partial["waves"]:
                        w["label"] = sub_label.get(w["label"], w["label"])
                    subwaves = sub_partial["waves"]
                    sub_current = sub_label.get(sub_partial["current"],
                                                sub_partial["current"])
                    sub_validation = sub_partial.get("validation", {})

    # ═══════════════════════════════════════════
    # ЦЕЛИ: старший + младший уровни
    # ═══════════════════════════════════════════
    # Цели старшего уровня (обратная конвертация из римских)
    rev_roman = {"0": "0", "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}
    senior_clean = [dict(w, label=rev_roman.get(w["label"], w["label"])) for w in waves]
    current_clean = rev_roman.get(current_wave, current_wave)
    senior_targets = _compute_targets(senior_clean, current_clean, current_price, is_uptrend)

    # Цели младшего уровня
    if subwaves:
        rev_sub = {"(0)": "0", "(i)": "1", "(ii)": "2", "(iii)": "3",
                   "(iv)": "4", "(v)": "5",
                   "(start_corr)": "start_corr", "(a)": "A", "(b)": "B", "(c)": "C"}
        sub_clean = [dict(w, label=rev_sub.get(w["label"], w["label"])) for w in subwaves]
        sub_current_clean = rev_sub.get(sub_current, sub_current)
        sub_targets = _compute_targets(sub_clean, sub_current_clean, current_price, is_uptrend)

    # Объединить все цели, отфильтровать достигнутые
    all_targets = []
    for t in senior_targets:
        t["level"] = "senior"
        all_targets.append(t)
    for t in sub_targets:
        t["level"] = "sub"
        all_targets.append(t)

    all_targets = [t for t in all_targets
                   if (is_uptrend and t["price"] > current_price)
                   or (not is_uptrend and t["price"] < current_price)]

    return {
        "waves": waves,
        "current_wave": current_wave,
        "wave_targets": all_targets,
        "pattern": pattern,
        "is_uptrend": is_uptrend,
        "validation": validation,
        "guidelines": guidelines,
        "subwaves": subwaves,
        "sub_current_wave": sub_current,
        "senior_dev": senior_dev,
    }


def _find_anchor(points, target_price):
    """Найти точку ближайшую к якорной цене."""
    min_dist = float("inf")
    idx = 0
    for i, p in enumerate(points):
        d = abs(p["price"] - target_price)
        if d < min_dist:
            min_dist = d
            idx = i
    return idx


def _extract_alternating(points, is_uptrend):
    """Извлечь чередующиеся high/low (без повторов типа)."""
    if not points:
        return []
    result = [points[0]]
    for p in points[1:]:
        if p["type"] != result[-1]["type"]:
            result.append(p)
        else:
            # Обновить если лучше
            if p["type"] == "high" and p["price"] > result[-1]["price"]:
                result[-1] = p
            elif p["type"] == "low" and p["price"] < result[-1]["price"]:
                result[-1] = p
    return result


def _wave_len(p1, p2):
    """Длина волны (абсолютная)."""
    return abs(p2 - p1)


def _retrace_pct(wave_start, wave_end, retrace_end):
    """Процент ретрейсмента."""
    move = wave_end - wave_start
    if abs(move) < 1e-10:
        return 0
    retrace = retrace_end - wave_end
    return abs(retrace / move)


def _fit_impulse(points, is_uptrend):
    """Попытка разметить 5-волновой импульс с валидацией."""
    if len(points) < 6:
        # Частичный импульс
        return _fit_partial_impulse(points, is_uptrend)

    w = points[:6]  # 0, end_w1, end_w2, end_w3, end_w4, end_w5
    p = [x["price"] for x in w]

    validation = {}
    guidelines = []
    valid = True

    # ═══════════════════════════════════════════
    # АБСОЛЮТНЫЕ ПРАВИЛА
    # ═══════════════════════════════════════════

    # Правило 1: W2 не ретрейсит > 100% W1
    w1_len = _wave_len(p[0], p[1])
    w2_retrace = _retrace_pct(p[0], p[1], p[2])
    rule1 = w2_retrace < 1.0
    validation["rule1_w2_less_100pct"] = rule1
    if not rule1:
        valid = False

    # Правило 2: W3 не самая короткая
    w3_len = _wave_len(p[2], p[3])
    w5_len = _wave_len(p[4], p[5])
    rule2 = not (w3_len < w1_len and w3_len < w5_len)
    validation["rule2_w3_not_shortest"] = rule2
    if not rule2:
        valid = False

    # Правило 3: W4 не заходит на территорию W1
    if is_uptrend:
        rule3 = p[4] > p[1]  # W4 low > W1 high — ОК для импульса
    else:
        rule3 = p[4] < p[1]  # W4 high < W1 low
    validation["rule3_w4_no_overlap_w1"] = rule3
    if not rule3:
        # Может быть диагональ — не сразу отбрасываем
        guidelines.append("W4 перекрывает W1 — возможна диагональ")

    # ═══════════════════════════════════════════
    # РУКОВОДЯЩИЕ ПРИНЦИПЫ (Пректер)
    # ═══════════════════════════════════════════

    # W2 обычно 50-61.8% W1
    if 0.382 <= w2_retrace <= 0.786:
        guidelines.append(f"W2 ретрейс {w2_retrace:.1%} — типичный")
    elif w2_retrace > 0.786:
        guidelines.append(f"W2 ретрейс {w2_retrace:.1%} — глубокий (зигзаг?)")
    elif w2_retrace < 0.382:
        guidelines.append(f"W2 ретрейс {w2_retrace:.1%} — мелкий")

    # W3 обычно самая длинная, 1.618× W1
    w3_to_w1 = w3_len / w1_len if w1_len > 0 else 0
    if w3_len > w1_len and w3_len > w5_len:
        guidelines.append(f"W3 самая длинная ({w3_to_w1:.2f}× W1) — норма")
    else:
        guidelines.append(f"W3 = {w3_to_w1:.2f}× W1 — {'расширена' if w3_to_w1 > 1.618 else 'не расширена'}")

    # W4 обычно 23.6-38.2% W3
    w4_retrace = _retrace_pct(p[2], p[3], p[4])
    validation["w4_retrace_pct"] = round(w4_retrace, 3)
    if 0.236 <= w4_retrace <= 0.500:
        guidelines.append(f"W4 ретрейс {w4_retrace:.1%} W3 — типичный")
    elif w4_retrace < 0.236:
        guidelines.append(f"W4 ретрейс {w4_retrace:.1%} W3 — мелкий (сильный тренд)")
    else:
        guidelines.append(f"W4 ретрейс {w4_retrace:.1%} W3 — глубокий")

    # W5 часто = W1 или 0.618× W1
    w5_to_w1 = w5_len / w1_len if w1_len > 0 else 0
    validation["w5_to_w1_ratio"] = round(w5_to_w1, 3)
    if 0.8 <= w5_to_w1 <= 1.2:
        guidelines.append(f"W5 ≈ W1 ({w5_to_w1:.2f}×) — типичное равенство")
    elif 0.5 <= w5_to_w1 <= 0.75:
        guidelines.append(f"W5 = {w5_to_w1:.2f}× W1 — укороченная")

    # Усечение (truncation): W5 не пробивает конец W3
    if is_uptrend:
        truncation = p[5] < p[3]
    else:
        truncation = p[5] > p[3]
    validation["truncation"] = truncation
    if truncation:
        guidelines.append("УСЕЧЕНИЕ: W5 не достигла уровня W3 — слабость тренда")

    # Альтернация W2/W4
    w2_is_sharp = w2_retrace > 0.5
    w4_is_sharp = w4_retrace > 0.382
    if w2_is_sharp != w4_is_sharp:
        guidelines.append("Альтернация W2/W4 соблюдена")
    else:
        guidelines.append("Альтернация W2/W4 НЕ соблюдена (обе " +
                         ("резкие" if w2_is_sharp else "плоские") + ")")

    # Расширение: какая волна расширена
    extended = "нет"
    if w3_len > w1_len * 1.618 and w3_len > w5_len * 1.618:
        extended = "W3"
    elif w1_len > w3_len * 1.618 and w1_len > w5_len * 1.618:
        extended = "W1"
    elif w5_len > w1_len * 1.618 and w5_len > w3_len * 1.618:
        extended = "W5"
    validation["extended_wave"] = extended
    guidelines.append(f"Расширенная волна: {extended}")

    # Каналирование: проверка что W1-W3-W5 и W2-W4 формируют канал
    if is_uptrend:
        channel_upper = (p[1] + p[3] + p[5]) / 3  # средняя вершин
        channel_lower = (p[0] + p[2] + p[4]) / 3  # средняя впадин
    else:
        channel_upper = (p[0] + p[2] + p[4]) / 3
        channel_lower = (p[1] + p[3] + p[5]) / 3
    validation["channel_width_pct"] = round(abs(channel_upper - channel_lower) / max(channel_upper, 0.001) * 100, 2)

    # Валидность
    validation["valid"] = valid and rule1 and rule2
    validation["w2_retrace_pct"] = round(w2_retrace, 3)
    validation["w3_to_w1_ratio"] = round(w3_to_w1, 3)

    if not (rule1 and rule2):
        return None

    # Собрать волны
    labels = ["0", "1", "2", "3", "4", "5"]
    waves = []
    for i in range(6):
        waves.append({
            "label": labels[i],
            "price": round(w[i]["price"], 4),
            "time": w[i].get("time", ""),
            "type": w[i]["type"],
        })

    return {
        "waves": waves,
        "complete": True,
        "current": "5",
        "used_points": 6,
        "validation": validation,
        "guidelines": guidelines,
    }


def _fit_partial_impulse(points, is_uptrend):
    """Частичный импульс (менее 6 точек)."""
    if len(points) < 3:
        return None

    labels = ["0", "1", "2", "3", "4", "5"]
    waves = []
    validation = {}
    guidelines = []

    for i in range(min(len(points), 6)):
        waves.append({
            "label": labels[i],
            "price": round(points[i]["price"], 4),
            "time": points[i].get("time", ""),
            "type": points[i]["type"],
        })

    n = len(waves)

    # Валидация доступных правил
    if n >= 3:
        w2_retrace = _retrace_pct(points[0]["price"], points[1]["price"], points[2]["price"])
        validation["w2_retrace_pct"] = round(w2_retrace, 3)
        validation["rule1_w2_less_100pct"] = w2_retrace < 1.0
        if w2_retrace >= 1.0:
            return None

    if n >= 5:
        p4 = points[4]["price"]
        p1 = points[1]["price"]
        if is_uptrend:
            validation["rule3_w4_no_overlap_w1"] = p4 > p1
        else:
            validation["rule3_w4_no_overlap_w1"] = p4 < p1

    validation["valid"] = True
    current = str(n - 1)

    return {
        "waves": waves,
        "complete": False,
        "current": current,
        "used_points": n,
        "validation": validation,
        "guidelines": guidelines,
    }


def _fit_correction(points, is_uptrend):
    """Разметить коррекцию A-B-C с валидацией."""
    if len(points) < 3:
        return None

    w = points[:4]  # start, A, B, C
    guidelines = []
    waves = []

    waves.append({"label": "start_corr", "price": round(w[0]["price"], 4),
                   "time": w[0].get("time", ""), "type": w[0]["type"]})
    waves.append({"label": "A", "price": round(w[1]["price"], 4),
                   "time": w[1].get("time", ""), "type": w[1]["type"]})

    a_len = _wave_len(w[0]["price"], w[1]["price"])

    current = "A"

    if len(w) >= 3:
        waves.append({"label": "B", "price": round(w[2]["price"], 4),
                       "time": w[2].get("time", ""), "type": w[2]["type"]})
        current = "B"

        # B ретрейс A
        b_retrace = _retrace_pct(w[0]["price"], w[1]["price"], w[2]["price"])

        if b_retrace > 1.0:
            guidelines.append(f"B > 100% A ({b_retrace:.1%}) — расширенная плоская / нерегулярная")
        elif b_retrace > 0.9:
            guidelines.append(f"B ≈ A ({b_retrace:.1%}) — плоская коррекция")
        elif 0.382 <= b_retrace <= 0.786:
            guidelines.append(f"B ретрейс {b_retrace:.1%} A — зигзаг")
        else:
            guidelines.append(f"B ретрейс {b_retrace:.1%} A")

    if len(w) >= 4:
        waves.append({"label": "C", "price": round(w[3]["price"], 4),
                       "time": w[3].get("time", ""), "type": w[3]["type"]})
        current = "C"

        c_len = _wave_len(w[2]["price"], w[3]["price"])
        c_to_a = c_len / a_len if a_len > 0 else 0

        if 0.8 <= c_to_a <= 1.2:
            guidelines.append(f"C ≈ A ({c_to_a:.2f}×) — типичный зигзаг")
        elif c_to_a > 1.5:
            guidelines.append(f"C = {c_to_a:.2f}× A — расширенная C")
        elif c_to_a < 0.618:
            guidelines.append(f"C = {c_to_a:.2f}× A — укороченная C (усечение?)")

    # complete = True если есть все 4 точки (start_corr + A + B + C)
    complete = len(waves) >= 4 and current == "C"
    return {"waves": waves, "current": current, "guidelines": guidelines,
            "complete": complete, "used_points": min(len(w), 4)}


def _compute_targets(waves, current_wave, current_price, is_uptrend):
    """Расчётные цели завершения текущей волны по Фибо (Пректер)."""
    targets = []
    prices = {w["label"]: w["price"] for w in waves}

    if current_wave in ("1", "2") and "0" in prices and "1" in prices:
        # Цели W2: ретрейс W1
        w1_move = prices["1"] - prices["0"]
        base = prices["1"]
        for ratio, name in [(0.382, "38.2%"), (0.500, "50%"), (0.618, "61.8%"), (0.786, "78.6%")]:
            t = base - w1_move * ratio
            targets.append({"fib": f"W2 ретрейс {name}", "price": round(t, 4),
                            "pct": round((t - current_price) / current_price * 100, 2)})

    elif current_wave == "3" and "0" in prices and "1" in prices and "2" in prices:
        # Цели W3: расширения от W2
        w1_len = abs(prices["1"] - prices["0"])
        base = prices["2"]
        for ratio, name in [(1.0, "100% W1"), (1.618, "161.8% W1"), (2.618, "261.8% W1")]:
            t = base + w1_len * ratio * (1 if is_uptrend else -1)
            targets.append({"fib": f"W3 = {name}", "price": round(t, 4),
                            "pct": round((t - current_price) / current_price * 100, 2)})

    elif current_wave == "4" and "2" in prices and "3" in prices:
        # Цели W4: ретрейс W3
        w3_move = prices["3"] - prices["2"]
        base = prices["3"]
        for ratio, name in [(0.236, "23.6%"), (0.382, "38.2%"), (0.500, "50%")]:
            t = base - w3_move * ratio
            targets.append({"fib": f"W4 ретрейс {name} W3", "price": round(t, 4),
                            "pct": round((t - current_price) / current_price * 100, 2)})

    elif current_wave == "5" and "0" in prices and "1" in prices and "4" in prices:
        # Цели W5: от W4, часто = W1 или 0.618× W1
        w1_len = abs(prices["1"] - prices["0"])
        base = prices["4"]
        for ratio, name in [(0.618, "0.618× W1"), (1.0, "= W1"), (1.618, "1.618× W1")]:
            t = base + w1_len * ratio * (1 if is_uptrend else -1)
            targets.append({"fib": f"W5 {name}", "price": round(t, 4),
                            "pct": round((t - current_price) / current_price * 100, 2)})

    elif current_wave in ("A", "B"):
        # Цели B: ретрейс A (B идёт ПРОТИВ A)
        if "start_corr" in prices and "A" in prices:
            a_move = prices["A"] - prices["start_corr"]  # направленное движение A
            base = prices["A"]
            for ratio, name in [(0.382, "38.2%"), (0.500, "50%"), (0.618, "61.8%"), (0.786, "78.6%")]:
                t = base - a_move * ratio  # B идёт обратно от A
                targets.append({"fib": f"B ретрейс {name} A", "price": round(t, 4),
                                "pct": round((t - current_price) / current_price * 100, 2)})

    elif current_wave == "C":
        # Цели C: от конца B, C идёт В ТОМ ЖЕ НАПРАВЛЕНИИ что и A
        if "start_corr" in prices and "A" in prices:
            a_len = abs(prices["A"] - prices["start_corr"])
            base_c = prices.get("B", prices["A"])
            # Направление A: start_corr → A
            a_goes_down = prices["A"] < prices["start_corr"]
            for ratio, name in [(1.0, "C = A"), (1.272, "C = 1.272× A"), (1.618, "C = 1.618× A")]:
                if a_goes_down:
                    t = base_c - a_len * ratio
                else:
                    t = base_c + a_len * ratio
                targets.append({"fib": name, "price": round(t, 4),
                                "pct": round((t - current_price) / current_price * 100, 2)})

            # Расширения от полного импульса (волна 0→5) после коррекции
            if "0" in prices and "5" in prices:
                impulse_len = abs(prices["0"] - prices["5"])
                # Расширения от конца коррекции (B или C)
                ext_base = prices.get("B", prices.get("C", base_c))
                for ratio, name in [(0.618, "имп×0.618"), (1.0, "имп×1.0"), (1.618, "имп×1.618")]:
                    if is_uptrend:
                        t = ext_base + impulse_len * ratio
                    else:
                        t = ext_base - impulse_len * ratio
                    targets.append({"fib": name, "price": round(t, 4),
                                    "pct": round((t - current_price) / current_price * 100, 2)})

    return targets


def label_subwaves(zigzag_minor: list[dict], wave_start_price: float,
                   wave_end_price: float, is_uptrend: bool) -> list[dict]:
    """Разметить подволны (i)-(v) внутри текущей волны."""
    lo = min(wave_start_price, wave_end_price) * 0.99
    hi = max(wave_start_price, wave_end_price) * 1.01

    in_range = [p for p in zigzag_minor if lo <= p["price"] <= hi]
    if len(in_range) < 3:
        return []

    labels = ["(i)", "(ii)", "(iii)", "(iv)", "(v)"]
    subwaves = []
    for i, p in enumerate(in_range[:5]):
        lbl = labels[i] if i < len(labels) else f"({i + 1})"
        subwaves.append({
            "label": lbl,
            "price": round(p["price"], 4),
            "time": p.get("time", ""),
        })

    return subwaves
