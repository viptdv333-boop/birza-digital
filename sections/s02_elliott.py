"""
Биржа-цифровой — Раздел 2: ВОЛНОВОЙ АНАЛИЗ.

Алгоритм (v8, по спецификации Босса):
  1. Выбор масштаба: ZigZag 5% (главные I-V), ZigZag 1-2% (подволны).
  2. Williams Fractals (2/2) — подтверждение разворотных пивотов
     (high[i] > high[i-1], high[i-2], high[i+1], high[i+2]; зеркально для low).
  3. Фильтр значимости: пивот оставляем, если движение от предыдущего
     >= max(2.5×ATR, 3% цены) — для главных волн (1×ATR, 1% — для подволн).
     Иначе — шум.
  4. Якорь = начало тренда из раздела 1 (trend_start_idx).
  5. Чередование low→high→low→high: два пивота одного типа подряд —
     оставляем более экстремальный.
  6. Правила Пректера для импульса:
      Bull: P2 > P0; волна 3 не самая короткая; P4 > P1; P3 > P1; P5 > P3.
      Bear: P2 < P0; волна 3 не самая короткая; P4 < P1; P3 < P1; P5 < P3.
  7. Если правила нарушены — помечаем как коррекцию A-B-C.
  8. Цели: вол.3 ≈ 1.618×вол.1; вол.5 ≈ 0.618..1.0×вол.1;
     вол.C ≈ 1.0..1.618×A; ретрейсы 0.382/0.5/0.618/0.786.
  9. Цены пивотов строго по high[i] / low[i] (НЕ close).
"""
import numpy as np
from sections.base import SectionProcessor


# ────────────────────────────────────────────────────────────────
# 1. FRACTALS (Williams 2/2 by default)
# ────────────────────────────────────────────────────────────────
def _find_fractals(high, low, left: int = 2, right: int = 2) -> list[dict]:
    """Williams Fractals: swing-high если high[i] > high всех left..right соседей
    (строго >), зеркально для swing-low."""
    n = len(high)
    pivots = []
    for i in range(left, n - right):
        h_i = float(high[i])
        l_i = float(low[i])
        is_high = True
        is_low = True
        for k in range(1, left + 1):
            if high[i - k] >= h_i:
                is_high = False
            if low[i - k] <= l_i:
                is_low = False
        for k in range(1, right + 1):
            if high[i + k] >= h_i:
                is_high = False
            if low[i + k] <= l_i:
                is_low = False
        if is_high:
            pivots.append({"index": i, "price": h_i, "type": "high"})
        if is_low:
            pivots.append({"index": i, "price": l_i, "type": "low"})
    pivots.sort(key=lambda p: (p["index"], 0 if p["type"] == "low" else 1))
    return pivots


# ────────────────────────────────────────────────────────────────
# 2. ZIGZAG (OHLC)
# ────────────────────────────────────────────────────────────────
def _zigzag_ohlc(high, low, dev_pct: float) -> list[dict]:
    """ZigZag по high/low: разворот при пробое на dev_pct от предыдущего экстремума."""
    n = len(high)
    if n < 3:
        return []
    dev = dev_pct / 100.0
    pivots: list[dict] = []

    last_hi = float(high[0]); last_hi_idx = 0
    last_lo = float(low[0]);  last_lo_idx = 0

    # Ждём первое значимое движение
    direction = 0  # 1 up, -1 down
    start_i = 0
    for i in range(1, n):
        h = float(high[i]); l = float(low[i])
        if h > last_hi:
            last_hi = h; last_hi_idx = i
        if l < last_lo:
            last_lo = l; last_lo_idx = i
        if last_hi > 0 and (last_hi - last_lo) / last_lo >= dev:
            if last_hi_idx < last_lo_idx:
                # хай был раньше — падение первое
                pivots.append({"index": last_hi_idx, "price": last_hi, "type": "high"})
                direction = -1
                last_lo = float(low[last_lo_idx])
                last_hi = last_lo  # reset hi
                last_hi_idx = last_lo_idx
                start_i = last_lo_idx + 1
            else:
                # лоу был раньше — рост первый
                pivots.append({"index": last_lo_idx, "price": last_lo, "type": "low"})
                direction = 1
                last_hi = float(high[last_hi_idx])
                last_lo = last_hi
                last_lo_idx = last_hi_idx
                start_i = last_hi_idx + 1
            break
    if direction == 0:
        return []

    cur_ext_idx = pivots[0]["index"]
    cur_ext_price = pivots[0]["price"]
    cur_type = pivots[0]["type"]  # последняя поставленная

    # after first pivot, next_ext is opposite
    next_type = "high" if cur_type == "low" else "low"
    next_ext_idx = pivots[0]["index"]
    next_ext_price = float(high[cur_ext_idx]) if next_type == "high" else float(low[cur_ext_idx])

    for i in range(start_i, n):
        h = float(high[i]); l = float(low[i])
        if next_type == "high":
            if h > next_ext_price:
                next_ext_price = h
                next_ext_idx = i
            # разворот — падение на dev от next_ext_price (потенциального high)
            if next_ext_price > 0 and (next_ext_price - l) / next_ext_price >= dev:
                pivots.append({"index": next_ext_idx, "price": next_ext_price, "type": "high"})
                cur_ext_idx = next_ext_idx
                cur_ext_price = next_ext_price
                next_type = "low"
                next_ext_idx = i
                next_ext_price = l
        else:
            if l < next_ext_price:
                next_ext_price = l
                next_ext_idx = i
            # разворот — рост на dev от next_ext_price
            if next_ext_price > 0 and (h - next_ext_price) / next_ext_price >= dev:
                pivots.append({"index": next_ext_idx, "price": next_ext_price, "type": "low"})
                cur_ext_idx = next_ext_idx
                cur_ext_price = next_ext_price
                next_type = "high"
                next_ext_idx = i
                next_ext_price = h

    # добавим финальный неподтверждённый разворот
    if next_ext_idx != cur_ext_idx:
        pivots.append({"index": next_ext_idx, "price": next_ext_price, "type": next_type})

    return pivots


# ────────────────────────────────────────────────────────────────
# 3. SIGNIFICANCE FILTER
# ────────────────────────────────────────────────────────────────
def _filter_significance(pivots: list[dict], atr: float, min_pct: float,
                         atr_mult: float) -> list[dict]:
    """Оставляем только пивоты, где движение от предыдущего превышает порог.
    Порог = max(atr_mult × ATR, min_pct% × предыдущая цена)."""
    if not pivots:
        return []
    filtered = [pivots[0]]
    for p in pivots[1:]:
        prev = filtered[-1]
        move = abs(p["price"] - prev["price"])
        thresh = max(atr_mult * atr, min_pct / 100.0 * abs(prev["price"]))
        if move >= thresh:
            filtered.append(p)
        else:
            # не значимо — но если тот же тип, что у последнего, и более экстремальный,
            # можно заменить последний
            if p["type"] == prev["type"]:
                if (p["type"] == "high" and p["price"] > prev["price"]) or \
                   (p["type"] == "low"  and p["price"] < prev["price"]):
                    filtered[-1] = p
    return filtered


# ────────────────────────────────────────────────────────────────
# 4. ENFORCE ALTERNATION
# ────────────────────────────────────────────────────────────────
def _enforce_alternation(pivots: list[dict]) -> list[dict]:
    """Схлопываем подряд идущие пивоты одного типа — оставляем более экстремальный."""
    if not pivots:
        return []
    out = [pivots[0]]
    for p in pivots[1:]:
        last = out[-1]
        if p["type"] == last["type"]:
            if p["type"] == "high" and p["price"] > last["price"]:
                out[-1] = p
            elif p["type"] == "low" and p["price"] < last["price"]:
                out[-1] = p
            # иначе — игнорируем, менее экстремальный того же типа
        else:
            out.append(p)
    return out


# ────────────────────────────────────────────────────────────────
# 5. IMPULSE VALIDATION (Prechter rules)
# ────────────────────────────────────────────────────────────────
def _validate_impulse(pivots: list[dict], direction: str) -> tuple[bool, list[str]]:
    """Проверка правил Пректера для последовательности P0..P5 (6 точек, 5 волн).

    Bullish (up): P0 low → P1 high → P2 low → P3 high → P4 low → P5 high.
    Bearish (down): P0 high → P1 low → P2 high → P3 low → P4 high → P5 low.
    """
    violations: list[str] = []
    if len(pivots) < 6:
        violations.append(
            f"❌ Недостаточно пивотов для импульса: {len(pivots)}/6"
        )
        return False, violations

    P = pivots[:6]
    expected_types = (
        ["low", "high", "low", "high", "low", "high"] if direction == "восходящий"
        else ["high", "low", "high", "low", "high", "low"]
    )
    for i, (p, t) in enumerate(zip(P, expected_types)):
        if p["type"] != t:
            violations.append(
                f"❌ Неверный тип пивота P{i}: ожидался {t}, получен {p['type']}"
            )
            return False, violations

    p0, p1, p2, p3, p4, p5 = [p["price"] for p in P]

    if direction == "восходящий":
        # Правило 1: волна 2 не ниже начала волны 1
        if p2 <= p0:
            violations.append(
                f"❌ Правило 1: P2 ({p2:.4g}) ≤ P0 ({p0:.4g}) — вторая волна пробила начало импульса"
            )
        # Правило 3 волна не самая короткая
        w1 = abs(p1 - p0); w3 = abs(p3 - p2); w5 = abs(p5 - p4)
        if w3 < w1 and w3 < w5:
            violations.append(
                f"❌ Правило 2: волна III ({w3:.3g}) самая короткая из I({w1:.3g})/V({w5:.3g})"
            )
        # Правило 4 не заходит в зону 1
        if p4 <= p1:
            violations.append(
                f"❌ Правило 3: P4 ({p4:.4g}) ≤ P1 ({p1:.4g}) — волна IV в зоне I (возможно диагональ)"
            )
        # Доп: P3 > P1 (третья обновляет хай первой)
        if p3 <= p1:
            violations.append(
                f"❌ Доп: P3 ({p3:.4g}) ≤ P1 ({p1:.4g}) — вол. III не обновила вершину I"
            )
        # Доп: P5 > P3 (пятая делает новый максимум)
        if p5 <= p3:
            violations.append(
                f"❌ Доп: P5 ({p5:.4g}) ≤ P3 ({p3:.4g}) — вол. V не обновила максимум III"
            )
    else:
        if p2 >= p0:
            violations.append(
                f"❌ Правило 1: P2 ({p2:.4g}) ≥ P0 ({p0:.4g}) — вторая волна пробила начало импульса"
            )
        w1 = abs(p0 - p1); w3 = abs(p2 - p3); w5 = abs(p4 - p5)
        if w3 < w1 and w3 < w5:
            violations.append(
                f"❌ Правило 2: волна III ({w3:.3g}) самая короткая из I({w1:.3g})/V({w5:.3g})"
            )
        if p4 >= p1:
            violations.append(
                f"❌ Правило 3: P4 ({p4:.4g}) ≥ P1 ({p1:.4g}) — волна IV в зоне I (возможно диагональ)"
            )
        if p3 >= p1:
            violations.append(
                f"❌ Доп: P3 ({p3:.4g}) ≥ P1 ({p1:.4g}) — вол. III не обновила минимум I"
            )
        if p5 >= p3:
            violations.append(
                f"❌ Доп: P5 ({p5:.4g}) ≥ P3 ({p3:.4g}) — вол. V не обновила минимум III"
            )

    is_valid = len(violations) == 0
    return is_valid, violations


# ────────────────────────────────────────────────────────────────
# 6. BUILD PIVOT SEQUENCE
# ────────────────────────────────────────────────────────────────
def _build_pivots(anchor_idx: int, end_idx: int,
                  high, low, atr: float,
                  direction: str,
                  min_pct: float, atr_mult: float,
                  zz_dev: float) -> list[dict]:
    """
    Построить последовательность пивотов в диапазоне [anchor_idx..end_idx]:
    1) ZigZag dev% + фракталы (объединяем для робастности)
    2) Фильтр значимости
    3) Чередование
    4) Принудительный первый пивот = якорь правильного типа
    """
    if end_idx <= anchor_idx + 3:
        return []
    h_seg = np.asarray(high[anchor_idx:end_idx + 1], dtype=float)
    l_seg = np.asarray(low[anchor_idx:end_idx + 1], dtype=float)

    # ZigZag pivots
    zz = _zigzag_ohlc(h_seg, l_seg, zz_dev)
    # Пересчитать индексы в абсолютные
    for p in zz:
        p["index"] = anchor_idx + p["index"]

    # Если ZZ совсем пуст — возьмём фрактальные точки
    if len(zz) < 2:
        fr = _find_fractals(high, low, left=2, right=2)
        zz = [p for p in fr if anchor_idx <= p["index"] <= end_idx]

    if not zz:
        return []

    # Якорь форсируем первой точкой нужного типа
    anchor_type = "high" if direction == "нисходящий" else "low"
    anchor_price = float(high[anchor_idx]) if anchor_type == "high" else float(low[anchor_idx])
    anchored = [{"index": anchor_idx, "price": anchor_price, "type": anchor_type}]
    for p in zz:
        if p["index"] > anchor_idx:
            anchored.append(p)

    # Чередование
    alt = _enforce_alternation(anchored)
    # Значимость
    sig = _filter_significance(alt, atr, min_pct, atr_mult)
    sig = _enforce_alternation(sig)
    return sig


# ────────────────────────────────────────────────────────────────
# 7. TARGETS
# ────────────────────────────────────────────────────────────────
def _compute_targets(pivots: list[dict], direction: str) -> dict:
    """Расчёт Фибо-целей по Эллиотту для найденных волн."""
    targets: dict = {}
    if len(pivots) < 2:
        return targets

    # wave 1 = P0→P1
    p0 = pivots[0]["price"]; p1 = pivots[1]["price"]
    len_w1 = abs(p1 - p0)
    sign = 1 if direction == "восходящий" else -1

    if len(pivots) >= 3:
        p2 = pivots[2]["price"]
        # цель вол.3 от p2 = p2 + sign × 1.618 × len_w1
        t3_1618 = p2 + sign * 1.618 * len_w1
        t3_2618 = p2 + sign * 2.618 * len_w1
        targets["III"] = {
            "1.618×I": round(t3_1618, 4),
            "2.618×I": round(t3_2618, 4),
        }

    if len(pivots) >= 5:
        p4 = pivots[4]["price"]
        t5_equal = p4 + sign * len_w1
        t5_0618 = p4 + sign * 0.618 * len_w1
        targets["V"] = {
            "0.618×I": round(t5_0618, 4),
            "1.0×I":   round(t5_equal, 4),
        }

    return targets


def _compute_corr_targets(corr_pivots: list[dict], direction: str) -> dict:
    """Цели коррекции A-B-C: C ≈ 1.0×A или 1.618×A."""
    targets: dict = {}
    if len(corr_pivots) < 3:
        return targets
    a_start = corr_pivots[0]["price"]
    a_end = corr_pivots[1]["price"]
    b_end = corr_pivots[2]["price"] if len(corr_pivots) > 2 else None
    len_a = abs(a_end - a_start)
    if b_end is None:
        return targets
    # Для коррекции направление = против главного тренда
    # sign A: +1 если корр против down-тренда, -1 если против up
    sign = 1 if direction == "нисходящий" else -1
    tC_1  = b_end + sign * len_a
    tC_161 = b_end + sign * 1.618 * len_a
    targets["C"] = {
        "1.0×A":   round(tC_1, 4),
        "1.618×A": round(tC_161, 4),
    }
    return targets


# ────────────────────────────────────────────────────────────────
# MAIN PROCESSOR
# ────────────────────────────────────────────────────────────────
class ElliottProcessor(SectionProcessor):
    section_id = 2
    section_emoji = "🌊"
    section_title = "ВОЛНОВОЙ АНАЛИЗ"
    section_type = "partial"

    def compute(self, df, context: dict) -> dict:
        high = np.asarray(context["high"], dtype=float)
        low = np.asarray(context["low"], dtype=float)
        close = np.asarray(context["close"], dtype=float)
        current_price = float(close[-1])
        atr = float(context.get("atr_last") or 0.0)
        trend = context.get("trend_direction", "нисходящий")
        anchor_idx = int(context.get("trend_start_idx", 0))
        peak_idx = int(context.get("trend_peak_idx", len(high) - 1))
        n = len(high)

        times = df["time"].values if "time" in df.columns else None

        def _time(idx: int) -> str:
            if times is not None and 0 <= idx < len(times):
                return str(times[idx])[:16]
            return ""

        def _pct(price: float) -> float:
            if current_price == 0:
                return 0.0
            return round((price / current_price - 1) * 100, 2)

        # ═══════════════════════════════════════════════════════════
        # 1. Якорь и экстремум (из раздела 1)
        # ═══════════════════════════════════════════════════════════
        if trend == "нисходящий":
            anchor_price = float(high[anchor_idx])
            extremum_idx = peak_idx
            extremum_price = float(low[extremum_idx])
        else:
            anchor_price = float(low[anchor_idx])
            extremum_idx = peak_idx
            extremum_price = float(high[extremum_idx])

        anchor_dict = {
            "price": round(anchor_price, 4),
            "pct": _pct(anchor_price),
            "index": anchor_idx,
            "time": _time(anchor_idx),
        }
        extremum_dict = {
            "price": round(extremum_price, 4),
            "pct": _pct(extremum_price),
            "index": extremum_idx,
            "time": _time(extremum_idx),
        }

        # ═══════════════════════════════════════════════════════════
        # 2. Параметры фильтра значимости
        # ═══════════════════════════════════════════════════════════
        main_min_pct = 3.0
        main_atr_mult = 2.5
        sub_min_pct = 1.0
        sub_atr_mult = 1.0
        # ZZ deviation — ниже порога значимости, чтобы не пропустить кандидатов;
        # значимость отфильтруется _filter_significance.
        main_zz_dev = 2.0
        sub_zz_dev = 0.7

        significance_filter = {
            "main": {"atr_mult": main_atr_mult, "pct_min": main_min_pct, "zz_dev": main_zz_dev},
            "sub":  {"atr_mult": sub_atr_mult,  "pct_min": sub_min_pct,  "zz_dev": sub_zz_dev},
        }

        # ═══════════════════════════════════════════════════════════
        # 3. Фракталы (для диагностики)
        # ═══════════════════════════════════════════════════════════
        raw_fractals = _find_fractals(high, low, left=2, right=2)
        # Только в диапазоне анализа
        fractals_in_range = [
            {**p, "time": _time(p["index"])}
            for p in raw_fractals
            if anchor_idx <= p["index"] <= n - 1
        ]

        # ═══════════════════════════════════════════════════════════
        # 4. ГЛАВНЫЕ ПИВОТЫ: от якоря до конца данных
        # ═══════════════════════════════════════════════════════════
        main_pivots = _build_pivots(
            anchor_idx, n - 1, high, low, atr, trend,
            main_min_pct, main_atr_mult, main_zz_dev,
        )

        # ═══════════════════════════════════════════════════════════
        # 5. Разметка: импульс (I-V) или коррекция (A-B-C)
        # ═══════════════════════════════════════════════════════════
        impulse_waves: list[dict] = []
        correction_waves: list[dict] = []
        prechter_violations: list[str] = []
        is_impulse = False
        targets: dict = {}

        # main_pivots[0] — якорь; P1..P5 это волновые точки
        if len(main_pivots) >= 2:
            # Первые 6 пивотов кандидаты на импульс (P0..P5)
            # Если валидны — 5 волн, иначе коррекция A-B-C (первые 4 пивота)
            is_valid, violations = _validate_impulse(main_pivots, trend)
            prechter_violations = violations
            is_impulse = is_valid

            labels_imp = ["I", "II", "III", "IV", "V"]
            labels_corr = ["A", "B", "C"]

            if is_valid:
                # пивоты P1..P5 получают метки I..V
                wave_candidates = main_pivots[1:6]
                for lbl, piv in zip(labels_imp, wave_candidates):
                    impulse_waves.append({
                        "label": lbl,
                        "price": round(piv["price"], 4),
                        "pct": _pct(piv["price"]),
                        "time": _time(piv["index"]),
                        "index": int(piv["index"]),
                        "type": piv["type"],
                        "violations": [],
                        "fibo_notes": [],
                    })
                targets = _compute_targets(main_pivots[:6], trend)
            else:
                # Пытаемся пометить как импульс с нарушениями, если >= 5 пивотов
                # (главное — не ломать формат для форматтера)
                if len(main_pivots) >= 6:
                    wave_candidates = main_pivots[1:6]
                    for lbl, piv in zip(labels_imp, wave_candidates):
                        impulse_waves.append({
                            "label": lbl,
                            "price": round(piv["price"], 4),
                            "pct": _pct(piv["price"]),
                            "time": _time(piv["index"]),
                            "index": int(piv["index"]),
                            "type": piv["type"],
                            "violations": violations[:] if lbl == "I" else [],
                            "fibo_notes": [],
                        })
                    targets = _compute_targets(main_pivots[:6], trend)
                elif len(main_pivots) >= 4:
                    # 3 волны → A-B-C коррекция
                    wave_candidates = main_pivots[1:4]
                    for lbl, piv in zip(labels_corr, wave_candidates):
                        correction_waves.append({
                            "label": lbl,
                            "price": round(piv["price"], 4),
                            "pct": _pct(piv["price"]),
                            "time": _time(piv["index"]),
                            "index": int(piv["index"]),
                            "type": piv["type"],
                        })
                    targets = _compute_corr_targets(main_pivots[:4], trend)
                else:
                    # Мало пивотов — пометим как первую импульсную волну
                    for i, piv in enumerate(main_pivots[1:], start=1):
                        if i - 1 < len(labels_imp):
                            impulse_waves.append({
                                "label": labels_imp[i - 1],
                                "price": round(piv["price"], 4),
                                "pct": _pct(piv["price"]),
                                "time": _time(piv["index"]),
                                "index": int(piv["index"]),
                                "type": piv["type"],
                                "violations": violations[:] if i == 1 else [],
                                "fibo_notes": [],
                            })

        # ═══════════════════════════════════════════════════════════
        # 6. ПОСЛЕ 5 ВОЛН — ПРОДОЛЖЕНИЕ: коррекция A-B-C
        # ═══════════════════════════════════════════════════════════
        if is_impulse and len(main_pivots) >= 7:
            # После P5 могут идти ABC
            corr_candidates = main_pivots[5:9]  # P5, then A, B, C
            labels_corr = ["A", "B", "C"]
            for lbl, piv in zip(labels_corr, corr_candidates[1:]):
                correction_waves.append({
                    "label": lbl,
                    "price": round(piv["price"], 4),
                    "pct": _pct(piv["price"]),
                    "time": _time(piv["index"]),
                    "index": int(piv["index"]),
                    "type": piv["type"],
                })
            if len(correction_waves) >= 2:
                # Цели C
                ct = _compute_corr_targets(corr_candidates, trend)
                if ct:
                    targets.update(ct)

        # ═══════════════════════════════════════════════════════════
        # 7. ПОДВОЛНЫ внутри каждой волны (i)-(ii)-(iii)-(iv)-(v)
        # ═══════════════════════════════════════════════════════════
        wave_subwaves: dict[str, list[dict]] = {}

        def _build_sub(start_idx: int, end_idx: int,
                       sub_dir: str, labels: list[str]) -> list[dict]:
            """Подволны по тому же алгоритму, но со sub-порогами."""
            sub_pivots = _build_pivots(
                start_idx, end_idx, high, low, atr, sub_dir,
                sub_min_pct, sub_atr_mult, sub_zz_dev,
            )
            # Убираем якорь — это точка предыдущей волны
            if len(sub_pivots) < 2:
                return []
            out = []
            for i, piv in enumerate(sub_pivots[1:]):
                if i >= len(labels):
                    break
                out.append({
                    "label": labels[i],
                    "price": round(piv["price"], 4),
                    "pct": _pct(piv["price"]),
                    "time": _time(piv["index"]),
                    "index": int(piv["index"]),
                    "type": piv["type"],
                })
            return out

        sub_imp_labels = ["(i)", "(ii)", "(iii)", "(iv)", "(v)"]
        sub_corr_labels = ["(a)", "(b)", "(c)"]

        # Для импульсных волн чередуем направление: I, III, V — по тренду;
        # II, IV — против.
        def _sub_dir(main_dir: str, invert: bool) -> str:
            if invert:
                return "восходящий" if main_dir == "нисходящий" else "нисходящий"
            return main_dir

        if impulse_waves:
            prev_idx = anchor_idx
            for i, w in enumerate(impulse_waves):
                w_idx = w.get("index", prev_idx)
                if w_idx <= prev_idx:
                    prev_idx = w_idx
                    continue
                # I,III,V (индексы 0,2,4) — по тренду; II,IV — против
                invert = (i % 2 == 1)
                direction_sub = _sub_dir(trend, invert)
                is_corr_wave = invert
                labels = sub_corr_labels if is_corr_wave else sub_imp_labels
                sub = _build_sub(prev_idx, w_idx, direction_sub, labels)
                if sub:
                    wave_subwaves[w["label"]] = sub
                prev_idx = w_idx

        # Для коррекции A-B-C
        if correction_waves:
            # Старт коррекции — последняя импульсная или экстремум
            if impulse_waves:
                prev_idx = impulse_waves[-1].get("index", extremum_idx)
            else:
                prev_idx = extremum_idx
            for i, w in enumerate(correction_waves):
                w_idx = w.get("index", prev_idx)
                if w_idx <= prev_idx:
                    prev_idx = w_idx
                    continue
                # A,C против main trend; B по main trend
                invert = (i % 2 == 0)  # A, C — против
                direction_sub = _sub_dir(trend, invert)
                # A, C часто импульсные, B — коррекционная
                if i in (0, 2):  # A, C
                    labels = sub_imp_labels
                else:
                    labels = sub_corr_labels
                sub = _build_sub(prev_idx, w_idx, direction_sub, labels)
                if sub:
                    wave_subwaves[w["label"]] = sub
                prev_idx = w_idx

        # ═══════════════════════════════════════════════════════════
        # 8. FIBO NOTES & VIOLATIONS в первую волну (контракт форматтера)
        # ═══════════════════════════════════════════════════════════
        if impulse_waves:
            first = impulse_waves[0]
            first.setdefault("violations", [])
            first.setdefault("fibo_notes", [])
            # Скопируем все нарушения в первую волну (форматтер читает [0])
            first["violations"] = list(prechter_violations)
            # Заметки Фибо
            notes: list[str] = []
            pvs = main_pivots[:6] if len(main_pivots) >= 6 else main_pivots
            prices = [p["price"] for p in pvs]
            if len(prices) >= 6:
                if trend == "восходящий":
                    w1 = prices[1] - prices[0]
                    w3 = prices[3] - prices[2]
                    w5 = prices[5] - prices[4]
                else:
                    w1 = prices[0] - prices[1]
                    w3 = prices[2] - prices[3]
                    w5 = prices[4] - prices[5]
                if w1 > 0:
                    notes.append(f"III = {w3 / w1:.2f}×I (норма: 1.618–2.618)")
                if w1 > 0:
                    notes.append(f"V = {w5 / w1:.2f}×I (норма: 0.618–1.0)")
                if w3 > 0:
                    w2_retrace = abs(prices[2] - prices[1]) / w1 if w1 > 0 else 0
                    w4_retrace = abs(prices[4] - prices[3]) / w3 if w3 > 0 else 0
                    notes.append(f"II ретрейс = {w2_retrace:.1%} I (норма: 38–78%)")
                    notes.append(f"IV ретрейс = {w4_retrace:.1%} III (норма: 23–50%)")
            if is_impulse:
                notes.insert(0, "✅ Импульс валиден по Пректеру")
            else:
                notes.insert(0, "⚠ Импульс нарушен — см. violations")
            first["fibo_notes"] = notes

        # ═══════════════════════════════════════════════════════════
        # 9. RESULT
        # ═══════════════════════════════════════════════════════════
        return {
            "trend": trend,
            "anchor": anchor_dict,
            "extremum": extremum_dict,
            "impulse_waves": impulse_waves,
            "impulse_count": len(impulse_waves),
            "correction_waves": correction_waves,
            "correction_count": len(correction_waves),
            "wave_subwaves": wave_subwaves,
            "subwaves": [],  # legacy поле — оставлено для совместимости
            "subwaves_count": 0,
            "wave_proportions": [],
            "current_price": current_price,
            # Новые поля
            "is_impulse": is_impulse,
            "fractals": fractals_in_range[:50],  # ограничим размер
            "fractals_count": len(fractals_in_range),
            "significance_filter": significance_filter,
            "prechter_violations": prechter_violations,
            "targets": targets,
            "main_pivots_count": len(main_pivots),
        }
