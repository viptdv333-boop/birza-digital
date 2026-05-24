"""
Биржа-цифровой — Раздел 1: ТРЕНДЫ (ЯДРО v8).

Алгоритм цифрового определения начала тренда (по csv_periods v8):
  1) Определяем окно истории по ТФ (csv_periods):
        15M: 5–7 торговых дней (≈672 баров 15м)
        1H : 50–60 торговых дней (≈1440 баров)
        4H : 15–25 торговых дней (≈150 баров)
        1D : 2–3 месяца (≈90 баров)
        1W : 6–12 месяцев (≈52 бара)
  2) Внутри этого окна строим ZigZag/фрактальные swing-точки (шум чистим).
  3) Структурный режим: HH/HL = восходящий, LH/LL = нисходящий.
     Старт — последний ключевой разворотный экстремум ПЕРЕД последовательностью.
  4) Подтверждение: минимум 2 swing-звена после старта.
  5) LinReg/ALMA/объём/CVD — только ВАЛИДАЦИЯ, не выбор.

Выводим ДВЕ точки (строгий вариант из регламента):
  senior_trend — старший тренд от главного экстремума окна
  local_trend  — локальное движение от последнего подтверждённого pivot

Fallback: если структурный HH/LL не найден, берём argmax/argmin в окне.

Сохраняемые в context ключи (backward compat):
  trend_direction          — направление senior
  current_direction        — направление local
  trend_start_idx          — senior_start
  trend_peak_idx           — конечная точка импульса
  senior_trend_start_idx   — алиас trend_start_idx
  local_trend_start_idx    — последний чередующийся пивот
"""
import numpy as np
from sections.base import SectionProcessor
from core.linreg import linear_regression_channel
from core.zigzag import zigzag
from core.utils import calc_alma


class TrendsProcessor(SectionProcessor):
    section_id = 1
    section_emoji = "📈"
    section_title = "ТРЕНДЫ"
    section_type = "partial"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        open_ = context["open_"]
        atr_last = context["atr_last"]
        swing_points = context.get("swing_points", [])
        available = context.get("available_cols", {})
        n = len(close)
        current_price = float(close[-1])
        tf_hours = context.get("tf_hours", 4.0)
        times = df["time"].values

        # ================================================================
        #  1. ТРЕНД ПО ДОУ (структурный, в окне csv_periods v8)
        # ================================================================
        # Окно анализа по ТФ (csv_periods): тренд ищется ВНУТРИ этого окна.
        window_bars = _tf_window_bars(tf_hours, n)
        window_offset = max(0, n - window_bars)  # абсолютный индекс начала окна

        # Адаптивный порог ZZ для поиска структурных пивотов.
        # Старшие ТФ должны видеть КРУПНЫЕ развороты, а не мелкие качели:
        #   15m: 3% — внутридневные колебания
        #   1H:  5% — дневные/недельные свинги
        #   4H:  8% — недельные/месячные структуры
        #   1D: 15% — месячные/квартальные развороты
        #   1W: 25% — только мега-тренды
        if tf_hours <= 0.25:    # 15m
            zz_threshold = 3.0
        elif tf_hours <= 1:     # 1H
            zz_threshold = 5.0
        elif tf_hours <= 4:     # 4H
            zz_threshold = 8.0
        elif tf_hours <= 24:    # 1D
            zz_threshold = 15.0
        else:                   # 1W+
            zz_threshold = 25.0

        # ZigZag внутри окна — пивоты имеют абсолютные индексы
        zz_structural_full = zigzag(high, low, deviation_pct=zz_threshold, times=times)
        # Фильтруем пивоты по окну: start точка должна быть в диапазоне
        zz_structural = [p for p in zz_structural_full
                         if int(p.get("index", 0)) >= window_offset]

        senior_direction, senior_start_price, senior_start_idx, senior_start_time, \
        senior_end_price, senior_end_idx, senior_end_time, trend_source = \
            _dow_structural_trend(
                zz_structural, high, low, times, current_price, df, n,
                window_offset=window_offset,
                tf_hours=tf_hours,
            )

        # ZZ 20% для канала — в окне csv_periods
        zz_20 = zigzag(high[window_offset:], low[window_offset:],
                       deviation_pct=20.0, times=times[window_offset:])

        # Senior stage via LinReg от anchor до текущего
        senior_segment_len = n - senior_start_idx
        senior_lr_period = max(50, min(senior_segment_len, n))
        senior_linreg = linear_regression_channel(close, period=senior_lr_period)
        senior_stage = _compute_stage(
            senior_linreg, senior_direction,
            end_price=senior_end_price, current_price=current_price,
        )

        # Senior channel из ZZ 20% пивотов
        senior_channel = _build_channel(zz_20, current_price)

        senior_trend = {
            "direction": senior_direction,
            "start": {
                "price": senior_start_price,
                "time": senior_start_time,
                "pct_from_current": _pct(senior_start_price, current_price),
            },
            "end": {
                "price": senior_end_price,
                "time": senior_end_time,
                "pct_from_current": _pct(senior_end_price, current_price),
            },
            "stage": senior_stage,
            "channel": senior_channel,
            "zz_pivots_count": len(zz_20),
            "source": trend_source,  # "dow_structural" | "window_fallback"
            "zz_threshold_pct": zz_threshold,
            "structural_pivots": [
                {"time": str(p.get("time", "")), "type": p["type"],
                 "price": round(float(p["price"]), 4)}
                for p in zz_structural[-10:]
            ],
            "analysis_window": {
                "bars": int(n - window_offset),
                "start_time": str(df["time"].iloc[window_offset]),
                "end_time": str(df["time"].iloc[-1]),
                "tf_hours": float(tf_hours),
            },
        }

        # ================================================================
        #  2. LOCAL TREND — от предпоследнего значимого пивота ZZ 10-15%
        # ================================================================
        # Используем ZZ 10-15% (не 5%) чтобы получить значимые точки.
        # Предпоследний пивот = начало текущего движения.
        zz_5 = context.get("zigzag_5pct", [])

        local_start_idx = senior_start_idx
        local_start_price = senior_start_price

        # ZigZag Period 8 после senior_start — как на TradingView
        from core.zigzag import zigzag_period
        LOCAL_ZZ_PERIOD = 8
        h_local = high[senior_start_idx:]
        l_local = low[senior_start_idx:]
        t_local = times[senior_start_idx:]
        local_pivots = zigzag_period(h_local, l_local, period=LOCAL_ZZ_PERIOD, times=t_local)

        if len(local_pivots) >= 3:
            # Предпоследний подтверждённый пивот = начало текущего движения
            pivot = local_pivots[-2]
            local_start_idx = senior_start_idx + pivot["index"]
            local_start_price = pivot["price"]
        elif len(local_pivots) >= 2:
            pivot = local_pivots[-2]
            local_start_idx = senior_start_idx + pivot["index"]
            local_start_price = pivot["price"]
        elif len(local_pivots) == 1:
            pivot = local_pivots[0]
            local_start_idx = senior_start_idx + pivot["index"]
            local_start_price = pivot["price"]

        # Используем local_pivots для определения HH/HL структуры
        pivots_after = local_pivots

        local_start_time = str(df["time"].iloc[local_start_idx])

        # Направление: по структуре swing-точек после senior_start
        # HH+HL = восходящий, LH+LL = нисходящий
        if len(pivots_after) >= 4:
            recent_hi = [p["price"] for p in pivots_after if p["type"] in ("high", "H")][-3:]
            recent_lo = [p["price"] for p in pivots_after if p["type"] in ("low", "L")][-3:]
            hi_rising = len(recent_hi) >= 2 and recent_hi[-1] > recent_hi[-2]
            lo_rising = len(recent_lo) >= 2 and recent_lo[-1] > recent_lo[-2]
            hi_falling = len(recent_hi) >= 2 and recent_hi[-1] < recent_hi[-2]
            lo_falling = len(recent_lo) >= 2 and recent_lo[-1] < recent_lo[-2]

            if hi_rising and lo_rising:
                local_direction = "восходящий"
            elif hi_falling and lo_falling:
                local_direction = "нисходящий"
            elif hi_falling or lo_falling:
                local_direction = "нисходящий"
            elif hi_rising or lo_rising:
                local_direction = "восходящий"
            else:
                # Боковика нет — форсируем по направлению senior
                local_direction = senior_direction
        elif len(pivots_after) >= 2:
            # Мало пивотов — сравниваем первый и последний
            first_p = pivots_after[0]["price"]
            last_p = pivots_after[-1]["price"]
            if last_p < first_p:
                local_direction = "нисходящий"
            elif last_p > first_p:
                local_direction = "восходящий"
            else:
                local_direction = senior_direction
        else:
            # Один пивот или нет — по senior
            local_direction = senior_direction

        local_end_idx = n - 1
        local_end_price = current_price
        local_end_time = str(df["time"].iloc[-1])

        # Local stage
        local_segment_len = n - local_start_idx
        local_lr_period = max(20, min(local_segment_len, 100))
        local_linreg = linear_regression_channel(close, period=local_lr_period)
        local_stage = _compute_stage(local_linreg, local_direction)

        # Local channel
        local_channel = _build_channel(zz_5, current_price)

        local_trend = {
            "direction": local_direction,
            "start": {
                "price": local_start_price,
                "time": local_start_time,
                "pct_from_current": _pct(local_start_price, current_price),
            },
            "end": {
                "price": local_end_price,
                "time": local_end_time,
                "pct_from_current": _pct(local_end_price, current_price),
            },
            "stage": local_stage,
            "channel": local_channel,
        }

        # ================================================================
        #  3. CONTEXT KEYS (backward compat + new)
        # ================================================================
        # Backward compat: trend_start_idx / trend_peak_idx — senior trend
        context["trend_start_idx"] = senior_start_idx
        # peak_idx для S02: конец импульса = мин/макс от senior_start до текущей
        if senior_direction == "нисходящий":
            # Импульс вниз: peak = минимум после senior_start
            _seg = low[senior_start_idx:]
            context["trend_peak_idx"] = senior_start_idx + int(np.argmin(_seg))
        elif senior_direction == "восходящий":
            # Импульс вверх: peak = максимум после senior_start
            _seg = high[senior_start_idx:]
            context["trend_peak_idx"] = senior_start_idx + int(np.argmax(_seg))
        else:
            context["trend_peak_idx"] = n - 1
        context["trend_direction"] = senior_direction
        context["current_direction"] = local_direction

        # New v7 explicit keys
        context["senior_trend_start_idx"] = senior_start_idx
        context["local_trend_start_idx"] = local_start_idx

        # ================================================================
        #  4. STATE (описание текущего состояния)
        # ================================================================
        state = _compute_state(
            senior_direction, local_direction,
            senior_end_price, local_start_price,
            current_price, swing_points,
        )

        # ================================================================
        #  5. LINREG (сводная, как раньше)
        # ================================================================
        linreg_short = linear_regression_channel(close, period=50)
        linreg = linear_regression_channel(close, period=100)

        r2_short = linreg_short["r_squared"]
        r2 = linreg["r_squared"]
        output_linreg = linreg_short if r2_short > r2 else linreg

        # Structure break
        structure_break = _detect_structure_break(
            swing_points, local_direction, current_price,
        )

        # Stage (итоговая, берём local как более практичную)
        stage = local_stage

        # ================================================================
        #  6. ALMA 200
        # ================================================================
        alma_200_val = None
        alma_200_position = None
        if "alma" in available and "ALMA 200" in available["alma"]:
            a200 = df["ALMA 200"].values
            valid = a200[~np.isnan(a200)]
            if len(valid) > 0:
                alma_200_val = float(valid[-1])
        elif n >= 200:
            a200 = calc_alma(close, period=200)
            alma_200_val = float(a200[-1])

        if alma_200_val is not None:
            alma_200_position = "выше" if current_price > alma_200_val else "ниже"

        # ================================================================
        #  7. ALMA stack (20/50)
        # ================================================================
        alma_stack = {}
        for name, period in [("ALMA 20", 20), ("ALMA 50", 50)]:
            if "alma" in available and name in available["alma"]:
                vals = df[name].values
                valid = vals[~np.isnan(vals)]
                if len(valid) > 0:
                    alma_stack[name] = float(valid[-1])
            elif n >= period:
                computed = calc_alma(close, period=period)
                alma_stack[name] = float(computed[-1])

        alma_order = "нейтральный"
        a20 = alma_stack.get("ALMA 20", 0)
        a50 = alma_stack.get("ALMA 50", 0)
        if alma_200_val and a20 and a50:
            if a20 > a50 > alma_200_val:
                alma_order = "бычий (20>50>200)"
            elif a20 < a50 < alma_200_val:
                alma_order = "медвежий (20<50<200)"
            elif a20 > a50:
                alma_order = "смешанный (20>50)"
            else:
                alma_order = "смешанный (20<50)"

        # ================================================================
        #  8. Swing-points (recent, для вывода)
        # ================================================================
        recent_out = list(swing_points[-6:])

        # ================================================================
        #  9. ВИЛЫ ЭНДРЮСА (Andrews' Pitchfork) — v8
        #     A = первый значимый пивот, B и C — два следующих противоположных
        #     Медиана: из A через середину BC
        #     Верхняя/нижняя параллели: B и C параллельно медиане
        # ================================================================
        pitchfork = _build_andrews_pitchfork(
            local_pivots, senior_start_idx, close, high, low, times,
            current_price, atr_last,
        )

        # ================================================================
        #  RETURN
        # ================================================================
        return {
            # ── v7: два тренда ──
            "senior_trend": senior_trend,
            "local_trend": local_trend,
            # ── v8: Вилы Эндрюса ──
            "pitchfork": pitchfork,
            # ── backward compat (top-level) ──
            "direction": senior_direction,
            "current_direction": local_direction,
            "state": state,
            "stage": stage,
            "trend_start": {
                "price": senior_start_price,
                "time": senior_start_time,
                "pct_from_current": _pct(senior_start_price, current_price),
            },
            "trend_peak": {
                "price": senior_end_price,
                "time": senior_end_time,
                "pct_from_current": _pct(senior_end_price, current_price),
            },
            "trend_range_pct": _pct(senior_start_price, senior_end_price),
            # ── LinReg (сводная) ──
            "linreg": {
                "slope_pct_per_bar": round(output_linreg["slope_pct"], 6),
                "position_pct": round(output_linreg["position_pct"], 1),
                "r_squared": round(output_linreg["r_squared"], 3),
                "period": output_linreg["period"],
                "channel_upper": round(float(linreg["upper"][-1]), 4),
                "channel_lower": round(float(linreg["lower"][-1]), 4),
                "channel_center": round(float(linreg["center"][-1]), 4),
            },
            "structure_break": structure_break,
            # ── ALMA ──
            "alma_200_value": alma_200_val,
            "alma_200_position": alma_200_position,
            "alma_stack": alma_stack,
            "alma_order": alma_order,
            # ── Swing ──
            "swing_points": recent_out,
            "current_price": current_price,
            "atr_last": atr_last,
        }


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────

def _pct(base: float, target: float) -> float:
    """Процент изменения от base к target."""
    if base and base != 0:
        return round((target - base) / base * 100, 2)
    return 0.0


def _fmt_price(p, current):
    pct = (p - current) / current * 100
    p_str = f"{p:.2f}" if p >= 1000 else f"{p:.4g}"
    return f"{p_str} ({pct:+.2f}%)"


def _compute_stage(linreg: dict, direction: str,
                   end_price: float = None,
                   current_price: float = None) -> str:
    """Стадия тренда по v8.1: Начало / Развитие / Затухание / Коррекция.

    Коррекция — приоритетна: если цена отошла от end_price ПРОТИВ направления
    тренда (отскок от low в даунтренде / откат от high в аптренде) — это
    локальная коррекция, а не продолжение основного импульса.
    """
    # ── Коррекция (отскок против тренда от end_price) ──
    if end_price is not None and current_price is not None and end_price > 0:
        delta_pct = (current_price - end_price) / end_price * 100
        # Отход на ≥0.3% против тренда → коррекция
        if direction == "нисходящий" and delta_pct >= 0.3:
            return "Коррекция"
        if direction == "восходящий" and delta_pct <= -0.3:
            return "Коррекция"

    slope = linreg["slope_pct"]
    r2 = linreg["r_squared"]
    position = linreg["position_pct"]

    if r2 > 0.4 and abs(slope) > 0.05:
        return "Развитие"
    if abs(slope) > 0.02 and r2 > 0.2:
        return "Развитие"
    if abs(slope) < 0.01 and r2 < 0.15:
        return "Затухание"
    if abs(slope) > 0.02:
        if direction == "восходящий" and position < 30:
            return "Начало"
        if direction == "нисходящий" and position > 70:
            return "Начало"
        return "Развитие"
    return "Затухание"


def _detect_local_direction(swing_points: list, close: np.ndarray) -> str:
    """Направление локального тренда по последним swing-точкам.

    HH + HL = восходящий, LH + LL = нисходящий.
    ЯДРО v8: бокового направления нет — если структура смешанная,
    форсируем выбор по сравнению first/last.
    """
    current_price = float(close[-1])
    if not swing_points:
        # Нет пивотов — по первой/текущей цене
        first = float(close[0]) if len(close) > 0 else current_price
        return "восходящий" if current_price >= first else "нисходящий"

    recent = swing_points[-6:] if len(swing_points) >= 6 else swing_points

    # Подсчёт классификаций
    hh = sum(1 for p in recent if p.get("swing") == "HH")
    hl = sum(1 for p in recent if p.get("swing") == "HL")
    lh = sum(1 for p in recent if p.get("swing") == "LH")
    ll = sum(1 for p in recent if p.get("swing") == "LL")
    bull = hh + hl
    bear = lh + ll

    # Простая проверка по последним хаям/лоям
    highs = [p["price"] for p in recent if p["type"] == "high"]
    lows = [p["price"] for p in recent if p["type"] == "low"]

    direction = None
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[0] and lows[-1] > lows[0]:
            direction = "восходящий"
        elif highs[-1] < highs[0] and lows[-1] < lows[0]:
            direction = "нисходящий"
        elif bear > bull:
            direction = "нисходящий"
        elif bull > bear:
            direction = "восходящий"

    if direction is None:
        # Форсируем по первому vs последнему пивоту
        first_p = recent[0]["price"]
        last_p = recent[-1]["price"]
        direction = "восходящий" if last_p >= first_p else "нисходящий"

    # Пробой структуры
    if direction == "восходящий" and lows:
        if current_price < lows[-1] * 0.995:
            direction = "нисходящий"
    elif direction == "нисходящий" and highs:
        if current_price > highs[-1] * 1.005:
            direction = "восходящий"

    return direction


def _find_local_anchor(swing_points: list, local_direction: str,
                       current_price: float) -> dict | None:
    """Найти якорь локального тренда — начало текущего движения.

    Для восходящего: последний swing low ниже текущей цены.
    Для нисходящего: последний swing high выше текущей цены.
    """
    if not swing_points:
        return None

    if local_direction == "восходящий":
        # Ищем последний значимый лоу (начало роста)
        lows = [p for p in swing_points if p["type"] == "low"]
        if lows:
            # Берём последний лоу, который ниже текущей цены
            for sp in reversed(lows):
                if sp["price"] < current_price:
                    return sp
            return lows[-1]
    elif local_direction == "нисходящий":
        # Ищем последний значимый хай (начало падения)
        highs = [p for p in swing_points if p["type"] == "high"]
        if highs:
            for sp in reversed(highs):
                if sp["price"] > current_price:
                    return sp
            return highs[-1]
    else:
        # Боковик: последний swing
        return swing_points[-1] if swing_points else None

    return None


def _build_channel(zz_points: list, current_price: float) -> dict:
    """Канал: resistance по swing highs, support по swing lows.

    Позиция = где текущая цена внутри канала (0% = support, 100% = resistance).
    """
    if len(zz_points) < 2:
        return {"resistance": [], "support": [], "position_pct": 50.0}

    swing_highs = [p for p in zz_points if p["type"] == "high"]
    swing_lows = [p for p in zz_points if p["type"] == "low"]

    # Берём последние 2 точки каждого типа для линии
    res_points = swing_highs[-2:] if len(swing_highs) >= 2 else swing_highs
    sup_points = swing_lows[-2:] if len(swing_lows) >= 2 else swing_lows

    # Resistance line value (экстраполяция к текущему бару не нужна —
    # просто берём ценовые уровни как ориентиры)
    res_prices = [p["price"] for p in res_points]
    sup_prices = [p["price"] for p in sup_points]

    # Позиция в канале
    if res_prices and sup_prices:
        top = max(res_prices)
        bottom = min(sup_prices)
        width = top - bottom
        if width > 0:
            pos = (current_price - bottom) / width * 100
            pos = round(float(np.clip(pos, 0, 100)), 1)
        else:
            pos = 50.0
    else:
        pos = 50.0

    return {
        "resistance": [
            {"price": p["price"], "time": str(p.get("time", "")), "index": p["index"]}
            for p in res_points
        ],
        "support": [
            {"price": p["price"], "time": str(p.get("time", "")), "index": p["index"]}
            for p in sup_points
        ],
        "position_pct": pos,
    }


def _detect_structure_break(swing_points: list, direction: str,
                            current_price: float) -> bool:
    """Пробой структуры — цена пробила предыдущий swing high/low."""
    if len(swing_points) < 3:
        return False

    recent_hi = [p["price"] for p in swing_points[-6:] if p["type"] == "high"]
    recent_lo = [p["price"] for p in swing_points[-6:] if p["type"] == "low"]

    if direction == "восходящий" and len(recent_hi) >= 2:
        if current_price > max(recent_hi[:-1]):
            return True
    elif direction == "нисходящий" and len(recent_lo) >= 2:
        if current_price < min(recent_lo[:-1]):
            return True
    return False


def _compute_state(senior_dir: str, local_dir: str,
                   senior_end_price: float, local_start_price: float,
                   current_price: float, swing_points: list) -> str:
    """Текстовое описание текущего состояния."""
    if not swing_points:
        return "неопределено"

    last_sp = swing_points[-1]
    last_price = last_sp["price"]
    last_type = last_sp["type"]

    # Совпадение направлений = тренд в силе
    if senior_dir == local_dir:
        if local_dir == "нисходящий":
            return f"Нисходящий тренд, импульс от {_fmt_price(last_price, current_price)}"
        elif local_dir == "восходящий":
            return f"Восходящий тренд, импульс от {_fmt_price(last_price, current_price)}"

    # Расхождение = коррекция внутри старшего тренда
    if senior_dir == "нисходящий" and local_dir == "восходящий":
        return f"Коррекция вверх от {_fmt_price(local_start_price, current_price)}"
    if senior_dir == "восходящий" and local_dir == "нисходящий":
        return f"Коррекция вниз от {_fmt_price(local_start_price, current_price)}"

    # Fallback
    if last_type == "low" and current_price > last_price:
        return f"Рост от {_fmt_price(last_price, current_price)}"
    if last_type == "high" and current_price < last_price:
        return f"Снижение от {_fmt_price(last_price, current_price)}"

    return f"Тест {_fmt_price(last_price, current_price)}"


def _tf_window_bars(tf_hours: float, n: int) -> int:
    """
    Глубина окна анализа по ТФ.

    Возвращает количество баров окна, кэппированное длиной данных.
      15M: 7 торговых дней = 672 баров
      1H:  60 торговых дней = 1440 баров
      4H:  80 торговых дней = 480 баров
      1D:  ВСЕ данные (крупные свинги нужны для senior тренда)
      1W:  ВСЕ данные (мега-тренды)

    Старшие ТФ (1D/1W) используют ВЕСЬ массив данных, чтобы видеть
    структурные тренды (рост от 800 до 2200 за несколько лет),
    а не только последние 3 месяца.
    """
    if tf_hours <= 0.25:
        bars = 672        # 15M: 7 торговых дней
    elif tf_hours <= 1:
        bars = 1440       # 1H:  60 торговых дней
    elif tf_hours <= 4:
        bars = 480        # 4H:  80 торговых дней
    elif tf_hours <= 24:
        bars = n          # 1D:  ВСЕ данные
    else:
        bars = n          # 1W:  ВСЕ данные
    return min(bars, max(20, n))


def _find_sequence_start(pivots: list, rising: bool):
    """Найти начало непрерывной серии HL (rising=True) или LH (rising=False).

    Идёт от конца назад: пока каждый предыдущий пивот ниже (rising)
    или выше (!rising) следующего — серия продолжается.
    Возвращает самый ранний пивот серии.
    """
    if len(pivots) < 2:
        return pivots[0] if pivots else None
    earliest = len(pivots) - 2
    for i in range(len(pivots) - 2, 0, -1):
        if rising:
            if pivots[i - 1]["price"] < pivots[i]["price"]:
                earliest = i - 1
            else:
                break
        else:
            if pivots[i - 1]["price"] > pivots[i]["price"]:
                earliest = i - 1
            else:
                break
    return pivots[earliest]


def _dow_structural_trend(zz_pivots: list, high: np.ndarray, low: np.ndarray,
                          times, current_price: float, df, n: int,
                          window_offset: int = 0, tf_hours: float = 1.0):
    """
    Структурный тренд по Доу — двухуровневый алгоритм:

    1D/1W (старшие ТФ):
      Цена >20% выше глобального ZZ-минимума → восходящий от того дна.
      Это защищает от ложных LL внутри коррекций мега-тренда.

    4H/1H/15m (младшие ТФ):
      Последние 2 ZZ-лоу: HL → восходящий, LL → нисходящий.
      Начало = самый ранний пивот непрерывной HL/LH-серии.

    Разные ТФ дают разные результаты, потому что ZZ-пороги
    масштабируются (1H:5%, 4H:8%, 1D:15%), и структура пивотов различается.

    Возвращает: (direction, start_price, start_idx, start_time,
                 end_price, end_idx, end_time, source)
    """
    if len(zz_pivots) >= 2:
        # ── Структурный анализ HH/HL/LH/LL ──
        zz_highs = [p for p in zz_pivots if p.get("type") in ("high", "H")]
        zz_lows = [p for p in zz_pivots if p.get("type") in ("low", "L")]

        direction = None
        start_pivot = None
        end_pivot = None
        source_tag = None

        # ── 1D/1W: якорь по глобальному минимуму ──
        # На старших ТФ цена может делать LL в коррекции, но если она
        # значительно выше абсолютного дна — мега-тренд восходящий.
        if tf_hours >= 24 and len(zz_lows) >= 3:
            min_low = min(zz_lows, key=lambda p: p["price"])
            min_pct = (current_price - min_low["price"]) / max(min_low["price"], 0.01) * 100
            if min_pct > 20:
                # Цена >20% выше дна → восходящий от глобального минимума
                direction = "восходящий"
                start_pivot = min_low
                end_pivot = zz_highs[-1] if zz_highs else zz_pivots[-1]
                source_tag = "global_low_anchor"
            else:
                # Цена у дна — проверяем: может быть нисходящий от глобального max
                max_high = max(zz_highs, key=lambda p: p["price"]) if zz_highs else None
                if max_high:
                    max_pct = (max_high["price"] - current_price) / max(current_price, 0.01) * 100
                    if max_pct > 20:
                        direction = "нисходящий"
                        start_pivot = max_high
                        end_pivot = zz_lows[-1]
                        source_tag = "global_high_anchor"

        # ── 4H и ниже: последние 2 лоу HL/LL ──
        if direction is None and len(zz_lows) >= 2:
            if zz_lows[-1]["price"] > zz_lows[-2]["price"]:
                # Higher Low → восходящий тренд
                direction = "восходящий"
                start_pivot = _find_sequence_start(zz_lows, rising=True)
                end_pivot = zz_highs[-1] if zz_highs else zz_lows[-1]
                source_tag = "structural_HL"
            elif zz_lows[-1]["price"] < zz_lows[-2]["price"]:
                # Lower Low → нисходящий тренд
                direction = "нисходящий"
                if len(zz_highs) >= 2:
                    start_pivot = _find_sequence_start(zz_highs, rising=False)
                else:
                    start_pivot = zz_highs[0] if zz_highs else zz_pivots[0]
                end_pivot = zz_lows[-1]
                source_tag = "structural_LL"

        # Если лоу равны или <2 лоу — проверяем хаи
        if direction is None and len(zz_highs) >= 2:
            if zz_highs[-1]["price"] > zz_highs[-2]["price"]:
                direction = "восходящий"
                start_pivot = zz_lows[-1] if zz_lows else zz_pivots[0]
                end_pivot = zz_highs[-1]
                source_tag = "structural_HH"
            elif zz_highs[-1]["price"] < zz_highs[-2]["price"]:
                direction = "нисходящий"
                start_pivot = zz_highs[-2]
                end_pivot = zz_lows[-1] if zz_lows else zz_pivots[-1]
                source_tag = "structural_LH"

        # Fallback: направление последнего swing-leg
        if direction is None:
            prev_p = zz_pivots[-2]
            last_p = zz_pivots[-1]
            direction = "восходящий" if last_p["price"] > prev_p["price"] else "нисходящий"
            start_pivot = prev_p
            end_pivot = last_p
            source_tag = "last_swing_leg"

        # ── Sanity: rally/drop override ──
        # Цена может уйти от последнего ZZ-пивота дальше ZZ-порога,
        # фактически образуя новый swing, который ZZ ещё не зафиксировал.
        # Пример: structural_LL → "нисходящий", но цена отскочила от дна
        # на 6% при ZZ-пороге 5% → нисходящий тренд сломан.
        _zz_dev = (3.0 if tf_hours <= 0.25 else
                   5.0 if tf_hours <= 1 else
                   8.0 if tf_hours <= 4 else
                   15.0 if tf_hours <= 24 else 25.0)

        # Порог override = 75% от ZZ-deviation: цена прошла ¾ полного
        # ZZ-свинга от последнего пивота → структура уже сломана.
        _override_thr = _zz_dev * 0.75

        if (direction == "нисходящий"
                and source_tag in ("structural_LL", "structural_LH")
                and zz_lows):
            last_low_price = zz_lows[-1]["price"]
            rally_pct = (current_price - last_low_price) / max(last_low_price, 0.01) * 100
            if rally_pct > _override_thr:
                direction = "восходящий"
                start_pivot = zz_lows[-1]
                end_pivot = zz_highs[-1] if zz_highs else zz_pivots[-1]
                source_tag = "rally_override"

        elif (direction == "восходящий"
                and source_tag in ("structural_HL", "structural_HH")
                and zz_highs):
            last_high_price = zz_highs[-1]["price"]
            drop_pct = (last_high_price - current_price) / max(last_high_price, 0.01) * 100
            if drop_pct > _override_thr:
                direction = "нисходящий"
                start_pivot = zz_highs[-1]
                end_pivot = zz_lows[-1] if zz_lows else zz_pivots[-1]
                source_tag = "drop_override"

        start_idx = max(0, min(int(start_pivot["index"]), n - 1))
        end_idx = max(0, min(int(end_pivot["index"]), n - 1))
        start_price = float(start_pivot["price"])
        end_price = float(end_pivot["price"])
        start_time = str(df["time"].iloc[start_idx])
        end_time = str(df["time"].iloc[end_idx])
        return (direction, start_price, start_idx, start_time,
                end_price, end_idx, end_time, source_tag)

    if len(zz_pivots) == 1:
        # Только один пивот — берём его как start, end = последний экстремум
        last_pivot = zz_pivots[0]
        start_idx = max(0, min(int(last_pivot["index"]), n - 1))
        start_price = float(last_pivot["price"])
        last_type = last_pivot.get("type")
        if last_type in ("high", "H"):
            direction = "нисходящий"
            seg = low[start_idx:]
            end_idx = start_idx + (int(np.argmin(seg)) if len(seg) else 0)
            end_price = float(low[end_idx])
        else:
            direction = "восходящий"
            seg = high[start_idx:]
            end_idx = start_idx + (int(np.argmax(seg)) if len(seg) else 0)
            end_price = float(high[end_idx])
        end_idx = max(0, min(end_idx, n - 1))
        start_time = str(df["time"].iloc[start_idx])
        end_time = str(df["time"].iloc[end_idx])
        return (direction, start_price, start_idx, start_time,
                end_price, end_idx, end_time, "single_pivot")

    # Fallback: пивотов нет — берём экстремумы окна
    offset = max(0, int(window_offset))
    if offset >= n:
        offset = max(0, n - min(n, 500))
    hi_rel = int(np.argmax(high[offset:]))
    lo_rel = int(np.argmin(low[offset:]))
    hi_idx, lo_idx = offset + hi_rel, offset + lo_rel
    hi_price, lo_price = float(high[hi_idx]), float(low[lo_idx])
    # Самая поздняя точка определяет направление импульса
    if hi_idx >= lo_idx:
        direction = "нисходящий"
        start_idx, start_price = hi_idx, hi_price
        end_idx, end_price = lo_idx, lo_price
    else:
        direction = "восходящий"
        start_idx, start_price = lo_idx, lo_price
        end_idx, end_price = hi_idx, hi_price
    start_idx = max(0, min(int(start_idx), n - 1))
    end_idx = max(0, min(int(end_idx), n - 1))
    start_time = str(df["time"].iloc[start_idx])
    end_time = str(df["time"].iloc[end_idx])
    return (direction, start_price, start_idx, start_time,
            end_price, end_idx, end_time, "window_fallback")


def _build_andrews_pitchfork(local_pivots: list, senior_start_idx: int,
                             close: np.ndarray, high: np.ndarray, low: np.ndarray,
                             times, current_price: float, atr_last: float) -> dict:
    """Вилы Эндрюса по ЯДРУ v8.

    A — первый значимый пивот, B — следующий противоположный, C — третий.
    Медиана: из A через среднюю точку BC.
    Верхняя/нижняя параллели: из B и C параллельно медиане.
    Возвращает: точки, угол медианы, положение цены, количество касаний.
    """
    empty = {
        "available": False,
        "reason": "мало пивотов (нужно ≥3 после якоря)",
    }
    if not local_pivots or len(local_pivots) < 3:
        return empty

    # Берём 3 последние значимые точки из local_pivots (уже строятся ZZ period 8)
    # Предпочтительно: последние 3 чередующихся по типу high/low
    # local_pivots индексированы от senior_start_idx
    n = len(close)
    pts = list(local_pivots[-3:])
    # abs_index = senior_start_idx + pivot['index']
    A = {
        "index": senior_start_idx + pts[0]["index"],
        "price": float(pts[0]["price"]),
        "type": pts[0]["type"],
    }
    B = {
        "index": senior_start_idx + pts[1]["index"],
        "price": float(pts[1]["price"]),
        "type": pts[1]["type"],
    }
    C = {
        "index": senior_start_idx + pts[2]["index"],
        "price": float(pts[2]["price"]),
        "type": pts[2]["type"],
    }

    for pt in (A, B, C):
        pt["time"] = str(times[pt["index"]]) if 0 <= pt["index"] < n else ""

    # Midpoint BC
    mid_idx = (B["index"] + C["index"]) / 2.0
    mid_price = (B["price"] + C["price"]) / 2.0

    # Наклон медианы (цена/бар)
    dx = mid_idx - A["index"]
    if dx <= 0:
        return {**empty, "reason": "нулевой шаг медианы"}
    slope = (mid_price - A["price"]) / dx

    # Угол в градусах (нормируем по ATR — чтоб было осмысленно)
    # angle = atan(slope_pct / bar_width_pct). Приближённо:
    if current_price > 0 and atr_last > 0:
        slope_pct_per_bar = slope / current_price * 100.0
        angle_deg = float(np.degrees(np.arctan(slope_pct_per_bar / (atr_last / current_price * 100.0))))
    else:
        angle_deg = 0.0

    # Значения медианы и параллелей на последнем баре
    last_idx = n - 1
    median_at_now = A["price"] + slope * (last_idx - A["index"])
    # Параллели — смещение равно смещению B (или C) от медианы на их x-позиции
    offset_B = B["price"] - (A["price"] + slope * (B["index"] - A["index"]))
    offset_C = C["price"] - (A["price"] + slope * (C["index"] - A["index"]))
    upper_offset = max(offset_B, offset_C)
    lower_offset = min(offset_B, offset_C)
    upper_at_now = median_at_now + upper_offset
    lower_at_now = median_at_now + lower_offset

    # Положение текущей цены
    if current_price > upper_at_now:
        position = "выше верхней параллели"
    elif current_price < lower_at_now:
        position = "ниже нижней параллели"
    elif current_price > median_at_now:
        position = "между медианой и верхней параллелью"
    elif current_price < median_at_now:
        position = "между медианой и нижней параллелью"
    else:
        position = "на медиане"

    # Касания: сколько раз цена приходила в зону ±0.2×ATR от медианы или параллелей
    touch_tol = 0.2 * atr_last if atr_last > 0 else 0.0
    touches_median = 0
    touches_upper = 0
    touches_lower = 0
    start_scan = max(A["index"], n - 200)
    for i in range(start_scan, n):
        m = A["price"] + slope * (i - A["index"])
        u = m + upper_offset
        l = m + lower_offset
        bar_hi = float(high[i])
        bar_lo = float(low[i])
        if bar_lo <= m <= bar_hi or abs(float(close[i]) - m) <= touch_tol:
            touches_median += 1
        if bar_lo <= u <= bar_hi or abs(float(close[i]) - u) <= touch_tol:
            touches_upper += 1
        if bar_lo <= l <= bar_hi or abs(float(close[i]) - l) <= touch_tol:
            touches_lower += 1

    return {
        "available": True,
        "A": A,
        "B": B,
        "C": C,
        "median": {
            "slope_per_bar": round(float(slope), 6),
            "angle_deg": round(angle_deg, 2),
            "value_now": round(float(median_at_now), 4),
        },
        "upper_parallel": {"value_now": round(float(upper_at_now), 4)},
        "lower_parallel": {"value_now": round(float(lower_at_now), 4)},
        "position": position,
        "touches": {
            "median": int(touches_median),
            "upper": int(touches_upper),
            "lower": int(touches_lower),
            "total": int(touches_median + touches_upper + touches_lower),
        },
    }
