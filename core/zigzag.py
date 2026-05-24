"""
Биржа-цифровой — ZigZag, Williams Fractal, Swing Points.

ZigZag с настраиваемым отклонением (1% и 5%).
Williams Fractal (period=5) для валидации.
Классификация swing-точек: HH, HL, LH, LL.
"""
import numpy as np


def zigzag(high: np.ndarray, low: np.ndarray, deviation_pct: float = 5.0,
           times=None) -> list[dict]:
    """ZigZag — поворотные точки с минимальным отклонением deviation_pct%.

    Возвращает список точек:
    [{"index": int, "price": float, "type": "high"|"low", "time": ...}, ...]
    """
    n = len(high)
    if n < 3:
        return []

    dev = deviation_pct / 100.0
    points = []

    # Инициализация: первый экстремум
    first_high_idx = 0
    first_low_idx = 0
    first_high = high[0]
    first_low = low[0]

    # Ищем первый значимый экстремум
    for i in range(1, n):
        if high[i] > first_high:
            first_high = high[i]
            first_high_idx = i
        if low[i] < first_low:
            first_low = low[i]
            first_low_idx = i

        # Если уже набралось достаточное отклонение
        up_move = (first_high - first_low) / first_low if first_low > 0 else 0
        if up_move >= dev:
            if first_low_idx < first_high_idx:
                points.append({"index": first_low_idx, "price": first_low, "type": "low"})
                points.append({"index": first_high_idx, "price": first_high, "type": "high"})
            else:
                points.append({"index": first_high_idx, "price": first_high, "type": "high"})
                points.append({"index": first_low_idx, "price": first_low, "type": "low"})
            break
    else:
        return []

    # Основной цикл: продолжаем от последней точки
    start_i = max(first_high_idx, first_low_idx) + 1
    last = points[-1]
    last_type = last["type"]

    for i in range(start_i, n):
        if last_type == "high":
            # Ищем low
            if low[i] < last["price"] * (1 - dev):
                # Новый low — достаточное отклонение от предыдущего high
                points.append({"index": i, "price": low[i], "type": "low"})
                last = points[-1]
                last_type = "low"
            elif high[i] > last["price"]:
                # Обновление high
                last["index"] = i
                last["price"] = high[i]
        else:
            # Ищем high
            if high[i] > last["price"] * (1 + dev):
                points.append({"index": i, "price": high[i], "type": "high"})
                last = points[-1]
                last_type = "high"
            elif low[i] < last["price"]:
                # Обновление low
                last["index"] = i
                last["price"] = low[i]

    # Добавить временные метки
    if times is not None:
        for p in points:
            p["time"] = str(times[p["index"]])

    return points


def zigzag_period(high: np.ndarray, low: np.ndarray, period: int = 8,
                   times=None) -> list[dict]:
    """ZigZag по периоду (как на TradingView).

    Разворот фиксируется когда за последние `period` баров
    хай/лоу не обновлялся.

    Args:
        period: количество баров без обновления экстремума = разворот.
    """
    n = len(high)
    if n < period + 1:
        return []

    points = []
    trend = 0  # 0=undefined, 1=up, -1=down
    last_hi = high[0]
    last_hi_idx = 0
    last_lo = low[0]
    last_lo_idx = 0

    for i in range(1, n):
        if high[i] > last_hi:
            last_hi = high[i]
            last_hi_idx = i
        if low[i] < last_lo:
            last_lo = low[i]
            last_lo_idx = i

        if trend == 0:
            # Ждём первый разворот
            if i - last_hi_idx >= period and last_hi_idx > last_lo_idx:
                # Хай не обновлялся period баров → разворот вниз от хая
                points.append({
                    "index": last_hi_idx, "price": float(last_hi),
                    "type": "high",
                    "time": str(times[last_hi_idx])[:19] if times is not None else "",
                })
                trend = -1
                last_lo = low[i]
                last_lo_idx = i
            elif i - last_lo_idx >= period and last_lo_idx > last_hi_idx:
                points.append({
                    "index": last_lo_idx, "price": float(last_lo),
                    "type": "low",
                    "time": str(times[last_lo_idx])[:19] if times is not None else "",
                })
                trend = 1
                last_hi = high[i]
                last_hi_idx = i

        elif trend == 1:  # растём
            if high[i] > last_hi:
                last_hi = high[i]
                last_hi_idx = i
            if i - last_hi_idx >= period:
                # Хай не обновлялся → разворот вниз
                points.append({
                    "index": last_hi_idx, "price": float(last_hi),
                    "type": "high",
                    "time": str(times[last_hi_idx])[:19] if times is not None else "",
                })
                trend = -1
                last_lo = low[last_hi_idx]
                last_lo_idx = last_hi_idx
                for j in range(last_hi_idx, i + 1):
                    if low[j] < last_lo:
                        last_lo = low[j]
                        last_lo_idx = j

        elif trend == -1:  # падаем
            if low[i] < last_lo:
                last_lo = low[i]
                last_lo_idx = i
            if i - last_lo_idx >= period:
                # Лоу не обновлялся → разворот вверх
                points.append({
                    "index": last_lo_idx, "price": float(last_lo),
                    "type": "low",
                    "time": str(times[last_lo_idx])[:19] if times is not None else "",
                })
                trend = 1
                last_hi = high[last_lo_idx]
                last_hi_idx = last_lo_idx
                for j in range(last_lo_idx, i + 1):
                    if high[j] > last_hi:
                        last_hi = high[j]
                        last_hi_idx = j

    # Последний пивот
    if trend == 1:
        points.append({
            "index": last_hi_idx, "price": float(last_hi),
            "type": "high",
            "time": str(times[last_hi_idx])[:19] if times is not None else "",
        })
    elif trend == -1:
        points.append({
            "index": last_lo_idx, "price": float(last_lo),
            "type": "low",
            "time": str(times[last_lo_idx])[:19] if times is not None else "",
        })

    return points


def williams_fractal(high: np.ndarray, low: np.ndarray, period: int = 5
                     ) -> dict[str, list[dict]]:
    """Williams Fractal — фрактальные вершины и впадины.

    Фрактал вверх: high[i] > всех high в окне [i-period..i+period] (кроме i).
    Фрактал вниз: low[i] < всех low в окне [i-period..i+period] (кроме i).

    Возвращает {"up": [...], "down": [...]}.
    """
    n = len(high)
    half = period // 2
    up_fractals = []
    down_fractals = []

    for i in range(half, n - half):
        # Фрактал вверх
        is_up = True
        for j in range(i - half, i + half + 1):
            if j != i and high[j] >= high[i]:
                is_up = False
                break
        if is_up:
            up_fractals.append({"index": i, "price": float(high[i])})

        # Фрактал вниз
        is_down = True
        for j in range(i - half, i + half + 1):
            if j != i and low[j] <= low[i]:
                is_down = False
                break
        if is_down:
            down_fractals.append({"index": i, "price": float(low[i])})

    return {"up": up_fractals, "down": down_fractals}


def classify_swing_points(zz_points: list[dict]) -> list[dict]:
    """Классификация swing-точек: HH, HL, LH, LL.

    HH — Higher High: high выше предыдущего high.
    HL — Higher Low: low выше предыдущего low.
    LH — Lower High: high ниже предыдущего high.
    LL — Lower Low: low ниже предыдущего low.

    Определяет тренд:
    - HH + HL = восходящий
    - LH + LL = нисходящий
    - иначе = боковик
    """
    if len(zz_points) < 3:
        return zz_points

    prev_highs = []
    prev_lows = []
    classified = []

    for p in zz_points:
        p = dict(p)  # копия
        if p["type"] == "high":
            if prev_highs:
                p["swing"] = "HH" if p["price"] > prev_highs[-1] else "LH"
            else:
                p["swing"] = "HH"  # первый high
            prev_highs.append(p["price"])
        else:
            if prev_lows:
                p["swing"] = "HL" if p["price"] > prev_lows[-1] else "LL"
            else:
                p["swing"] = "HL"  # первый low
            prev_lows.append(p["price"])
        classified.append(p)

    return classified


def detect_trend(swing_points: list[dict], lookback: int = 6,
                 close=None, alma_200=None, tf_hours: float = 4.0) -> str:
    """Определить тренд по swing-точкам с проверкой пробоя структуры.

    Логика:
    1. Базовый тренд — по swing highs/lows (последние N точек)
    2. Пробой структуры — если цена ниже последнего HL (в up-тренде) или
       выше последнего LH (в down-тренде) → тренд СЛОМАН, переключить
    3. Подтверждение по slope последних 20 close + ALMA 200
    """
    import numpy as np

    if len(swing_points) < 2:
        return "боковик"

    recent = swing_points[-lookback:] if len(swing_points) >= lookback else swing_points

    highs = [p["price"] for p in recent if p["type"] == "high"]
    lows = [p["price"] for p in recent if p["type"] == "low"]

    trend = "боковик"

    if len(highs) >= 2 and len(lows) >= 2:
        highs_declining = highs[-1] < highs[0]
        lows_declining = lows[-1] < lows[0]
        highs_rising = highs[-1] > highs[0]
        lows_rising = lows[-1] > lows[0]

        if highs_declining and lows_declining:
            trend = "нисходящий"
        elif highs_rising and lows_rising:
            trend = "восходящий"
        else:
            hh = sum(1 for p in recent if p.get("swing") == "HH")
            hl = sum(1 for p in recent if p.get("swing") == "HL")
            lh = sum(1 for p in recent if p.get("swing") == "LH")
            ll = sum(1 for p in recent if p.get("swing") == "LL")
            bull = hh + hl
            bear = lh + ll
            if bear > bull:
                trend = "нисходящий"
            elif bull > bear:
                trend = "восходящий"

    # ── ПРОБОЙ СТРУКТУРЫ (переключение тренда) ──
    # В up-тренде последний swing-low = HL. Если цена ниже него → сломан.
    # В down-тренде последний swing-high = LH. Если цена выше → сломан.
    if close is not None and len(close) > 0:
        curr = float(close[-1])
        last_low = lows[-1] if lows else None
        last_high = highs[-1] if highs else None

        if trend == "восходящий" and last_low and curr < last_low * 0.995:
            # Пробой восходящей структуры вниз
            trend = "нисходящий"
        elif trend == "нисходящий" and last_high and curr > last_high * 1.005:
            # Пробой нисходящей структуры вверх
            trend = "восходящий"

    # ── Подтверждение по slope (для боковика) ──
    if trend == "боковик" and close is not None and len(close) >= 20:
        recent_close = close[-20:]
        x = np.arange(len(recent_close))
        slope = np.polyfit(x, recent_close, 1)[0]
        slope_pct = slope / recent_close[0] * 100

        if slope_pct < -0.1:
            trend = "нисходящий"
        elif slope_pct > 0.1:
            trend = "восходящий"

    # ── ALMA 200 (сильное отклонение) ──
    if close is not None and alma_200 is not None:
        curr_price = float(close[-1])
        if trend == "боковик":
            if curr_price < alma_200 * 0.98:
                trend = "нисходящий"
            elif curr_price > alma_200 * 1.02:
                trend = "восходящий"

    # ── ПРИОРИТЕТ slope: адаптивное временное окно ~5 торговых дней
    # независимо от ТФ. На 4H это 30 баров, на 15m — 500 баров, на 1D — 5 баров.
    # Используется 2 окна одновременно:
    #   - short: 5 дней (тактический импульс)
    #   - long:  20 дней (стратегический контекст)
    # Переключаем только если оба согласны (или long сильно доминирует).
    if close is not None and tf_hours > 0:
        bars_5d = max(20, int(5 * 24 / tf_hours))
        bars_20d = max(50, int(20 * 24 / tf_hours))

        def _slope_r2(arr):
            if len(arr) < 10:
                return 0.0, 0.0
            x = np.arange(len(arr))
            s, i = np.polyfit(x, arr, 1)
            m = float(np.mean(arr))
            spct = s / m * 100 if m > 0 else 0.0
            fit = s * x + i
            ssr = np.sum((arr - fit) ** 2)
            sst = np.sum((arr - m) ** 2)
            r2 = 1 - ssr / sst if sst > 0 else 0.0
            return spct, r2

        short_arr = close[-min(bars_5d, len(close)):]
        long_arr = close[-min(bars_20d, len(close)):]
        short_slope, short_r2 = _slope_r2(short_arr)
        long_slope, long_r2 = _slope_r2(long_arr)

        # «Сильный» slope: |>0.01%/бар| и R²>0.4
        def _classify(s, r2):
            if abs(s) < 0.01 or r2 < 0.4:
                return "flat"
            return "down" if s < 0 else "up"

        short_dir = _classify(short_slope, short_r2)
        long_dir = _classify(long_slope, long_r2)

        # Override: трендовая классификация должна подтверждаться
        # либо long (стратегический контекст), либо обоими окнами.
        # Для коротких откатов (short≠long) оставляем swing-вердикт и
        # не переключаем в "противоположный тренд", а понижаем до "боковик".
        if trend == "восходящий":
            if long_dir == "down" and short_dir == "down":
                trend = "нисходящий"
            elif long_dir == "down" and short_dir == "flat":
                trend = "боковик"
        elif trend == "нисходящий":
            if long_dir == "up" and short_dir == "up":
                trend = "восходящий"
            elif long_dir == "up" and short_dir == "flat":
                trend = "боковик"
        elif trend == "боковик":
            if long_dir == "up" and short_dir != "down":
                trend = "восходящий"
            elif long_dir == "down" and short_dir != "up":
                trend = "нисходящий"

    return trend
