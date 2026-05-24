"""
Биржа-цифровой — Алгоритмический детектор графических паттернов.

Детектирует по swing-точкам (ZigZag):
- H&S / Inverse H&S
- Double Top / Bottom
- Triple Top / Bottom
- Ascending / Descending / Symmetrical Triangle
- Wedge (Rising / Falling)
- Flag / Pennant
- Cup and Handle
"""
import numpy as np


def detect_patterns(swing_points: list[dict], current_price: float,
                    atr: float) -> list[dict]:
    """Найти графические паттерны по swing-точкам.

    Returns: [{name, name_ru, status, target, direction, points}, ...]

    Ограничение: берём только последние 12 swing-точек, чтобы не ловить
    устаревшие паттерны с неадекватными целями (раньше Double Top находился
    в 2022 году с target 1200 при цене 1970).
    """
    if len(swing_points) < 4:
        return []

    # Только недавние swing (текущий сегмент)
    swing_points = swing_points[-20:]

    results = []
    tolerance = 0.015  # 1.5% для сравнения уровней
    MAX_TARGET_PCT = 25.0  # фильтр далёких целей (±25% от цены)

    highs = [(i, p) for i, p in enumerate(swing_points) if p["type"] == "high"]
    lows = [(i, p) for i, p in enumerate(swing_points) if p["type"] == "low"]

    # --- Double Top ---
    for dt in _detect_double_top(highs, lows, current_price, tolerance, atr):
        results.append(dt)

    # --- Double Bottom ---
    for db in _detect_double_bottom(highs, lows, current_price, tolerance, atr):
        results.append(db)

    # --- Head & Shoulders ---
    for hs in _detect_head_shoulders(highs, lows, current_price, tolerance, atr):
        results.append(hs)

    # --- Inverse H&S ---
    for ihs in _detect_inv_head_shoulders(highs, lows, current_price, tolerance, atr):
        results.append(ihs)

    # --- Triangles ---
    for tri in _detect_triangles(swing_points, current_price, atr):
        results.append(tri)

    # --- Wedge ---
    for w in _detect_wedge(swing_points, current_price, atr):
        results.append(w)

    # --- Flag ---
    for f in _detect_flag(swing_points, current_price, atr):
        results.append(f)

    # Фильтр неадекватных целей: отбрасываем паттерны с target дальше MAX_TARGET_PCT
    filtered = []
    for p in results:
        tp = p.get("target_pct")
        if tp is None or abs(tp) <= MAX_TARGET_PCT:
            filtered.append(p)
    return filtered


def _near(a, b, tolerance):
    """Цены примерно равны (в пределах tolerance %)."""
    return abs(a - b) / max(a, 0.001) < tolerance


def _detect_double_top(highs, lows, price, tol, atr):
    """Двойная вершина: два хая примерно на одном уровне, между ними лоу."""
    results = []
    if len(highs) < 2:
        return results

    for i in range(len(highs) - 1):
        idx1, h1 = highs[i]
        idx2, h2 = highs[i + 1]

        if _near(h1["price"], h2["price"], tol):
            # Найти лоу между ними
            between_lows = [l for j, l in lows if idx1 < j < idx2]
            if between_lows:
                neckline = min(l["price"] for l in between_lows)
                height = h1["price"] - neckline
                target = neckline - height

                # Статус
                if price < neckline:
                    status = "В реализации"
                elif price < h2["price"]:
                    status = "Подтверждён"
                else:
                    status = "Формируется"

                results.append({
                    "name": "Double Top",
                    "name_ru": "Двойная вершина",
                    "direction": "медвежий",
                    "status": status,
                    "target": round(target, 4),
                    "target_pct": round((target - price) / price * 100, 2),
                    "neckline": round(neckline, 4),
                    "top": round(h1["price"], 4),
                })
                break  # один паттерн

    return results


def _detect_double_bottom(highs, lows, price, tol, atr):
    """Двойное дно: два лоу примерно на одном уровне."""
    results = []
    if len(lows) < 2:
        return results

    for i in range(len(lows) - 1):
        idx1, l1 = lows[i]
        idx2, l2 = lows[i + 1]

        if _near(l1["price"], l2["price"], tol):
            between_highs = [h for j, h in highs if idx1 < j < idx2]
            if between_highs:
                neckline = max(h["price"] for h in between_highs)
                height = neckline - l1["price"]
                target = neckline + height

                if price > neckline:
                    status = "В реализации"
                elif price > l2["price"]:
                    status = "Подтверждён"
                else:
                    status = "Формируется"

                results.append({
                    "name": "Double Bottom",
                    "name_ru": "Двойное дно",
                    "direction": "бычий",
                    "status": status,
                    "target": round(target, 4),
                    "target_pct": round((target - price) / price * 100, 2),
                    "neckline": round(neckline, 4),
                    "bottom": round(l1["price"], 4),
                })
                break

    return results


def _detect_head_shoulders(highs, lows, price, tol, atr):
    """Голова и плечи: левое плечо, голова (выше), правое плечо (≈ левое)."""
    results = []
    if len(highs) < 3:
        return results

    for i in range(len(highs) - 2):
        _, h_left = highs[i]
        _, h_head = highs[i + 1]
        _, h_right = highs[i + 2]

        if (h_head["price"] > h_left["price"] and
                h_head["price"] > h_right["price"] and
                _near(h_left["price"], h_right["price"], tol * 2)):

            # Neckline — линия между лоу
            idx_l = highs[i][0]
            idx_h = highs[i + 1][0]
            idx_r = highs[i + 2][0]

            neck_lows = [l["price"] for j, l in lows if idx_l < j < idx_r]
            if neck_lows:
                neckline = np.mean(neck_lows)
                height = h_head["price"] - neckline
                target = neckline - height

                if price < neckline:
                    status = "В реализации"
                elif price < h_right["price"]:
                    status = "Подтверждён"
                else:
                    status = "Формируется"

                results.append({
                    "name": "Head & Shoulders",
                    "name_ru": "Голова и плечи",
                    "direction": "медвежий",
                    "status": status,
                    "target": round(target, 4),
                    "target_pct": round((target - price) / price * 100, 2),
                    "neckline": round(neckline, 4),
                    "head": round(h_head["price"], 4),
                })
                break

    return results


def _detect_inv_head_shoulders(highs, lows, price, tol, atr):
    """Перевёрнутая голова и плечи."""
    results = []
    if len(lows) < 3:
        return results

    for i in range(len(lows) - 2):
        _, l_left = lows[i]
        _, l_head = lows[i + 1]
        _, l_right = lows[i + 2]

        if (l_head["price"] < l_left["price"] and
                l_head["price"] < l_right["price"] and
                _near(l_left["price"], l_right["price"], tol * 2)):

            idx_l = lows[i][0]
            idx_r = lows[i + 2][0]

            neck_highs = [h["price"] for j, h in highs if idx_l < j < idx_r]
            if neck_highs:
                neckline = np.mean(neck_highs)
                height = neckline - l_head["price"]
                target = neckline + height

                if price > neckline:
                    status = "В реализации"
                else:
                    status = "Формируется"

                results.append({
                    "name": "Inverse H&S",
                    "name_ru": "Перевёрнутая голова и плечи",
                    "direction": "бычий",
                    "status": status,
                    "target": round(target, 4),
                    "target_pct": round((target - price) / price * 100, 2),
                    "neckline": round(neckline, 4),
                    "head": round(l_head["price"], 4),
                })
                break

    return results


def _detect_triangles(swings, price, atr):
    """Треугольники: восходящий, нисходящий, симметричный."""
    results = []
    if len(swings) < 6:
        return results

    recent = swings[-8:]
    rh = [p["price"] for p in recent if p["type"] == "high"]
    rl = [p["price"] for p in recent if p["type"] == "low"]

    if len(rh) < 2 or len(rl) < 2:
        return results

    highs_slope = (rh[-1] - rh[0]) / max(len(rh), 1)
    lows_slope = (rl[-1] - rl[0]) / max(len(rl), 1)

    # Сходимость: highs падают и lows растут
    converging = highs_slope < 0 and lows_slope > 0
    h_flat = abs(highs_slope) < atr * 0.05
    l_flat = abs(lows_slope) < atr * 0.05

    if h_flat and lows_slope > 0:
        # Восходящий треугольник
        resistance = np.mean(rh)
        target = resistance + (resistance - min(rl))
        results.append({
            "name": "Ascending Triangle",
            "name_ru": "Восходящий треугольник",
            "direction": "бычий",
            "status": "Формируется",
            "target": round(target, 4),
            "target_pct": round((target - price) / price * 100, 2),
            "resistance": round(resistance, 4),
        })
    elif l_flat and highs_slope < 0:
        # Нисходящий треугольник
        support = np.mean(rl)
        target = support - (max(rh) - support)
        results.append({
            "name": "Descending Triangle",
            "name_ru": "Нисходящий треугольник",
            "direction": "медвежий",
            "status": "Формируется",
            "target": round(target, 4),
            "target_pct": round((target - price) / price * 100, 2),
            "support": round(support, 4),
        })
    elif converging:
        # Симметричный
        mid = (np.mean(rh) + np.mean(rl)) / 2
        span = max(rh) - min(rl)
        results.append({
            "name": "Symmetrical Triangle",
            "name_ru": "Симметричный треугольник",
            "direction": "нейтральный (пробой определит)",
            "status": "Формируется",
            "target_up": round(mid + span, 4),
            "target_down": round(mid - span, 4),
        })

    return results


def _detect_wedge(swings, price, atr):
    """Клин: растущий (медвежий) / падающий (бычий).

    Пробуем несколько размеров окна (8, 12, 16) и возвращаем
    первый валидный паттерн — более широкое окно даёт более чёткую структуру.
    """
    if len(swings) < 6:
        return []

    for window_size in [8, 12, 16]:
        if len(swings) < window_size:
            recent = swings[:]
        else:
            recent = swings[-window_size:]

        rh = [p["price"] for p in recent if p["type"] == "high"]
        rl = [p["price"] for p in recent if p["type"] == "low"]

        if len(rh) < 2 or len(rl) < 2:
            continue

        highs_up = rh[-1] > rh[0]
        lows_up = rl[-1] > rl[0]

        # Rising wedge: оба растут, но сходятся
        if highs_up and lows_up:
            h_slope = rh[-1] - rh[0]
            l_slope = rl[-1] - rl[0]
            if l_slope > h_slope * 0.5:  # лоу растут быстрее — сходимость
                target = min(rl)
                return [{
                    "name": "Rising Wedge",
                    "name_ru": "Восходящий клин (разворот)",
                    "direction": "медвежий",
                    "status": "Формируется",
                    "target": round(target, 4),
                    "target_pct": round((target - price) / price * 100, 2),
                }]

        # Falling wedge: оба падают, верхняя граница падает круче
        if not highs_up and not lows_up:
            h_slope = abs(rh[-1] - rh[0])
            l_slope = abs(rl[-1] - rl[0])
            # Сходимость: верхняя граница падает быстрее ИЛИ обе падают (клин)
            if h_slope > l_slope * 0.3:
                target = max(rh)
                return [{
                    "name": "Falling Wedge",
                    "name_ru": "Нисходящий клин (разворот)",
                    "direction": "бычий",
                    "status": "Формируется",
                    "target": round(target, 4),
                    "target_pct": round((target - price) / price * 100, 2),
                }]

    return []


def _detect_flag(swings, price, atr):
    """Флаг: сильный импульс + узкая консолидация."""
    results = []
    if len(swings) < 5:
        return results

    # Последние 4-6 точек — проверяем диапазон
    recent = swings[-6:]
    prices_r = [p["price"] for p in recent]
    rng = max(prices_r) - min(prices_r)

    # Предшествующие точки — импульс
    prev = swings[-10:-6] if len(swings) >= 10 else swings[:4]
    if not prev:
        return results

    prices_p = [p["price"] for p in prev]
    impulse = max(prices_p) - min(prices_p)

    # Флаг: диапазон консолидации < 30% от импульса
    if impulse > 0 and rng / impulse < 0.3 and rng > atr * 0.5:
        # Направление импульса
        if prices_p[-1] > prices_p[0]:
            direction = "бычий"
            target = max(prices_r) + impulse
        else:
            direction = "медвежий"
            target = min(prices_r) - impulse

        results.append({
            "name": "Flag",
            "name_ru": f"Флаг ({direction})",
            "direction": direction,
            "status": "Формируется",
            "target": round(target, 4),
            "target_pct": round((target - price) / price * 100, 2),
            "impulse_size": round(impulse, 4),
            "flag_range": round(rng, 4),
        })

    return results
