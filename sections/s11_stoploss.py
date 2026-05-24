"""
Биржа-цифровой — Раздел 11: ЗОНЫ СБОРА СТОПОВ.

Тип: full. Кластеры стопов на основе swing-точек и структуры.
v6: KDE/binning кластеризация High/Low, консолидации 3-5 баров,
    tail-бары (тень > 2x тело), over-under (пробой + возврат).
"""
import numpy as np
from sections.base import SectionProcessor


class StopLossProcessor(SectionProcessor):
    section_id = 11
    section_emoji = "📍"
    section_title = "ЗОНЫ СБОРА СТОПОВ"
    section_type = "full"

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _cluster_levels(prices: np.ndarray, atr: float, max_clusters: int = 8):
        """
        KDE-подобная кластеризация уровней по биннингу.
        Bin width = 0.25 * ATR; бины с >= 2 точками → кластер.
        Возвращает список dict(level, count, lo, hi).
        """
        if len(prices) == 0:
            return []
        bin_w = 0.25 * atr
        if bin_w <= 0:
            return []
        lo, hi = float(np.min(prices)), float(np.max(prices))
        if hi - lo < bin_w:
            return [{"level": round(float(np.mean(prices)), 4),
                      "count": int(len(prices)),
                      "lo": round(lo, 4), "hi": round(hi, 4)}]
        edges = np.arange(lo, hi + bin_w, bin_w)
        counts, _ = np.histogram(prices, bins=edges)
        clusters = []
        for idx in range(len(counts)):
            if counts[idx] >= 2:
                b_lo = float(edges[idx])
                b_hi = float(edges[idx + 1])
                mask = (prices >= b_lo) & (prices < b_hi)
                pts = prices[mask]
                clusters.append({
                    "level": round(float(np.mean(pts)), 4),
                    "count": int(counts[idx]),
                    "lo": round(b_lo, 4),
                    "hi": round(b_hi, 4),
                })
        # сортировка по кол-ву точек, обрезка
        clusters.sort(key=lambda c: c["count"], reverse=True)
        return clusters[:max_clusters]

    # ── compute ─────────────────────────────────────────────

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        open_ = context["open_"]
        atr_last = context["atr_last"]
        swing_points = context.get("swing_points", [])
        current_price = float(close[-1])
        n = len(close)

        # ═══════════════════════════════════════════════════
        # 1. Кластеризация swing-уровней (KDE/binning)
        # ═══════════════════════════════════════════════════
        swing_low_prices = np.array(
            [p["price"] for p in swing_points if p["type"] == "low"], dtype=float
        )
        swing_high_prices = np.array(
            [p["price"] for p in swing_points if p["type"] == "high"], dtype=float
        )

        # Берём ВСЕ swing-точки из массива.
        # Глобальная структура важна для определения ключевых уровней ликвидности.
        recent_cutoff = 0
        recent_lows_arr = np.array(
            [p["price"] for p in swing_points
             if p["type"] == "low" and p.get("index", 0) >= recent_cutoff],
            dtype=float,
        )
        recent_highs_arr = np.array(
            [p["price"] for p in swing_points
             if p["type"] == "high" and p.get("index", 0) >= recent_cutoff],
            dtype=float,
        )

        # Добавить пивотные уровни из контекста (S06 кладёт через preprocessor)
        pivot_supports = context.get("pivot_supports", [])
        pivot_resistances = context.get("pivot_resistances", [])
        extra_below = np.array([p for p in pivot_supports if p < current_price], dtype=float)
        extra_above = np.array([p for p in pivot_resistances if p > current_price], dtype=float)

        below_arr = np.concatenate([
            recent_lows_arr[recent_lows_arr < current_price],
            extra_below
        ]) if len(extra_below) > 0 else recent_lows_arr[recent_lows_arr < current_price]
        above_arr = np.concatenate([
            recent_highs_arr[recent_highs_arr > current_price],
            extra_above
        ]) if len(extra_above) > 0 else recent_highs_arr[recent_highs_arr > current_price]
        # Fallback: если в недавних мало — берём все
        if len(below_arr) < 2:
            below_arr = swing_low_prices[swing_low_prices < current_price]
        if len(above_arr) < 2:
            above_arr = swing_high_prices[swing_high_prices > current_price]

        clusters_below = self._cluster_levels(below_arr, atr_last)
        clusters_above = self._cluster_levels(above_arr, atr_last)

        # Fallback: если кластеров нет — используем top-5 ближайших к цене
        # отдельных swing-точек как «кластеры count=1».
        if not clusters_below and len(below_arr) > 0:
            sorted_below = sorted(below_arr, reverse=True)[:5]
            clusters_below = [
                {"level": round(float(p), 4), "count": 1,
                 "lo": round(float(p), 4), "hi": round(float(p), 4)}
                for p in sorted_below
            ]
        if not clusters_above and len(above_arr) > 0:
            sorted_above = sorted(above_arr)[:5]
            clusters_above = [
                {"level": round(float(p), 4), "count": 1,
                 "lo": round(float(p), 4), "hi": round(float(p), 4)}
                for p in sorted_above
            ]

        # Стоп-зоны: чуть за кластером
        stops_below = []
        for cl in clusters_below:
            stop_zone = cl["lo"] - 0.5 * atr_last
            pct = (stop_zone - current_price) / current_price * 100
            stops_below.append({
                "level": cl["level"],
                "cluster_count": cl["count"],
                "zone_lo": cl["lo"],
                "zone_hi": cl["hi"],
                "stop_zone": round(stop_zone, 4),
                "pct": round(pct, 2),
            })
        # Прямые стопы от пивотных уровней (не зависят от кластеризации)
        existing_levels_below = {s["level"] for s in stops_below}
        for p_price in pivot_supports:
            p_price = round(float(p_price), 4)
            if p_price < current_price and p_price not in existing_levels_below:
                stop_zone = p_price - 0.5 * atr_last
                pct = (stop_zone - current_price) / current_price * 100
                stops_below.append({
                    "level": p_price,
                    "cluster_count": 1,
                    "zone_lo": p_price,
                    "zone_hi": p_price,
                    "stop_zone": round(stop_zone, 4),
                    "pct": round(pct, 2),
                })

        # ближайшие 5 (по уровню — сверху вниз)
        stops_below.sort(key=lambda s: s["level"], reverse=True)
        stops_below = stops_below[:5]

        stops_above = []
        for cl in clusters_above:
            stop_zone = cl["hi"] + 0.5 * atr_last
            pct = (stop_zone - current_price) / current_price * 100
            stops_above.append({
                "level": cl["level"],
                "cluster_count": cl["count"],
                "zone_lo": cl["lo"],
                "zone_hi": cl["hi"],
                "stop_zone": round(stop_zone, 4),
                "pct": round(pct, 2),
            })
        # Прямые стопы от пивотных сопротивлений (не зависят от кластеризации)
        existing_levels_above = {s["level"] for s in stops_above}
        for p_price in pivot_resistances:
            p_price = round(float(p_price), 4)
            if p_price > current_price and p_price not in existing_levels_above:
                stop_zone = p_price + 0.5 * atr_last
                pct = (stop_zone - current_price) / current_price * 100
                stops_above.append({
                    "level": p_price,
                    "cluster_count": 1,
                    "zone_lo": p_price,
                    "zone_hi": p_price,
                    "stop_zone": round(stop_zone, 4),
                    "pct": round(pct, 2),
                })

        stops_above.sort(key=lambda s: s["level"])
        stops_above = stops_above[:5]

        # ═══════════════════════════════════════════════════
        # 2. Узкие консолидации — окно 3-5 баров (v6)
        # ═══════════════════════════════════════════════════
        narrow_zones = []
        for window in (3, 4, 5):
            step = max(1, window // 2)
            for i in range(window, n, step):
                seg_c = close[max(0, i - window):i]
                seg_h = high[max(0, i - window):i]
                seg_l = low[max(0, i - window):i]
                range_pct = (float(np.max(seg_h)) - float(np.min(seg_l))) / float(np.mean(seg_c)) * 100
                if range_pct < 1.0:
                    narrow_zones.append({
                        "zone_low": round(float(np.min(seg_l)), 4),
                        "zone_high": round(float(np.max(seg_h)), 4),
                        "range_pct": round(float(range_pct), 3),
                        "window": window,
                        "bar_end_offset": n - i,
                    })
        # дедупликация по зоне (оставляем с наименьшим range)
        seen = set()
        deduped = []
        for z in sorted(narrow_zones, key=lambda x: x["range_pct"]):
            key = (z["zone_low"], z["zone_high"])
            if key not in seen:
                seen.add(key)
                deduped.append(z)
        narrow_zones = deduped[-5:]  # последние 5

        # ═══════════════════════════════════════════════════
        # 3. Tail-бары: тень > 2x тела (ловушки ликвидности)
        # ═══════════════════════════════════════════════════
        tail_bars = []
        lookback = min(50, n)
        for i in range(n - lookback, n):
            body = abs(float(close[i]) - float(open_[i]))
            upper_shadow = float(high[i]) - max(float(close[i]), float(open_[i]))
            lower_shadow = min(float(close[i]), float(open_[i])) - float(low[i])
            body_safe = max(body, 1e-10)

            tail_type = None
            tail_shadow = 0.0
            if lower_shadow > 2.0 * body_safe and lower_shadow > upper_shadow:
                tail_type = "lower"
                tail_shadow = lower_shadow
            elif upper_shadow > 2.0 * body_safe and upper_shadow > lower_shadow:
                tail_type = "upper"
                tail_shadow = upper_shadow

            if tail_type is not None:
                tail_bars.append({
                    "bar_offset": n - 1 - i,
                    "tail_type": tail_type,
                    "tail_size": round(tail_shadow, 4),
                    "tail_body_ratio": min(round(tail_shadow / body_safe, 2), 100.0),
                    "price": round(float(low[i]) if tail_type == "lower" else float(high[i]), 4),
                })
        tail_bars = tail_bars[-10:]  # последние 10

        # ═══════════════════════════════════════════════════
        # 4. Over-under: цена пробила уровень, затем вернулась
        #    (sweep liquidity / fakeout)
        # ═══════════════════════════════════════════════════
        over_under = []
        all_levels = (
            [cl["level"] for cl in clusters_below]
            + [cl["level"] for cl in clusters_above]
        )
        lookback_ou = min(30, n - 1)
        for lvl in all_levels:
            for i in range(n - lookback_ou, n):
                # цена выходит за уровень хвостом, но close возвращается
                exceeded_below = float(low[i]) < lvl and float(close[i]) > lvl
                exceeded_above = float(high[i]) > lvl and float(close[i]) < lvl
                if exceeded_below:
                    over_under.append({
                        "bar_offset": n - 1 - i,
                        "level": round(lvl, 4),
                        "direction": "sweep_below",
                        "low": round(float(low[i]), 4),
                        "close": round(float(close[i]), 4),
                    })
                elif exceeded_above:
                    over_under.append({
                        "bar_offset": n - 1 - i,
                        "level": round(lvl, 4),
                        "direction": "sweep_above",
                        "high": round(float(high[i]), 4),
                        "close": round(float(close[i]), 4),
                    })
        # дедупликация по (bar_offset, level)
        seen_ou = set()
        deduped_ou = []
        for ou in over_under:
            key = (ou["bar_offset"], ou["level"])
            if key not in seen_ou:
                seen_ou.add(key)
                deduped_ou.append(ou)
        over_under = sorted(deduped_ou, key=lambda x: x["bar_offset"])[:10]

        return {
            "stops_below_supports": stops_below,
            "stops_above_resistances": stops_above,
            "narrow_consolidations": narrow_zones,
            "tail_bars": tail_bars,
            "over_under_sweeps": over_under,
            "current_price": current_price,
            "atr_buffer": round(0.5 * atr_last, 4),
        }
