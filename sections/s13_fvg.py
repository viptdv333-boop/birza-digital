"""
Биржа-цифровой — Раздел 13: ИМБАЛАНСЫ / ЛИКВИДНОСТЬ / ГЭПЫ / FVG.

Тип: full.
v6: Гэпы по High[i-1] vs Low[i], пулы ликвидности
    (swing-точки + объёмные кластеры).
"""
import numpy as np
from sections.base import SectionProcessor
from config import FVG_MIN_SIZE_ATR_MULT


class FVGProcessor(SectionProcessor):
    section_id = 13
    section_emoji = "💧"
    section_title = "ИМБАЛАНСЫ / ЛИКВИДНОСТЬ / ГЭПЫ / FVG"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        open_ = context["open_"]
        volume = context["volume"]
        atr_last = context["atr_last"]
        swing_points = context.get("swing_points", [])
        n = len(close)
        current_price = float(close[-1])

        min_fvg_size = FVG_MIN_SIZE_ATR_MULT * atr_last

        # ═══════════════════════════════════════════════════
        # 1. FVG (Fair Value Gap) — 3-свечная модель
        #    Бычий: low[i] > high[i-2]  (гэп между баром i и i-2)
        #    Медвежий: high[i] < low[i-2]
        #    Заполнение: проверяем ВСЕ последующие бары.
        # ═══════════════════════════════════════════════════
        fvgs = []
        lookback = min(200, n - 2)
        start = max(2, n - lookback)
        for i in range(start, n):
            # Бычий FVG: low[i] > high[i-2]
            if low[i] > high[i - 2]:
                gap_size = float(low[i] - high[i - 2])
                if gap_size >= min_fvg_size:
                    # Заполнение бычьего FVG: цена должна опуститься обратно
                    # в зону гэпа И полностью закрыть его (low <= bottom)
                    fvg_bottom = float(high[i - 2])
                    filled = any(low[j] <= fvg_bottom for j in range(i + 1, n))
                    fvgs.append({
                        "type": "bullish",
                        "top": round(float(low[i]), 4),
                        "bottom": round(fvg_bottom, 4),
                        "size": round(gap_size, 4),
                        "bar_offset": n - 1 - i,
                        "filled": filled,
                    })
            # Медвежий FVG: high[i] < low[i-2]
            if high[i] < low[i - 2]:
                gap_size = float(low[i - 2] - high[i])
                if gap_size >= min_fvg_size:
                    # Заполнение медвежьего FVG: цена должна подняться обратно
                    # к ВЕРХНЕЙ границе гэпа (high >= top), а не просто коснуться нижней
                    fvg_top = float(low[i - 2])
                    filled = any(high[j] >= fvg_top for j in range(i + 1, n))
                    fvgs.append({
                        "type": "bearish",
                        "top": round(fvg_top, 4),
                        "bottom": round(float(high[i]), 4),
                        "size": round(gap_size, 4),
                        "bar_offset": n - 1 - i,
                        "filled": filled,
                    })

        # Только незаполненные FVG
        open_fvgs = [f for f in fvgs if not f["filled"]]

        # ═══════════════════════════════════════════════════
        # 2. Гэпы: High[i-1] vs Low[i]
        #    Gap up:   low[i] > high[i-1]
        #    Gap down: high[i] < low[i-1]
        #    Заполнение: проверяем все последующие бары.
        # ═══════════════════════════════════════════════════
        gaps = []
        gap_lookback = min(200, n - 1)
        gap_start = max(1, n - gap_lookback)
        for i in range(gap_start, n):
            # Gap up: весь бар i выше предыдущего
            if low[i] > high[i - 1]:
                gap_size = float(low[i] - high[i - 1])
                if gap_size >= min_fvg_size:
                    filled = any(low[j] <= high[i - 1] for j in range(i + 1, n))
                    gaps.append({
                        "type": "up",
                        "size": round(gap_size, 4),
                        "gap_top": round(float(low[i]), 4),
                        "gap_bottom": round(float(high[i - 1]), 4),
                        "bar_offset": n - 1 - i,
                        "filled": filled,
                    })
            # Gap down: весь бар i ниже предыдущего
            if high[i] < low[i - 1]:
                gap_size = float(low[i - 1] - high[i])
                if gap_size >= min_fvg_size:
                    filled = any(high[j] >= low[i - 1] for j in range(i + 1, n))
                    gaps.append({
                        "type": "down",
                        "size": round(gap_size, 4),
                        "gap_top": round(float(low[i - 1]), 4),
                        "gap_bottom": round(float(high[i]), 4),
                        "bar_offset": n - 1 - i,
                        "filled": filled,
                    })

        open_gaps = [g for g in gaps if not g["filled"]]

        # ═══════════════════════════════════════════════════
        # 3. Пулы ликвидности (v6)
        #    a) swing lows / highs — зоны скопления стопов
        #    b) объёмные кластеры — бары с объёмом > 1.5x среднего
        # ═══════════════════════════════════════════════════
        liquidity_pools = []

        # 3a. Swing-point liquidity
        for sp in swing_points:
            dist = abs(sp["price"] - current_price) / atr_last if atr_last > 0 else 999
            if dist <= 5.0:  # в пределах 5 ATR
                liquidity_pools.append({
                    "source": "swing",
                    "type": sp["type"],  # "high" или "low"
                    "price": round(sp["price"], 4),
                    "distance_atr": round(dist, 2),
                })

        # 3b. Volume clusters — бары с объёмом > 1.5x среднего,
        #     группировка по ценовым бинам (0.5 ATR)
        vol_lookback = min(100, n)
        vol_slice = volume[-vol_lookback:]
        vol_mean = float(np.mean(vol_slice)) if len(vol_slice) > 0 else 0
        vol_threshold = 1.5 * vol_mean

        if vol_mean > 0 and atr_last > 0:
            bin_w = 0.5 * atr_last
            heavy_prices = []
            heavy_volumes = []
            for i in range(n - vol_lookback, n):
                if volume[i] > vol_threshold:
                    typical = (float(high[i]) + float(low[i]) + float(close[i])) / 3
                    heavy_prices.append(typical)
                    heavy_volumes.append(float(volume[i]))

            if len(heavy_prices) >= 2:
                hp = np.array(heavy_prices)
                hv = np.array(heavy_volumes)
                lo_edge = float(np.min(hp))
                hi_edge = float(np.max(hp)) + bin_w
                edges = np.arange(lo_edge, hi_edge, bin_w)
                if len(edges) >= 2:
                    indices = np.digitize(hp, edges) - 1
                    for b_idx in range(len(edges) - 1):
                        mask = indices == b_idx
                        cnt = int(np.sum(mask))
                        if cnt >= 2:
                            avg_price = float(np.mean(hp[mask]))
                            total_vol = float(np.sum(hv[mask]))
                            dist = abs(avg_price - current_price) / atr_last
                            liquidity_pools.append({
                                "source": "volume_cluster",
                                "type": "concentration",
                                "price": round(avg_price, 4),
                                "bar_count": cnt,
                                "total_volume": round(total_vol, 2),
                                "distance_atr": round(dist, 2),
                            })

        # Сортировка по близости
        liquidity_pools.sort(key=lambda p: p["distance_atr"])
        liquidity_pools = liquidity_pools[:10]

        # ═══════════════════════════════════════════════════
        # 4. Ближайший магнит — FVG/гэп/swing пул (v8: proximity ≤ 0.6×ATR)
        #    Min distance 0.1 ATR (отсекаем «магниты» в цене текущего бара),
        #    Max distance 0.6 ATR — по регламенту v8 (magnet rule).
        #    Магниты с distance > 0.6×ATR остаются в magnets_all как справочные.
        # ═══════════════════════════════════════════════════
        min_distance = 0.1 * atr_last
        proximity_threshold = 0.6 * atr_last  # v8
        nearest_magnet = None
        all_targets = []

        for f in open_fvgs:
            mid = (f["top"] + f["bottom"]) / 2
            dist = abs(mid - current_price)
            all_targets.append({
                "type": f"FVG_{f['type']}",
                "price": mid,
                "distance": dist,
                "source": "FVG",
            })

        for g in open_gaps:
            mid = (g["gap_top"] + g["gap_bottom"]) / 2
            dist = abs(mid - current_price)
            all_targets.append({
                "type": f"gap_{g['type']}",
                "price": mid,
                "distance": dist,
                "source": "gap",
            })

        # Fallback: swing-пулы ликвидности как магниты
        for lp in liquidity_pools:
            if lp.get("source") == "swing":
                dist = abs(lp["price"] - current_price)
                all_targets.append({
                    "type": f"swing_{lp['type']}",
                    "price": lp["price"],
                    "distance": dist,
                    "source": "swing",
                })

        # Фильтр: отбрасываем магниты слишком близко к текущей цене
        all_targets = [t for t in all_targets if t["distance"] >= min_distance]
        all_targets.sort(key=lambda t: t["distance"])

        magnets_all = [
            {
                "type": t["type"],
                "price": round(float(t["price"]), 4),
                "distance_atr": round(t["distance"] / atr_last, 2) if atr_last > 0 else None,
                "source": t.get("source"),
            }
            for t in all_targets[:20]
        ]

        if all_targets and all_targets[0]["distance"] <= proximity_threshold:
            nearest_magnet = all_targets[0]

        # 5. Первый шаг + first_step_dir / first_step_reason (v8)
        first_step = None
        first_step_dir = None
        first_step_reason = None
        if nearest_magnet:
            direction = "вверх" if nearest_magnet["price"] > current_price else "вниз"
            first_step_dir = direction
            first_step_reason = (
                f"ближайший магнит {nearest_magnet['type']} на "
                f"{round(nearest_magnet['price'], 4)} "
                f"({round(nearest_magnet['distance'] / atr_last, 2)}×ATR, "
                f"≤ 0.6×ATR)"
            )
            first_step = {
                "direction": direction,
                "target": round(nearest_magnet["price"], 4),
                "type": nearest_magnet["type"],
                "distance_atr": round(nearest_magnet["distance"] / atr_last, 2),
                "reason": first_step_reason,
            }
        else:
            # Нет магнита в пределах 0.6×ATR — первый шаг определяется структурой
            first_step_reason = "нет магнита в радиусе 0.6×ATR — первый шаг по структуре"

        return {
            "open_fvgs": open_fvgs[-10:],
            "open_gaps": open_gaps[-10:],
            "liquidity_pools": liquidity_pools,
            "magnets": magnets_all,
            "nearest_magnet": nearest_magnet,
            "first_step": first_step,
            "first_step_dir": first_step_dir,
            "first_step_reason": first_step_reason,
            "current_price": current_price,
            "proximity_threshold_atr": 0.6,
        }
