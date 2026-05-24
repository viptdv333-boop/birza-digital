"""
Биржа-цифровой — Раздел 9: ОБЪЁМНЫЕ ЗОНЫ.

Тип: full. Два Volume Profile + TPO (Time Price Opportunity):
  Profile A — от якоря раздела 1 (trend start) до последнего бара.
  Profile B — от начала последнего значимого импульса до последнего бара.
  TPO — временной профиль рынка для каждого среза.
Thin areas = бины с объёмом < 20% от POC-бина (магниты).
Single Prints = уровни с 1 TPO (магниты для возврата).
"""
import numpy as np
import pandas as pd
from sections.base import SectionProcessor
from config import VP_BINS, VP_VAH_VAL_THRESHOLD, TPO_PERIOD_MINUTES, TPO_TICK_DIVISOR, TPO_VA_THRESHOLD

THIN_AREA_THRESHOLD = 0.20  # 20% от объёма POC-бина


class VolumeProfileProcessor(SectionProcessor):
    section_id = 9
    section_emoji = "📊"
    section_title = "ОБЪЁМНЫЕ ЗОНЫ"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        current_price = float(close[-1])
        atr_last = context["atr_last"]
        swing_points = context.get("swing_points", [])

        n = len(df)

        # ── ЯДРО v8.1 §IV: VP строится на ВСЁМ диапазоне глубины анализа ──
        # 1W = 6–12 мес / 1D = 2–3 мес / 4H = 15–25 дн / 1H = 50–60 дн / 15M = 5–7 дн
        # «Активная фаза» (волна/боковик/накопление/распределение) — это ОПИСАНИЕ
        # того, что в окне, а не якорь. Якорь = начало TF-окна.
        tf_hours = float(context.get("tf_hours", 1.0))
        # Используем тот же window helper, что и s01
        if tf_hours <= 0.25:    # 15M = 7 дней
            window_bars = 7 * 24 * 4
        elif tf_hours <= 1:     # 1H = 60 дней
            window_bars = 60 * 24
        elif tf_hours <= 4:     # 4H = 25 дней
            window_bars = 25 * 6
        elif tf_hours <= 24:    # 1D = 90 дней
            window_bars = 90
        else:                    # 1W = 52 недели (12 мес)
            window_bars = 52
        window_bars = min(window_bars, n)
        window_idx = max(0, n - window_bars)

        # Profile A — ВЕСЬ TF-диапазон (v8.1 §IV)
        # Profile B — старший тренд (если короче окна, иначе совпадёт с A)
        senior_idx = context.get("senior_trend_start_idx", context.get("trend_start_idx", 0))
        senior_idx = max(0, min(senior_idx, n - 1))
        # senior может быть РАНЬШЕ окна — обрезаем по правилу v8.1 «не использовать точки за окном»
        senior_idx = max(senior_idx, window_idx)

        # ── Определить ОПИСАНИЕ активной фазы (для контекста, не для якоря) ──
        s1_state = (context.get("s01_state") or "").lower()
        senior_dir = (context.get("trend_direction") or "").lower()
        local_dir = (context.get("current_direction") or "").lower()
        if "коррекция" in s1_state or (senior_dir and local_dir and senior_dir != local_dir):
            phase = "коррекция внутри старшего тренда"
        elif "боковик" in s1_state:
            phase = "боковик"
        elif local_dir == "восходящий":
            phase = "волна роста"
        elif local_dir == "нисходящий":
            phase = "волна падения"
        else:
            phase = "активная фаза"

        # ── Вычислить профили (Volume + TPO) ──
        profile_a = self._compute_profile(
            df.iloc[window_idx:], current_price, atr_last
        )
        profile_b = self._compute_profile(
            df.iloc[senior_idx:], current_price, atr_last
        )
        tpo_a = self._compute_tpo(
            df.iloc[window_idx:], current_price, atr_last
        )
        tpo_b = self._compute_tpo(
            df.iloc[senior_idx:], current_price, atr_last
        )

        return {
            "profile_a": {
                "description": (
                    f"полный TF-диапазон ({window_bars} баров, v8.1 §IV); "
                    f"текущая фаза: {phase}"
                ),
                "phase": phase,
                "anchor_reason": "tf_window (v8.1 §IV. Глубина анализа по ТФ)",
                "start_bar_index": int(window_idx),
                "bars_count": int(n - window_idx),
                "window_bars": int(window_bars),
                **profile_a,
                "tpo": tpo_a,
            },
            "profile_b": {
                "description": "старший тренд (от senior anchor, обрезан по TF-окну)",
                "anchor_reason": "senior_trend_start clipped to tf_window",
                "start_bar_index": int(senior_idx),
                "bars_count": int(n - senior_idx),
                **profile_b,
                "tpo": tpo_b,
            },
            "current_price": current_price,
        }

    @staticmethod
    def _compute_profile(data, current_price: float, atr_last: float) -> dict:
        """Вычислить Volume Profile для среза данных.

        Возвращает POC/VAH/VAL + thin_areas.
        """
        low_all = data["low"].values.astype(float)
        high_all = data["high"].values.astype(float)
        volume_all = data["volume"].values.astype(float)

        if len(data) < 2:
            return {
                "POC": _dist(current_price, current_price, atr_last),
                "VAH": _dist(current_price, current_price, atr_last),
                "VAL": _dist(current_price, current_price, atr_last),
                "nearest_magnet": "POC",
                "position": "недостаточно данных",
                "thin_areas": [],
            }

        price_min = float(low_all.min())
        price_max = float(high_all.max())
        if price_max <= price_min:
            price_max = price_min + 1e-8  # защита от деления на 0

        bins = VP_BINS
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        vol_per_bin = np.zeros(bins)

        for i in range(len(data)):
            lo, hi, vol = low_all[i], high_all[i], volume_all[i]
            if hi <= lo:
                idx = np.searchsorted(bin_edges, lo, side="right") - 1
                idx = max(0, min(idx, bins - 1))
                vol_per_bin[idx] += vol
            else:
                idx_lo = np.searchsorted(bin_edges, lo, side="right") - 1
                idx_hi = np.searchsorted(bin_edges, hi, side="right") - 1
                idx_lo = max(0, min(idx_lo, bins - 1))
                idx_hi = max(0, min(idx_hi, bins - 1))
                n_bins = idx_hi - idx_lo + 1
                for b in range(idx_lo, idx_hi + 1):
                    vol_per_bin[b] += vol / n_bins

        # POC
        poc_idx = int(np.argmax(vol_per_bin))
        poc = float(0.5 * (bin_edges[poc_idx] + bin_edges[poc_idx + 1]))

        # VAH / VAL (70% зона)
        total_vol = vol_per_bin.sum()
        target = total_vol * VP_VAH_VAL_THRESHOLD
        cum = vol_per_bin[poc_idx]
        lo_idx = poc_idx
        hi_idx = poc_idx
        while cum < target and (lo_idx > 0 or hi_idx < bins - 1):
            expand_lo = vol_per_bin[lo_idx - 1] if lo_idx > 0 else 0
            expand_hi = vol_per_bin[hi_idx + 1] if hi_idx < bins - 1 else 0
            if expand_lo >= expand_hi and lo_idx > 0:
                lo_idx -= 1
                cum += expand_lo
            elif hi_idx < bins - 1:
                hi_idx += 1
                cum += expand_hi
            else:
                lo_idx -= 1
                cum += expand_lo

        val = float(bin_edges[lo_idx])
        vah = float(bin_edges[hi_idx + 1])

        # ── Thin areas (магниты) ──
        poc_volume = vol_per_bin[poc_idx]
        thin_threshold = poc_volume * THIN_AREA_THRESHOLD
        thin_areas = []
        for b in range(bins):
            if vol_per_bin[b] < thin_threshold and vol_per_bin[b] > 0:
                mid_price = float(0.5 * (bin_edges[b] + bin_edges[b + 1]))
                thin_areas.append({
                    "price": round(mid_price, 4),
                    "volume_pct_of_poc": round(
                        vol_per_bin[b] / poc_volume * 100, 1
                    ) if poc_volume > 0 else 0.0,
                })
        # Также включить пустые бины между заполненными (нулевой объём)
        for b in range(bins):
            if vol_per_bin[b] == 0:
                # Проверить, что это не край (есть объём и выше, и ниже)
                has_below = any(vol_per_bin[j] > 0 for j in range(0, b))
                has_above = any(vol_per_bin[j] > 0 for j in range(b + 1, bins))
                if has_below and has_above:
                    mid_price = float(0.5 * (bin_edges[b] + bin_edges[b + 1]))
                    thin_areas.append({
                        "price": round(mid_price, 4),
                        "volume_pct_of_poc": 0.0,
                    })

        # Сортировка thin_areas по цене
        thin_areas.sort(key=lambda x: x["price"])

        # Расстояние до уровней
        poc_d = _dist(poc, current_price, atr_last)
        vah_d = _dist(vah, current_price, atr_last)
        val_d = _dist(val, current_price, atr_last)

        # Ближайший магнит
        distances = {
            "POC": abs(poc - current_price),
            "VAH": abs(vah - current_price),
            "VAL": abs(val - current_price),
        }
        nearest = min(distances, key=distances.get)

        # Положение цены относительно VA
        if current_price > vah:
            position = "выше VAH (вне зоны стоимости, перекуплено)"
        elif current_price < val:
            position = "ниже VAL (вне зоны стоимости, перепродано)"
        elif current_price > poc:
            position = "между POC и VAH (верхняя часть зоны стоимости)"
        else:
            position = "между VAL и POC (нижняя часть зоны стоимости)"

        return {
            "POC": poc_d,
            "VAH": vah_d,
            "VAL": val_d,
            "nearest_magnet": nearest,
            "position": position,
            "thin_areas": thin_areas,
        }


    @staticmethod
    def _compute_tpo(data, current_price: float, atr_last: float) -> dict:
        """Вычислить TPO (Time Price Opportunity) для среза данных.

        Группирует бары по TPO-периодам (30 мин по умолчанию),
        считает кол-во периодов на каждом ценовом уровне.
        Возвращает TPO POC/VAH/VAL, single prints, poor high/low, IB.
        """
        if len(data) < 2:
            return {"error": "недостаточно данных для TPO"}

        low_all = data["low"].values.astype(float)
        high_all = data["high"].values.astype(float)

        price_min = float(low_all.min())
        price_max = float(high_all.max())
        if price_max <= price_min:
            return {"error": "нулевой диапазон цен"}

        # Размер тика — адаптивный по диапазону
        tick_size = (price_max - price_min) / TPO_TICK_DIVISOR
        if tick_size <= 0:
            return {"error": "нулевой tick_size"}

        levels = np.arange(price_min, price_max + tick_size, tick_size)
        n_levels = len(levels)
        tpo_count = np.zeros(n_levels, dtype=int)

        # Группировка по TPO-периодам
        if "time" in data.columns:
            time_col = pd.to_datetime(data["time"])
        else:
            time_col = pd.to_datetime(data.index)

        # .floor() работает только на DatetimeIndex, не Series
        period_key = pd.DatetimeIndex(time_col).floor(f"{TPO_PERIOD_MINUTES}min")
        grouped = data.groupby(period_key)

        first_period_key = None
        ib_high = None
        ib_low = None

        for period_val, period_df in grouped:
            p_high = period_df["high"].max()
            p_low = period_df["low"].min()

            # Initial Balance — первый период
            if first_period_key is None:
                first_period_key = period_val
                ib_high = float(p_high)
                ib_low = float(p_low)

            # Отметить все уровни в диапазоне (один раз за период)
            idx_lo = np.searchsorted(levels, p_low, side="right") - 1
            idx_hi = np.searchsorted(levels, p_high, side="right") - 1
            idx_lo = max(0, min(idx_lo, n_levels - 1))
            idx_hi = max(0, min(idx_hi, n_levels - 1))
            tpo_count[idx_lo:idx_hi + 1] += 1

        # TPO POC — уровень с максимальным count
        tpo_poc_idx = int(np.argmax(tpo_count))
        tpo_poc = float(levels[tpo_poc_idx])

        # TPO Value Area (70%)
        total_tpo = int(tpo_count.sum())
        target = total_tpo * TPO_VA_THRESHOLD
        cum = tpo_count[tpo_poc_idx]
        lo_i = tpo_poc_idx
        hi_i = tpo_poc_idx
        while cum < target and (lo_i > 0 or hi_i < n_levels - 1):
            expand_lo = tpo_count[lo_i - 1] if lo_i > 0 else 0
            expand_hi = tpo_count[hi_i + 1] if hi_i < n_levels - 1 else 0
            if expand_lo >= expand_hi and lo_i > 0:
                lo_i -= 1
                cum += expand_lo
            elif hi_i < n_levels - 1:
                hi_i += 1
                cum += expand_hi
            else:
                lo_i -= 1
                cum += expand_lo

        tpo_val = float(levels[lo_i])
        tpo_vah = float(levels[hi_i])

        # Single Prints — уровни с ровно 1 TPO (магниты для возврата)
        active_mask = tpo_count > 0
        single_mask = tpo_count == 1
        singles = []
        for i in range(n_levels):
            if single_mask[i]:
                singles.append({
                    "price": round(float(levels[i]), 4),
                    "pct": round((float(levels[i]) - current_price) / current_price * 100, 2),
                })
        # Ограничить до 10 ближайших к цене
        singles.sort(key=lambda x: abs(x["price"] - current_price))
        singles = singles[:10]

        # Poor High / Poor Low — если крайний уровень имеет >1 TPO
        active_indices = np.where(active_mask)[0]
        if len(active_indices) >= 2:
            top_idx = active_indices[-1]
            bot_idx = active_indices[0]
            poor_high = bool(tpo_count[top_idx] > 1)
            poor_low = bool(tpo_count[bot_idx] > 1)
        else:
            poor_high = False
            poor_low = False

        return {
            "tpo_poc": _dist(tpo_poc, current_price, atr_last),
            "tpo_vah": _dist(tpo_vah, current_price, atr_last),
            "tpo_val": _dist(tpo_val, current_price, atr_last),
            "single_prints": singles,
            "poor_high": poor_high,
            "poor_low": poor_low,
            "initial_balance": {
                "high": round(ib_high, 4) if ib_high is not None else None,
                "low": round(ib_low, 4) if ib_low is not None else None,
            },
            "total_tpo_periods": int(len(grouped)) if 'grouped' in dir() else 0,
        }


def _dist(price: float, current_price: float, atr_last: float) -> dict:
    """Расстояние от уровня до текущей цены."""
    pct = (price - current_price) / current_price * 100 if current_price != 0 else 0
    atr_distance = abs(price - current_price) / atr_last if atr_last > 0 else 0
    return {
        "price": round(price, 4),
        "pct": round(pct, 2),
        "atr_distance": round(atr_distance, 2),
    }
