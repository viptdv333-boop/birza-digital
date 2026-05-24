"""
Биржа-цифровой — Раздел 12: ТЕМП РЫНКА.

Тип: full.
K-темпа: k = Mean(TrueRange, 10-20) / ATR(ТФ)
Дневной ATR: ресемплинг -> ATR(14) по дневкам
Сроки: Дней = ceil(Dist% / (k * ATR_дн% * F))
v6: добавлен расчёт сроков (timeline) внутри секции.
"""
import math
import numpy as np
import pandas as pd
from sections.base import SectionProcessor
from core.utils import calc_atr


class TempoProcessor(SectionProcessor):
    section_id = 12
    section_emoji = "🔄"
    section_title = "ТЕМП РЫНКА"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        open_ = context["open_"]
        atr = context["atr"]
        atr_last = context["atr_last"]
        tf_hours = context.get("tf_hours", 1.0)
        n = len(close)
        current_price = float(close[-1])

        # ──────────────────────────────────────
        # 1. ATR текущего ТФ
        # ──────────────────────────────────────
        atr_valid = atr[~np.isnan(atr)]
        atr_tf = float(atr_valid[-1]) if len(atr_valid) > 0 else atr_last
        atr_tf_pct = atr_tf / current_price * 100

        # ──────────────────────────────────────
        # 2. Дневной ATR(14) + k_daily
        #    Группировка по дате (groupby, не resample — надёжнее).
        #    TR дневной = max(DH-DL, |DH-prev_close|, |DL-prev_close|)
        # ──────────────────────────────────────
        k_daily = None
        if tf_hours >= 24:
            atr_daily = atr_tf
        else:
            dfc = df.copy()
            dfc["_date"] = dfc["time"].dt.date
            daily = (
                dfc.groupby("_date")
                .agg(dh=("high", "max"), dl=("low", "min"), dc=("close", "last"))
                .reset_index()
            )
            daily["pc"] = daily["dc"].shift(1)
            daily["tr"] = np.maximum(
                daily["dh"] - daily["dl"],
                np.maximum(
                    (daily["dh"] - daily["pc"]).abs(),
                    (daily["dl"] - daily["pc"]).abs(),
                ),
            )
            daily["atr14"] = daily["tr"].rolling(14).mean()

            if len(daily) >= 15 and not np.isnan(daily["atr14"].iloc[-1]):
                atr_daily = float(daily["atr14"].iloc[-1])
                # k_daily: средний TR за 10 дней / ATR14
                daily_tr_10 = daily["tr"].iloc[-10:].values
                k_daily = float(np.nanmean(daily_tr_10) / atr_daily) if atr_daily > 0 else None
            else:
                # Мало дней — оценка из ATR(TF)
                atr_daily = atr_tf * (24 / tf_hours) ** 0.5

        atr_daily_pct = atr_daily / current_price * 100

        # ──────────────────────────────────────
        # 3. K-темпа по формуле v6:
        #    k = Mean(TrueRange, 10-20 баров) / ATR(ТФ)
        # ──────────────────────────────────────
        window = min(20, n - 1)
        true_ranges = np.maximum(
            high[-window:] - low[-window:],
            np.maximum(
                np.abs(high[-window:] - np.concatenate([[close[-window - 1]], close[-window:-1]])),
                np.abs(low[-window:] - np.concatenate([[close[-window - 1]], close[-window:-1]]))
            )
        )
        mean_tr = float(np.mean(true_ranges))
        # k_tempo: отношение среднего TR к ATR текущего ТФ (по регламенту)
        # k < 0.8 замедлен / 0.8-1.2 норма / > 1.2 перегрет
        k_tempo = mean_tr / atr_tf if atr_tf > 0 else 1.0
        k_tempo = round(k_tempo, 3)

        # Классификация
        if k_tempo >= 1.2:
            tempo_class = "перегрет"
        elif k_tempo < 0.8:
            tempo_class = "замедлен"
        else:
            tempo_class = "нормальный"

        # v8: подсказка роутеру — как k-темпа влияет на ранжирование целей
        #   k>=1.2   → второстепенные повышаются (promote_secondary), но не более 8
        #   k<0.8    → ключевые понижаются (demote_key)
        #   иначе    → нейтрально
        if k_tempo >= 1.2:
            k_influence_hint = "promote_secondary"
        elif k_tempo < 0.8:
            k_influence_hint = "demote_key"
        else:
            k_influence_hint = "neutral"

        # ──────────────────────────────────────
        # 4. ATR динамика
        # ──────────────────────────────────────
        if len(atr_valid) >= 10:
            atr_change = np.mean(atr_valid[-5:]) / np.mean(atr_valid[-10:-5]) - 1
            if atr_change > 0.1:
                atr_dynamics = "растёт"
            elif atr_change < -0.1:
                atr_dynamics = "падает"
            else:
                atr_dynamics = "стабильный"
        else:
            atr_dynamics = "недостаточно данных"

        # ──────────────────────────────────────
        # 5. Направленность
        # ──────────────────────────────────────
        bull_bars = int(np.sum(close[-20:] > open_[-20:])) if n >= 20 else 0
        bull_pct = round(bull_bars / min(20, n) * 100, 1)

        bodies = np.abs(close[-20:] - open_[-20:])
        ranges = high[-20:] - low[-20:]
        ranges[ranges == 0] = 0.001
        body_ratio = round(float(np.mean(bodies / ranges)), 3)

        # ──────────────────────────────────────
        # 6. Расчёт сроков (Timeline) — v6
        #    Дней = ceil(Dist% / (k * ATR_дн% * F))
        #    F: 1.0 тренд / 0.7 коррекция / 1.3 пробой
        #    Dist% = proxy-расстояние до ключевых целей
        #    (используем +-1 ATR и +-2 ATR как прокси)
        # ──────────────────────────────────────
        f_factors = {
            "тренд": 1.0,
            "коррекция": 0.7,
            "пробой": 1.3,
        }

        # Прокси-расстояния: 1 ATR и 2 ATR от текущей цены
        proxy_targets = {
            "1_ATR_up": current_price + atr_daily,
            "1_ATR_down": current_price - atr_daily,
            "2_ATR_up": current_price + 2 * atr_daily,
            "2_ATR_down": current_price - 2 * atr_daily,
        }

        timelines = {}
        for target_name, target_price in proxy_targets.items():
            dist_pct = abs(target_price - current_price) / current_price * 100
            target_days = {}
            for regime, f_val in f_factors.items():
                denominator = k_tempo * atr_daily_pct * f_val
                if denominator > 0:
                    days = math.ceil(dist_pct / denominator)
                else:
                    days = None  # невозможно оценить
                target_days[regime] = days
            timelines[target_name] = {
                "target_price": round(target_price, 4),
                "dist_pct": round(dist_pct, 2),
                "days_by_regime": target_days,
            }

        return {
            "atr_tf": round(atr_tf, 4),
            "atr_tf_pct": round(atr_tf_pct, 2),
            "atr_daily": round(atr_daily, 4),
            "atr_daily_pct": round(atr_daily_pct, 2),
            "atr_dynamics": atr_dynamics,
            "k_tempo": k_tempo,
            "k_daily": round(k_daily, 3) if k_daily is not None else None,
            "tempo_class": tempo_class,
            "body_ratio_20": body_ratio,
            "bull_bars_pct_20": bull_pct,
            "timelines": timelines,
            "f_factors": f_factors,
            "k_influence_hint": k_influence_hint,
        }
