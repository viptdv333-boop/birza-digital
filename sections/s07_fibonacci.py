"""
Биржа-цифровой — Раздел 7: ФИБОНАЧЧИ (ЯДРО v8.1).

Задание v8.1: «Построй сетку Фибо по актуальному тренду».
ОДНА сетка от якоря раздела 1 (trend start) до пика тренда.
Уровни коррекции 0.236/0.382/0.5/0.618/0.786, расширения 1.272/1.618/2.0/2.618.
"""
import numpy as np
from sections.base import SectionProcessor
from config import FIBO_RETRACEMENT, FIBO_EXTENSION


class FibonacciProcessor(SectionProcessor):
    section_id = 7
    section_emoji = "📐"
    section_title = "ФИБОНАЧЧИ"
    section_type = "full"

    # ── helpers ──
    @staticmethod
    def _fibo_grid(start: float, end: float, current_price: float,
                   retracement_levels, extension_levels) -> dict:
        """Build retracement + extension lists for one Fibonacci grid."""
        diff = end - start
        is_uptrend = end > start

        retracements = []
        for r in retracement_levels:
            price = end - diff * r
            pct = (price - current_price) / current_price * 100
            retracements.append({
                "level": r,
                "price": round(price, 4),
                "pct_from_current": round(pct, 2),
            })

        extensions = []
        for e in extension_levels:
            if diff > 0:
                price = end + diff * (e - 1)
            else:
                price = end - abs(diff) * (e - 1)
            pct = (price - current_price) / current_price * 100
            extensions.append({
                "level": e,
                "price": round(price, 4),
                "pct_from_current": round(pct, 2),
            })

        return {
            "direction": "восходящий" if is_uptrend else "нисходящий",
            "start": round(start, 4),
            "end": round(end, 4),
            "retracements": retracements,
            "extensions": extensions,
        }

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        current_price = float(close[-1])
        n = len(close)

        trend_direction = context.get("trend_direction", "")
        trend_start_idx = context.get("trend_start_idx")
        trend_peak_idx = context.get("trend_peak_idx")

        # ── Сетка A: senior trend (trend_start → trend_peak) ──
        if trend_start_idx is not None and trend_direction:
            if trend_direction == "нисходящий":
                s_start = float(high[trend_start_idx])
                s_end = float(low[trend_peak_idx]) if trend_peak_idx is not None else float(np.min(low[trend_start_idx:]))
            elif trend_direction == "восходящий":
                s_start = float(low[trend_start_idx])
                s_end = float(high[trend_peak_idx]) if trend_peak_idx is not None else float(np.max(high[trend_start_idx:]))
            else:
                s_start = float(np.min(low))
                s_end = float(np.max(high))
        else:
            global_low = float(np.min(low))
            global_high = float(np.max(high))
            low_idx = int(np.argmin(low))
            high_idx = int(np.argmax(high))
            if low_idx < high_idx:
                s_start, s_end = global_low, global_high
            else:
                s_start, s_end = global_high, global_low

        senior = self._fibo_grid(s_start, s_end, current_price,
                                 FIBO_RETRACEMENT, FIBO_EXTENSION)

        # v8.1: одна сетка по актуальному тренду. Локальная убрана как лишнее.
        return {
            "senior_trend": senior,
            "current_price": current_price,
        }
