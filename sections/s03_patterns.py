"""
Биржа-цифровой — Раздел 3: ГРАФИЧЕСКИЕ ПАТТЕРНЫ.

Тип: partial (скрипт детектирует паттерны, ИИ уточняет).
"""
import numpy as np
from sections.base import SectionProcessor
from core.patterns import detect_patterns, _detect_wedge
from core.zigzag import zigzag, classify_swing_points
from core.linreg import linear_regression_channel


class PatternsProcessor(SectionProcessor):
    section_id = 3
    section_emoji = "🔺"
    section_title = "ГРАФИЧЕСКИЕ ПАТТЕРНЫ"
    section_type = "partial"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        swing_points = context.get("swing_points", [])
        atr_last = context["atr_last"]
        current_price = float(close[-1])

        # Алгоритмический детект (ZZ 5% swings)
        patterns = detect_patterns(swing_points, current_price, atr_last)

        # Дополнительная проверка клинов на более крупном масштабе (ZZ 10%)
        # ZZ 5% на 15m даёт слишком много шумных точек; клин виден на ZZ 10-20%
        has_wedge = any(
            "Wedge" in p.get("name", "") for p in patterns
        )
        if not has_wedge:
            times = df["time"].values if "time" in df.columns else None
            for coarse_dev in (10.0, 15.0, 20.0):
                zz_coarse = zigzag(high, low, coarse_dev, times)
                if len(zz_coarse) >= 6:
                    coarse_swings = classify_swing_points(zz_coarse)
                    wedges = _detect_wedge(coarse_swings, current_price, atr_last)
                    if wedges:
                        for w in wedges:
                            w["name_ru"] = w.get("name_ru", "") + f" (ZZ {coarse_dev:.0f}%)"
                        patterns.extend(wedges)
                        break  # Found wedge on coarser scale, stop

        # LinReg для контекста
        linreg = linear_regression_channel(close, period=50)

        # Диапазоны
        ranges = []
        for window in [10, 20, 50]:
            if len(close) >= window:
                segment = close[-window:]
                r = float((np.max(segment) - np.min(segment)) / current_price * 100)
                ranges.append({"window": window, "range_pct": round(r, 2)})

        return {
            "patterns": patterns,
            "pattern_count": len(patterns),
            "linreg_slope_pct": linreg["slope_pct"],
            "linreg_r_squared": linreg["r_squared"],
            "price_ranges": ranges,
            "current_price": current_price,
        }
