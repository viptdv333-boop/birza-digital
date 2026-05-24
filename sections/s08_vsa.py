"""
Биржа-цифровой — Раздел 8: VSA.

Тип: partial (скрипт считает V/L, OBV, CVD; ИИ интерпретирует).
"""
import numpy as np
from sections.base import SectionProcessor
from core.utils import calc_obv


class VSAProcessor(SectionProcessor):
    section_id = 8
    section_emoji = "🔊"
    section_title = "VSA"
    section_type = "partial"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        open_ = context["open_"]
        volume = context["volume"]
        available = context.get("available_cols", {})
        n = len(close)

        # 1. V/L ratio (Volume / (High-Low)) за последние 20 баров
        ranges = high - low
        ranges[ranges == 0] = 0.001
        vl_ratio = volume / ranges

        # Средний V/L
        window = min(20, n)
        vl_recent = vl_ratio[-window:]
        vl_avg = float(np.mean(vl_recent))
        vl_current = float(vl_ratio[-1])
        vl_relative = vl_current / vl_avg if vl_avg > 0 else 1.0

        # 2. Последние 10 баров: V/L + направление
        bars_vsa = []
        for i in range(max(0, n - 10), n):
            body = close[i] - open_[i]
            bar_range = high[i] - low[i]
            bars_vsa.append({
                "bar": i - (n - min(10, n)),
                "close": round(float(close[i]), 4),
                "volume": float(volume[i]),
                "range": round(float(bar_range), 4),
                "vl_ratio": round(float(vl_ratio[i]), 2),
                "direction": "bull" if body > 0 else "bear",
                "body_pct": round(abs(body) / bar_range * 100, 1) if bar_range > 0 else 0,
            })

        # 3. OBV
        obv = calc_obv(close, volume)
        obv_direction = "растёт" if obv[-1] > obv[-10] else "падает"
        obv_slope_5 = float(obv[-1] - obv[-5]) if n >= 5 else 0

        # 4. CVD из CSV
        cvd_data = {}
        if "cvd" in available and "CVD (Close)" in available["cvd"]:
            cvd = df["CVD (Close)"].values.astype(float)
            cvd_valid = cvd[~np.isnan(cvd)]
            if len(cvd_valid) >= 5:
                cvd_data = {
                    "current": round(float(cvd_valid[-1]), 2),
                    "direction": "покупатели" if cvd_valid[-1] > cvd_valid[-5] else "продавцы",
                    "change_5": round(float(cvd_valid[-1] - cvd_valid[-5]), 2),
                }

        # 5. Аномалии объёма
        vol_mean = np.mean(volume[-50:]) if n >= 50 else np.mean(volume)
        vol_std = np.std(volume[-50:]) if n >= 50 else np.std(volume)
        anomalies = []
        for i in range(max(0, n - 10), n):
            if vol_std > 0 and volume[i] > vol_mean + 2 * vol_std:
                body = close[i] - open_[i]
                bar_range = high[i] - low[i]
                anomalies.append({
                    "bar": i - (n - min(10, n)),
                    "volume_zscore": round((volume[i] - vol_mean) / vol_std, 2),
                    "direction": "bull" if body > 0 else "bear",
                    "body_pct": round(abs(body) / bar_range * 100, 1) if bar_range > 0 else 0,
                })

        return {
            "vl_current": round(vl_current, 2),
            "vl_avg_20": round(vl_avg, 2),
            "vl_relative": round(vl_relative, 3),
            "bars_vsa": bars_vsa,
            "obv_direction": obv_direction,
            "obv_slope_5": round(obv_slope_5, 2),
            "cvd": cvd_data,
            "volume_anomalies": anomalies,
        }
