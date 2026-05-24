"""
Биржа-цифровой — Раздел 18: КОРРЕЛЯЦИИ И КОНВЕРГЕНЦИЯ.

Тип: full. Close-Volume корреляция, ALMA-веер, автокорреляция.
"""
import numpy as np
from sections.base import SectionProcessor
from core.utils import calc_alma
from config import CORR_WINDOW


class CorrelationsProcessor(SectionProcessor):
    section_id = 18
    section_emoji = "🔗"
    section_title = "КОРРЕЛЯЦИИ И КОНВЕРГЕНЦИЯ"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        volume = context["volume"]
        available = context.get("available_cols", {})
        atr_last = context["atr_last"]
        n = len(close)

        # 1. Корреляция Close-Volume (50 баров)
        window = min(CORR_WINDOW, n - 1)
        close_w = close[-window:]
        vol_w = volume[-window:]

        if np.std(close_w) > 0 and np.std(vol_w) > 0:
            corr_cv = float(np.corrcoef(close_w, vol_w)[0, 1])
        else:
            corr_cv = 0.0

        vol_confirms = "подтверждает" if abs(corr_cv) > 0.3 else "не подтверждает"

        # 2. ALMA-веер
        alma_spreads = {}

        # ALMA 20
        if "alma" in available and "ALMA 20" in available["alma"]:
            alma20 = df["ALMA 20"].values.astype(float)
        elif n >= 20:
            alma20 = calc_alma(close, 20)
        else:
            alma20 = None

        # ALMA 50
        if "alma" in available and "ALMA 50" in available["alma"]:
            alma50 = df["ALMA 50"].values.astype(float)
        elif n >= 50:
            alma50 = calc_alma(close, 50)
        else:
            alma50 = None

        # ALMA 200
        if "alma" in available and "ALMA 200" in available["alma"]:
            alma200 = df["ALMA 200"].values.astype(float)
        elif n >= 200:
            alma200 = calc_alma(close, 200)
        else:
            alma200 = None

        # Spread 20-50
        if alma20 is not None and alma50 is not None:
            a20 = alma20[-1]
            a50 = alma50[-1]
            if not np.isnan(a20) and not np.isnan(a50):
                spread_2050 = (a20 - a50) / atr_last if atr_last > 0 else 0
                alma_spreads["spread_20_50_atr"] = round(float(spread_2050), 3)

                # Динамика (5 баров назад)
                if len(alma20) >= 5 and len(alma50) >= 5:
                    prev_spread = (alma20[-5] - alma50[-5]) / atr_last if atr_last > 0 else 0
                    if not np.isnan(prev_spread):
                        if spread_2050 > prev_spread + 0.05:
                            alma_spreads["dynamics_20_50"] = "расширение"
                        elif spread_2050 < prev_spread - 0.05:
                            alma_spreads["dynamics_20_50"] = "сужение"
                        else:
                            alma_spreads["dynamics_20_50"] = "стабильно"

        # Spread 50-200
        if alma50 is not None and alma200 is not None:
            a50 = alma50[-1]
            a200 = alma200[-1]
            if not np.isnan(a50) and not np.isnan(a200):
                spread_50200 = (a50 - a200) / atr_last if atr_last > 0 else 0
                alma_spreads["spread_50_200_atr"] = round(float(spread_50200), 3)

        # Конвергенция/дивергенция ALMA-веера
        fan_state = "нейтрально"
        if "spread_20_50_atr" in alma_spreads and "spread_50_200_atr" in alma_spreads:
            s1 = alma_spreads["spread_20_50_atr"]
            s2 = alma_spreads["spread_50_200_atr"]
            if s1 > 0 and s2 > 0:
                fan_state = "бычий веер (20>50>200)"
            elif s1 < 0 and s2 < 0:
                fan_state = "медвежий веер (20<50<200)"
            elif abs(s1) < 0.2 and abs(s2) < 0.2:
                fan_state = "конвергенция (ALMA сжимаются)"
            else:
                fan_state = "смешанный"

        # 3. Автокорреляция приращений (lag=1)
        returns = np.diff(close[-min(50, n):])
        if len(returns) >= 10:
            autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        else:
            autocorr = 0.0

        if autocorr > 0.1:
            regime = "трендовость"
        elif autocorr < -0.1:
            regime = "mean-reversion"
        else:
            regime = "случайность"

        return {
            "close_volume_corr": round(corr_cv, 3),
            "volume_confirms_movement": vol_confirms,
            "alma_spreads": alma_spreads,
            "alma_fan_state": fan_state,
            "autocorrelation_lag1": round(autocorr, 4),
            "market_regime": regime,
        }
