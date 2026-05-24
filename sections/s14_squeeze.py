"""
Биржа-цифровой — Раздел 14: BOLLINGER / KELTNER / SQUEEZE.

Тип: full.
"""
import numpy as np
from sections.base import SectionProcessor
from core.utils import calc_ema, calc_sma, calc_atr
from config import BB_PERIOD, BB_STD, KC_PERIOD, KC_ATR_PERIOD, KC_MULT


class SqueezeProcessor(SectionProcessor):
    section_id = 14
    section_emoji = "📊"
    section_title = "BOLLINGER / KELTNER / SQUEEZE"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        available = context.get("available_cols", {})
        n = len(close)
        current_price = float(close[-1])

        # 1. Bollinger Bands
        if "bollinger" in available:
            bb_upper = df["Upper"].values.astype(float) if "Upper" in df.columns else None
            bb_basis = df["Basis"].values.astype(float) if "Basis" in df.columns else None
            bb_lower = df["Lower"].values.astype(float) if "Lower" in df.columns else None
        else:
            bb_upper = bb_basis = bb_lower = None

        if bb_basis is None:
            sma = calc_sma(close, BB_PERIOD)
            std = np.full_like(close, np.nan)
            for i in range(BB_PERIOD - 1, n):
                std[i] = np.std(close[i - BB_PERIOD + 1:i + 1])
            bb_basis = sma
            bb_upper = sma + BB_STD * std
            bb_lower = sma - BB_STD * std

        # %B
        bb_width_arr = bb_upper - bb_lower
        last_width = float(bb_width_arr[-1]) if not np.isnan(bb_width_arr[-1]) else 1
        pct_b = (current_price - float(bb_lower[-1])) / last_width if last_width > 0 else 0.5

        # BB Width
        bb_width = last_width / float(bb_basis[-1]) * 100 if not np.isnan(bb_basis[-1]) and bb_basis[-1] > 0 else 0

        # BB Width Percentile (за 100 баров)
        valid_widths = bb_width_arr[~np.isnan(bb_width_arr)]
        if len(valid_widths) >= 20:
            window = min(100, len(valid_widths))
            pct_rank = float(np.searchsorted(np.sort(valid_widths[-window:]), bb_width_arr[-1]) / window)
        else:
            pct_rank = 0.5

        # 2. Keltner Channel
        kc_mid = calc_ema(close, KC_PERIOD)
        kc_atr = calc_atr(high, low, close, KC_ATR_PERIOD)
        kc_upper = kc_mid + KC_MULT * kc_atr
        kc_lower = kc_mid - KC_MULT * kc_atr

        # 3. Squeeze Detection: BB внутри KC = Squeeze
        squeeze_active = False
        if not np.isnan(bb_upper[-1]) and not np.isnan(kc_upper[-1]):
            squeeze_active = (bb_upper[-1] < kc_upper[-1]) and (bb_lower[-1] > kc_lower[-1])

        # Фаза волатильности
        if squeeze_active:
            vol_phase = "сжатие (Squeeze активен)"
        elif pct_rank > 0.8:
            vol_phase = "расширение"
        else:
            vol_phase = "нормальная"

        # Squeeze history (последние 10 баров)
        squeeze_bars = 0
        for i in range(n - 1, max(0, n - 30), -1):
            if (not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i]) and
                    bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]):
                squeeze_bars += 1
            else:
                break

        return {
            "bollinger": {
                "upper": round(float(bb_upper[-1]), 4) if not np.isnan(bb_upper[-1]) else None,
                "basis": round(float(bb_basis[-1]), 4) if not np.isnan(bb_basis[-1]) else None,
                "lower": round(float(bb_lower[-1]), 4) if not np.isnan(bb_lower[-1]) else None,
                "pct_b": round(float(pct_b), 3),
                "bb_width": round(float(bb_width), 3),
                "width_percentile": round(float(pct_rank), 2),
            },
            "keltner": {
                "upper": round(float(kc_upper[-1]), 4) if not np.isnan(kc_upper[-1]) else None,
                "mid": round(float(kc_mid[-1]), 4) if not np.isnan(kc_mid[-1]) else None,
                "lower": round(float(kc_lower[-1]), 4) if not np.isnan(kc_lower[-1]) else None,
            },
            "squeeze_active": squeeze_active,
            "squeeze_bars": squeeze_bars,
            "vol_phase": vol_phase,
            "current_price": current_price,
        }
