"""
Биржа-цифровой — Раздел 16: ПОТОКОВЫЕ ИНДИКАТОРЫ.

Тип: full. MFI/CMF/CVD.
"""
import numpy as np
from sections.base import SectionProcessor
from core.utils import calc_mfi, calc_cmf


class FlowProcessor(SectionProcessor):
    section_id = 16
    section_emoji = "💰"
    section_title = "ПОТОКОВЫЕ ИНДИКАТОРЫ"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        volume = context["volume"]
        available = context.get("available_cols", {})
        n = len(close)

        # 1. MFI(14)
        mfi_val = None
        mfi_zone = "нейтрально"
        mfi_trend = None

        flow_cols = available.get("flow", [])
        mf_col = None
        for col_name in ["MF", "MF (Money Flow Index)"]:
            if col_name in flow_cols:
                mf_col = col_name
                break

        if mf_col:
            mfi_arr = df[mf_col].values.astype(float)
        else:
            mfi_arr = calc_mfi(high, low, close, volume, period=14)

        mfi_valid = mfi_arr[~np.isnan(mfi_arr)]
        if len(mfi_valid) > 0:
            mfi_val = round(float(mfi_valid[-1]), 2)
            if mfi_val > 80:
                mfi_zone = "перекупленность"
            elif mfi_val < 20:
                mfi_zone = "перепроданность"

            if len(mfi_valid) >= 5:
                mfi_trend = "растёт" if mfi_valid[-1] > mfi_valid[-5] else "падает"

        # 2. CMF(20)
        cmf_val = None
        cmf_signal = None

        if "wyckoff" in available and "CMF" in available["wyckoff"]:
            cmf_arr = df["CMF"].values.astype(float)
        else:
            cmf_arr = calc_cmf(high, low, close, volume, period=20)

        cmf_valid = cmf_arr[~np.isnan(cmf_arr)]
        if len(cmf_valid) > 0:
            cmf_val = round(float(cmf_valid[-1]), 4)
            cmf_signal = "покупатели" if cmf_val > 0 else "продавцы"

        cmf_trend = None
        if len(cmf_valid) >= 5:
            cmf_trend = "усиливается" if abs(cmf_valid[-1]) > abs(cmf_valid[-5]) else "ослабевает"

        # 3. CVD
        cvd_data = {}
        if "cvd" in available and "CVD (Close)" in available["cvd"]:
            cvd = df["CVD (Close)"].values.astype(float)
            cvd_valid = cvd[~np.isnan(cvd)]
            if len(cvd_valid) >= 5:
                # Показываем ОБА окна: 20 и 50 баров. Если знаки разные —
                # это сигнал перелома потока (развилка), флаг divergence.
                ref_20 = min(20, len(cvd_valid))
                ref_50 = min(50, len(cvd_valid))
                change_5 = float(cvd_valid[-1] - cvd_valid[-5])
                change_20 = float(cvd_valid[-1] - cvd_valid[-ref_20])
                change_50 = float(cvd_valid[-1] - cvd_valid[-ref_50])

                dir_20 = "покупатели" if change_20 > 0 else "продавцы"
                dir_50 = "покупатели" if change_50 > 0 else "продавцы"
                conflict = dir_20 != dir_50

                # Основное направление — по 50-барному окну (стратегия).
                # Если 20 и 50 согласны — сильный сигнал.
                if conflict:
                    direction = f"развилка (20: {dir_20}, 50: {dir_50})"
                else:
                    direction = dir_20

                cvd_data = {
                    "current": round(float(cvd_valid[-1]), 2),
                    "direction": direction,
                    "change_5": round(change_5, 2),
                    "change_20": round(change_20, 2),
                    "change_50": round(change_50, 2),
                    "dir_20": dir_20,
                    "dir_50": dir_50,
                    "conflict": conflict,
                }

        # 4. Дивергенции MFI/CMF vs цена
        divs = []
        if mfi_val is not None and len(mfi_valid) >= 20 and len(close) >= 20:
            price_up = close[-1] > close[-20]
            mfi_down = mfi_valid[-1] < mfi_valid[-20]
            if price_up and mfi_down:
                divs.append("MFI медвежья (цена↑, MFI↓)")
            elif not price_up and not mfi_down:
                divs.append("MFI бычья (цена↓, MFI↑)")

        if cmf_val is not None and len(cmf_valid) >= 20 and len(close) >= 20:
            price_up = close[-1] > close[-20]
            cmf_down = cmf_valid[-1] < cmf_valid[-20]
            if price_up and cmf_down:
                divs.append("CMF медвежья (цена↑, CMF↓)")
            elif not price_up and not cmf_down:
                divs.append("CMF бычья (цена↓, CMF↑)")

        # 5. Общий вывод
        signals = []
        if mfi_zone == "перекупленность":
            signals.append("MFI: перекупленность")
        elif mfi_zone == "перепроданность":
            signals.append("MFI: перепроданность")
        if cmf_signal:
            signals.append(f"CMF: {cmf_signal}")
        if cvd_data.get("direction"):
            signals.append(f"CVD: {cvd_data['direction']}")

        return {
            "mfi": {"value": mfi_val, "zone": mfi_zone, "trend": mfi_trend},
            "cmf": {"value": cmf_val, "signal": cmf_signal, "trend": cmf_trend},
            "cvd": cvd_data,
            "divergences": divs,
            "summary_signals": signals,
        }
