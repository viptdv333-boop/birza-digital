"""
Биржа-цифровой — Раздел 10: ВАЙКОФФ (по регламенту v6).

Алгоритмическая классификация:
- Тип: Накопление / Распределение
- Фаза A-E
- Узлы: SC, AR, ST, SPRING, TEST, SOS, LPS / BC, AR, ST, UT, UTAD, SOW, LPSY
"""
import numpy as np
from sections.base import SectionProcessor
from core.utils import calc_ad, calc_elder_force, calc_cmf


class WyckoffProcessor(SectionProcessor):
    section_id = 10
    section_emoji = "⚖"
    section_title = "ВАЙКОФФ"
    section_type = "partial"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        open_ = context["open_"]
        volume = context["volume"]
        available = context.get("available_cols", {})
        vp = context.get("vp", {})
        swing_points = context.get("swing_points", [])
        atr_last = context["atr_last"]
        n = len(close)
        current_price = float(close[-1])

        # ──────── Индикаторы ────────
        # EFI
        if "wyckoff" in available and "Elder Force Index" in available["wyckoff"]:
            efi = df["Elder Force Index"].values.astype(float)
        else:
            efi = calc_elder_force(close, volume, period=13)
        efi_valid = efi[~np.isnan(efi)]
        efi_current = float(efi_valid[-1]) if len(efi_valid) > 0 else 0

        # A/D
        if "wyckoff" in available and "Accumulation/Distribution" in available["wyckoff"]:
            ad = df["Accumulation/Distribution"].values.astype(float)
        else:
            ad = calc_ad(high, low, close, volume)
        ad_valid = ad[~np.isnan(ad)]

        # CMF
        if "wyckoff" in available and "CMF" in available["wyckoff"]:
            cmf = df["CMF"].values.astype(float)
        else:
            cmf = calc_cmf(high, low, close, volume, period=20)
        cmf_valid = cmf[~np.isnan(cmf)]
        cmf_current = float(cmf_valid[-1]) if len(cmf_valid) > 0 else 0

        # ──────── Определение типа: Накопление / Распределение ────────
        # По совокупности: A/D направление, CMF, позиция цены vs VP
        poc = vp.get("POC", current_price)
        val_price = vp.get("VAL", current_price * 0.95)
        vah_price = vp.get("VAH", current_price * 1.05)

        ad_slope = 0
        if len(ad_valid) >= 20:
            ad_slope = float(ad_valid[-1] - ad_valid[-20])

        # Направление цены за последние 20 баров (для контекста)
        price_slope_20 = 0.0
        if n >= 20:
            price_slope_20 = (close[-1] - close[-20]) / close[-20] * 100

        accumulation_score = 0
        distribution_score = 0

        # A/D растёт → накопление
        if ad_slope > 0:
            accumulation_score += 2
        else:
            distribution_score += 2

        # CMF > 0 → покупатели → накопление
        if cmf_current > 0.05:
            accumulation_score += 1
        elif cmf_current < -0.05:
            distribution_score += 1

        # EFI > 0 → сила покупателей
        if efi_current > 0:
            accumulation_score += 1
        else:
            distribution_score += 1

        # Ключевой фактор: тренд цены (приоритет над позицией vs POC)
        # Нисходящий импульс = распределение даже если цена ниже POC
        if price_slope_20 < -2.0:
            distribution_score += 3  # сильный медвежий тренд
        elif price_slope_20 > 2.0:
            accumulation_score += 3  # сильный бычий тренд
        else:
            # Позиция vs POC работает только в боковике
            if current_price < poc:
                accumulation_score += 1
            else:
                distribution_score += 1

        structure_type = "Накопление" if accumulation_score > distribution_score else "Распределение"

        # ──────── Определение фазы A-E ────────
        phase, phase_description = self._detect_phase(
            swing_points, close, high, low, volume, atr_last,
            structure_type, current_price, poc, val_price, vah_price
        )

        # ──────── Определение текущего узла ────────
        node = self._detect_node(
            swing_points, close, high, low, volume, atr_last,
            structure_type, phase, current_price
        )

        # ──────── Ближайший ожидаемый узел ────────
        next_node = self._predict_next_node(structure_type, phase, node)

        return {
            "structure_type": structure_type,
            "phase": phase,
            "phase_description": phase_description,
            "current_node": node,
            "next_node": next_node,
            "evidence": {
                "ad_slope_20": round(ad_slope, 2),
                "cmf": round(cmf_current, 4),
                "efi": round(efi_current, 2),
                "price_vs_poc": "ниже" if current_price < poc else "выше",
                "accumulation_score": accumulation_score,
                "distribution_score": distribution_score,
            },
            "volume_profile": {
                "POC": round(poc, 4),
                "VAH": round(vah_price, 4),
                "VAL": round(val_price, 4),
            },
            "current_price": current_price,
        }

    def _detect_phase(self, swings, close, high, low, volume, atr,
                      stype, price, poc, val_p, vah_p):
        """Определить фазу A-E."""
        n = len(close)
        if n < 50:
            return "A", "Недостаточно данных"

        # Волатильность последних 50 баров vs предыдущих 50
        recent_range = np.std(close[-50:])
        prev_range = np.std(close[-100:-50]) if n >= 100 else recent_range

        # Объём
        vol_recent = np.mean(volume[-20:])
        vol_prev = np.mean(volume[-50:-20]) if n >= 50 else vol_recent

        # Направление последних swing
        recent_swings = swings[-4:] if len(swings) >= 4 else swings

        if stype == "Накопление":
            # A: Кульминация продаж, падение замедляется
            # B: Формирование диапазона, тесты поддержки
            # C: SPRING / тест минимума
            # D: SOS, цена пробивает вверх
            # E: Финальный откат перед трендом

            if price < val_p:
                if vol_recent > vol_prev * 1.5:
                    return "A", "Кульминация продаж (SC). Высокий объём у поддержки"
                else:
                    return "C", "Тест минимума / SPRING. Цена ниже VAL"
            elif price < poc:
                if recent_range < prev_range * 0.7:
                    return "B", "Формирование диапазона. Волатильность сужается"
                else:
                    return "B", "Вторичный тест (ST). Проторговка у поддержки"
            elif price > poc and price < vah_p:
                return "D", "Признак силы (SOS). Цена выше POC"
            else:
                return "E", "Выход из диапазона. Цена выше VAH"
        else:
            # Распределение
            if price > vah_p:
                if vol_recent > vol_prev * 1.5:
                    return "A", "Кульминация покупок (BC). Высокий объём у сопротивления"
                else:
                    return "C", "Тест максимума / UT/UTAD. Цена выше VAH"
            elif price > poc:
                if recent_range < prev_range * 0.7:
                    return "B", "Формирование диапазона. Волатильность сужается"
                else:
                    return "B", "Вторичный тест (ST). Проторговка у сопротивления"
            elif price < poc and price > val_p:
                return "D", "Признак слабости (SOW). Цена ниже POC"
            else:
                return "E", "Выход из диапазона вниз. Цена ниже VAL"

    def _detect_node(self, swings, close, high, low, volume, atr,
                     stype, phase, price):
        """Определить текущий узел Вайкоффа."""
        n = len(close)
        if n < 20:
            return "—"

        vol_spike = np.mean(volume[-3:]) > np.mean(volume[-20:]) * 1.5
        price_at_low = price <= np.min(low[-20:]) * 1.01
        price_at_high = price >= np.max(high[-20:]) * 0.99

        if stype == "Накопление":
            if phase == "A" and vol_spike and price_at_low:
                return "SC (Selling Climax)"
            elif phase == "A":
                return "AR (Automatic Rally)"
            elif phase == "B":
                return "ST (Secondary Test)"
            elif phase == "C" and price_at_low:
                return "SPRING"
            elif phase == "C":
                return "TEST"
            elif phase == "D":
                return "SOS (Sign of Strength)"
            elif phase == "E":
                return "LPS (Last Point of Support)"
        else:
            if phase == "A" and vol_spike and price_at_high:
                return "BC (Buying Climax)"
            elif phase == "A":
                return "AR (Automatic Reaction)"
            elif phase == "B":
                return "ST (Secondary Test)"
            elif phase == "C" and price_at_high:
                return "UT (Upthrust) / UTAD"
            elif phase == "C":
                return "TEST"
            elif phase == "D":
                return "SOW (Sign of Weakness)"
            elif phase == "E":
                return "LPSY (Last Point of Supply)"
        return "—"

    def _predict_next_node(self, stype, phase, current_node):
        """Следующий ожидаемый узел."""
        if stype == "Накопление":
            seq = {"SC": "AR", "AR": "ST", "ST": "SPRING", "SPRING": "TEST",
                   "TEST": "SOS", "SOS": "LPS", "LPS": "Выход вверх"}
        else:
            seq = {"BC": "AR", "AR": "ST", "ST": "UT", "UT": "UTAD",
                   "UTAD": "SOW", "SOW": "LPSY", "LPSY": "Выход вниз"}

        for key, nxt in seq.items():
            if key in (current_node or ""):
                return nxt
        return "Следующий тест структуры"
