"""
Биржа-цифровой — Раздел 5: ДИВЕРГЕНЦИИ.

Тип: full (скрипт считает всё).
"""
import numpy as np
from sections.base import SectionProcessor
from core.utils import calc_rsi


def _calc_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     k_period: int = 14, d_period: int = 3,
                     smooth_k: int = 3) -> tuple:
    """Stochastic %K (slow) и %D.

    %K_raw = 100 * (close - LL) / (HH - LL)  за k_period
    %K     = SMA(%K_raw, smooth_k)
    %D     = SMA(%K, d_period)
    """
    n = len(close)
    k_raw = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        window_hi = np.max(high[i - k_period + 1:i + 1])
        window_lo = np.min(low[i - k_period + 1:i + 1])
        rng = window_hi - window_lo
        if rng > 0:
            k_raw[i] = 100.0 * (close[i] - window_lo) / rng
        else:
            k_raw[i] = 50.0

    def _sma(arr, p):
        out = np.full_like(arr, np.nan, dtype=float)
        for i in range(p - 1, len(arr)):
            w = arr[i - p + 1:i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) > 0:
                out[i] = float(np.mean(valid))
        return out

    k = _sma(k_raw, smooth_k)
    d = _sma(k, d_period)
    return k, d


class DivergencesProcessor(SectionProcessor):
    section_id = 5
    section_emoji = "📉"
    section_title = "ДИВЕРГЕНЦИИ"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        available = context.get("available_cols", {})
        n = len(close)

        # 1. RSI
        if "rsi" in available and "RSI" in available["rsi"]:
            rsi = df["RSI"].values.astype(float)
        else:
            rsi = calc_rsi(close, period=14)

        rsi_current = float(rsi[-1]) if not np.isnan(rsi[-1]) else None

        # Зона RSI (с учётом пограничных значений)
        rsi_zone = "нейтральная"
        if rsi_current is not None:
            if rsi_current >= 70:
                rsi_zone = "перекупленность"
            elif rsi_current >= 65:
                rsi_zone = "близко к перекупленности"
            elif rsi_current <= 30:
                rsi_zone = "перепроданность"
            elif rsi_current <= 35:
                rsi_zone = "близко к перепроданности"

        # RSI тренд (направление за последние 10 баров)
        rsi_valid = rsi[~np.isnan(rsi)]
        rsi_trend = "нейтральный"
        if len(rsi_valid) >= 10:
            rsi_10 = rsi_valid[-10:]
            if rsi_10[-1] > rsi_10[0] + 3:
                rsi_trend = "растущий"
            elif rsi_10[-1] < rsi_10[0] - 3:
                rsi_trend = "падающий"

        # 2. Дивергенции из CSV (Regular/Hidden Bullish/Bearish)
        div_data = {}
        div_cols = {
            "regular_bullish": "Regular Bullish",
            "hidden_bullish": "Hidden Bullish",
            "regular_bearish": "Regular Bearish",
            "hidden_bearish": "Hidden Bearish",
        }

        has_csv_divs = "rsi_divergences" in available

        # Проверяем, какие конкретные колонки есть в CSV
        csv_available_keys = set()
        for key, col_name in div_cols.items():
            if has_csv_divs and col_name in available.get("rsi_divergences", []):
                csv_available_keys.add(key)
                col_data = df[col_name].values
                # Найти последний сигнал (ненулевое значение)
                signals = np.where(~np.isnan(col_data) & (col_data != 0))[0]
                if len(signals) > 0:
                    last_idx = signals[-1]
                    bars_since = n - 1 - last_idx
                    div_data[key] = {
                        "detected": True,
                        "bars_since": int(bars_since),
                        "price_at_signal": float(close[last_idx]),
                    }
                else:
                    div_data[key] = {"detected": False}

        # 3. Собственный детект дивергенций для тех типов, которых нет в CSV
        self_detected = self._detect_divergences(close, rsi)

        # Для каждого типа: если не было в CSV — берём self-detected
        for key in div_cols:
            if key not in csv_available_keys:
                if key in self_detected:
                    div_data[key] = self_detected[key]
                else:
                    div_data[key] = {"detected": False}

        # 3b. Фильтр взаимоисключения:
        # Regular Bullish и Regular Bearish не могут быть одновременно
        rb = div_data.get("regular_bullish", {})
        rbe = div_data.get("regular_bearish", {})
        if rb.get("detected") and rbe.get("detected"):
            # Оставляем более свежий (меньше bars_since)
            if rb.get("bars_since", 999) <= rbe.get("bars_since", 999):
                div_data["regular_bearish"] = {"detected": False}
            else:
                div_data["regular_bullish"] = {"detected": False}

        # Hidden Bullish и Hidden Bearish не могут быть одновременно
        hb = div_data.get("hidden_bullish", {})
        hbe = div_data.get("hidden_bearish", {})
        if hb.get("detected") and hbe.get("detected"):
            if hb.get("bars_since", 999) <= hbe.get("bars_since", 999):
                div_data["hidden_bearish"] = {"detected": False}
            else:
                div_data["hidden_bullish"] = {"detected": False}

        # Regular и Hidden одного направления тоже конфликтуют:
        # Regular Bullish (разворот вверх) + Hidden Bearish (продолжение вниз) = конфликт
        # Оставляем regular как более сильный сигнал
        if rb.get("detected") and hbe.get("detected"):
            div_data["hidden_bearish"] = {"detected": False}
        if rbe.get("detected") and hb.get("detected"):
            div_data["hidden_bullish"] = {"detected": False}

        # 4. CVD дивергенция
        cvd_div = None
        if "cvd" in available and "CVD (Close)" in available["cvd"]:
            cvd = df["CVD (Close)"].values.astype(float)
            cvd_valid = cvd[~np.isnan(cvd)]
            if len(cvd_valid) >= 20:
                # Простая проверка: цена растёт, CVD падает = медвежья дивергенция
                price_change = close[-1] - close[-20]
                cvd_change = cvd_valid[-1] - cvd_valid[-20] if len(cvd_valid) >= 20 else 0

                if price_change > 0 and cvd_change < 0:
                    cvd_div = "медвежья (цена↑, CVD↓)"
                elif price_change < 0 and cvd_change > 0:
                    cvd_div = "бычья (цена↓, CVD↑)"
                else:
                    cvd_div = "нет"

        # 5. Согласованность
        active_bull = any(
            d.get("detected") and d.get("bars_since", 999) < 20
            for k, d in div_data.items() if "bullish" in k
        )
        active_bear = any(
            d.get("detected") and d.get("bars_since", 999) < 20
            for k, d in div_data.items() if "bearish" in k
        )

        signal = "нейтрально"
        if active_bull and not active_bear:
            signal = "бычий разворот"
        elif active_bear and not active_bull:
            signal = "медвежий разворот"
        elif active_bull and active_bear:
            signal = "конфликт сигналов"

        # 6. MACD — критичный для определения разворотов
        macd_data = None
        oscs = available.get("oscillators", [])
        if "MACD" in oscs and "Signal line" in oscs and "Histogram" in oscs:
            macd_arr = df["MACD"].values.astype(float)
            sig_arr = df["Signal line"].values.astype(float)
            hist_arr = df["Histogram"].values.astype(float)

            m_val = float(macd_arr[-1]) if not np.isnan(macd_arr[-1]) else None
            s_val = float(sig_arr[-1]) if not np.isnan(sig_arr[-1]) else None
            h_val = float(hist_arr[-1]) if not np.isnan(hist_arr[-1]) else None

            if m_val is not None and s_val is not None and h_val is not None:
                # Позиция MACD vs ноль
                macd_pos = "выше нуля" if m_val > 0 else "ниже нуля"
                # Кросс
                macd_cross = "бычий (MACD>Signal)" if m_val > s_val else "медвежий (MACD<Signal)"

                # Гистограмма: направление за последние 5 баров
                hist_valid = hist_arr[~np.isnan(hist_arr)]
                hist_trend = "нейтрально"
                hist_reversal = None
                if len(hist_valid) >= 5:
                    h5 = hist_valid[-5:]
                    if h5[-1] > h5[-3] > h5[-5]:
                        hist_trend = "растёт"
                    elif h5[-1] < h5[-3] < h5[-5]:
                        hist_trend = "падает"
                    else:
                        hist_trend = "неопределено"

                    # Разворот: гистограмма меняет знак или достигла локального экстремума
                    # и начала возвращаться
                    if len(hist_valid) >= 10:
                        h10 = hist_valid[-10:]
                        min_idx = int(np.argmin(h10))
                        max_idx = int(np.argmax(h10))
                        # Разворот вверх: минимум был в середине и сейчас растёт
                        if min_idx < 8 and h10[-1] > h10[min_idx] + abs(h10[min_idx]) * 0.2:
                            if h10[-1] < 0:
                                hist_reversal = "бычий разворот (гистограмма растёт из минимума)"
                            else:
                                hist_reversal = "бычий импульс"
                        elif max_idx < 8 and h10[-1] < h10[max_idx] - abs(h10[max_idx]) * 0.2:
                            if h10[-1] > 0:
                                hist_reversal = "медвежий разворот (гистограмма падает от максимума)"
                            else:
                                hist_reversal = "медвежий импульс"

                macd_data = {
                    "macd": round(m_val, 4),
                    "signal": round(s_val, 4),
                    "histogram": round(h_val, 4),
                    "position": macd_pos,
                    "cross": macd_cross,
                    "histogram_trend": hist_trend,
                    "histogram_reversal": hist_reversal,
                }

        # 7. STOCHASTIC (%K/%D) + дивергенция по Stochastic — v8
        stoch_data = None
        try:
            k_arr, d_arr = _calc_stochastic(high, low, close,
                                            k_period=14, d_period=3, smooth_k=3)
            k_cur = float(k_arr[-1]) if not np.isnan(k_arr[-1]) else None
            d_cur = float(d_arr[-1]) if not np.isnan(d_arr[-1]) else None
            if k_cur is not None and d_cur is not None:
                if k_cur >= 80:
                    stoch_zone = "перекупленность"
                elif k_cur <= 20:
                    stoch_zone = "перепроданность"
                else:
                    stoch_zone = "нейтральная"
                stoch_cross = "бычий (%K>%D)" if k_cur > d_cur else "медвежий (%K<%D)"
                # дивергенция Stochastic vs price — по тем же локальным экстремумам
                stoch_div = self._divergence_vs_price(close, k_arr)
                stoch_data = {
                    "k": round(k_cur, 2),
                    "d": round(d_cur, 2),
                    "zone": stoch_zone,
                    "cross": stoch_cross,
                    "divergence": stoch_div,
                }
        except Exception:
            stoch_data = None

        # 8. MACD дивергенция (по гистограмме/MACD vs price) — v8
        macd_divergence = None
        if macd_data is not None:
            oscs2 = available.get("oscillators", [])
            if "MACD" in oscs2:
                macd_arr_full = df["MACD"].values.astype(float)
                macd_divergence = self._divergence_vs_price(close, macd_arr_full)

        if macd_data is not None:
            macd_data["divergence"] = macd_divergence

        return {
            "rsi_current": rsi_current,
            "rsi_zone": rsi_zone,
            "rsi_trend": rsi_trend,
            "macd": macd_data,
            "macd_divergence": macd_divergence,
            "stochastic": stoch_data,
            "divergences": div_data,
            "cvd_divergence": cvd_div,
            "signal": signal,
        }

    def _divergence_vs_price(self, close: np.ndarray, ind: np.ndarray) -> str | None:
        """Ищет regular bullish / bearish дивергенцию индикатора vs цены
        на последних 50 барах. Возвращает строку или None.
        """
        n = len(close)
        lookback = min(50, n - 1)
        if lookback < 10:
            return None
        seg_p = close[-lookback:]
        seg_i = ind[-lookback:]
        valid = ~np.isnan(seg_i)
        if valid.sum() < 10:
            return None
        # Локальные экстремумы с окном 5
        window = 5
        lows_idx, highs_idx = [], []
        for i in range(window, len(seg_p) - window):
            if not valid[i]:
                continue
            if seg_p[i] == np.min(seg_p[max(0, i - window):i + window + 1]):
                lows_idx.append(i)
            if seg_p[i] == np.max(seg_p[max(0, i - window):i + window + 1]):
                highs_idx.append(i)
        # Regular bullish: price LL, ind HL
        if len(lows_idx) >= 2:
            i1, i2 = lows_idx[-2], lows_idx[-1]
            if seg_p[i2] < seg_p[i1] and seg_i[i2] > seg_i[i1]:
                return f"бычья (цена LL, индикатор HL, {lookback - 1 - i2} баров назад)"
        # Regular bearish: price HH, ind LH
        if len(highs_idx) >= 2:
            i1, i2 = highs_idx[-2], highs_idx[-1]
            if seg_p[i2] > seg_p[i1] and seg_i[i2] < seg_i[i1]:
                return f"медвежья (цена HH, индикатор LH, {lookback - 1 - i2} баров назад)"
        return "нет"

    def _detect_divergences(self, close: np.ndarray, rsi: np.ndarray) -> dict:
        """Детект regular + hidden дивергенций по экстремумам цены и RSI.

        Regular Bullish:  price LL, RSI HL  (momentum slowing on new low)
        Regular Bearish:  price HH, RSI LH  (momentum slowing on new high)
        Hidden Bullish:   price HL, RSI LL  (trend continuation signal up)
        Hidden Bearish:   price LH, RSI HH  (trend continuation signal down)
        """
        n = len(close)
        result = {}

        # Ищем последние 2 значимых лоу и хая (lookback 50 баров)
        lookback = min(50, n - 1)
        segment_close = close[-lookback:]
        segment_rsi = rsi[-lookback:]

        valid_mask = ~np.isnan(segment_rsi)
        if valid_mask.sum() < 10:
            return result

        # Скользящее окно для поиска локальных экстремумов
        window = 5
        lows_idx = []
        highs_idx = []

        for i in range(window, len(segment_close) - window):
            if valid_mask[i]:
                if segment_close[i] == np.min(segment_close[max(0, i - window):i + window + 1]):
                    lows_idx.append(i)
                if segment_close[i] == np.max(segment_close[max(0, i - window):i + window + 1]):
                    highs_idx.append(i)

        # --- Regular Bullish: price LL, RSI HL ---
        if len(lows_idx) >= 2:
            i1, i2 = lows_idx[-2], lows_idx[-1]
            if (segment_close[i2] < segment_close[i1] and
                    segment_rsi[i2] > segment_rsi[i1]):
                result["regular_bullish"] = {
                    "detected": True,
                    "bars_since": len(segment_close) - 1 - i2,
                    "price_at_signal": float(segment_close[i2]),
                }

        # --- Regular Bearish: price HH, RSI LH ---
        if len(highs_idx) >= 2:
            i1, i2 = highs_idx[-2], highs_idx[-1]
            if (segment_close[i2] > segment_close[i1] and
                    segment_rsi[i2] < segment_rsi[i1]):
                result["regular_bearish"] = {
                    "detected": True,
                    "bars_since": len(segment_close) - 1 - i2,
                    "price_at_signal": float(segment_close[i2]),
                }

        # --- Hidden Bullish: price HL, RSI LL ---
        if len(lows_idx) >= 2:
            i1, i2 = lows_idx[-2], lows_idx[-1]
            if (segment_close[i2] > segment_close[i1] and
                    segment_rsi[i2] < segment_rsi[i1]):
                result["hidden_bullish"] = {
                    "detected": True,
                    "bars_since": len(segment_close) - 1 - i2,
                    "price_at_signal": float(segment_close[i2]),
                }

        # --- Hidden Bearish: price LH, RSI HH ---
        if len(highs_idx) >= 2:
            i1, i2 = highs_idx[-2], highs_idx[-1]
            if (segment_close[i2] < segment_close[i1] and
                    segment_rsi[i2] > segment_rsi[i1]):
                result["hidden_bearish"] = {
                    "detected": True,
                    "bars_since": len(segment_close) - 1 - i2,
                    "price_at_signal": float(segment_close[i2]),
                }

        return result
