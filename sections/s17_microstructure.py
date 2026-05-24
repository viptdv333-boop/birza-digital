"""
Биржа-цифровой — Раздел 17: МИКРОСТРУКТУРА.

Тип: full. Bid-ask proxy, VWAP deviation, volume anomalies.
v6: Fallback VWAP из OHLCV, скан аномалий 50 баров.
"""
import numpy as np
from sections.base import SectionProcessor
from config import VOL_ANOMALY_ZSCORE


class MicrostructureProcessor(SectionProcessor):
    section_id = 17
    section_emoji = "🔬"
    section_title = "МИКРОСТРУКТУРА"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        open_ = context["open_"]
        volume = context["volume"]
        atr_last = context["atr_last"]
        available = context.get("available_cols", {})
        n = len(close)
        current_price = float(close[-1])

        # ═══════════════════════════════════════════════════
        # 1. Bid-Ask Proxy (изменение CVD / суммарный объём)
        # ═══════════════════════════════════════════════════
        bid_ask_proxy = None
        if "cvd" in available and "CVD (Close)" in available["cvd"]:
            cvd = df["CVD (Close)"].values.astype(float)
            cvd_valid = cvd[~np.isnan(cvd)]
            window = min(20, len(cvd_valid) - 1)
            if window > 0:
                cvd_change = float(cvd_valid[-1] - cvd_valid[-1 - window])
                total_vol = float(np.sum(volume[-window:]))
                if total_vol > 0:
                    bid_ask_proxy = round(cvd_change / total_vol, 4)

        # ═══════════════════════════════════════════════════
        # 2. VWAP — из CSV если есть, иначе fallback из OHLCV
        #    VWAP = cumsum(typical_price * volume) / cumsum(volume)
        # ═══════════════════════════════════════════════════
        vwap_data = {}
        vwap_source = None

        if "vwap" in available and "VWAP" in available["vwap"]:
            vwap = df["VWAP"].values.astype(float)
            vwap_valid = vwap[~np.isnan(vwap)]
            if len(vwap_valid) > 0:
                vwap_current = float(vwap_valid[-1])
                vwap_source = "csv"
            else:
                vwap_current = None
        else:
            vwap_current = None

        # Fallback: рассчитать VWAP из OHLCV
        if vwap_current is None:
            typical_price = (high + low + close) / 3.0
            cum_tp_vol = np.cumsum(typical_price * volume)
            cum_vol = np.cumsum(volume)
            # Избегаем деления на 0
            safe_cum_vol = np.where(cum_vol > 0, cum_vol, 1.0)
            vwap_array = cum_tp_vol / safe_cum_vol
            vwap_valid = vwap_array[~np.isnan(vwap_array)]
            if len(vwap_valid) > 0:
                vwap_current = float(vwap_valid[-1])
                vwap_source = "computed"

        if vwap_current is not None:
            deviation = current_price - vwap_current
            deviation_atr = deviation / atr_last if atr_last > 0 else 0
            vwap_data = {
                "vwap": round(vwap_current, 4),
                "deviation": round(deviation, 4),
                "deviation_atr": round(deviation_atr, 2),
                "position": "выше VWAP" if deviation > 0 else "ниже VWAP",
                "source": vwap_source,
            }

        # ═══════════════════════════════════════════════════
        # 3. Объёмные аномалии — скан 50 баров (v6)
        #    z-score рассчитывается по окну 50 баров
        # ═══════════════════════════════════════════════════
        anomaly_window = min(200, n)  # v6: окно 200 баров для z-score
        vol_window = volume[-anomaly_window:]
        vol_mean = float(np.mean(vol_window))
        vol_std = float(np.std(vol_window))

        anomalies = []
        scan_lookback = min(200, n)  # v6: сканируем 200 баров
        for i in range(n - scan_lookback, n):
            if vol_std > 0:
                zscore = (volume[i] - vol_mean) / vol_std
                if zscore > VOL_ANOMALY_ZSCORE:
                    body = close[i] - open_[i]
                    anomalies.append({
                        "bar_offset": n - 1 - i,
                        "volume_zscore": round(float(zscore), 2),
                        "direction": "bull" if body > 0 else "bear",
                        "close_price": round(float(close[i]), 4),
                    })

        # ═══════════════════════════════════════════════════
        # 4. Институциональный след
        # ═══════════════════════════════════════════════════
        institutional_signal = "отсутствует"
        if len(anomalies) >= 2:
            # Несколько аномалий одного направления
            bull_anom = sum(1 for a in anomalies if a["direction"] == "bull")
            bear_anom = sum(1 for a in anomalies if a["direction"] == "bear")
            if bull_anom >= 2 and bull_anom > bear_anom:
                institutional_signal = "вероятен вход крупного покупателя"
            elif bear_anom >= 2 and bear_anom > bull_anom:
                institutional_signal = "вероятен вход крупного продавца"
        elif len(anomalies) == 1:
            institutional_signal = "единичная аномалия (неопределённо)"

        # ═══════════════════════════════════════════════════
        # 5. Баланс спроса/предложения
        # ═══════════════════════════════════════════════════
        if bid_ask_proxy is not None:
            if bid_ask_proxy > 0.1:
                balance = "спрос преобладает"
            elif bid_ask_proxy < -0.1:
                balance = "предложение преобладает"
            else:
                balance = "баланс"
        else:
            balance = "нет данных CVD"

        return {
            "bid_ask_proxy": bid_ask_proxy,
            "vwap": vwap_data,
            "volume_anomalies": anomalies,
            "institutional_signal": institutional_signal,
            "supply_demand_balance": balance,
        }
