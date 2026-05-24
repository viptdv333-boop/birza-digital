"""
Биржа-цифровой — Раздел 15: ЭФФЕКТИВНОСТЬ ДВИЖЕНИЙ.

Тип: full. ER, Profit Factor, Win Rate, ADX(14) — v8.
"""
import numpy as np
from sections.base import SectionProcessor
from config import ER_PERIOD


def _calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              period: int = 14) -> dict:
    """Классический ADX(14) по Wilder.

    +DM = high[i]-high[i-1] если > low[i-1]-low[i] и > 0, иначе 0
    -DM = low[i-1]-low[i]   если > high[i]-high[i-1] и > 0, иначе 0
    TR  = max(H-L, |H-prev_close|, |L-prev_close|)
    Smoothed с Wilder (RMA).
    +DI = 100 * smooth(+DM) / smooth(TR)
    -DI = 100 * smooth(-DM) / smooth(TR)
    DX  = 100 * |+DI - -DI| / (+DI + -DI)
    ADX = RMA(DX, period)
    """
    n = len(close)
    if n < period * 2 + 1:
        return {"adx": None, "plus_di": None, "minus_di": None,
                "regime": "недостаточно данных", "trend_dir": None}

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    def _rma(arr, p):
        out = np.full(len(arr), np.nan)
        # инициализация — сумма первых p
        if len(arr) < p + 1:
            return out
        seed = float(np.sum(arr[1:p + 1]))
        out[p] = seed
        for i in range(p + 1, len(arr)):
            out[i] = out[i - 1] - (out[i - 1] / p) + arr[i]
        return out

    tr_s = _rma(tr, period)
    pdm_s = _rma(plus_dm, period)
    mdm_s = _rma(minus_dm, period)

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(tr_s[i]) and tr_s[i] > 0:
            plus_di[i] = 100.0 * pdm_s[i] / tr_s[i]
            minus_di[i] = 100.0 * mdm_s[i] / tr_s[i]

    dx = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            s = plus_di[i] + minus_di[i]
            if s > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s

    # ADX = Wilder smooth(DX)
    adx = np.full(n, np.nan)
    # seed в 2*period
    start_seed = 2 * period
    if n > start_seed:
        seed_slice = dx[period:start_seed]
        valid = seed_slice[~np.isnan(seed_slice)]
        if len(valid) > 0:
            adx[start_seed - 1] = float(np.mean(valid))
            for i in range(start_seed, n):
                if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                    adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    adx_cur = float(adx[-1]) if not np.isnan(adx[-1]) else None
    pdi_cur = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else None
    mdi_cur = float(minus_di[-1]) if not np.isnan(minus_di[-1]) else None

    # Классификация режима по v8: >25 тренд, <20 флэт, между — промежуточный
    if adx_cur is None:
        regime = "нет данных"
    elif adx_cur >= 25:
        regime = "тренд"
    elif adx_cur < 20:
        regime = "флэт"
    else:
        regime = "переходный"

    trend_dir = None
    if pdi_cur is not None and mdi_cur is not None:
        if pdi_cur > mdi_cur:
            trend_dir = "восходящий (+DI>-DI)"
        elif mdi_cur > pdi_cur:
            trend_dir = "нисходящий (-DI>+DI)"
        else:
            trend_dir = "нейтрально"

    return {
        "adx": round(adx_cur, 2) if adx_cur is not None else None,
        "plus_di": round(pdi_cur, 2) if pdi_cur is not None else None,
        "minus_di": round(mdi_cur, 2) if mdi_cur is not None else None,
        "regime": regime,
        "trend_dir": trend_dir,
        "period": period,
    }


class EfficiencyProcessor(SectionProcessor):
    section_id = 15
    section_emoji = "⚡"
    section_title = "ЭФФЕКТИВНОСТЬ ДВИЖЕНИЙ"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        open_ = context["open_"]
        high = context.get("high")
        low = context.get("low")
        n = len(close)

        # 1. ER / PF / WR для каждого окна (10, 20, 50, 100)
        windows = [10, 20, 50, 100]
        er_values = {}
        pf_values = {}
        wr_values = {}

        for N in windows:
            if n > N:
                seg = close[-N:]
                net = abs(float(seg[-1] - seg[0]))
                path = float(np.sum(np.abs(np.diff(seg))))
                er_values[N] = round(net / path if path > 0 else 0, 4)

                # Profit Factor: сумма ростов / сумма падений (bar-to-bar)
                deltas = np.diff(seg)
                bull = float(np.sum(deltas[deltas > 0]))
                bear = abs(float(np.sum(deltas[deltas < 0])))
                pf_values[N] = round(bull / bear if bear > 0 else float("inf"), 3)

                # Win Rate: доля бычьих баров (close > open)
                wr_values[N] = round(
                    float(np.sum(close[-N:] > open_[-N:])) / N * 100, 1
                )
            else:
                er_values[N] = None
                pf_values[N] = None
                wr_values[N] = None

        # Основной ER — по ER_PERIOD (default 20)
        period = min(ER_PERIOD, n - 1)
        er = er_values.get(period, 0) or 0

        # Порог ER: >0.5 тренд, 0.2-0.5 боковик, <0.2 шум
        if er > 0.5:
            er_class = "тренд"
        elif er > 0.2:
            er_class = "боковик"
        else:
            er_class = "шум"

        # Profit Factor / Win Rate основного периода
        profit_factor = pf_values.get(period, 0) or 0
        win_rate = wr_values.get(period, 50) or 50

        # Подтверждение тренда
        if er_class == "тренд" and win_rate > 55:
            confirmation = "подтверждает восходящий тренд"
        elif er_class == "тренд" and win_rate < 45:
            confirmation = "подтверждает нисходящий тренд"
        elif er_class == "боковик":
            confirmation = "боковик — тренд не подтверждён"
        else:
            confirmation = "шум — направление неопределено"

        # ADX(14) — v8
        adx_data = None
        if high is not None and low is not None:
            try:
                adx_data = _calc_adx(
                    np.asarray(high, dtype=float),
                    np.asarray(low, dtype=float),
                    np.asarray(close, dtype=float),
                    period=14,
                )
            except Exception:
                adx_data = None

        return {
            "efficiency_ratio": round(float(er), 4),
            "er_by_period": er_values,
            "pf_by_period": pf_values,
            "wr_by_period": wr_values,
            "er_classification": er_class,
            "profit_factor": round(float(profit_factor), 3),
            "win_rate_pct": round(float(win_rate), 1),
            "period": period,
            "confirmation": confirmation,
            "adx": adx_data,
        }
