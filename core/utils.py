"""
Математическая модель v2.6.3 — утилиты.
"""
import numpy as np


def human_time(bars: float, tf_hours: float) -> str:
    """Перевод количества баров в человекочитаемое время.

    Минимум = 1 бар × tf_hours (не может быть меньше одного бара).
    Слово «свечей» ЗАПРЕЩЕНО (модель v2.6.3, п.1.3.4).
    """
    bars = max(bars, 1.0)  # минимум 1 бар
    total_hours = bars * tf_hours
    # Минимум = 1 бар ТФ
    total_hours = max(total_hours, tf_hours)

    if total_hours < 24:
        h = round(total_hours)
        h = max(h, round(tf_hours))  # не меньше 1 бара
        return f"~{h} ч"
    else:
        d = round(total_hours / 24, 1)
        if d == int(d):
            return f"~{int(d)} дн"
        return f"~{d} дн"


def detect_tf_hours(time_series) -> float:
    """Определить таймфрейм по медиане шага между свечами (в часах).

    Использует dtype для определения единицы измерения времени, а не
    порядок величины (раньше 4H = 1.44e13 ns ошибочно классифицировался
    как микросекунды и давал ~4000 ч → метку 1W).
    """
    import pandas as pd
    ts = time_series
    # Приводим к pandas datetime если ещё не
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    ts = pd.to_datetime(ts)
    # Timedelta в секундах через pandas — надёжно вне зависимости от units
    diffs = ts.diff().dropna().dt.total_seconds()
    diffs_positive = diffs[diffs > 0]
    if len(diffs_positive) == 0:
        return 1.0
    median_seconds = float(diffs_positive.median())
    return median_seconds / 3600.0


def tf_label(tf_hours: float) -> str:
    """Человекочитаемая метка ТФ."""
    if tf_hours < 1:
        return f"{int(tf_hours * 60)}m"
    elif tf_hours < 24:
        h = int(tf_hours) if tf_hours == int(tf_hours) else tf_hours
        return f"{h}H"
    elif tf_hours < 168:
        return f"{int(tf_hours / 24)}D"
    else:
        return f"{int(tf_hours / 168)}W"


def snap_to_level(price: float, levels: list[float], max_dist: float) -> float | None:
    """Привязка цены к ближайшему уровню, если расстояние < max_dist."""
    if not levels:
        return None
    dists = np.abs(np.array(levels) - price)
    idx = np.argmin(dists)
    if dists[idx] < max_dist:
        return levels[idx]
    return None


def robust_standardize(X: np.ndarray) -> np.ndarray:
    """Робастная стандартизация: медиана и MAD × 1.4826."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    mad[mad < 1e-10] = 1.0
    return (X - med) / mad


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> np.ndarray:
    """ATR(period) — Average True Range."""
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = np.full_like(tr, np.nan)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calc_ema(data: np.ndarray, period: int) -> np.ndarray:
    """EMA — Exponential Moving Average."""
    ema = np.full_like(data, np.nan, dtype=float)
    ema[period - 1] = np.mean(data[:period])
    mult = 2.0 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = data[i] * mult + ema[i - 1] * (1 - mult)
    return ema


def calc_sma(data: np.ndarray, period: int) -> np.ndarray:
    """SMA — Simple Moving Average."""
    sma = np.full_like(data, np.nan, dtype=float)
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    return sma


def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI — Relative Strength Index."""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = np.full(len(close), np.nan)
    valid = ~np.isnan(rs)
    rsi[valid] = 100.0 - 100.0 / (1.0 + rs[valid])
    return rsi


def calc_alma(data: np.ndarray, period: int = 20, offset: float = 0.85,
              sigma: float = 6.0) -> np.ndarray:
    """ALMA — Arnaud Legoux Moving Average."""
    alma = np.full_like(data, np.nan, dtype=float)
    m = offset * (period - 1)
    s = period / sigma
    w = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
    w /= w.sum()
    for i in range(period - 1, len(data)):
        alma[i] = np.dot(w, data[i - period + 1:i + 1])
    return alma


def calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26,
              signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, Signal line, Histogram."""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    valid = ~np.isnan(macd_line)
    signal_line = np.full_like(close, np.nan)
    first_valid = np.argmax(valid)
    if np.sum(valid) >= signal:
        signal_line[first_valid:] = calc_ema(
            macd_line[first_valid:], signal
        )
        # pad with NaN to match length
        pad = np.full(first_valid, np.nan)
        signal_line = np.concatenate([pad, calc_ema(macd_line[first_valid:], signal)])
        signal_line = signal_line[:len(close)]
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    k_period: int = 14, d_period: int = 3
                    ) -> tuple[np.ndarray, np.ndarray]:
    """%K and %D stochastic oscillator."""
    k = np.full_like(close, np.nan)
    for i in range(k_period - 1, len(close)):
        hh = np.max(high[i - k_period + 1:i + 1])
        ll = np.min(low[i - k_period + 1:i + 1])
        denom = hh - ll
        k[i] = ((close[i] - ll) / denom * 100) if denom > 0 else 50.0
    d = calc_sma(k, d_period)
    return k, d


def calc_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    obv = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def calc_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    mfi = np.full_like(close, np.nan)
    for i in range(period, len(close)):
        pos = 0.0
        neg = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0 and tp[j] > tp[j - 1]:
                pos += rmf[j]
            elif j > 0:
                neg += rmf[j]
        ratio = pos / max(neg, 1e-10)
        mfi[i] = 100.0 - 100.0 / (1.0 + ratio)
    return mfi


def calc_cmf(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Chaikin Money Flow."""
    hl = high - low
    hl[hl == 0] = 1e-10
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    cmf = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        sv = np.sum(volume[i - period + 1:i + 1])
        if sv > 0:
            cmf[i] = np.sum(mfv[i - period + 1:i + 1]) / sv
        else:
            cmf[i] = 0.0
    return cmf


def calc_ad(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            volume: np.ndarray) -> np.ndarray:
    """Accumulation/Distribution line."""
    hl = high - low
    hl[hl == 0] = 1e-10
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    return np.cumsum(mfv)


def calc_elder_force(close: np.ndarray, volume: np.ndarray,
                     period: int = 13) -> np.ndarray:
    """Elder Force Index."""
    fi = np.zeros_like(close)
    fi[1:] = (close[1:] - close[:-1]) * volume[1:]
    return calc_ema(fi, period)
