"""
Биржа-цифровой — Подготовка данных.

Парсинг CSV, определение колонок, базовая статистика.
"""
import numpy as np
import pandas as pd

from core.utils import detect_tf_hours, calc_atr


REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume"]

OPTIONAL_GROUPS = {
    "alma": ["ALMA 20", "ALMA 50", "ALMA 200"],
    "bollinger": ["Upper", "Basis", "Lower"],
    "vwap": ["VWAP", "Upper Band #1", "Lower Band #1"],
    "cvd": ["CVD (Open)", "CVD (High)", "CVD (Low)", "CVD (Close)"],
    "rsi": ["RSI"],
    "rsi_divergences": [
        "Regular Bullish", "Hidden Bullish", "Regular Bearish", "Hidden Bearish",
    ],
    "oscillators": ["%K", "%D", "MACD", "Signal line", "Histogram"],
    "flow": ["MF", "MF (Money Flow Index)"],
    "wyckoff": ["Elder Force Index", "Accumulation/Distribution", "CMF"],
    "service": ["Volume MA", "ATR"],
}


def load_csv(path: str) -> pd.DataFrame:
    """Загрузить CSV с auto-detect разделителя, привести колонки."""
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if len(df.columns) >= 6:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Не удалось прочитать CSV: {path}")

    df.columns = [c.strip() for c in df.columns]

    RU_EN_MAP = {
        "CVD (Цена откр.)": "CVD (Open)",
        "CVD (Макс.)": "CVD (High)",
        "CVD (Макс.": "CVD (High)",
        "CVD (Мин.)": "CVD (Low)",
        "CVD (Мин.": "CVD (Low)",
        "CVD (Закрыть)": "CVD (Close)",
        "CVD (Закрыть": "CVD (Close)",
    }
    df.columns = [RU_EN_MAP.get(c, c) for c in df.columns]

    # Дедупликация колонок
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    # Проверка обязательных колонок (case-insensitive)
    col_lower = {c.lower(): c for c in df.columns}
    for req in REQUIRED_COLUMNS:
        if req.lower() not in col_lower:
            raise ValueError(f"Отсутствует обязательная колонка: {req}")

    rename_map = {}
    for req in REQUIRED_COLUMNS:
        actual = col_lower[req.lower()]
        if actual != req:
            rename_map[actual] = req
    if rename_map:
        df = df.rename(columns=rename_map)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    return df


def detect_available_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Какие опциональные колонки есть в CSV."""
    available = {}
    cols_set = set(df.columns)
    for group, columns in OPTIONAL_GROUPS.items():
        found = [c for c in columns if c in cols_set]
        if found:
            available[group] = found
    return available


def compute_base_stats(df: pd.DataFrame) -> dict:
    """Базовая статистика: S0, mu, sigma, returns."""
    close = df["close"].values.astype(float)
    returns = np.diff(np.log(close))
    returns = np.concatenate([[0.0], returns])

    mu = float(np.mean(returns[1:]))
    sigma = float(np.std(returns[1:]))
    S0 = float(close[-1])

    return {"S0": S0, "mu": mu, "sigma": sigma, "returns": returns}


def prepare_data(csv_path: str, original_filename: str = "") -> dict:
    """Полная подготовка данных — точка входа.

    Args:
        csv_path: путь к CSV файлу на диске.
        original_filename: оригинальное имя файла (без UUID-префикса).
            Если не задано, имя берётся из csv_path.
    """
    df = load_csv(csv_path)
    available_cols = detect_available_columns(df)
    base = compute_base_stats(df)

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)

    atr = calc_atr(high, low, close, period=14)
    atr_last = float(atr[~np.isnan(atr)][-1]) if np.any(~np.isnan(atr)) else 0.01

    tf_hours = detect_tf_hours(df["time"])

    ticker = ""
    exchange = ""
    # TradingView экспортирует в формате: EXCHANGE_TICKER, TIMEFRAME_HASH.csv
    # Примеры: NYMEX_NG1!, 60_abcde.csv → биржа=NYMEX, тикер=NG1!
    #          RUS_CNYRUB.P, 240_abc.csv → биржа=RUS, тикер=CNYRUB.P
    import os
    # Используем оригинальное имя, если передано (мульти-ТФ добавляет UUID-префикс)
    fname_src = original_filename if original_filename else os.path.basename(csv_path)
    fname = os.path.splitext(fname_src)[0]
    parts = fname.split("_")
    if len(parts) >= 2:
        exchange = parts[0].strip().upper()
        # Тикер = вторая часть до запятой (если есть)
        ticker_raw = parts[1].split(",")[0].strip()
        ticker = ticker_raw.upper()

    return {
        "df": df,
        "close": close,
        "high": high,
        "low": low,
        "open_": open_,
        "volume": volume,
        "atr": atr,
        "atr_last": atr_last,
        "available_cols": available_cols,
        "base_stats": base,
        "tf_hours": tf_hours,
        "n_bars": len(df),
        "period_start": df["time"].iloc[0],
        "period_end": df["time"].iloc[-1],
        "ticker": ticker,
        "exchange": exchange,
    }
