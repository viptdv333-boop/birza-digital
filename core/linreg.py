"""
Биржа-цифровой — Linear Regression Channel ±2σ.
"""
import numpy as np


def linear_regression_channel(close: np.ndarray, period: int = 100
                              ) -> dict:
    """Linear Regression Channel ±2σ по последним period барам.

    Возвращает:
    {
        "slope": float — наклон (цена/бар),
        "slope_pct": float — наклон в % за бар,
        "intercept": float,
        "upper": np.ndarray — верхняя граница (+2σ),
        "lower": np.ndarray — нижняя граница (-2σ),
        "center": np.ndarray — линия регрессии,
        "position_pct": float — положение текущей цены в канале (0-100%),
        "r_squared": float — R² (качество фита),
    }
    """
    n = len(close)
    actual_period = min(period, n)
    data = close[-actual_period:]
    x = np.arange(actual_period, dtype=float)

    # Линейная регрессия y = a*x + b
    coeffs = np.polyfit(x, data, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    center = slope * x + intercept
    residuals = data - center
    sigma = np.std(residuals)

    upper = center + 2 * sigma
    lower = center - 2 * sigma

    # Положение текущей цены в канале
    current_price = data[-1]
    channel_width = upper[-1] - lower[-1]
    if channel_width > 0:
        position_pct = (current_price - lower[-1]) / channel_width * 100
    else:
        position_pct = 50.0

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Наклон в % за бар
    mid_price = np.mean(data)
    slope_pct = (slope / mid_price * 100) if mid_price > 0 else 0.0

    return {
        "slope": float(slope),
        "slope_pct": float(slope_pct),
        "intercept": float(intercept),
        "upper": upper,
        "lower": lower,
        "center": center,
        "sigma": float(sigma),
        "position_pct": float(np.clip(position_pct, 0, 100)),
        "r_squared": float(r_squared),
        "period": actual_period,
    }
