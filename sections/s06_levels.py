"""
Биржа-цифровой — Раздел 6: УРОВНИ.

Тип: full. Пивоты (дневные/недельные/месячные) + swing уровни S/R.
Формула: P=(H+L+C)/3, R1..R4, S1..S4 по закрытым периодам.
"""
import numpy as np
import pandas as pd
from sections.base import SectionProcessor
from config import LEVELS_MERGE_ATR_MULT


def _calc_pivots(h: float, l: float, c: float) -> dict:
    """Classic Pivot Points: P, R1-R4, S1-S4 (+ R5/S5 extension)."""
    P = (h + l + c) / 3
    return {
        "P": round(P, 4),
        "R1": round(2 * P - l, 4),
        "R2": round(P + (h - l), 4),
        "R3": round(h + 2 * (P - l), 4),
        "R4": round(P + 3 * (h - l), 4),
        "R5": round(h + 4 * (P - l), 4),
        "S1": round(2 * P - h, 4),
        "S2": round(P - (h - l), 4),
        "S3": round(l - 2 * (h - P), 4),
        "S4": round(P - 3 * (h - P), 4),
        "S5": round(l - 4 * (h - P), 4),
    }


def _compute_pivots(df: pd.DataFrame, period: str) -> dict | None:
    """Рассчитать пивоты по закрытым периодам.

    period: "D" (дневные), "W" (недельные), "M" (месячные).
    Логика: группируем по дате/неделе/месяцу — берём предпоследнюю полную группу.
    Возвращает dict с ключами P, R1..R5, S1..S5 или None.
    """
    if len(df) < 2:
        return None

    dfc = df.copy()
    dfc["_date"] = dfc["time"].dt.date

    if period == "D":
        # Дневные: unique dates, берём предпоследний полный день
        dates = sorted(dfc["_date"].unique())
        if len(dates) < 2:
            return None
        prev_date = dates[-2]
        mask = dfc["_date"] == prev_date

    elif period == "W":
        # Недельные: year*100+week — простая целочисленная группировка
        iso = dfc["time"].dt.isocalendar()
        dfc["_yearweek"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
        yearweeks = sorted(dfc["_yearweek"].unique())
        if len(yearweeks) < 2:
            return None
        prev_yw = yearweeks[-2]
        mask = dfc["_yearweek"] == prev_yw

    elif period in ("M", "ME"):
        # Месячные: year*100+month
        dfc["_yearmonth"] = dfc["time"].dt.year * 100 + dfc["time"].dt.month
        yearmonths = sorted(dfc["_yearmonth"].unique())
        if len(yearmonths) < 2:
            return None
        prev_ym = yearmonths[-2]
        mask = dfc["_yearmonth"] == prev_ym

    else:
        return None

    period_data = dfc[mask]
    if period_data.empty:
        return None

    H = float(period_data["high"].max())
    L = float(period_data["low"].min())
    C = float(period_data["close"].iloc[-1])

    return _calc_pivots(H, L, C)


class LevelsProcessor(SectionProcessor):
    section_id = 6
    section_emoji = "📊"
    section_title = "УРОВНИ"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        high = context["high"]
        low = context["low"]
        atr_last = context["atr_last"]
        tf_hours = context.get("tf_hours", 1.0)
        swing_points = context.get("swing_points", [])
        current_price = float(close[-1])

        # ── ЯДРО v8.1 §IV: исторические уровни ИЩУТСЯ ВНУТРИ TF-окна ──
        # 1W/1D/4H/1H/15M = 52 нед / 90 дн / 150 / 1440 / 672 баров
        n = len(df)
        if tf_hours <= 0.25:
            window_bars = 7 * 24 * 4
        elif tf_hours <= 1:
            window_bars = 60 * 24
        elif tf_hours <= 4:
            window_bars = 25 * 6
        elif tf_hours <= 24:
            window_bars = 90
        else:
            window_bars = 52
        window_idx = max(0, n - min(window_bars, n))
        # Свинги/зигзаг-точки за окном НЕ используем (v8.1: «не использовать
        # точки как якоря, если экстремумы раньше окна»)
        swing_points = [p for p in swing_points
                        if int(p.get("index", 0)) >= window_idx]

        # 1. Пивоты по закрытым периодам
        pivots = {}

        # Дневные — только если ТФ < 1D
        if tf_hours < 24:
            daily = _compute_pivots(df, "D")
            if daily:
                pivots["daily"] = daily

        # Недельные — только если ТФ < 1W
        if tf_hours < 168:
            weekly = _compute_pivots(df, "W")
            if weekly:
                pivots["weekly"] = weekly

        # Месячные
        monthly = _compute_pivots(df, "ME")
        if monthly:
            pivots["monthly"] = monthly

        # 2. Все пивотные уровни в один список
        pivot_levels = []  # (price, label)
        for period_name, period_key, prefix in [
            ("daily", "daily", "D."),
            ("weekly", "weekly", "W."),
            ("monthly", "monthly", "M."),
        ]:
            pv = pivots.get(period_key)
            if pv:
                for lbl, val in pv.items():
                    pivot_levels.append((val, f"{prefix}{lbl}"))

        # 3. Swing S/R (классифицированные свинги + ZigZag 5% В TF-окне)
        zigzag_points = [
            p for p in context.get("zigzag_5pct", [])
            if int(p.get("index", 0)) >= window_idx
        ]

        swing_highs = sorted(set(
            [p["price"] for p in swing_points if p["type"] == "high"] +
            [p["price"] for p in zigzag_points if p.get("type") == "H"]
        ), reverse=True)
        swing_lows = sorted(set(
            [p["price"] for p in swing_points if p["type"] == "low"] +
            [p["price"] for p in zigzag_points if p.get("type") == "L"]
        ))

        # 4. Собрать все R и S
        merge_dist = LEVELS_MERGE_ATR_MULT * atr_last

        def fmt(p, label=""):
            pct = (p - current_price) / current_price * 100
            return {"price": round(p, 4), "pct": round(pct, 2), "label": label}

        # Сопротивления: пивоты выше цены + swing highs
        resistances_raw = []
        for val, lbl in pivot_levels:
            if val > current_price:
                resistances_raw.append((val, lbl))
        for sh in swing_highs:
            if sh > current_price:
                resistances_raw.append((sh, "Swing"))

        resistances_raw.sort(key=lambda x: x[0])

        # Поддержки: пивоты ниже цены + swing lows
        supports_raw = []
        for val, lbl in pivot_levels:
            if val < current_price:
                supports_raw.append((val, lbl))
        for sl in swing_lows:
            if sl < current_price:
                supports_raw.append((sl, "Swing"))

        supports_raw.sort(key=lambda x: x[0], reverse=True)

        # 5. Мердж близких + ограничение 5
        # Приоритет при слиянии: D. > W. > M. > Swing
        PRIORITY = {"D.": 0, "W.": 1, "M.": 2, "Swing": 3}

        def _label_priority(lbl):
            for prefix, prio in PRIORITY.items():
                if lbl.startswith(prefix):
                    return prio
            return 3

        def merge_and_limit(levels, limit=5):
            if not levels:
                return []
            merged = [levels[0]]
            for val, lbl in levels[1:]:
                prev_val, prev_lbl = merged[-1]
                if abs(val - prev_val) < merge_dist:
                    # При слиянии — побеждает более приоритетный
                    if _label_priority(lbl) < _label_priority(prev_lbl):
                        merged[-1] = ((prev_val + val) / 2, lbl)
                    else:
                        merged[-1] = ((prev_val + val) / 2, prev_lbl)
                else:
                    merged.append((val, lbl))
            return merged[:limit]

        resistances = merge_and_limit(resistances_raw)
        supports = merge_and_limit(supports_raw)

        return {
            "pivots": pivots,
            "resistances_5": [fmt(p, lbl) for p, lbl in resistances],
            "supports_5": [fmt(p, lbl) for p, lbl in supports],
            "conclusion": "5 ближайших сопротивлений / 5 ближайших поддержек",
            "current_price": current_price,
            "merge_distance": round(merge_dist, 4),
        }
