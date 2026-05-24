"""
Биржа-цифровой — Раздел 4: СВЕЧНЫЕ ПАТТЕРНЫ (по регламенту v6).

Алгоритмический детект: Молот, Повешенный, Падающая звезда, Доджи,
Поглощение, Харами, Утренняя/Вечерняя звезда, Три солдата/вороны, Марабоцу.
Характер: ускорение/торможение/разворот.
"""
import numpy as np
from sections.base import SectionProcessor


class CandlesProcessor(SectionProcessor):
    section_id = 4
    section_emoji = "🕯"
    section_title = "СВЕЧНЫЕ ПАТТЕРНЫ"
    section_type = "partial"

    def compute(self, df, context: dict) -> dict:
        close = context["close"]
        open_ = context["open_"]
        high = context["high"]
        low = context["low"]
        n = len(close)
        window = min(50, n)
        current_price = float(close[-1])

        # Геометрия свечей
        candles = []
        for i in range(n - window, n):
            o, h, l, c = open_[i], high[i], low[i], close[i]
            body = abs(c - o)
            full_range = h - l if h > l else 0.001
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l

            candles.append({
                "bar": i - (n - window),
                "open": round(float(o), 4),
                "high": round(float(h), 4),
                "low": round(float(l), 4),
                "close": round(float(c), 4),
                "body_pct": round(body / full_range * 100, 1),
                "upper_shadow_pct": round(upper_shadow / full_range * 100, 1),
                "lower_shadow_pct": round(lower_shadow / full_range * 100, 1),
                "direction": "bull" if c > o else ("bear" if c < o else "doji"),
            })

        # Статистика
        bull_count = sum(1 for c in candles if c["direction"] == "bull")
        doji_count = sum(1 for c in candles if c["direction"] == "doji")
        bear_count = window - bull_count - doji_count
        avg_body = np.mean([c["body_pct"] for c in candles])

        # ──────── Детект именных паттернов ────────
        patterns = []
        for i in range(max(0, len(candles) - 5), len(candles)):
            c = candles[i]
            bp = c["body_pct"]
            usp = c["upper_shadow_pct"]
            lsp = c["lower_shadow_pct"]
            d = c["direction"]

            # Доджи: тело < 10%
            if bp < 10:
                patterns.append({"bar": c["bar"], "name": "Доджи",
                                 "signal": "нейтральный/разворот"})

            # Молот: маленькое тело вверху, длинная нижняя тень > 2× тела
            elif lsp > 2 * bp and usp < bp and d in ("bull", "doji"):
                patterns.append({"bar": c["bar"], "name": "Молот",
                                 "signal": "бычий разворот"})

            # Повешенный: как молот но в восходящем тренде (направление bear)
            elif lsp > 2 * bp and usp < bp and d == "bear":
                patterns.append({"bar": c["bar"], "name": "Повешенный",
                                 "signal": "медвежий разворот"})

            # Падающая звезда: маленькое тело внизу, длинная верхняя тень
            elif usp > 2 * bp and lsp < bp:
                patterns.append({"bar": c["bar"], "name": "Падающая звезда",
                                 "signal": "медвежий разворот"})

            # Марабоцу: тело > 80%, тени минимальные
            elif bp > 80:
                sig = "бычий импульс" if d == "bull" else "медвежий импульс"
                patterns.append({"bar": c["bar"], "name": f"Марабоцу ({d})",
                                 "signal": sig})

        # Двухсвечные паттерны
        for i in range(max(1, len(candles) - 5), len(candles)):
            prev = candles[i - 1]
            curr = candles[i]

            # Бычье поглощение
            if (prev["direction"] == "bear" and curr["direction"] == "bull" and
                    curr["close"] > prev["open"] and curr["open"] < prev["close"]):
                patterns.append({"bar": curr["bar"], "name": "Бычье поглощение",
                                 "signal": "бычий разворот"})

            # Медвежье поглощение
            elif (prev["direction"] == "bull" and curr["direction"] == "bear" and
                  curr["close"] < prev["open"] and curr["open"] > prev["close"]):
                patterns.append({"bar": curr["bar"], "name": "Медвежье поглощение",
                                 "signal": "медвежий разворот"})

            # Харами бычья
            elif (prev["direction"] == "bear" and curr["direction"] == "bull" and
                  curr["close"] < prev["open"] and curr["open"] > prev["close"] and
                  curr["body_pct"] < prev["body_pct"] * 0.5):
                patterns.append({"bar": curr["bar"], "name": "Бычья харами",
                                 "signal": "бычий разворот"})

            # Харами медвежья
            elif (prev["direction"] == "bull" and curr["direction"] == "bear" and
                  curr["close"] > prev["open"] and curr["open"] < prev["close"] and
                  curr["body_pct"] < prev["body_pct"] * 0.5):
                patterns.append({"bar": curr["bar"], "name": "Медвежья харами",
                                 "signal": "медвежий разворот"})

        # Трёхсвечные
        for i in range(max(2, len(candles) - 5), len(candles)):
            c0, c1, c2 = candles[i - 2], candles[i - 1], candles[i]

            # Три белых солдата
            if (c0["direction"] == "bull" and c1["direction"] == "bull" and
                    c2["direction"] == "bull" and
                    c1["close"] > c0["close"] and c2["close"] > c1["close"] and
                    c0["body_pct"] > 40 and c1["body_pct"] > 40 and c2["body_pct"] > 40):
                patterns.append({"bar": c2["bar"], "name": "Три белых солдата",
                                 "signal": "бычий импульс"})

            # Три чёрных вороны
            elif (c0["direction"] == "bear" and c1["direction"] == "bear" and
                  c2["direction"] == "bear" and
                  c1["close"] < c0["close"] and c2["close"] < c1["close"] and
                  c0["body_pct"] > 40 and c1["body_pct"] > 40 and c2["body_pct"] > 40):
                patterns.append({"bar": c2["bar"], "name": "Три чёрных вороны",
                                 "signal": "медвежий импульс"})

            # Утренняя звезда
            if (c0["direction"] == "bear" and c0["body_pct"] > 50 and
                    c1["body_pct"] < 20 and
                    c2["direction"] == "bull" and c2["body_pct"] > 50 and
                    c2["close"] > (c0["open"] + c0["close"]) / 2):
                patterns.append({"bar": c2["bar"], "name": "Утренняя звезда",
                                 "signal": "бычий разворот"})

            # Вечерняя звезда
            if (c0["direction"] == "bull" and c0["body_pct"] > 50 and
                    c1["body_pct"] < 20 and
                    c2["direction"] == "bear" and c2["body_pct"] > 50 and
                    c2["close"] < (c0["open"] + c0["close"]) / 2):
                patterns.append({"bar": c2["bar"], "name": "Вечерняя звезда",
                                 "signal": "медвежий разворот"})

        # ──────── Характер движения ────────
        # Сравниваем абсолютные тела и диапазоны последних 5 vs предыдущих 5
        if len(candles) >= 10:
            # Абсолютные размеры тел (не относительные body_pct)
            recent_abs_bodies = np.mean([abs(c["close"] - c["open"]) for c in candles[-5:]])
            prev_abs_bodies = np.mean([abs(c["close"] - c["open"]) for c in candles[-10:-5]])
            # Абсолютные диапазоны (range)
            recent_ranges = np.mean([c["high"] - c["low"] for c in candles[-5:]])
            prev_ranges = np.mean([c["high"] - c["low"] for c in candles[-10:-5]])
            recent_bull = sum(1 for c in candles[-5:] if c["direction"] == "bull")

            # Ускорение: тела растут И/ИЛИ диапазоны растут
            bodies_growing = recent_abs_bodies > prev_abs_bodies * 1.3 if prev_abs_bodies > 0 else False
            ranges_growing = recent_ranges > prev_ranges * 1.3 if prev_ranges > 0 else False
            bodies_shrinking = recent_abs_bodies < prev_abs_bodies * 0.7 if prev_abs_bodies > 0 else False
            ranges_shrinking = recent_ranges < prev_ranges * 0.7 if prev_ranges > 0 else False

            if bodies_growing and ranges_growing:
                character = "ускорение тренда"
            elif bodies_shrinking or ranges_shrinking:
                character = "замедление тренда"
            elif abs(recent_bull - 2.5) <= 0.5 and np.mean([c["body_pct"] for c in candles[-5:]]) < 30:
                character = "смена тренда (неопределённость)"
            else:
                character = "развитие тренда"
        else:
            character = "недостаточно данных"

        # Доминирование
        if bull_count > bear_count + 3:
            dominance = "покупатель"
        elif bear_count > bull_count + 3:
            dominance = "продавец"
        else:
            dominance = "баланс"

        return {
            "candles_20": candles,
            "last_3": candles[-3:] if len(candles) >= 3 else candles,
            "named_patterns": patterns,
            "stats": {
                "bull_count": bull_count,
                "bear_count": bear_count,
                "doji_count": doji_count,
                "avg_body_pct": round(float(avg_body), 1),
            },
            "character": character,
            "dominance": dominance,
        }
