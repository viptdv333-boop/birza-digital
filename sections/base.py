"""
Биржа-цифровой — Базовый класс секции.
"""
from abc import ABC, abstractmethod


class SectionProcessor(ABC):
    """Базовый класс для процессора раздела ТА.

    section_type:
        "full"     — скрипт считает всё, ИИ интерпретирует числа
        "partial"  — скрипт даёт числа, ИИ добавляет паттерны/стадии
        "ai_only"  — скрипт даёт сырые данные, ИИ выполняет полный анализ
    """

    section_id: int = 0
    section_emoji: str = ""
    section_title: str = ""
    section_type: str = "full"  # "full" | "partial" | "ai_only"

    @abstractmethod
    def compute(self, df, context: dict) -> dict:
        """Вычислить числовые данные раздела.

        Args:
            df: DataFrame с OHLCV + опциональные колонки.
            context: Shared context (zigzag, swing, atr, vp, ...).

        Returns:
            dict с вычисленными данными.
        """

    def to_json(self, computed: dict) -> dict:
        """Сериализовать результат для JSON.

        По умолчанию — прямая сериализация.
        Переопределить, если есть numpy arrays или сложные объекты.
        """
        import numpy as np

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        return {
            "section_id": self.section_id,
            "section_emoji": self.section_emoji,
            "section_title": self.section_title,
            "section_type": self.section_type,
            "data": _convert(computed),
        }
