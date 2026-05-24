"""
Биржа-цифровой — JSON схема препроцессор → ИИ.
"""


def build_analysis_json(meta: dict, sections_data: list[dict]) -> dict:
    """Собрать финальный JSON для отправки в ИИ.

    Args:
        meta: {ticker, exchange, timeframe, analysis_type, current_price, ...}
        sections_data: [{section_id, section_emoji, section_title, section_type, data}, ...]

    Returns:
        Полный JSON документ.
    """
    sections = {}
    for s in sections_data:
        key = str(s["section_id"])
        sections[key] = {
            "emoji": s["section_emoji"],
            "title": s["section_title"],
            "type": s["section_type"],
            "data": s["data"],
        }

    return {
        "meta": meta,
        "sections": sections,
    }
