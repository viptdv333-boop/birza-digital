"""
Биржа-цифровой — Препроцессор (оркестратор).

CSV → shared context → 18 sections → JSON → AI → Report.
"""
import json
import numpy as np

from core.data_prep import prepare_data
from core.zigzag import zigzag, williams_fractal, classify_swing_points
from core.utils import calc_atr, tf_label
from config import (
    ZIGZAG_DEV_MAJOR, ZIGZAG_DEV_MINOR, FRACTAL_PERIOD,
    VP_WINDOW, VP_BINS, VP_VAH_VAL_THRESHOLD,
)
from ai.schema import build_analysis_json
from ai.prompts import build_user_message, build_system_prompt_with_sections
from ai.client import create_client


def _load_processors():
    """Загрузить все доступные процессоры секций."""
    processors = {}
    section_imports = [
        (1, "sections.s01_trends", "TrendsProcessor"),
        (2, "sections.s02_elliott", "ElliottProcessor"),
        (3, "sections.s03_patterns", "PatternsProcessor"),
        (4, "sections.s04_candles", "CandlesProcessor"),
        (5, "sections.s05_rsi_div", "DivergencesProcessor"),
        (6, "sections.s06_levels", "LevelsProcessor"),
        (7, "sections.s07_fibonacci", "FibonacciProcessor"),
        (8, "sections.s08_vsa", "VSAProcessor"),
        (9, "sections.s09_volume_profile", "VolumeProfileProcessor"),
        (10, "sections.s10_wyckoff", "WyckoffProcessor"),
        (11, "sections.s11_stoploss", "StopLossProcessor"),
        (12, "sections.s12_tempo", "TempoProcessor"),
        (13, "sections.s13_fvg", "FVGProcessor"),
        (14, "sections.s14_squeeze", "SqueezeProcessor"),
        (15, "sections.s15_efficiency", "EfficiencyProcessor"),
        (16, "sections.s16_flow", "FlowProcessor"),
        (17, "sections.s17_microstructure", "MicrostructureProcessor"),
        (18, "sections.s18_correlations", "CorrelationsProcessor"),
        # (20, "sections.s19_math", "MathModelProcessor"),  # убрана — запускается отдельно
    ]
    for sid, module_path, class_name in section_imports:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            processors[sid] = cls()
        except (ImportError, AttributeError):
            pass
    return processors


def compute_volume_profile(df, window=VP_WINDOW, bins=VP_BINS,
                           threshold=VP_VAH_VAL_THRESHOLD) -> dict:
    """Volume Profile: POC, VAH, VAL."""
    data = df.tail(window)
    low_all = data["low"].values
    high_all = data["high"].values
    volume_all = data["volume"].values

    price_min = low_all.min()
    price_max = high_all.max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_per_bin = np.zeros(bins)

    for i in range(len(data)):
        lo, hi, vol = low_all[i], high_all[i], volume_all[i]
        if hi <= lo:
            idx = np.searchsorted(bin_edges, lo, side="right") - 1
            idx = max(0, min(idx, bins - 1))
            vol_per_bin[idx] += vol
        else:
            idx_lo = np.searchsorted(bin_edges, lo, side="right") - 1
            idx_hi = np.searchsorted(bin_edges, hi, side="right") - 1
            idx_lo = max(0, min(idx_lo, bins - 1))
            idx_hi = max(0, min(idx_hi, bins - 1))
            n_bins = idx_hi - idx_lo + 1
            for b in range(idx_lo, idx_hi + 1):
                vol_per_bin[b] += vol / n_bins

    poc_idx = np.argmax(vol_per_bin)
    poc = float(0.5 * (bin_edges[poc_idx] + bin_edges[poc_idx + 1]))

    total_vol = vol_per_bin.sum()
    target = total_vol * threshold
    cum = vol_per_bin[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx
    while cum < target and (lo_idx > 0 or hi_idx < bins - 1):
        expand_lo = vol_per_bin[lo_idx - 1] if lo_idx > 0 else 0
        expand_hi = vol_per_bin[hi_idx + 1] if hi_idx < bins - 1 else 0
        if expand_lo >= expand_hi and lo_idx > 0:
            lo_idx -= 1
            cum += expand_lo
        elif hi_idx < bins - 1:
            hi_idx += 1
            cum += expand_hi
        else:
            lo_idx -= 1
            cum += expand_lo

    val = float(bin_edges[lo_idx])
    vah = float(bin_edges[hi_idx + 1])

    return {"POC": poc, "VAH": vah, "VAL": val}


def build_shared_context(data: dict) -> dict:
    """Построить shared context для всех секций."""
    df = data["df"]
    close = data["close"]
    high = data["high"]
    low = data["low"]
    times = df["time"].values

    # ZigZag 5% (основные волны) — одинаковая dev для всех ТФ.
    # Направление тренда теперь определяется по глобальному min/max (s01_trends),
    # а не по swing-точкам, поэтому масштабирование dev не нужно.
    zz_major = zigzag(high, low, ZIGZAG_DEV_MAJOR, times)
    # ZigZag 1% (подволны)
    zz_minor = zigzag(high, low, ZIGZAG_DEV_MINOR, times)
    # Williams Fractal
    fractals = williams_fractal(high, low, FRACTAL_PERIOD)
    # Классификация swing-точек
    swing_points = classify_swing_points(zz_major)
    # Volume Profile
    vp = compute_volume_profile(df)

    context = {
        **data,  # df, close, high, low, open_, volume, atr, ...
        "zigzag_5pct": zz_major,
        "zigzag_1pct": zz_minor,
        "williams_fractal": fractals,
        "swing_points": swing_points,
        "vp": vp,
    }
    return context


def run_preprocessor(csv_path: str, on_progress=None,
                     original_filename: str = "") -> dict:
    """Запустить препроцессор: CSV → JSON со всеми разделами.

    Args:
        csv_path: путь к CSV файлу.
        on_progress: callback(step, total, message).
        original_filename: оригинальное имя файла (для правильного парсинга
            тикера/биржи, если файл сохранён с UUID-префиксом).

    Returns:
        {"meta": {...}, "sections": {...}, "sections_json": [...]}
    """
    processors = _load_processors()
    total_steps = 3 + len(processors)  # data + context + sections + done

    def progress(step, msg):
        if on_progress:
            on_progress(step, total_steps, msg)

    # Шаг 1: Загрузка данных
    progress(1, "Загрузка и парсинг CSV...")
    data = prepare_data(csv_path, original_filename=original_filename)

    # Шаг 2: Shared context
    progress(2, "Вычисление ZigZag, фракталов, Volume Profile...")
    context = build_shared_context(data)
    context["csv_path"] = csv_path  # для мат.модели (s19)

    # Шаг 3+: Секции
    sections_data = []
    for i, (sid, proc) in enumerate(sorted(processors.items())):
        progress(3 + i, f"Раздел {sid}: {proc.section_title}...")
        try:
            computed = proc.compute(data["df"], context)
            section_json = proc.to_json(computed)
            sections_data.append(section_json)
            # После S06 — передать пивотные уровни в контекст для S11
            if sid == 6 and "error" not in computed:
                sup_prices = []
                res_prices = []
                for item in computed.get("supports_5", computed.get("supports", [])):
                    if isinstance(item, dict) and "price" in item:
                        sup_prices.append(float(item["price"]))
                for item in computed.get("resistances_5", computed.get("resistances", [])):
                    if isinstance(item, dict) and "price" in item:
                        res_prices.append(float(item["price"]))
                # Также добавить все пивоты (не только 5 ближайших)
                for period_key in ["daily", "weekly", "monthly"]:
                    pivots = computed.get("pivots", {}).get(period_key, {})
                    if isinstance(pivots, dict):
                        for lbl, val in pivots.items():
                            if isinstance(val, (int, float)):
                                if val < float(context["close"][-1]):
                                    sup_prices.append(float(val))
                                else:
                                    res_prices.append(float(val))
                context["pivot_supports"] = sorted(set(sup_prices))
                context["pivot_resistances"] = sorted(set(res_prices))
        except Exception as e:
            sections_data.append({
                "section_id": sid,
                "section_emoji": proc.section_emoji,
                "section_title": proc.section_title,
                "section_type": proc.section_type,
                "data": {"error": str(e)},
            })

    # Meta
    meta = {
        "ticker": data["ticker"],
        "exchange": data["exchange"],
        "timeframe": tf_label(data["tf_hours"]),
        "current_price": data["base_stats"]["S0"],
        "n_bars": data["n_bars"],
        "period_start": str(data["period_start"]),
        "period_end": str(data["period_end"]),
        "analysis_type_name": "",
        "horizon": "",
    }

    # JSON для ИИ
    analysis_json = build_analysis_json(meta, sections_data)

    progress(total_steps, "Препроцессор завершён")

    return {
        "meta": meta,
        "sections_data": sections_data,
        "analysis_json": analysis_json,
    }


def run_full_pipeline(csv_path: str, on_progress=None,
                      skip_ai: bool = False,
                      provider: str = None, model: str = None,
                      horizon: str = None,
                      analysis_type_name: str = None) -> dict:
    """Полный пайплайн: CSV → Препроцессор → AI → Отчёт.

    Args:
        csv_path: путь к CSV.
        on_progress: callback(step, total, message).
        skip_ai: True = только препроцессор, без вызова ИИ.
        provider: "anthropic" / "openrouter" / "nvidia".
        model: model ID.
        horizon: горизонт прогноза (из ANALYSIS_TYPES).
        analysis_type_name: название типа анализа.

    Returns:
        {"meta": {...}, "analysis_json": {...}, "ai_report": str | None}
    """
    result = run_preprocessor(csv_path, on_progress)

    if skip_ai:
        result["ai_report"] = None
        result["provider"] = "none"
        result["model"] = ""
        return result

    prov_name = provider or "anthropic"
    if on_progress:
        on_progress(99, 100, f"Вызов AI ({prov_name})...")

    section_ids = [s["section_id"] for s in result["sections_data"]]
    system_prompt = build_system_prompt_with_sections(
        section_ids, horizon=horizon, analysis_type_name=analysis_type_name,
    )
    user_message = build_user_message(result["analysis_json"])

    client = create_client(provider_id=provider, model=model)
    ai_report = client.analyze(system_prompt, user_message)
    result["provider"] = prov_name
    result["model"] = model or ""

    result["ai_report"] = ai_report

    if on_progress:
        on_progress(100, 100, "Готово")

    return result
