"""
Биржа-цифровой — Форматтер отчёта из данных препроцессора.

Генерирует оформленный текст по заданию без обращения к AI.
"""


def format_report(meta: dict, sections_data: list[dict],
                   include_conclusions: bool = True) -> str:
    """Сформировать текстовый отчёт из данных препроцессора."""
    parts = []

    # ── Шапка по ЯДРУ v8 ──
    from ai.prompts import _infer_analysis_type
    ticker = meta.get("ticker", "—")
    exchange = meta.get("exchange", "—")
    tf = meta.get("timeframe", "—")
    price = meta.get("current_price", 0)
    period_end = meta.get("period_end", "")[:16]
    analysis_type = (
        meta.get("analysis_type_name")
        or meta.get("analysis_type")
        or _infer_analysis_type(tf)
    )

    parts.append(f"📘ТИКЕР: #{ticker} (#{exchange})")
    parts.append(f"БИРЖА: #{exchange}")
    parts.append(f"ТИП АНАЛИЗА: {analysis_type}")
    parts.append(f"ЦЕНА: {price}")
    parts.append(f"ДАТА И ВРЕМЯ: {period_end} UTC+3")
    parts.append(f"❗ НЕ ЯВЛЯЕТСЯ ИИР ❗")
    parts.append("")

    # Секции
    sections_map = {s["section_id"]: s for s in sections_data}

    for sid in sorted(sections_map.keys()):
        if sid == 20:
            continue  # мат.модель выводится после вывода
        s = sections_map[sid]
        emoji = s.get("section_emoji", "")
        title = s.get("section_title", "")
        data = s.get("data", {})

        parts.append(f"{emoji} {sid}. {title}")
        parts.append("")

        formatter = SECTION_FORMATTERS.get(sid)
        # ЯДРО v8.1: per-section «📍 Вывод» убираем — финальный вывод один
        # (раздел 19 в режиме Б/В, раздел 14 в режиме А).
        if formatter:
            parts.append(formatter(data, price, include_conclusion=False))
        else:
            parts.append(_format_generic(data, price))

        parts.append("")

    # Раздел 19 — ВЫВОД (агрегация из всех разделов, v8: был 20)
    parts.append("🧠 Раздел 19 ВЫВОД")
    parts.append("")
    parts.append(_format_s20(meta, sections_map, price))
    parts.append("")

    # Раздел 20 — МАТЕМАТИЧЕСКАЯ МОДЕЛЬ (убрана, запускается отдельно)
    if False and 20 in sections_map:
        s20 = sections_map[20]
        parts.append(f"{s20.get('section_emoji', '🔢')} 20. {s20.get('section_title', 'МАТЕМАТИЧЕСКАЯ МОДЕЛЬ')}")
        parts.append("")
        fmt20 = SECTION_FORMATTERS.get(20)
        if fmt20:
            parts.append(fmt20(s20.get("data", {}), price, include_conclusions))
        else:
            parts.append(_format_generic(s20.get("data", {}), price))
        parts.append("")

    # ── Блок подписок ──
    parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    parts.append("🟥 Платная аналитика по ГАЗУ и ПЛАТИНЕ: @Siroezhkin_bot")
    parts.append("💰 Для донатов:")
    parts.append("💳 https://pay.cloudtips.ru/p/562cbedb")
    parts.append("💳 2200 7006 2350 2977 (Т-Банк)")
    parts.append("🔥 Больше инструментов в профиле")
    parts.append("")

    # v8.1: после вывода — вопрос Боссу про ФОМО
    parts.append("Нужен ФОМО?")
    parts.append("")

    return "\n".join(parts)


def _fmt_price(val):
    """Адаптивный формат цены: без научной нотации."""
    if val >= 1000:
        return f"{val:.2f}"
    elif val >= 1:
        return f"{val:.4g}"
    else:
        return f"{val:.4g}"


def _pct(val, current):
    """Форматировать цену с % от текущей."""
    if current and current > 0:
        pct = (val - current) / current * 100
        sign = "+" if pct >= 0 else ""
        return f"{_fmt_price(val)} ({sign}{pct:.2f}%)"
    return _fmt_price(val)


def _format_s01(data, price, include_conclusion=True):
    """Раздел 1: ТРЕНДЫ (ЯДРО v8.1).
    Только: старший тренд + точка начала + стадия + вилы Эндрюса.
    """
    lines = []
    senior = data.get("senior_trend", {})
    direction = senior.get("direction") or data.get("direction", "—")
    start = senior.get("start", {}) if senior else {}
    stage = senior.get("stage") or data.get("stage", "—")

    lines.append(f"Тренд: {direction}")
    if start and start.get("price"):
        lines.append(
            f"Начало: {_pct(start['price'], price)} от {str(start.get('time', ''))[:10]}"
        )
    lines.append(f"Стадия: {stage}")

    # Вилы Эндрюса (только если построены)
    pf = data.get("pitchfork") or {}
    if pf.get("available"):
        A = pf.get("A", {}); B = pf.get("B", {}); C = pf.get("C", {})
        med = pf.get("median", {})
        lines.append("")
        lines.append("Вилы Эндрюса:")
        lines.append(f"  A: {_pct(A.get('price', 0), price)}")
        lines.append(f"  B: {_pct(B.get('price', 0), price)}")
        lines.append(f"  C: {_pct(C.get('price', 0), price)}")
        slope_pct = med.get("slope_pct_per_bar")
        if slope_pct is not None:
            lines.append(f"  Угол медианы: {slope_pct:+.4f}%/бар")

    if include_conclusion:
        lines.append("")
        lines.append(f"📍 Вывод: Тренд {direction}, стадия — {stage}.")

    return "\n".join(lines)


def _format_s02(data, price, include_conclusion=True):
    """Раздел 2: ВОЛНОВОЙ АНАЛИЗ (ЯДРО v8.1).
    Только: классическая разметка Элиотта + стадия волны + цель окончания.
    """
    lines = []
    anchor = data.get("anchor", {}) or {}
    anchor_price = anchor.get("price", 0)

    imp_waves = data.get("impulse_waves", []) or []
    corr_waves = data.get("correction_waves", []) or []
    wave_subwaves = data.get("wave_subwaves", {}) or {}

    def _print(waves, start_price, title):
        lines.append(title)
        prev = start_price
        for w in waves:
            label = w.get("label", "?")
            lines.append(f"  Волна {label}: {_pct(prev, price)} → {_pct(w['price'], price)}")
            for sw in wave_subwaves.get(label, []) or []:
                sub_lbl = sw.get("label", "?")
                lines.append(f"      {sub_lbl}: {_pct(prev, price)} → {_pct(sw['price'], price)}")
                prev = sw["price"]
            prev = w["price"]

    if imp_waves:
        _print(imp_waves, anchor_price, f"Импульс от {_pct(anchor_price, price)}:")

    if corr_waves:
        last_imp_price = imp_waves[-1]["price"] if imp_waves else anchor_price
        lines.append("")
        _print(corr_waves, last_imp_price, f"Коррекция от {_pct(last_imp_price, price)}:")

    # Стадия волны + цель окончания — ОБЯЗАТЕЛЬНО по v8.1 (тело раздела)
    if corr_waves:
        current = f"коррекция, волна {corr_waves[-1].get('label', '?')}"
    elif imp_waves:
        last_label = imp_waves[-1].get("label", "?")
        current = (
            f"завершение импульса (волна {last_label}), начало коррекции"
            if last_label == "V"
            else f"импульс, волна {last_label} в развитии"
        )
    else:
        current = "разметка не сформирована"

    lines.append("")
    lines.append(f"Стадия: {current}")

    wave_targets = data.get("wave_targets", []) or []
    if wave_targets:
        lines.append(f"Цель окончания: {_pct(wave_targets[0]['price'], price)}")

    return "\n".join(lines)


def _format_s05(data, price, include_conclusion=True):
    """Раздел 5: ДИВЕРГЕНЦИИ (ЯДРО v8.1).
    RSI / CVD / Stochastic / MACD — зоны, кроссы, дивергенции.
    Без сырых числовых полей MACD/Signal/Hist (debug).
    """
    lines = []
    rsi = data.get("rsi_current")
    if rsi is not None:
        lines.append(f"RSI: {rsi:.1f} ({data.get('rsi_zone', '—')})")

    # MACD — только направление/кросс/дивергенция
    macd = data.get("macd")
    if macd:
        lines.append(
            f"MACD: {macd.get('position', '—')}, кросс — {macd.get('cross', '—')}, "
            f"гистограмма {macd.get('histogram_trend', '—')}"
        )
    macd_div = data.get("macd_divergence")
    if macd_div and macd_div != "нет":
        lines.append(f"MACD дивергенция: {macd_div}")

    # RSI дивергенции
    divs = data.get("divergences", {}) or {}
    for key, d in divs.items():
        if d.get("detected"):
            name = key.replace("_", " ").title()
            lines.append(f"RSI дивергенция: {name}")

    cvd = data.get("cvd_divergence")
    if cvd and cvd != "нет":
        lines.append(f"CVD дивергенция: {cvd}")

    # Stochastic — зона, кросс, дивергенция
    stoch = data.get("stochastic") or {}
    if stoch:
        lines.append(
            f"Stochastic: {stoch.get('zone','—')}, кросс — {stoch.get('cross','—')}"
        )
        sd = stoch.get("divergence")
        if sd and sd != "нет":
            lines.append(f"Stochastic дивергенция: {sd}")

    if include_conclusion:
        rsi_val = data.get("rsi_current")
        active_divs = []
        for key, d in data.get("divergences", {}).items():
            if d.get("detected"):
                active_divs.append(key.replace("_", " "))
        cvd_div = data.get("cvd_divergence", "нет")
        div_str = ", ".join(active_divs) if active_divs else "нет"

        lines.append("")
        rsi_disp = f"{rsi_val:.1f}" if isinstance(rsi_val, (int, float)) else "—"
        lines.append(
            f"📍 Вывод: RSI {rsi_disp}. Дивергенции: {div_str}. "
            f"CVD: {cvd_div}. MACD дивергенция: {macd_div or '—'}. "
            f"Stochastic: {stoch.get('zone','—') if stoch else '—'}."
        )

    return "\n".join(lines)


def _format_s06(data, price, include_conclusion=True):
    """Раздел 6: УРОВНИ (ЯДРО v8.1).
    Только: 5R + 5S с подписью источника.
    Формат: «R2 — 3.297 (+2.14%)» (label — price (+%)).
    """
    lines = []
    res = data.get("resistances", data.get("resistances_5", [])) or []
    sup = data.get("supports", data.get("supports_5", [])) or []

    lines.append("Сопротивления:")
    for r in res[:5]:
        lbl = r.get("label", "?")
        lines.append(f"  {lbl} — {_pct(r['price'], price)}")
    lines.append("")
    lines.append("Поддержки:")
    for s in sup[:5]:
        lbl = s.get("label", "?")
        lines.append(f"  {lbl} — {_pct(s['price'], price)}")

    if include_conclusion:
        tgts = [_pct(r["price"], price) for r in res[:2]] + [_pct(s["price"], price) for s in sup[:2]]
        lines.append("")
        lines.append(
            f"📍 Вывод: Ближайшие магниты — {', '.join(tgts)}." if tgts
            else "📍 Вывод: Уровни не определены."
        )

    return "\n".join(lines)


def _format_s07(data, price, include_conclusion=True):
    """Раздел 7: ФИБОНАЧЧИ (ЯДРО v8.1 — одна сетка по актуальному тренду)."""
    lines = []

    block = data.get("senior_trend", {})
    if not block or "error" in block:
        return "—"

    start_val = block.get("start", "?")
    end_val = block.get("end", "?")
    if isinstance(start_val, dict):
        start_str = _pct(start_val.get("price", 0), price) if start_val.get("price") else "?"
    else:
        start_str = str(start_val)
    if isinstance(end_val, dict):
        end_str = _pct(end_val.get("price", 0), price) if end_val.get("price") else "?"
    else:
        end_str = str(end_val)
    lines.append(f"Актуальный тренд: {start_str} → {end_str}")

    for r in block.get("retracements", []):
        lines.append(f"  {r['level']}: {_pct(r['price'], price)}")
    for e in block.get("extensions", []):
        lines.append(f"  ext {e['level']}: {_pct(e['price'], price)}")

    return "\n".join(lines)


def _format_s09(data, price, include_conclusion=True):
    """Раздел 9: ОБЪЁМНЫЕ ЗОНЫ (ЯДРО v8.1).
    Один профиль по активному диапазону. Фаза, POC/VAH/VAL, TPO.
    """
    lines = []
    p = data.get("profile_a") or data
    if not p or "POC" not in p:
        return "—"

    phase = p.get("phase") or p.get("anchor_reason", "")
    bars = p.get("bars_count")
    if phase:
        lines.append(f"Фаза: {phase}")
    if bars:
        lines.append(f"Баров в профиле: {bars}")

    for k in ("POC", "VAH", "VAL"):
        v = p.get(k, {})
        if isinstance(v, dict) and "price" in v:
            lines.append(f"{k}: {_pct(v['price'], price)}")

    # TPO
    tpo = p.get("tpo", {})
    if tpo and "error" not in tpo:
        tpo_poc = tpo.get("tpo_poc", {})
        tpo_vah = tpo.get("tpo_vah", {})
        tpo_val = tpo.get("tpo_val", {})
        if tpo_poc.get("price"):
            lines.append("")
            lines.append("TPO:")
            lines.append(f"  POC: {_pct(tpo_poc['price'], price)}")
            if tpo_vah.get("price"):
                lines.append(f"  VAH: {_pct(tpo_vah['price'], price)}")
            if tpo_val.get("price"):
                lines.append(f"  VAL: {_pct(tpo_val['price'], price)}")

    # Итоговый вывод (по profile_a, если есть)
    main = data.get("profile_a") if "profile_a" in data else data
    position = (main or {}).get("position", "—")
    magnet = (main or {}).get("nearest_magnet", "—")

    if include_conclusion:
        poc_v = (main or {}).get("POC", {})
        poc_p = poc_v.get("price", 0) if isinstance(poc_v, dict) else 0
        lines.append(f"📍 Вывод: POC {_pct(poc_p, price)}, {position}. Магнит: {magnet}.")

    return "\n".join(lines)


def _format_s12(data, price, include_conclusion=True):
    """Раздел 12: ТЕМП РЫНКА (ЯДРО v8.1).
    ATR(ТФ), ATR(D14), k-темпа, формула сроков ceil(Dist% / (k × ATR_д% × F)).
    """
    lines = []
    atr_tf = data.get("atr_tf", "?"); atr_tf_pct = data.get("atr_tf_pct", "?")
    atr_d = data.get("atr_daily", "?"); atr_d_pct = data.get("atr_daily_pct", "?")
    k = data.get("k_tempo", "?"); tc = data.get("tempo_class", "—")

    lines.append(f"ATR(ТФ): {atr_tf} ({atr_tf_pct}%)")
    lines.append(f"ATR(D14): {atr_d} ({atr_d_pct}%)")
    lines.append(f"k-темпа: {k} ({tc})")
    lines.append("")
    lines.append("Сроки: ceil(Dist% / (k × ATR_д% × F))")
    lines.append("  F: 1.0 тренд / 0.7 коррекция / 1.3 пробой")

    if include_conclusion:
        lines.append("")
        lines.append(f"📍 Вывод: ATR(D)={atr_d_pct}%, k={k} ({tc}).")

    return "\n".join(lines)


def _format_s13(data, price, include_conclusion=True):
    """Раздел 13: FVG / ГЭПЫ."""
    lines = []
    fvgs = data.get("open_fvgs", [])
    if fvgs:
        lines.append(f"Незакрытых FVG: {len(fvgs)}")
        for f in fvgs[:3]:
            lines.append(f"  {f.get('type','')}: {f.get('bottom','?')}—{f.get('top','?')}")
    gaps = data.get("open_gaps", [])
    if gaps:
        lines.append(f"Незакрытых гэпов: {len(gaps)}")

    # v8: first_step (явные поля)
    fs_dir = data.get("first_step_dir")
    fs_reason = data.get("first_step_reason")
    fs = data.get("first_step") or {}
    if fs_dir:
        lines.append(
            f"Первый шаг: {fs_dir} "
            f"→ {_pct(fs.get('target', 0), price) if fs.get('target') else '—'} "
            f"({fs.get('type','—')}, {fs.get('distance_atr','—')}×ATR)"
        )
    if fs_reason:
        lines.append(f"  Причина: {fs_reason}")

    if include_conclusion:
        lines.append("")
        lines.append(
            f"📍 Вывод: FVG: {len(fvgs)}, гэпов: {len(gaps)}. "
            f"Первый шаг: {fs_dir or '—'} ({fs_reason or '—'})."
        )

    return "\n".join(lines)


def _format_s14(data, price, include_conclusion=True):
    """Раздел 14: BB / KC / SQUEEZE (ЯДРО v8.1).
    %B, Width, Percentile, Squeeze.
    """
    lines = []
    bb = data.get("bollinger", {}) or {}
    if bb:
        lines.append(
            f"Bollinger: %B={bb.get('pct_b', '?')}, "
            f"Width={bb.get('bb_width', '?')}%, "
            f"Percentile={bb.get('width_percentile', '?')}"
        )
    sq = "активен" if data.get("squeeze_active") else "нет"
    lines.append(f"Squeeze: {sq} ({data.get('squeeze_bars', 0)} баров)")
    lines.append(f"Фаза волатильности: {data.get('vol_phase', '—')}")

    if include_conclusion:
        lines.append("")
        lines.append(f"📍 Вывод: Фаза волатильности — {data.get('vol_phase', '—')}. "
                     f"{'Squeeze активен — готовность к импульсу.' if data.get('squeeze_active') else ''}")

    return "\n".join(lines)


def _format_s15(data, price, include_conclusion=True):
    """Раздел 15: ЭФФЕКТИВНОСТЬ."""
    lines = []

    # ER по всем окнам
    er_by_period = data.get("er_by_period", {})
    if er_by_period:
        er_parts = []
        for w in [10, 20, 50, 100]:
            val = er_by_period.get(w)
            if val is not None:
                er_parts.append(f"ER({w})={val}")
        if er_parts:
            lines.append(", ".join(er_parts))
    else:
        lines.append(f"ER: {data.get('efficiency_ratio', '?')}")

    lines.append(f"Классификация: {data.get('er_classification', '—')}")
    lines.append(f"Profit Factor: {data.get('profit_factor', '?')}")
    lines.append(f"Win Rate: {data.get('win_rate_pct', '?')}%")

    # v8: ADX(14)
    adx = data.get("adx") or {}
    if adx.get("adx") is not None:
        lines.append(
            f"ADX(14): {adx.get('adx')} (режим: {adx.get('regime','—')}), "
            f"+DI={adx.get('plus_di','?')}, -DI={adx.get('minus_di','?')}, "
            f"направление: {adx.get('trend_dir','—')}"
        )
    elif adx:
        lines.append(f"ADX(14): {adx.get('regime','нет данных')}")

    if include_conclusion:
        lines.append("")
        er_class = (data.get("er_classification") or "—").strip()
        conf = (data.get("confirmation") or "").strip()
        # Не дублируем er_class в confirmation (часто confirmation начинается с него)
        conf_part = ""
        if conf and not conf.lower().startswith(er_class.lower()):
            conf_part = f" {conf}."
        adx_part = (
            f" ADX(14)={adx.get('adx','—')} ({adx.get('regime','—')})."
            if adx.get("adx") is not None else ""
        )
        lines.append(f"📍 Вывод: {er_class}.{conf_part}{adx_part}")

    return "\n".join(lines)


def _format_s16(data, price, include_conclusion=True):
    """Раздел 16: ПОТОКОВЫЕ."""
    """Раздел 16: ПОТОКОВЫЕ ИНДИКАТОРЫ (ЯДРО v8.1).
    MFI(14), CMF(20), CVD направление, дивергенции MFI/CMF.
    """
    lines = []
    mfi = data.get("mfi", {}) or {}
    if mfi.get("value") is not None:
        lines.append(f"MFI(14): {mfi['value']} ({mfi.get('zone', '—')})")
    cmf = data.get("cmf", {}) or {}
    if cmf.get("value") is not None:
        lines.append(f"CMF(20): {cmf['value']} ({cmf.get('signal', '—')})")
    cvd = data.get("cvd", {}) or {}
    if cvd:
        lines.append(f"CVD направление: {cvd.get('direction', '—')}")
        if cvd.get("conflict"):
            lines.append("  ⚠ Конфликт окон 20/50 — перелом потока")
    divs = data.get("divergences", []) or []
    for d in divs:
        lines.append(f"Дивергенция: {d}")

    if include_conclusion:
        signals = data.get("summary_signals", []) or []
        cvd_dir = cvd.get("direction", "—") if cvd else "—"
        # Фильтруем дубли: убираем CVD-сигналы (CVD уже выведен отдельно)
        filtered = [s for s in signals if not s.lower().startswith("cvd")]
        lines.append("")
        if filtered:
            lines.append(f"📍 Вывод: CVD {cvd_dir}. {'; '.join(filtered[:2])}.")
        else:
            lines.append(f"📍 Вывод: CVD {cvd_dir}.")

    return "\n".join(lines)


def _format_s18(data, price, include_conclusion=True):
    """Раздел 18: КОРРЕЛЯЦИИ."""
    lines = []
    lines.append(f"Close-Volume корреляция: {data.get('close_volume_corr', '?')} — {data.get('volume_confirms_movement', '—')}")
    spreads = data.get("alma_spreads", {})
    if spreads:
        for k, v in spreads.items():
            lines.append(f"  {k}: {v}")
    lines.append(f"ALMA веер: {data.get('alma_fan_state', '—')}")
    lines.append(f"Автокорреляция: {data.get('autocorrelation_lag1', '?')} → {data.get('market_regime', '—')}")

    if include_conclusion:
        lines.append("")
        lines.append(f"📍 Вывод: Режим — {data.get('market_regime', '—')}. "
                     f"ALMA — {data.get('alma_fan_state', '—')}.")

    return "\n".join(lines)


def _format_generic(data, price):
    """Универсальный форматтер для секций без специального."""
    lines = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for k2, v2 in v.items():
                lines.append(f"  {k2}: {v2}")
        elif isinstance(v, list):
            lines.append(f"{k}: [{len(v)} элементов]")
            for item in v[:5]:
                if isinstance(item, dict):
                    summary = ", ".join(f"{kk}={vv}" for kk, vv in list(item.items())[:4])
                    lines.append(f"  {summary}")
                else:
                    lines.append(f"  {item}")
            if len(v) > 5:
                lines.append(f"  ... и ещё {len(v) - 5}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _format_s20(meta: dict, sections_map: dict, price: float) -> str:
    """Раздел 20 — ВЫВОД по регламенту v6 (бывший 19).

    Абзац >=300 слов + ровно 6 буллитов.
    Все цены с (±X.XX%).
    """
    import math
    lines = []
    ticker = meta.get("ticker", "—")

    # --- Шапка раздела 19 (по ЯДРУ v8) ---
    from ai.prompts import _infer_analysis_type
    exchange = meta.get("exchange", "—")
    tf = meta.get("timeframe", "—")
    period_end = str(meta.get("period_end", ""))[:16]
    analysis_type = (
        meta.get("analysis_type_name")
        or meta.get("analysis_type")
        or _infer_analysis_type(tf)
    )
    lines.append(f"📘ТИКЕР: #{ticker} (#{exchange})")
    lines.append(f"БИРЖА: #{exchange}")
    lines.append(f"ТИП АНАЛИЗА: {analysis_type}")
    lines.append(f"ЦЕНА: {price}")
    lines.append(f"ДАТА И ВРЕМЯ: {period_end} UTC+3")
    lines.append("❗ НЕ ЯВЛЯЕТСЯ ИИР ❗")
    lines.append("")

    # --- Данные из всех разделов ---
    s1 = sections_map.get(1, {}).get("data", {})
    s2 = sections_map.get(2, {}).get("data", {})
    s3 = sections_map.get(3, {}).get("data", {})
    s4 = sections_map.get(4, {}).get("data", {})
    s5 = sections_map.get(5, {}).get("data", {})
    s6 = sections_map.get(6, {}).get("data", {})
    s7 = sections_map.get(7, {}).get("data", {})
    s8 = sections_map.get(8, {}).get("data", {})
    s9 = sections_map.get(9, {}).get("data", {})
    s10 = sections_map.get(10, {}).get("data", {})
    s11 = sections_map.get(11, {}).get("data", {})
    s12 = sections_map.get(12, {}).get("data", {})
    s13 = sections_map.get(13, {}).get("data", {})
    s14 = sections_map.get(14, {}).get("data", {})
    s15 = sections_map.get(15, {}).get("data", {})
    s16 = sections_map.get(16, {}).get("data", {})
    s17 = sections_map.get(17, {}).get("data", {})
    s18 = sections_map.get(18, {}).get("data", {})

    direction = s1.get("direction", "—")
    current_direction = s1.get("current_direction", direction)
    alma_order = s1.get("alma_order", "—")
    alma_200 = s1.get("alma_200_position", "—")
    trend_start = s1.get("trend_start", {})
    linreg = s1.get("linreg", {})

    rsi_val = s5.get("rsi_current")
    rsi_zone = s5.get("rsi_zone", "—")
    div_signal = s5.get("signal", "—")

    er_class = s15.get("er_classification", "—")
    er_val = s15.get("efficiency_ratio", 0)
    pf = s15.get("profit_factor", 0)
    wr = s15.get("win_rate_pct", 50)

    vol_phase = s14.get("vol_phase", "—")
    squeeze = s14.get("squeeze_active", False)

    regime = s18.get("market_regime", "—")
    alma_fan = s18.get("alma_fan_state", "—")
    autocorr = s18.get("autocorrelation_lag1", 0)

    k_tempo = s12.get("k_tempo", 1.0)
    tempo_class = s12.get("tempo_class", "—")
    atr_current = s12.get("atr_daily", s12.get("atr_current", 0))
    atr_pct = s12.get("atr_daily_pct", s12.get("atr_pct", 0))
    atr_dynamics = s12.get("atr_dynamics", "—")

    # VP: данные могут быть в profile_a/profile_b или напрямую
    _vp_main = s9.get("profile_a", s9) if "profile_a" in s9 else s9
    poc = _vp_main.get("POC", {})
    vah = _vp_main.get("VAH", {})
    val = _vp_main.get("VAL", {})
    vp_position = _vp_main.get("position", "—")

    mfi = s16.get("mfi", {})
    cmf = s16.get("cmf", {})
    cvd = s16.get("cvd", {})

    inst_signal = s17.get("institutional_signal", "—")
    balance = s17.get("supply_demand_balance", "—")

    first_step = s13.get("first_step")

    # --- Route Engine v6 (9-шаговый алгоритм) ---
    from pipeline.route_engine import build_route
    # tf_hours из S12 или meta
    _tf_str = meta.get("timeframe", "4H")
    _tf_map = {"15m": 0.25, "1H": 1, "4H": 4, "1D": 24}
    _tf_h = _tf_map.get(_tf_str, 4)
    for _k, _v in _tf_map.items():
        if _k.lower() in _tf_str.lower():
            _tf_h = _v
            break
    route_result = build_route(sections_map, price, current_direction, tf_hours=_tf_h)

    key_targets_r = route_result["key_targets"]
    # v8: второстепенные цели упразднены, sec_targets пуст для обратной совместимости
    other_targets_r = route_result["other_targets"]
    route_str = route_result["route_str"]
    slam_price = route_result["slam_price"]
    slam_str = route_result["slam_str"]
    pause_str = route_result.get("pause_str", "—")
    days = route_result["days"]
    days_detail = route_result["days_detail"]
    manip = route_result["manipulation"]

    # (Route Engine v6 вычислил всё выше)

    # ═══════════════════════════════════════════
    # АБЗАЦ (>=300 слов, собран из выводов разделов, ЯДРО v8.1)
    # Структура: opening → история прихода → ТА-разбор → характер движения → слом
    # ═══════════════════════════════════════════
    para = []
    # Раздел 1 — тренд + v8.1 «история прихода к цене»
    ts_price = trend_start.get("price", price)
    ts_pct = trend_start.get("pct_from_current", 0)
    # v8.1 obligatory opening phrase
    from ai.prompts import _ticker_display_name
    name_human = _ticker_display_name(ticker)
    para.append(f"Сейчас {name_human} (#{ticker}) торгуется по {_fmt_price(price)}.")
    # v8.1 история прихода — упали/выросли от … за N баров
    sen_bars = trend_start.get("bars_since_start") or trend_start.get("bars")
    if ts_price and price and ts_price != price:
        try:
            d_pct = (price - float(ts_price)) / float(ts_price) * 100
            # _pct сам возвращает «цена (%)», поэтому _fmt_price отдельно не нужен
            verb = "Выросли от" if d_pct >= 0 else "Упали от"
            bars_phrase = f" за {int(sen_bars)} баров" if sen_bars else ""
            para.append(
                f"{verb} {_pct(ts_price, price)}{bars_phrase} — "
                f"это структурный якорь старшего движения."
            )
        except Exception:
            para.append(
                f"Стоим на {_fmt_price(price)} после движения от уровня {_fmt_price(ts_price)}."
            )
    else:
        para.append(f"Удерживается на {_fmt_price(price)}, балансируя в текущей структуре.")
    para.append(
        f"Направление {current_direction} (глобально {direction}). "
        f"Порядок ALMA {alma_order}, цена {alma_200} ALMA 200. "
        f"LinReg канал показывает наклон {linreg.get('slope_pct_per_bar', 0):.4f}%/бар "
        f"при R²={linreg.get('r_squared', 0):.2f}, положение в канале {linreg.get('position_pct', 50):.1f}%."
    )
    # Раздел 2 — волны (impulse_count / correction_count из S02)
    imp_cnt = s2.get("impulse_count", 0)
    corr_cnt = s2.get("correction_count", 0)
    para.append(
        f"Волновая структура содержит {imp_cnt} импульсных и {corr_cnt} коррекционных поворотных точек."
    )
    # Раздел 4 — свечи
    stats = s4.get("stats", {})
    para.append(
        f"Свечной анализ за 20 баров: {stats.get('bull_count', 0)} бычьих, "
        f"{stats.get('bear_count', 0)} медвежьих, среднее тело {stats.get('avg_body_pct', 0)}%."
    )
    # Раздел 5 — дивергенции
    rsi_str = f"RSI {rsi_val:.1f}" if rsi_val else "RSI —"
    para.append(
        f"{rsi_str} ({rsi_zone}). Дивергенции: {div_signal}."
    )
    # Раздел 6 — пивот P
    _pivots = s6.get("pivots", {})
    _daily_pv = _pivots.get("daily", {})
    _weekly_pv = _pivots.get("weekly", {})
    if _daily_pv.get("P"):
        _p_parts = [f"дневной P {_pct(_daily_pv['P'], price)}"]
        if _weekly_pv.get("P"):
            _p_parts.append(f"недельный P {_pct(_weekly_pv['P'], price)}")
        para.append(
            f"Классические пивоты: {', '.join(_p_parts)} — ключевые уровни баланса."
        )
    # Раздел 8 — VSA
    obv_dir = s8.get("obv_direction", "—")
    vl_rel = s8.get("vl_relative", 1.0)
    para.append(
        f"VSA: OBV {obv_dir}, относительный V/L = {vl_rel:.2f}. "
        f"Аномалии объёма: {len(s8.get('volume_anomalies', []))}."
    )
    # Раздел 9 — VP
    poc_p = poc.get("price", 0) if isinstance(poc, dict) else 0
    para.append(
        f"Volume Profile: POC {_pct(poc_p, price)}, {vp_position}."
    )
    # Раздел 10 — Вайкофф
    wyckoff_struct = s10.get("structure_type", "—")
    wyckoff_phase = s10.get("phase", "—")
    wyckoff_node = s10.get("current_node", "—")
    para.append(
        f"Вайкофф: {wyckoff_struct}, фаза {wyckoff_phase}, узел: {wyckoff_node}."
    )
    # Раздел 12 — темп
    para.append(
        f"ATR {atr_current} ({atr_pct}%), K-темпа {k_tempo} ({tempo_class}), динамика {atr_dynamics}."
    )
    # Раздел 13 — FVG
    fvg_count = len(s13.get("open_fvgs", []))
    gap_count = len(s13.get("open_gaps", []))
    para.append(
        f"Открытых FVG: {fvg_count}, гэпов: {gap_count}."
    )
    # Раздел 14 — squeeze
    para.append(
        f"Волатильность: {vol_phase}. {'Squeeze активен.' if squeeze else ''}"
    )
    # Раздел 15 — эффективность
    para.append(
        f"Efficiency Ratio {er_val:.3f} ({er_class}), Profit Factor {pf:.2f}, Win Rate {wr:.1f}%."
    )
    # Раздел 16 — потоки
    mfi_v = mfi.get("value")
    cmf_v = cmf.get("value")
    cvd_dir = cvd.get("direction", "—")
    para.append(
        f"MFI {'%.1f' % mfi_v if mfi_v else '—'} ({mfi.get('zone', '—')}), "
        f"CMF {'%.4f' % cmf_v if cmf_v else '—'} ({cmf.get('signal', '—')}), "
        f"CVD: {cvd_dir}."
    )
    # Раздел 17 — микроструктура
    para.append(
        f"Институциональный след: {inst_signal}. Баланс: {balance}."
    )
    # Раздел 18 — корреляции
    para.append(
        f"ALMA веер: {alma_fan}. Режим рынка: {regime} (автокорр. {autocorr:.4f})."
    )
    # Обобщение
    para.append(
        f"Характер движения: {'импульс' if er_class == 'тренд' else 'консолидация' if er_class == 'боковик' else 'неопределённость'}. "
        f"Сценарий сохраняется при удержании цены {'выше' if direction == 'восходящий' else 'ниже'} "
        f"{slam_str}. "
        f"Пробой этого уровня с закреплением потребует пересмотра маршрута."
    )

    lines.append(" ".join(para))
    lines.append("")

    # ═══════════════════════════════════════════
    # РОВНО 6 БУЛЛИТОВ (ЯДРО v8) — БЕЗ ТА-терминов
    # Порядок (обязателен): ключевые / второстепенные / остальные /
    # плановый маршрут / вероятные сроки / уровень слома
    # ═══════════════════════════════════════════

    # v8.1: чек-лист манипуляции и отдельная строка ⚠ МАНИПУЛЯЦИЯ — убраны.
    # По заданию манипуляция должна быть В АБЗАЦЕ как обоснование шага маршрута,
    # а не диагностической таблицей.

    # ── Горизонт из route_engine (v8): реальный диапазон из days_min/days_max ──
    horizon_label = route_result.get("horizon_label", "")
    days_min = route_result.get("days_min") or days
    days_max = route_result.get("days_max") or days
    # интрадей — специальный случай (TF ≤ 15m)
    if horizon_label == "интрадей":
        dur_range = "интрадей"
    elif days_min and days_max and days_min != days_max:
        # правильное склонение: 1 день / 2-4 дня / 5+ дней
        def _days_word(n):
            n_last = n % 10
            n_two = n % 100
            if 11 <= n_two <= 14:
                return "дней"
            if n_last == 1:
                return "день"
            if 2 <= n_last <= 4:
                return "дня"
            return "дней"
        dur_range = f"{days_min}–{days_max} {_days_word(days_max)}"
    else:
        dur_range = f"{days} дней"

    # 5 буллитов v8.1 — строго без ТА-терминов, без диагностики K/ATR/F
    # «отмена маршрута» = объединённый slam (приостановка+слом из v8 свёрнуты в один уровень)
    lines.append(f"📌 ключевые цели: {', '.join(key_targets_r) if key_targets_r else '—'}")
    lines.append(f"📌 остальные цели: {', '.join(other_targets_r) if other_targets_r else '—'}")
    lines.append(f"📌 вероятный маршрут: {route_str}")
    lines.append(f"📌 вероятные сроки: {dur_range}")
    lines.append(f"📌 отмена маршрута: {slam_str}")

    return "\n".join(lines)


def _validate_conclusion_v8(text: str) -> dict:
    """Проверка блока вывода на соответствие ЯДРУ v8.

    Возвращает dict с найденными проблемами и количеством буллитов.
    Используется как подстраховка, когда AI формирует вывод.
    """
    issues = []
    # v8.1: ровно 5 буллитов
    n_bullets = text.count("📌")
    if n_bullets != 5:
        issues.append(f"буллитов {n_bullets}, ожидалось 5")
    # Порядок буллитов v8.1
    expected = [
        "ключевые цели",
        "остальные цели",
        "вероятный маршрут",
        "вероятные сроки",
        "отмена маршрута",
    ]
    idx = 0
    for line in text.split("\n"):
        if "📌" in line and idx < 5:
            if expected[idx] not in line.lower():
                issues.append(f"буллит {idx + 1}: ожидалось '{expected[idx]}'")
            idx += 1

    # ТА-термины в буллитах запрещены
    ta_terms = [
        "rsi", "alma", "wyckoff", "вайкофф", "фибо", "fibo", "fvg", "poc",
        "vah", "val", "mfi", "cmf", "cvd", "atr", "obv", "macd", "bollinger",
        "keltner", "squeeze", "stoch", "adx", "efi", "k-темпа",
    ]
    for line in text.split("\n"):
        if "📌" in line:
            low = line.lower()
            for t in ta_terms:
                if t in low:
                    issues.append(f"ТА-термин '{t}' в буллите: {line[:70]}")
                    break

    # Абзац ≥300 слов (берём самый длинный абзац между шапкой и буллитами)
    lines_raw = text.split("\n")
    paragraphs = []
    buf = []
    for ln in lines_raw:
        if ln.strip() == "":
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
        else:
            if "📌" in ln or ln.startswith("━"):
                if buf:
                    paragraphs.append(" ".join(buf))
                    buf = []
                continue
            buf.append(ln)
    if buf:
        paragraphs.append(" ".join(buf))
    max_words = max((len(p.split()) for p in paragraphs), default=0)
    if max_words < 300:
        issues.append(f"абзац короче 300 слов (макс {max_words})")

    # v8: обязательная первая фраза «Сейчас ... торгуется по ...»
    has_opening = False
    for p in paragraphs:
        low = p.lower().strip()
        if low.startswith("сейчас ") and "торгуется по" in low:
            has_opening = True
            break
    if not has_opening:
        issues.append("нет обязательной вступительной фразы «Сейчас … торгуется по …»")

    # v8: шапка должна содержать «ТИП АНАЛИЗА»
    if "ТИП АНАЛИЗА" not in text:
        issues.append("в шапке отсутствует поле «ТИП АНАЛИЗА»")

    # v8.1: запрещены старые формулировки v8
    forbidden = [
        "условия отмены",
        "приостановка и наблюдение",
        "слом структуры и пересмотр",
    ]
    low_full = text.lower()
    for ph in forbidden:
        if ph in low_full:
            issues.append(f"запрещённая v8-фраза «{ph}» найдена (используй «отмена маршрута»)")

    # v8.1: вторая фраза — «история прихода к цене» (упали/выросли/стоим)
    has_history = False
    for p in paragraphs:
        low = p.lower()
        if any(w in low for w in (
            "упали от", "выросли от", "стоим на", "падали от", "подросли от",
            "снизились от", "поднялись от", "удерживается на", "торгуется в диапазоне",
            "формируется на",
        )):
            has_history = True
            break
    if not has_history:
        issues.append("нет краткой истории прихода к цене (упали/выросли/стоим…)")

    # v8.1: характер движения — импульс / манипуляция / сползание
    char_terms = ("импульс", "манипуляц", "сползан", "тренд", "коррекц")
    if not any(t in low_full for t in char_terms):
        issues.append("не описан характер движения (импульс / манипуляция / сползание / тренд / коррекция)")

    # v8.1: «отмена маршрута» обязательна (заменяет старые приостановку/слом)
    if "отмена маршрута" not in low_full:
        issues.append("отсутствует буллит «отмена маршрута» (v8.1)")

    return {"n_bullets": n_bullets, "issues": issues, "paragraph_words": max_words}


def _build_fallback_prose(analysis_json: dict, sections_map: dict,
                          meta: dict) -> str:
    """Резервный прозаический абзац (≥300 слов) с русскими связками.

    Используется, когда AI-вывод не прошёл валидацию v8 даже после retry.
    Собирает факты из секций в связный текст через «кроме того», «в свою
    очередь», «наконец» и добавляет деталей из самой глубокой доступной
    секции, если слов < 300.
    """
    price = meta.get("price") or meta.get("current_price") or 0
    ticker = meta.get("ticker", "—")
    tf = meta.get("tf", "—")

    def _s(sid):
        s = sections_map.get(sid) or {}
        return s.get("data", s) if isinstance(s, dict) else {}

    s1 = _s(1); s2 = _s(2); s5 = _s(5); s6 = _s(6); s7 = _s(7)
    s9 = _s(9); s10 = _s(10); s12 = _s(12); s13 = _s(13); s15 = _s(15)

    sent = []
    sen_dir = (s1.get("senior_trend", {}) or {}).get("direction", s1.get("direction", "—"))
    loc_dir = (s1.get("local_trend", {}) or {}).get("direction", s1.get("current_direction", "—"))
    # v8: обязательная вступительная фраза
    from ai.prompts import _ticker_display_name
    name = _ticker_display_name(ticker)
    price_str = _fmt_price(price) if price else "—"
    sent.append(
        f"Сейчас {name} (#{ticker}) торгуется по {price_str}."
    )
    # v8.1: краткая история прихода к цене (упали/выросли от … за N сессий)
    sen_start = (s1.get("senior_trend", {}) or {}).get("start_price")
    sen_bars = (s1.get("senior_trend", {}) or {}).get("bars_since_start")
    if sen_start and price:
        try:
            delta_pct = (float(price) - float(sen_start)) / float(sen_start) * 100
            verb = "выросли от" if delta_pct >= 0 else "упали от"
            bars_phrase = f" за {sen_bars} баров" if sen_bars else ""
            sent.append(
                f"Котировки {verb} {sen_start}{bars_phrase}, что задаёт "
                f"контекст текущего движения."
            )
        except Exception:
            sent.append(
                f"Стоим на текущем уровне после движения от {sen_start}."
            )
    else:
        sent.append(
            f"Стоим на {price_str}, отрабатывая структуру предыдущих сессий."
        )
    sent.append(
        f"На таймфрейме {tf} старший тренд определён как {sen_dir}, "
        f"тогда как локальная структура идёт {loc_dir}."
    )

    stage = (s1.get("senior_trend", {}) or {}).get("stage") or s1.get("stage", "—")
    sent.append(
        f"Стадия старшего движения оценивается как «{stage}», "
        f"что задаёт контекст для распределения вероятностей на ближайшие сессии."
    )

    wave = s2.get("current_wave") or s2.get("wave_label") or "—"
    sent.append(
        f"В разметке волн текущее положение — {wave}, и это согласуется с "
        f"наблюдаемой динамикой импульсов."
    )

    rsi = s5.get("rsi_current"); rsi_zone = s5.get("rsi_zone", "—")
    sent.append(
        f"Кроме того, осцилляторы дают следующий срез: RSI на уровне "
        f"{rsi if rsi is not None else '—'} ({rsi_zone}), "
        f"а общий сигнал дивергенций — «{s5.get('signal', '—')}»."
    )

    res5 = (s6.get("resistances_5") or [])[:2]
    sup5 = (s6.get("supports_5") or [])[:2]
    res_str = ", ".join(f"{r.get('label','R')}={r.get('price','?')}" for r in res5) or "—"
    sup_str = ", ".join(f"{s.get('label','S')}={s.get('price','?')}" for s in sup5) or "—"
    sent.append(
        f"В свою очередь, ключевые уровни сопротивления распределены как {res_str}, "
        f"а поддержки — как {sup_str}, что формирует рабочий коридор цены."
    )

    fs = s13.get("first_step") or {}
    fs_dir = s13.get("first_step_dir") or fs.get("direction", "—")
    fs_reason = s13.get("first_step_reason") or "—"
    sent.append(
        f"Ближайший шаг направлен {fs_dir}: {fs_reason}."
    )

    k_hint = s12.get("k_influence_hint", "neutral")
    k_tempo = s12.get("k_tempo", "—")
    sent.append(
        f"Темп рынка характеризуется коэффициентом k={k_tempo} "
        f"(подсказка роутеру: {k_hint}), что влияет на ранжирование целей."
    )

    adx = (s15.get("adx") or {}).get("adx")
    adx_regime = (s15.get("adx") or {}).get("regime", "—")
    sent.append(
        f"Оценка ADX на уровне {adx if adx is not None else '—'} "
        f"говорит о режиме «{adx_regime}», что подтверждает или ограничивает "
        f"валидность трендовых сигналов."
    )

    phase = s10.get("phase") or s10.get("current_phase") or "—"
    node = s10.get("current_node") or "—"
    sent.append(
        f"Наконец, по Вайкоффу структура находится в фазе {phase}, "
        f"ближайший узел — {node}, что позволяет оценить готовность "
        f"крупного игрока к следующему импульсу."
    )

    prose = " ".join(sent)

    # pad if <300 words — добавляем факты из самой глубокой доступной секции
    while len(prose.split()) < 300:
        extra = None
        poc_a = (s9.get("profile_a", {}) or {}).get("POC")
        if poc_a and isinstance(poc_a, dict) and poc_a.get("price"):
            extra = (
                f"Дополнительно отметим, что POC профиля A расположен на "
                f"{poc_a.get('price')} и выступает магнитом для цены в пределах сессии."
            )
        elif s7.get("senior_grid"):
            fib = (s7.get("senior_grid") or {}).get("retracements") or {}
            if fib:
                parts = [f"{k}={v}" for k, v in list(fib.items())[:3]]
                extra = (
                    "Кроме того, сетка Фибоначчи старшего тренда показывает "
                    f"ключевые откаты: {', '.join(parts)}."
                )
        if not extra:
            extra = (
                "Таким образом, совокупность факторов указывает на необходимость "
                "дождаться подтверждения ближайшего уровня, прежде чем наращивать "
                "позицию в направлении старшего тренда."
            )
        prose += " " + extra
        if len(prose.split()) > 340:
            break

    return prose


def _format_s03(data, price, include_conclusion=True):
    """Раздел 3: ГРАФИЧЕСКИЕ ПАТТЕРНЫ (ЯДРО v8.1).
    Только: формирующиеся / подтверждённые паттерны с направлением и целью.
    """
    lines = []
    patterns = data.get("patterns", []) or []
    # Нормализация статуса в две категории v8.1
    def _norm_status(s):
        s = (s or "").lower()
        if "формир" in s:
            return "формирующийся"
        return "подтверждённый"

    if patterns:
        for p in patterns:
            name_ru = p.get("name_ru", p.get("name", "?"))
            status = _norm_status(p.get("status"))
            direction = p.get("direction", "?")
            target = p.get("target")
            line = f"  {name_ru} — {status}, {direction}"
            if target:
                line += f", цель {_pct(target, price)}"
            lines.append(line)
    else:
        lines.append("Явных паттернов не обнаружено.")

    if include_conclusion:
        names = ", ".join(p.get("name_ru", "?") for p in patterns[:3])
        lines.append("")
        lines.append(
            f"📍 Вывод: {names if patterns else 'паттерны не выявлены'}."
        )

    return "\n".join(lines)


def _format_s04(data, price, include_conclusion=True):
    """Раздел 4: СВЕЧНЫЕ ПАТТЕРНЫ (ЯДРО v8.1).
    Только: классические свечные паттерны японских свечей.
    """
    lines = []
    named = data.get("named_patterns", []) or []

    if named:
        for p in named:
            lines.append(f"  {p.get('name', '?')} (бар {p.get('bar', '?')}) — {p.get('signal', '—')}")
    else:
        lines.append("Свечные паттерны не выявлены.")

    if include_conclusion:
        pat_names = ", ".join(p.get("name", "?") for p in named[:3]) if named else "нет"
        lines.append("")
        lines.append(f"📍 Вывод: Паттерны — {pat_names}.")

    return "\n".join(lines)


def _format_s08(data, price, include_conclusion=True):
    """Раздел 8: VSA (Volume Spread Analysis)."""
    lines = []
    vl_cur = data.get("vl_current", 0)
    vl_avg = data.get("vl_avg_20", 0)
    vl_rel = data.get("vl_relative", 0)

    lines.append(f"Объём текущий: {vl_cur:.0f}")
    lines.append(f"Объём средний (20): {vl_avg:.0f}")
    lines.append(f"Относительный объём: {vl_rel:.3f}")

    obv_dir = data.get("obv_direction", "—")
    obv_slope = data.get("obv_slope_5", 0)
    lines.append(f"OBV: {obv_dir} (наклон 5: {obv_slope:.0f})")

    cvd = data.get("cvd", {})
    if cvd:
        lines.append(f"CVD: {cvd.get('current', '?')} ({cvd.get('direction', '—')}), "
                     f"Δ5: {cvd.get('change_5', '?')}")

    # Аномалии объёма
    anomalies = data.get("volume_anomalies", [])
    if anomalies:
        lines.append("")
        lines.append(f"Аномалии объёма ({len(anomalies)}):")
        for a in anomalies[:4]:
            zscore = a.get("volume_zscore", 0)
            direction = a.get("direction", "?")
            body = a.get("body_pct", 0)
            lines.append(f"  бар {a.get('bar', '?')}: z={zscore:.1f}, {direction}, тело {body:.1f}%")

    if include_conclusion:
        anom_count = len(anomalies)
        lines.append("")
        lines.append(f"📍 Вывод: OBV {obv_dir}, V/L = {vl_rel:.2f}. "
                     f"Аномалий: {anom_count}. CVD: {cvd.get('direction', '—')}.")

    return "\n".join(lines)


def _format_s10(data, price, include_conclusion=True):
    """Раздел 10: ВАЙКОФФ (ЯДРО v8.1).
    Только: тип (Накопление/Распределение) + фаза A-E + ближайший узел.
    """
    lines = []
    struct = data.get("structure_type", "—")
    phase = data.get("phase", "—")
    phase_desc = data.get("phase_description", "")
    node = data.get("current_node", "—")
    next_node = data.get("next_node", "—")

    lines.append(f"Структура: {struct}")
    if phase_desc:
        lines.append(f"Фаза: {phase} — {phase_desc}")
    else:
        lines.append(f"Фаза: {phase}")
    lines.append(f"Текущий узел: {node}")
    lines.append(f"Ближайший узел: {next_node}")

    if include_conclusion:
        lines.append("")
        lines.append(f"📍 Вывод: {struct}, фаза {phase}. Узел: {node} → {next_node}.")

    return "\n".join(lines)


def _format_s11(data, price, include_conclusion=True):
    """Раздел 11: ЗОНЫ СБОРА СТОПОВ (ЯДРО v8.1).
    Только: хвосты, выносы (sweep), консолидации, ложные пробои — кластеры
    swing-точек. Без debug-дампов dict.
    """
    lines = []
    stops_below = data.get("stops_below_supports", []) or []
    stops_above = data.get("stops_above_resistances", []) or []
    narrow = data.get("narrow_consolidations", []) or []
    tails = data.get("tail_bars", []) or []

    if stops_below:
        lines.append(f"Стопы под поддержками ({len(stops_below)}):")
        for s in stops_below[:5]:
            lo = s.get("zone_lo", 0); hi = s.get("zone_hi", 0)
            zone = f"зона {lo:.4g}—{hi:.4g}" if lo and hi and abs(hi - lo) > 0.01 else _pct(s.get("level", 0), price)
            lines.append(f"  {zone}")

    if stops_above:
        lines.append("")
        lines.append(f"Стопы над сопротивлениями ({len(stops_above)}):")
        for s in stops_above[:5]:
            lo = s.get("zone_lo", 0); hi = s.get("zone_hi", 0)
            zone = f"зона {lo:.4g}—{hi:.4g}" if lo and hi and abs(hi - lo) > 0.01 else _pct(s.get("level", 0), price)
            lines.append(f"  {zone}")

    if narrow:
        lines.append("")
        lines.append(f"Консолидации ({len(narrow)}):")
        for n in narrow[:3]:
            lo = n.get("zone_low", 0); hi = n.get("zone_high", 0)
            rng = n.get("range_pct", 0); win = n.get("window", 0)
            lines.append(f"  {lo:.4g}—{hi:.4g} ({rng:.2f}%, {win} баров)")

    if tails:
        lines.append("")
        lines.append(f"Длинные хвосты ({len(tails)}):")
        for t in tails[:4]:
            tt = t.get("tail_type", "?")
            side = "снизу" if "lower" in str(tt) else "сверху"
            lines.append(f"  бар -{t.get('bar_offset', '?')}: хвост {side}")

    if include_conclusion:
        lines.append("")
        lines.append(
            f"📍 Вывод: Стопы — {len(stops_below)} зон снизу, "
            f"{len(stops_above)} зон сверху, консолидаций {len(narrow)}."
        )

    return "\n".join(lines)


def _format_s17(data, price, include_conclusion=True):
    """Раздел 17: МИКРОСТРУКТУРА (ЯДРО v8.1).
    |ΔCVD|/V, VWAP отклонение в ATR, бары >2σ, инст. след.
    """
    lines = []
    delta_cvd_v = data.get("delta_cvd_over_volume")
    if delta_cvd_v is None:
        delta_cvd_v = data.get("abs_delta_cvd_per_volume")
    if delta_cvd_v is not None:
        lines.append(f"|ΔCVD|/Volume: {delta_cvd_v}")

    vwap = data.get("vwap", {}) or {}
    if vwap.get("deviation_atr") is not None:
        lines.append(f"VWAP отклонение: {vwap.get('deviation_atr')} ATR ({vwap.get('position', '—')})")

    anomalies = data.get("volume_anomalies", []) or []
    if anomalies:
        lines.append(f"Бары >2σ: {len(anomalies)}")

    inst = data.get("institutional_signal")
    if inst:
        lines.append(f"Институциональный след: {inst}")

    if include_conclusion:
        lines.append("")
        lines.append(f"📍 Вывод: {inst or 'нейтрально'}.")

    return "\n".join(lines) if lines else "—"


SECTION_FORMATTERS = {
    1: _format_s01,
    2: _format_s02,
    3: _format_s03,
    4: _format_s04,
    5: _format_s05,
    6: _format_s06,
    7: _format_s07,
    8: _format_s08,
    9: _format_s09,
    10: _format_s10,
    11: _format_s11,
    12: _format_s12,
    13: _format_s13,
    14: _format_s14,
    15: _format_s15,
    16: _format_s16,
    17: _format_s17,
    18: _format_s18,
}


def _format_s19_math(data, price, include_conclusion=True):
    """Раздел 20 — МАТЕМАТИЧЕСКАЯ МОДЕЛЬ (v2.6.3)."""
    lines = []

    if not isinstance(data, dict):
        lines.append("Ошибка: данные не являются словарём.")
        return "\n".join(lines)

    if data.get("error"):
        lines.append(f"Ошибка: {data['error']}")
        return "\n".join(lines)

    current = price or 0

    # ── helpers ──────────────────────────────────────────────────────

    def _time_human(val):
        """Время в человеческом формате (часы/дни) из числа часов."""
        if val is None:
            return "N/A"
        try:
            h = float(val)
        except (TypeError, ValueError):
            return str(val)
        if h < 1:
            return f"{h * 60:.0f} мин"
        if h < 24:
            return f"{h:.1f} ч"
        days = h / 24
        if days < 1.05:
            return "~1 день"
        return f"~{days:.1f} дн"

    def _safe_pct(target_price):
        """±% от текущей цены."""
        if not current or current <= 0:
            return 0.0
        try:
            return (float(target_price) - current) / current * 100
        except (TypeError, ValueError):
            return 0.0

    def _prices_close(a, b, tol=0.3):
        """True если цены в пределах tol% друг от друга."""
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return False
        if fa == 0 and fb == 0:
            return True
        if max(abs(fa), abs(fb)) == 0:
            return False
        return abs(fa - fb) / max(abs(fa), abs(fb)) * 100 <= tol

    def _safe_float(val, default=0.0):
        """Безопасное приведение к float."""
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _manip_verdict(score):
        """ДА / ВОЗМОЖНА / НЕТ по score (0-1 или 0-100)."""
        s = _safe_float(score, 0)
        # normalise: if passed as 0..1
        if 0 < s <= 1:
            s = s * 100
        if s > 60:
            return "ДА"
        if s >= 30:
            return "ВОЗМОЖНА"
        return "НЕТ"

    def _manip_score_pct(score):
        """Нормализовать score манипуляции к процентам."""
        s = _safe_float(score, 0)
        if 0 < s <= 1:
            s = s * 100
        return s

    def _format_route_step(step):
        """Форматировать один шаг маршрута."""
        if isinstance(step, dict):
            sp = _safe_float(step.get("price", 0))
            lbl = step.get("label", "") or step.get("level", "")
            if sp and current:
                price_str = _pct(sp, current)
            elif sp:
                price_str = _fmt_price(sp)
            else:
                price_str = "?"
            if lbl:
                return f"[{lbl} {price_str}]"
            return f"[{price_str}]"
        # scalar price or string
        try:
            fp = float(step)
            if current:
                return f"[{_pct(fp, current)}]"
            return f"[{_fmt_price(fp)}]"
        except (TypeError, ValueError):
            return f"[{step}]"

    # ================================================================
    # 1. ЦЕЛИ S/R С ВЕРОЯТНОСТЯМИ
    # ================================================================
    lines.append("1. ЦЕЛИ S/R С ВЕРОЯТНОСТЯМИ")
    lines.append("")

    scored = data.get("scored_targets") or []
    if not isinstance(scored, list):
        scored = []
    math_levels = data.get("math_levels") or []
    if not isinstance(math_levels, list):
        math_levels = []

    # Build merged list: {price, pct, p_touch, time, status, label}
    merged = []
    used_math_idx = set()

    for t in scored:
        if not isinstance(t, dict):
            continue
        tp = _safe_float(t.get("price", 0))
        if not tp:
            continue
        pct_val = t.get("pct")
        pct = _safe_float(pct_val) if pct_val is not None else _safe_pct(tp)
        p_touch = _safe_float(t.get("p_touch_pct", t.get("p_touch", 0)))
        time_val = t.get("t_first")
        status = "TECHNIQUE"
        label = t.get("label", "") or ""

        # check if any math level is close -> RESONANCE
        for mi, ml in enumerate(math_levels):
            if not isinstance(ml, dict):
                continue
            ml_price = _safe_float(ml.get("price", 0))
            if _prices_close(tp, ml_price):
                used_math_idx.add(mi)
                ml_p = _safe_float(ml.get("p_touch_pct", ml.get("p_touch", 0)))
                p_touch = max(p_touch, ml_p)
                if time_val is None:
                    time_val = ml.get("t_first")
                status = "RESONANCE"
                break

        merged.append({
            "price": tp, "pct": pct, "p_touch": p_touch,
            "time": time_val, "status": status, "label": label,
        })

    # add math levels not matched to scored
    for mi, ml in enumerate(math_levels):
        if mi in used_math_idx:
            continue
        if not isinstance(ml, dict):
            continue
        mp = _safe_float(ml.get("price", 0))
        if not mp:
            continue
        pct_val = ml.get("pct")
        merged.append({
            "price": mp,
            "pct": _safe_float(pct_val) if pct_val is not None else _safe_pct(mp),
            "p_touch": _safe_float(ml.get("p_touch_pct", ml.get("p_touch", 0))),
            "time": ml.get("t_first"),
            "status": "MATH",
            "label": ml.get("label", "") or "",
        })

    STATUS_ICONS = {
        "RESONANCE": "\U0001F525 RESONANCE",
        "TECHNIQUE": "\U0001F4CA TECHNIQUE",
        "MATH":      "\U0001F9EE MATH",
    }

    if merged:
        above = sorted([t for t in merged if t["pct"] > 0], key=lambda x: x["price"])
        below = sorted([t for t in merged if t["pct"] <= 0], key=lambda x: x["price"], reverse=True)

        header = f"  {'#':>3}  {'Цена':>12}  {'±%':>8}  {'P(touch)':>8}  {'Время':>10}  Статус"
        lines.append(header)

        idx = 1
        for t in above:
            tm = _time_human(t["time"])
            lbl = t["label"]
            name_str = f"{lbl} " if lbl else ""
            st_str = STATUS_ICONS.get(t["status"], t["status"])
            lines.append(
                f"  {idx:>3}  {name_str}{_fmt_price(t['price']):>12}  "
                f"{t['pct']:>+7.2f}%  {t['p_touch']:>6.1f}%  {tm:>10}  {st_str}"
            )
            idx += 1

        lines.append(f"  ======= S\u2080 {_fmt_price(current)} ==================")

        for t in below:
            tm = _time_human(t["time"])
            lbl = t["label"]
            name_str = f"{lbl} " if lbl else ""
            st_str = STATUS_ICONS.get(t["status"], t["status"])
            lines.append(
                f"  {idx:>3}  {name_str}{_fmt_price(t['price']):>12}  "
                f"{t['pct']:>+7.2f}%  {t['p_touch']:>6.1f}%  {tm:>10}  {st_str}"
            )
            idx += 1
    else:
        lines.append("  Нет данных по целям.")

    # ================================================================
    # 2. ПРИОРИТЕТ ЦЕЛЕЙ
    # ================================================================
    lines.append("")
    lines.append("2. ПРИОРИТЕТ ЦЕЛЕЙ")
    lines.append("")

    if merged:
        above_sorted = sorted(
            [t for t in merged if t["pct"] > 0], key=lambda x: x["price"],
        )
        below_sorted = sorted(
            [t for t in merged if t["pct"] <= 0], key=lambda x: x["price"], reverse=True,
        )

        nearest_up = above_sorted[0] if above_sorted else None
        nearest_down = below_sorted[0] if below_sorted else None
        far_up = above_sorted[-1] if len(above_sorted) > 1 else None
        far_down = below_sorted[-1] if len(below_sorted) > 1 else None

        def _target_line(t):
            lbl = t.get("label", "")
            prefix = f"{lbl} " if lbl else ""
            return (f"{prefix}{_pct(t['price'], current)} — "
                    f"P(touch)={t['p_touch']:.1f}%, ~{_time_human(t['time'])}")

        if nearest_up:
            lines.append(f"  \u2B06 Ближайшая сверху: {_target_line(nearest_up)}")
        if nearest_down:
            lines.append(f"  \u2B07 Ближайшая снизу: {_target_line(nearest_down)}")

        # which is reached first
        if nearest_up and nearest_down:
            p_up_val = nearest_up["p_touch"]
            p_down_val = nearest_down["p_touch"]
            diff = abs(p_up_val - p_down_val)
            if p_up_val >= p_down_val:
                first = nearest_up
                first_p = p_up_val
            else:
                first = nearest_down
                first_p = p_down_val
            first_lbl = first.get("label", "")
            first_name = f"{first_lbl} " if first_lbl else ""
            lines.append(
                f"  \U0001F3AF Первой достигается: {first_name}"
                f"{_fmt_price(first['price'])} — "
                f"P={first_p:.1f}% (разница P: +{diff:.1f} п.п.)"
            )

        if far_up:
            lines.append(f"  \u2B06 Дальняя сверху: {_target_line(far_up)}")
        if far_down:
            lines.append(f"  \u2B07 Дальняя снизу: {_target_line(far_down)}")

        # which far target is reached earlier
        if far_up and far_down:
            if far_up["p_touch"] >= far_down["p_touch"]:
                far_first = far_up
            else:
                far_first = far_down
            far_lbl = far_first.get("label", "")
            far_name = f"{far_lbl} " if far_lbl else ""
            lines.append(
                f"  \U0001F3AF Раньше достигается дальняя: "
                f"{far_name}{_fmt_price(far_first['price'])} "
                f"(P={far_first['p_touch']:.1f}%)"
            )
    else:
        lines.append("  Нет данных.")

    # ================================================================
    # 3. НАПРАВЛЕНИЕ ПО МОДЕЛЯМ
    # ================================================================
    lines.append("")
    lines.append("3. НАПРАВЛЕНИЕ ПО МОДЕЛЯМ")
    lines.append("")

    model_probs = data.get("model_probs") or {}
    if isinstance(model_probs, list):
        # convert list of dicts [{model: ..., p_up: ...}] to dict
        _mp = {}
        for item in model_probs:
            if isinstance(item, dict):
                name = item.get("model", item.get("name", f"model_{len(_mp)+1}"))
                _mp[str(name)] = item
        model_probs = _mp
    if not isinstance(model_probs, dict):
        model_probs = {}

    if model_probs:
        lines.append(
            f"  {'Модель':<14}  {'Направление':<10}  "
            f"{'P(up)':>7}  {'P(down)':>7}  {'P(flat)':>7}  {'Conf':>6}"
        )
        for model, probs in model_probs.items():
            if not isinstance(probs, dict):
                continue
            p_up = _safe_float(probs.get("p_up", 0))
            p_down = _safe_float(probs.get("p_down", 0))
            p_flat = _safe_float(probs.get("p_flat", 0))
            # Direction
            if p_flat > p_up and p_flat > p_down:
                direction = "БОКОВИК"
            elif p_up >= p_down:
                direction = "ВВЕРХ"
            else:
                direction = "ВНИЗ"
            # Confidence
            conf = abs(max(p_up, p_down) - min(p_up, p_down))
            # Format probabilities: normalise to display
            def _prob_str(v):
                if v <= 1:
                    return f"{v:>6.1%}"
                return f"{v:>5.1f}%"
            lines.append(
                f"  {model:<14}  {direction:<10}  "
                f"{_prob_str(p_up)}  {_prob_str(p_down)}  {_prob_str(p_flat)}  "
                f"{conf * 100 if conf <= 1 else conf:>5.1f}%"
            )
    else:
        lines.append("  Нет данных моделей.")

    # ================================================================
    # 4. МАНИПУЛЯЦИЯ ПО МОДЕЛЯМ
    # ================================================================
    lines.append("")
    lines.append("4. МАНИПУЛЯЦИЯ ПО МОДЕЛЯМ")
    lines.append("")

    manip = data.get("manipulation") or {}
    if not isinstance(manip, dict):
        manip = {}

    if manip:
        m_score = _manip_score_pct(manip.get("score", 0))
        m_dir = manip.get("direction", "—")
        if isinstance(m_dir, dict):
            m_dir = m_dir.get("direction", "—")
        m_verdict = _manip_verdict(manip.get("score", 0))
        m_depth_price = manip.get("depth_price")
        m_depth_pct = manip.get("depth_pct")
        depth_str = ""
        if m_depth_price is not None:
            depth_str = f", глубина до {_pct(_safe_float(m_depth_price), current)}"
        elif m_depth_pct is not None:
            depth_str = f", глубина {_safe_float(m_depth_pct):+.2f}%"
        lines.append(f"  Манипуляция: {m_verdict}")
        lines.append(f"  P(манип) = {m_score:.1f}%, направление: {m_dir}{depth_str}")
    else:
        lines.append("  Нет данных.")

    # ================================================================
    # 5. ТРИ МАРШРУТА-ПОБЕДИТЕЛЯ
    # ================================================================
    lines.append("")
    lines.append("5. ТРИ МАРШРУТА-ПОБЕДИТЕЛЯ")
    lines.append("")

    route_main = data.get("route_main") or []
    route_alt = data.get("route_alt") or []
    route_third = data.get("route_third") or []
    if not isinstance(route_main, list):
        route_main = [route_main] if route_main else []
    if not isinstance(route_alt, list):
        route_alt = [route_alt] if route_alt else []
    if not isinstance(route_third, list):
        route_third = [route_third] if route_third else []

    def _format_route(steps):
        if not steps:
            return "нет данных"
        parts = [f"S\u2080"]
        for step in steps:
            parts.append(_format_route_step(step))
        return " \u2192 ".join(parts)

    lines.append(f"  \U0001F947 Основной:       {_format_route(route_main)}")
    lines.append(f"  \U0001F948 Альтернативный: {_format_route(route_alt)}")
    if route_third:
        lines.append(f"  \U0001F949 Третий:         {_format_route(route_third)}")
    else:
        lines.append(f"  \U0001F949 Третий:         нет данных")

    # ================================================================
    # 5.1. ДЕТАЛЬНЫЙ МАРШРУТ (Route Engine)
    # ================================================================
    lines.append("")
    lines.append("5.1. ДЕТАЛЬНЫЙ МАРШРУТ (Route Engine)")
    lines.append("")

    # Use main route for detail; show table if steps have rich fields
    detail_route = route_main
    if detail_route and isinstance(detail_route[0], dict) and (
        "label" in detail_route[0] or "type" in detail_route[0] or "level" in detail_route[0]
    ):
        lines.append(f"  {'Шаг':>4}  {'Цена':>14}  {'Метка':<14}  {'Тип':<10}")
        for si, step in enumerate(detail_route, 1):
            if not isinstance(step, dict):
                continue
            sp = _safe_float(step.get("price", 0))
            lbl = step.get("label", "") or step.get("level", "") or ""
            stype = step.get("type", "") or step.get("kind", "") or "—"
            price_str = _pct(sp, current) if sp and current else _fmt_price(sp) if sp else "—"
            lines.append(f"  {si:>4}  {price_str:>14}  {lbl:<14}  {stype:<10}")
    elif detail_route:
        lines.append(f"  Маршрут: {_format_route(detail_route)}")
    else:
        lines.append("  Детальный маршрут не построен.")

    # ================================================================
    # 6. БАЙЕС-АНСАМБЛЬ
    # ================================================================
    lines.append("")
    lines.append("6. БАЙЕС-АНСАМБЛЬ")
    lines.append("")

    bayes = data.get("bayes") or {}
    if not isinstance(bayes, dict):
        bayes = {}
    posterior = bayes.get("posterior") or {}

    if posterior:
        lines.append(f"  {'Сценарий':<24}  {'P(j|D)':>8}")
        if isinstance(posterior, dict):
            for scenario, prob in posterior.items():
                p = _safe_float(prob)
                p_str = f"{p:.1%}" if p <= 1 else f"{p:.1f}%"
                lines.append(f"  {str(scenario):<24}  {p_str:>8}")
        elif isinstance(posterior, list):
            for i, prob in enumerate(posterior):
                p = _safe_float(prob)
                p_str = f"{p:.1%}" if p <= 1 else f"{p:.1f}%"
                lines.append(f"  {'Сценарий ' + str(i + 1):<24}  {p_str:>8}")

        # best_scenario — handle dict / int / str / None
        best_raw = bayes.get("best_scenario")
        if isinstance(best_raw, dict):
            center = best_raw.get("center")
            share = best_raw.get("share")
            rng = best_raw.get("range")
            bs_parts = []
            if center is not None:
                bs_parts.append(f"центр {_pct(_safe_float(center), current) if current else _fmt_price(_safe_float(center))}")
            if share is not None:
                s = _safe_float(share)
                bs_parts.append(f"доля {s:.1%}" if s <= 1 else f"доля {s:.1f}%")
            if rng is not None:
                if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                    bs_parts.append(
                        f"диапазон {_fmt_price(_safe_float(rng[0]))} — "
                        f"{_fmt_price(_safe_float(rng[1]))}"
                    )
                else:
                    bs_parts.append(f"диапазон {rng}")
            # fallback: show any other key-val
            if not bs_parts:
                for k, v in best_raw.items():
                    bs_parts.append(f"{k}={v}")
            lines.append(f"  Лучший сценарий: {', '.join(bs_parts)}")
        elif isinstance(best_raw, (int, float)):
            idx_bs = int(best_raw)
            lines.append(f"  Лучший сценарий: Сценарий {idx_bs}")
        elif best_raw is not None:
            lines.append(f"  Лучший сценарий: {best_raw}")
        else:
            lines.append("  Лучший сценарий: —")

        # weights
        weights = bayes.get("weights")
        if isinstance(weights, dict) and weights:
            lines.append("")
            lines.append(f"  {'Модель':<14}  {'Вес':>8}")
            for k, v in weights.items():
                lines.append(f"  {str(k):<14}  {_safe_float(v):>7.3f}")
        elif isinstance(weights, list) and weights:
            lines.append("")
            lines.append(f"  {'#':<6}  {'Вес':>8}")
            for wi, w in enumerate(weights):
                lines.append(f"  {wi + 1:<6}  {_safe_float(w):>7.3f}")
    else:
        lines.append("  Нет данных Байес-ансамбля.")

    # ================================================================
    # 7. ИТОГОВЫЙ ВЗВЕШЕННЫЙ РАСЧЁТ
    # ================================================================
    lines.append("")
    lines.append("7. ИТОГОВЫЙ ВЗВЕШЕННЫЙ РАСЧЁТ")
    lines.append("")

    # weighted averages from model_probs
    if model_probs:
        vals = [v for v in model_probs.values() if isinstance(v, dict)]
        n = len(vals) or 1
        avg_up = sum(_safe_float(p.get("p_up", 0)) for p in vals) / n
        avg_down = sum(_safe_float(p.get("p_down", 0)) for p in vals) / n
        avg_flat = sum(_safe_float(p.get("p_flat", 0)) for p in vals) / n
        # normalise display
        def _avg_str(v):
            return f"{v:.1%}" if v <= 1 else f"{v:.1f}%"
        lines.append(f"  P_up   = {_avg_str(avg_up)}")
        lines.append(f"  P_down = {_avg_str(avg_down)}")
        lines.append(f"  P_flat = {_avg_str(avg_flat)}")
    else:
        lines.append("  P_up / P_down / P_flat: нет данных")
        avg_up = avg_down = avg_flat = 0

    # manipulation verdict
    if manip:
        mv = _manip_verdict(manip.get("score", 0))
        lines.append(f"  Манипуляция: {mv}")

    # agreement from racing
    racing = data.get("racing") or {}
    if not isinstance(racing, dict):
        racing = {}
    agreement_raw = racing.get("agreement")

    if isinstance(agreement_raw, (int, float)):
        ag = _safe_float(agreement_raw)
        if ag <= 1:
            ag_pct = ag
        else:
            ag_pct = ag / 100
        if ag_pct >= 0.7:
            ag_label = "ВЫСОКАЯ"
        elif ag_pct >= 0.4:
            ag_label = "СРЕДНЯЯ"
        else:
            ag_label = "НИЗКАЯ"
        lines.append(f"  Согласованность: {ag_label} ({ag_pct:.0%})")
    elif isinstance(agreement_raw, str) and agreement_raw:
        lines.append(f"  Согласованность: {agreement_raw}")
    else:
        # compute from conflicts
        conflicts = racing.get("conflicts") or []
        if not isinstance(conflicts, list):
            conflicts = []
        dominant = racing.get("dominant", "—")
        if isinstance(dominant, dict):
            dominant = dominant.get("dominant", "—")
        if conflicts:
            ag_label = "НИЗКАЯ"
            lines.append(
                f"  Согласованность: {ag_label} "
                f"(конфликты: {', '.join(str(c) for c in conflicts)})"
            )
        elif dominant and dominant != "—":
            lines.append(f"  Согласованность: ВЫСОКАЯ (доминирует {dominant})")
        else:
            lines.append("  Согласованность: —")

    # conflicts list
    conflicts_list = racing.get("conflicts") or []
    if isinstance(conflicts_list, list) and conflicts_list:
        lines.append(f"  Конфликты: {', '.join(str(c) for c in conflicts_list)}")

    # ================================================================
    # 8. ВЕРДИКТ
    # ================================================================
    if include_conclusion:
        lines.append("")
        lines.append("8. ВЕРДИКТ")
        lines.append("")

        # ── consensus direction ──
        if model_probs:
            vals = [v for v in model_probs.values() if isinstance(v, dict)]
            n = len(vals) or 1
            c_up = sum(_safe_float(p.get("p_up", 0)) for p in vals) / n
            c_down = sum(_safe_float(p.get("p_down", 0)) for p in vals) / n
            if c_down > c_up + 0.05:
                cons_dir = "ВНИЗ"
            elif c_up > c_down + 0.05:
                cons_dir = "ВВЕРХ"
            else:
                cons_dir = "БЕЗ ВЫРАЖЕННОГО НАПРАВЛЕНИЯ"
            c_up_s = f"{c_up:.0%}" if c_up <= 1 else f"{c_up:.0f}%"
            c_dn_s = f"{c_down:.0%}" if c_down <= 1 else f"{c_down:.0f}%"
        else:
            cons_dir = "—"
            c_up_s = c_dn_s = "?"

        # ── dominant model ──
        dominant = racing.get("dominant", "—")
        if isinstance(dominant, dict):
            dominant = dominant.get("dominant", dominant.get("model", "—"))
        dominant_str = str(dominant) if dominant else "—"

        # ── manipulation short ──
        m_score_val = _manip_score_pct(manip.get("score", 0)) if manip else 0
        m_dir_val = manip.get("direction", "—") if manip else "—"
        if isinstance(m_dir_val, dict):
            m_dir_val = m_dir_val.get("direction", "—")

        # ── best target ──
        best_target_str = "—"
        if merged:
            best = max(merged, key=lambda x: x["p_touch"])
            best_lbl = best.get("label", "")
            best_prefix = f"{best_lbl} " if best_lbl else ""
            best_target_str = (
                f"{best_prefix}{_pct(best['price'], current)} "
                f"(P={best['p_touch']:.1f}%)"
            )

        # ── paragraph ──
        lines.append(
            f"  Консенсус моделей: {cons_dir} "
            f"(P\u2191={c_up_s}, P\u2193={c_dn_s}). "
            f"Доминирует: {dominant_str}. "
            + (f"Манипуляция: P={m_score_val:.0f}%, {m_dir_val}. " if m_score_val >= 30 else "")
            + f"Наиболее вероятная цель: {best_target_str}."
        )
        lines.append("")

        # ── 6 bullets v8 ──

        # ключевые цели
        key_targets = []
        if merged:
            top2 = sorted(merged, key=lambda x: x["p_touch"], reverse=True)[:2]
            for t in top2:
                lbl = t.get("label", "")
                prefix = f"{lbl} " if lbl else ""
                key_targets.append(f"{prefix}{_pct(t['price'], current)}")
        lines.append(
            f"  \U0001F4CC ключевые цели: {', '.join(key_targets) if key_targets else '—'}"
        )

        # остальные цели
        other_targets = []
        if merged and len(merged) > 2:
            rest = sorted(merged, key=lambda x: x["p_touch"], reverse=True)[2:]
            for t in rest:
                lbl = t.get("label", "")
                prefix = f"{lbl} " if lbl else ""
                other_targets.append(f"{prefix}{_pct(t['price'], current)}")
        lines.append(
            f"  \U0001F4CC остальные цели: {', '.join(other_targets) if other_targets else '—'}"
        )

        # вероятный маршрут
        if route_main:
            route_str = " \u2192 ".join(
                _format_route_step(s) for s in route_main
            )
        else:
            route_str = "не определён"
        lines.append(f"  \U0001F4CC вероятный маршрут: S\u2080 \u2192 {route_str}")

        # вероятные сроки
        if merged:
            senior_target = max(merged, key=lambda x: abs(x["pct"]))
            sr_time = _time_human(senior_target.get("time"))
            lines.append(f"  \U0001F4CC вероятные сроки: {sr_time}")
        else:
            lines.append(f"  \U0001F4CC вероятные сроки: —")

        # отмена маршрута (v8.1: объединённый уровень slam, без pause)
        break_level = data.get("break_level") or data.get("invalidation")
        if isinstance(break_level, dict):
            bl_price = _safe_float(break_level.get("price", 0))
            bl_str = _pct(bl_price, current) if bl_price and current else str(break_level)
        elif isinstance(break_level, (int, float)):
            bl_str = _pct(_safe_float(break_level), current) if current else _fmt_price(_safe_float(break_level))
        elif break_level:
            bl_str = str(break_level)
        else:
            bl_str = "—"
        lines.append(f"  \U0001F4CC отмена маршрута: {bl_str}")

    return "\n".join(lines)


SECTION_FORMATTERS[20] = _format_s19_math
