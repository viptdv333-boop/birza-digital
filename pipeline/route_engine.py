"""
Биржа-цифровой — Route Engine (ЯДРО v8).

Полный 9-шаговый алгоритм построения маршрута:
1. Собрать ВСЕ цели из всех разделов 1–18
2. Классифицировать (ключевые 3+, второстепенные 1–2, остальные 1). ВСЕ цели → в маршрут.
3. Магниты ликвидности (стопы, FVG, тонкие зоны VP)
4. Институциональные зоны (POC, аномалии объёма)
5. Проверить манипуляцию по 6-признаковому чек-листу v8
6. Фильтр очерёдности (статусы A/B/C, k_темпа)
7. Уровень слома = КЛЮЧЕВАЯ цель в маршруте, противоположная финальной
8. Очистка (<0.3% → зона X–Y)
9. Сроки: ceil(total_dist% / (k × ATR_d% × F)), F=1.0/0.7/1.3

Правила v8:
- Маршрут = зигзаг через 20–30 целей
- k-темпа: >1.2 → sec→key; <0.8 → key→sec
- Первый шаг — РЕАЛЬНЫЙ ближайший шаг по графику
- total_dist% = сумма всех плеч зигзага, не расстояние старт→конец
"""
import math
import numpy as np


def collect_all_targets(sections_map: dict, price: float) -> list[dict]:
    """Собрать ВСЕ цели из разделов 1-18 (шаг 1 алгоритма маршрута).

    Возвращает список: [{"price": float, "source": str, "section": int}, ...]
    Используется и в route_engine (маршрут), и в S19 (мат.модель для P(touch)).
    """
    s = lambda sid: sections_map.get(sid, {}).get("data", {})
    s2, s3, s6, s7, s9, s10, s11, s13 = s(2), s(3), s(6), s(7), s(9), s(10), s(11), s(13)

    raw = []

    # S06 — S/R (ключи: resistances или resistances_5)
    # Разделяем источник: Pivot для пивот-уровней (D./W./M.), Swing для свингов
    for r in s6.get("resistances", s6.get("resistances_5", [])):
        lbl = r.get("label", "")
        src = "Pivot" if any(lbl.startswith(p) for p in ("D.", "W.", "M.")) else "Swing"
        raw.append({"price": r["price"], "source": src, "section": 6, "label": lbl})
    for sup in s6.get("supports", s6.get("supports_5", [])):
        lbl = sup.get("label", "")
        src = "Pivot" if any(lbl.startswith(p) for p in ("D.", "W.", "M.")) else "Swing"
        raw.append({"price": sup["price"], "source": src, "section": 6, "label": lbl})

    # S07 — Фибо (senior_trend + local_trend: retracements и extensions)
    for trend_key in ("senior_trend", "local_trend"):
        block = s7.get(trend_key, {})
        for r in block.get("retracements", []):
            raw.append({"price": r["price"], "source": "Fibo", "section": 7})
        for e in block.get("extensions", []):
            raw.append({"price": e["price"], "source": "Fibo", "section": 7})

    # S09 — VP
    for pkey in ("profile_a", "profile_b"):
        prof = s9.get(pkey, {})
        for vk in ("POC", "VAH", "VAL"):
            item = prof.get(vk, {})
            if isinstance(item, dict) and "price" in item:
                raw.append({"price": item["price"], "source": vk, "section": 9})

    # S10 — Вайкофф
    vp_w = s10.get("volume_profile", {})
    for vk in ("POC", "VAH", "VAL"):
        val = vp_w.get(vk)
        if val and val > 0:
            raw.append({"price": val, "source": f"W.{vk}", "section": 10})

    # S11 — стопы
    for st in s11.get("stops_below", s11.get("stops_below_supports", [])):
        p = st.get("stop_zone") or st.get("level")
        if p:
            raw.append({"price": p, "source": "Stop", "section": 11})
    for st in s11.get("stops_above", s11.get("stops_above_resistances", [])):
        p = st.get("stop_zone") or st.get("level")
        if p:
            raw.append({"price": p, "source": "Stop", "section": 11})

    # S02 — волновые цели
    for wt in s2.get("wave_targets", []):
        raw.append({"price": wt["price"], "source": "Wave", "section": 2})

    # S03 — паттерны
    for pat in s3.get("patterns", []):
        if pat.get("target"):
            raw.append({"price": pat["target"], "source": "Pattern", "section": 3})

    # S13 — FVG/гэпы/ликвидность
    for fvg in s13.get("open_fvgs", []):
        mid = (fvg.get("top", 0) + fvg.get("bottom", 0)) / 2
        if mid > 0:
            raw.append({"price": mid, "source": "FVG", "section": 13})
    for gap in s13.get("open_gaps", []):
        mid = (gap.get("gap_top", 0) + gap.get("gap_bottom", 0)) / 2
        if mid > 0:
            raw.append({"price": mid, "source": "Gap", "section": 13})
    for lp in s13.get("liquidity_pools", []):
        if lp.get("price"):
            raw.append({"price": lp["price"], "source": "Liquidity", "section": 13})

    # S09 — тонкие зоны
    for pkey in ("profile_a", "profile_b"):
        for ta in s9.get(pkey, {}).get("thin_areas", []):
            if ta.get("price"):
                raw.append({"price": ta["price"], "source": "ThinVP", "section": 9})

    # Отсечь мусор > 50% от цены
    raw = [t for t in raw if abs(t["price"] - price) / price < 0.50]

    # Отсечь цели слишком близкие к текущей цене (<0.3% — Регламент v4, шаг 7)
    raw = [t for t in raw if abs(t["price"] - price) / price > 0.003]

    # Группировка: цели в пределах 0.3% → оставляем одну с наивысшим приоритетом
    raw = _group_close_targets(raw)

    return raw


# Семейства индикаторов (Регламент v4, Правило 1 шаг 2):
# ключевая цель = 2+ подтверждения из РАЗНЫХ семейств
_SOURCE_FAMILY = {
    "Pivot": "structure", "Swing": "structure",
    "Fibo": "fibo",
    "POC": "volume", "VAH": "volume", "VAL": "volume",
    "W.POC": "volume", "W.VAH": "volume", "W.VAL": "volume",
    "Wyckoff_POC": "volume", "Wyckoff_VAH": "volume", "Wyckoff_VAL": "volume",
    "ThinVP": "volume",
    "Stop": "liquidity", "Liquidity": "liquidity",
    "FVG": "imbalance", "Gap": "imbalance",
    "Wave": "waves",
    "Pattern": "patterns",
}


def _count_families(sources: list) -> int:
    """Сколько РАЗНЫХ семейств индикаторов подтверждают цель."""
    return len({_SOURCE_FAMILY.get(s, s) for s in sources if s})


# Приоритет источников для группировки (меньше = выше приоритет)
_SOURCE_PRIORITY = {
    "Pivot": 0, "Swing": 0,
    "Fibo": 1,
    "POC": 2, "VAH": 2, "VAL": 2,
    "W.POC": 2, "W.VAH": 2, "W.VAL": 2,
    "Stop": 3,
    "FVG": 4, "Gap": 4,
    "ThinVP": 5,
    "Liquidity": 6,
    "Wave": 1, "Pattern": 1,
}


def _group_close_targets(targets: list[dict], threshold_pct: float = 0.3) -> list[dict]:
    """Группировать цели в пределах threshold_pct% друг от друга.

    Из каждой группы оставляем цель с наивысшим приоритетом источника.

    Используем итеративное слияние: сортируем по цене, идём по списку,
    если следующая цель в пределах threshold_pct% от СРЕДНЕЙ цены текущей группы,
    добавляем в группу и пересчитываем среднюю. Это корректно обрабатывает
    цепочки близких целей (2.92, 2.93, 2.94, 2.95, 2.96, 2.97).
    """
    if not targets:
        return targets

    # Сортировать по цене
    sorted_t = sorted(targets, key=lambda t: t["price"])
    groups: list[list[dict]] = []
    current_group = [sorted_t[0]]
    group_sum = sorted_t[0]["price"]

    for t in sorted_t[1:]:
        # Сравниваем с СРЕДНЕЙ ценой текущей группы (не с первым элементом)
        group_avg = group_sum / len(current_group)
        if group_avg > 0 and abs(t["price"] - group_avg) / group_avg * 100 <= threshold_pct:
            current_group.append(t)
            group_sum += t["price"]
        else:
            groups.append(current_group)
            current_group = [t]
            group_sum = t["price"]
    groups.append(current_group)

    # Из каждой группы оставляем цель с наивысшим приоритетом,
    # но сохраняем ВСЕ уникальные источники из группы (через запятую).
    result = []
    for group in groups:
        best = min(group, key=lambda t: _SOURCE_PRIORITY.get(t.get("source", ""), 99))
        # Собрать все уникальные источники из всех целей группы
        all_srcs = set()
        for t in group:
            s = t.get("source", "")
            if s:
                for sub in s.split(","):
                    sub = sub.strip()
                    if sub:
                        all_srcs.add(sub)
        if len(all_srcs) > 1:
            best["source"] = ",".join(sorted(all_srcs))
        result.append(best)

    return result


def build_route(sections_map: dict, price: float, direction: str, tf_hours: float = 4.0,
                k_effect: float | None = None,
                extra_sections_maps: list | None = None) -> dict:
    """Построить маршрут по 9-шаговому алгоритму v6.

    Args:
        sections_map: {section_id: {"data": {...}}, ...} — старший ТФ
        price: текущая цена
        direction: "восходящий" / "нисходящий" / "боковик"
        k_effect: взвешенное k по ТФ (старший 0.5, рабочий 0.3, младший 0.2);
            если None — используется k_tempo старшего ТФ
        extra_sections_maps: секции младших/средних ТФ — их цели УТОЧНЯЮТ
            структуру маршрута (Регламент v4: младший ТФ детализирует)

    Returns:
        {
            "key_targets": [...],
            "sec_targets": [...],
            "other_targets": [...],
            "route": [...],
            "route_str": str,
            "slam_price": float,
            "slam_str": str,
            "days": int,
            "days_detail": str,
            "manipulation": {...},
            "first_step": {...},
        }
    """
    s = lambda sid: sections_map.get(sid, {}).get("data", {})

    s1 = s(1)
    s2 = s(2)
    s3 = s(3)
    s5 = s(5)
    s6 = s(6)
    s7 = s(7)
    s8 = s(8)
    s9 = s(9)
    s10 = s(10)
    s11 = s(11)
    s12 = s(12)
    s13 = s(13)
    s16 = s(16)
    s17 = s(17)

    atr_daily = s12.get("atr_daily", 0)
    atr_daily_pct = s12.get("atr_daily_pct", 1.0)
    k_tempo = s12.get("k_tempo", 1.0)

    # ═══════════════════════════════════════════
    # ШАГ 1: Собрать ВСЕ цели
    # ═══════════════════════════════════════════
    raw_targets = []  # (price, source_name, section_id)

    def _collect_targets_from(smap: dict) -> None:
        """Собрать цели из одной карты секций (одного ТФ) в raw_targets."""
        sx = lambda sid: smap.get(sid, {}).get("data", {})
        x2, x3, x6, x7, x9, x10, x11, x13 = (
            sx(2), sx(3), sx(6), sx(7), sx(9), sx(10), sx(11), sx(13))

        # S06 — уровни S/R (Pivot для пивот-уровней D./W./M., Swing для свингов)
        for r in x6.get("resistances", x6.get("resistances_5", [])):
            lbl = r.get("label", "")
            src = "Pivot" if any(lbl.startswith(p) for p in ("D.", "W.", "M.")) else "Swing"
            raw_targets.append((r["price"], src, 6))
        for sup in x6.get("supports", x6.get("supports_5", [])):
            lbl = sup.get("label", "")
            src = "Pivot" if any(lbl.startswith(p) for p in ("D.", "W.", "M.")) else "Swing"
            raw_targets.append((sup["price"], src, 6))

        # S07 — Фибо (senior_trend + local_trend: retracements и extensions)
        for trend_key in ("senior_trend", "local_trend"):
            block = x7.get(trend_key, {})
            for r in block.get("retracements", []):
                raw_targets.append((r["price"], "Fibo", 7))
            for e in block.get("extensions", []):
                raw_targets.append((e["price"], "Fibo", 7))

        # S09 — VP (два профиля)
        for pkey in ("profile_a", "profile_b"):
            prof = x9.get(pkey, {})
            for vk in ("POC", "VAH", "VAL"):
                item = prof.get(vk, {})
                if isinstance(item, dict) and "price" in item:
                    raw_targets.append((item["price"], vk, 9))

        # S11 — стопы
        for st in x11.get("stops_below", x11.get("stops_below_supports", [])):
            p = st.get("stop_zone") or st.get("level")
            if p:
                raw_targets.append((p, "Stop", 11))
        for st in x11.get("stops_above", x11.get("stops_above_resistances", [])):
            p = st.get("stop_zone") or st.get("level")
            if p:
                raw_targets.append((p, "Stop", 11))

        # S02 — волновые цели
        for wt in x2.get("wave_targets", []):
            raw_targets.append((wt["price"], "Wave", 2))

        # S03 — паттерны
        for pat in x3.get("patterns", []):
            if pat.get("target"):
                raw_targets.append((pat["target"], "Pattern", 3))

        # S13 — FVG/гэпы
        for fvg in x13.get("open_fvgs", []):
            mid = (fvg.get("top", 0) + fvg.get("bottom", 0)) / 2
            if mid > 0:
                raw_targets.append((mid, "FVG", 13))
        for gap in x13.get("open_gaps", []):
            mid = (gap.get("gap_top", 0) + gap.get("gap_bottom", 0)) / 2
            if mid > 0:
                raw_targets.append((mid, "Gap", 13))

        # Тонкие зоны VP
        for pkey in ("profile_a", "profile_b"):
            for ta in x9.get(pkey, {}).get("thin_areas", []):
                if ta.get("price"):
                    raw_targets.append((ta["price"], "ThinVP", 9))

        # Пулы ликвидности из S13
        for lp in x13.get("liquidity_pools", []):
            if lp.get("price"):
                raw_targets.append((lp["price"], "Liquidity", 13))

        # S10 — Вайкофф (POC/VAH/VAL как цели)
        vp_wyckoff = x10.get("volume_profile", {})
        for vk in ("POC", "VAH", "VAL"):
            val = vp_wyckoff.get(vk)
            if val and val > 0:
                raw_targets.append((val, f"Wyckoff_{vk}", 10))

    # Старший ТФ задаёт цели; младшие/средние ТФ УТОЧНЯЮТ структуру
    # (Регламент v4: «каждый младший ТФ уточняет структуру маршрута»)
    _collect_targets_from(sections_map)
    for extra_map in (extra_sections_maps or []):
        _collect_targets_from(extra_map)

    # ═══════════════════════════════════════════
    # ШАГ 2: Классифицировать (детерминированно)
    # Группировка: цели ближе 1% → одна группа
    # Подсчёт РАЗНЫХ типов источников
    # ═══════════════════════════════════════════
    # Сортировка стабильная: по цене, затем по имени источника
    sorted_raw = sorted(raw_targets, key=lambda t: (t[0], t[1], t[2]))

    # Порог группировки — Регламент v4 (шаг 7): цели ближе 0.3% = одна зона
    atr_tf = s12.get("atr_tf", price * 0.01)
    group_threshold = 0.003

    groups = []  # [sum_price, count, sources_list, section_ids_list]
    for tp, src, sid in sorted_raw:
        merged = False
        # Разбить запятые из _group_close_targets (напр. "Pivot,Fibo")
        sub_srcs = [s.strip() for s in src.split(",") if s.strip()]
        for g in groups:
            avg = g[0] / g[1]  # текущее среднее
            if abs(tp - avg) / max(avg, 0.001) < group_threshold:  # ATR-адаптивная группировка
                g[0] += tp
                g[1] += 1
                for ss in sub_srcs:
                    if ss not in g[2]:
                        g[2].append(ss)
                if sid not in g[3]:
                    g[3].append(sid)
                merged = True
                break
        if not merged:
            groups.append([tp, 1, list(sub_srcs) if sub_srcs else [src], [sid]])

    # Горизонт по ТФ: сколько ATR(D) покрывает маршрут
    # 15m=1.5 ATR(D), 1H=3, 4H=5, 1D=10
    if tf_hours <= 0.25:
        horizon_atr = 1.5
    elif tf_hours <= 1:
        horizon_atr = 3.0
    elif tf_hours <= 4:
        horizon_atr = 5.0
    else:
        horizon_atr = 10.0
    max_dist_pct = horizon_atr * atr_daily_pct / 100.0 if atr_daily_pct > 0 else 0.50

    valid_groups = []
    for g in groups:
        avg_p = g[0] / g[1]
        dist = abs(avg_p - price) / price
        if dist < max_dist_pct and dist > 0.003:
            valid_groups.append((avg_p, g[1], sorted(g[2]), sorted(g[3])))

    # Сортировка по расстоянию от цены (детерминированно)
    valid_groups.sort(key=lambda g: (abs(g[0] - price), g[0]))

    key_targets = []
    other_targets = []
    sec_targets = []  # устарело, оставлено пустым для обратной совместимости сигнатур

    # Регламент v4 (Правило 1, шаг 2):
    #   Ключевые = конфлюэнс 2+ подтверждения из РАЗНЫХ СЕМЕЙСТВ индикаторов
    #   Второстепенные = 1 подтверждение или кластер близких уровней
    # k-темпа: k>1.2 → other→key (кап 8 промоушенов); k<0.8 → key→other
    # ВСЕ цели остаются в маршруте.
    for avg_p, cnt, srcs, sids in valid_groups:
        n_src = len(srcs)
        n_sections = len(sids)
        n_families = _count_families(srcs)
        entry = _fmt(avg_p, price)
        entry_src = _fmt_with_sources(avg_p, price, srcs)
        item = {
            "price": avg_p,
            "str": entry,
            "str_src": entry_src,
            "sources": srcs,
            "sections": sids,
            "n_sections": n_sections,
            "n_families": n_families,
        }

        # Пивоты (D.P, D.R1, D.S1, W.P) — структурно важны
        has_pivot = "Pivot" in srcs
        pivot_combo = has_pivot and n_src >= 2

        # Регламент: ключевая = 2+ семейства индикаторов (или пивот-комбо)
        if n_families >= 2 or pivot_combo:
            tier = "key"
        else:
            tier = "other"

        # k-темпа: промоушен/демоушен (кап 8)
        if k_tempo >= 1.2 and tier == "other":
            promoted_count = sum(1 for it in key_targets if it.get("was_promoted"))
            if promoted_count < 8 and (n_sections >= 2 or has_pivot):
                tier = "key"
                item["was_promoted"] = True
        elif k_tempo < 0.8 and tier == "key" and not pivot_combo:
            tier = "other"
            item["was_demoted"] = True

        item["tier"] = tier
        if tier == "key":
            key_targets.append(item)
        else:
            other_targets.append(item)

    # Регламент: «ни одна цель не игнорируется»; маршрут = зигзаг через 20–30 целей.
    # Мягкий потолок 30 (12+18) — защита от вырожденных данных, не рабочий лимит.
    max_key, max_other = 12, 18
    key_targets = key_targets[:max_key]
    other_targets = other_targets[:max_other]

    # Если key пуст — промотируем первые other в key (для скелета маршрута)
    if not key_targets and other_targets:
        promote_n = min(3, len(other_targets))
        for it in other_targets[:promote_n]:
            it["tier"] = "key"
        key_targets = other_targets[:promote_n]
        other_targets = other_targets[promote_n:]

    # ═══════════════════════════════════════════
    # ШАГ 3: Магниты ликвидности
    # ═══════════════════════════════════════════
    magnets = []
    # Стопы
    for st in s11.get("stops_below", s11.get("stops_below_supports", []))[:3]:
        p = st.get("stop_zone") or st.get("level")
        if p:
            magnets.append({"price": p, "type": "stop_pool", "side": "below"})
    for st in s11.get("stops_above", s11.get("stops_above_resistances", []))[:3]:
        p = st.get("stop_zone") or st.get("level")
        if p:
            magnets.append({"price": p, "type": "stop_pool", "side": "above"})
    # FVG
    fs = s13.get("first_step")
    if fs and fs.get("target"):
        magnets.append({"price": fs["target"], "type": "FVG", "side": fs.get("direction", "?")})
    # Thin VP
    for pkey in ("profile_a", "profile_b"):
        for ta in s9.get(pkey, {}).get("thin_areas", [])[:3]:
            if ta.get("price"):
                magnets.append({"price": ta["price"], "type": "thin_area"})

    # ═══════════════════════════════════════════
    # ШАГ 4: Институциональные зоны
    # ═══════════════════════════════════════════
    institutional = []
    for pkey in ("profile_a", "profile_b"):
        poc_item = s9.get(pkey, {}).get("POC", {})
        if isinstance(poc_item, dict) and "price" in poc_item:
            institutional.append({"price": poc_item["price"], "type": "POC"})
    # Аномалии объёма
    for anom in s17.get("volume_anomalies", [])[:3]:
        institutional.append({"price": anom.get("close_price", 0), "type": "vol_anomaly"})

    # ═══════════════════════════════════════════
    # ШАГ 5: Дивергенции → манипуляция?
    # ═══════════════════════════════════════════
    div_signal = s5.get("signal", "—")
    cvd_div = s5.get("cvd_divergence")
    mfi_divs = s16.get("divergences", [])

    has_divergence = (
        "разворот" in div_signal.lower() or
        (cvd_div and "бычья" in str(cvd_div).lower()) or
        (cvd_div and "медвежья" in str(cvd_div).lower()) or
        len(mfi_divs) > 0
    )

    # ═══════════════════════════════════════════
    # ШАГ 6: Фильтр очерёдности + манипуляция
    # ═══════════════════════════════════════════

    # 6a. Статусы A/B/C по расстоянию в ATR_daily
    all_classified = []
    all_tgts = key_targets + sec_targets + other_targets
    for t in all_tgts:
        dist_atr = abs(t["price"] - price) / atr_daily if atr_daily > 0 else 0
        if dist_atr <= 1.0:
            status = "A"
        elif dist_atr <= 2.0:
            status = "B"
        else:
            status = "C"
        all_classified.append({**t, "status": status, "dist_atr": round(dist_atr, 2)})

    # 6b. Первый шаг — по структуре (волны, Вайкофф, ближайший магнит)
    first_step_dir = _determine_first_step(s1, s2, s5, s10, s13, magnets, price, direction)

    # Правило близости (Регламент v4, раздел 13): ближайший магнит ≤0.6×ATR
    # учитывается ПЕРВЫМ. S13 уже применил фильтр 0.6×ATR — его first_step
    # имеет приоритет над любыми эвристиками.
    _s13_fs = s13.get("first_step")
    if _s13_fs and _s13_fs.get("direction") in ("вверх", "вниз"):
        first_step_dir["direction"] = _s13_fs["direction"]
        first_step_dir.setdefault("reasons_v8", []).append(
            f"правило близости ≤0.6×ATR: {_s13_fs.get('reason', _s13_fs.get('target'))}"
        )
    else:
        # Магнита в радиусе 0.6×ATR нет — fallback: если один из ближайших
        # магнитов в 2× ближе другого, он определяет реальный первый шаг.
        _nearest_above = None
        _nearest_below = None
        for m in magnets:
            mp = m.get("price", 0)
            if mp <= 0:
                continue
            if mp > price and (_nearest_above is None or mp < _nearest_above):
                _nearest_above = mp
            elif mp < price and (_nearest_below is None or mp > _nearest_below):
                _nearest_below = mp
        if _nearest_above and _nearest_below:
            d_up = _nearest_above - price
            d_dn = price - _nearest_below
            if d_up > 0 and d_dn > 0:
                if d_up * 2 < d_dn:
                    # ближайший магнит сверху → реальный первый шаг вверх
                    first_step_dir["direction"] = "вверх"
                    first_step_dir.setdefault("reasons_v8", []).append(
                        f"v8: ближайший магнит сверху {_nearest_above:.4g} (d={d_up:.4g}) "
                        f"vs снизу {_nearest_below:.4g} (d={d_dn:.4g})"
                    )
                elif d_dn * 2 < d_up:
                    first_step_dir["direction"] = "вниз"
                    first_step_dir.setdefault("reasons_v8", []).append(
                        f"v8: ближайший магнит снизу {_nearest_below:.4g} (d={d_dn:.4g}) "
                        f"vs сверху {_nearest_above:.4g} (d={d_up:.4g})"
                    )

    # 6c. Чек-лист манипуляции — Правило 2 Регламента v4 (4 критерия, 3 из 4)
    s4 = s(4)
    # Текущее движение = локальный тренд (S01), если есть; иначе первый шаг; иначе старший тренд
    _local_dir = (s1.get("local_trend") or {}).get("direction")
    if _local_dir not in ("восходящий", "нисходящий"):
        _fs = first_step_dir.get("direction")
        _local_dir = ("восходящий" if _fs == "вверх" else
                      "нисходящий" if _fs == "вниз" else direction)
    manip = _check_manipulation(
        s8, s11, s13, s17, has_divergence, direction,
        s4=s4, s5=s5, s10=s10, s16=s16, s9=s9,
        price=price, atr_daily=atr_daily, move_dir=_local_dir,
    )

    # 6d. Глубина маршрута по k_темпа
    # По регламенту: k >= 1.2 = ускоренный тренд → против тренда shallow,
    # но ПО ТРЕНДУ глубина нормальная
    if k_tempo >= 1.2:
        depth = "normal"  # по тренду нормально, counter будет shallow
        counter_depth = "shallow"
    elif k_tempo < 0.8:
        depth = "deep"
        counter_depth = "deep"
    else:
        depth = "normal"
        counter_depth = "normal"

    # Swing points из S01 (нужны для маршрута и слома)
    s1_data = sections_map.get(1, {}).get("data", {})
    swing_points_s1 = s1_data.get("swing_points", [])

    # ═══════════════════════════════════════════
    # ШАГ 6 (продолжение): Построить цепочку
    # ═══════════════════════════════════════════
    atr_last = s12.get("atr_tf", price * 0.01)
    route = _build_chain(
        all_classified, price, first_step_dir, direction,
        depth, manip["is_manipulation"], magnets, institutional,
        key_targets=key_targets, sec_targets=sec_targets, other_targets=other_targets,
        swing_points=swing_points_s1, atr_last=atr_last,
        s10=s10,
        counter_depth=counter_depth,
    )

    # ═══════════════════════════════════════════
    # ШАГ 7: Уровень слома
    # ═══════════════════════════════════════════
    # Слом = ближайшая структурная точка, пробой которой инвалидирует сценарий.
    # Для восходящего тренда = ближайший swing-low под ценой.
    # Для нисходящего тренда = ближайший swing-high над ценой.
    # ВАЖНО: слом должен быть разумно близок к цене и желательно
    # совпадать с одной из целей (уровней S/R, Фибо и т.д.).
    slam_price = None

    # Собираем ВСЕ известные уровни для поиска ближайшего слома
    all_target_prices = {round(t["price"], 4) for t in all_classified}

    MIN_SLAM_DIST_PCT = 0.03  # Минимум 3% от цены — слом должен быть значимым

    # Собираем пивот-уровни (D.R1, W.P и т.д.) из S06 для предпочтения в сломе
    _pivot_prices_slam = set()
    for r in s6.get("resistances", s6.get("resistances_5", [])):
        lbl = r.get("label", "")
        if any(lbl.startswith(p) for p in ("D.", "W.", "M.")):
            _pivot_prices_slam.add(round(r["price"], 4))
    for sup in s6.get("supports", s6.get("supports_5", [])):
        lbl = sup.get("label", "")
        if any(lbl.startswith(p) for p in ("D.", "W.", "M.")):
            _pivot_prices_slam.add(round(sup["price"], 4))

    def _near_pivot(p, tol=0.005):
        """Проверяем, совпадает ли цена p с каким-либо пивот-уровнем (±0.5%)."""
        return any(abs(pp - round(p, 4)) / max(p, 0.001) < tol for pp in _pivot_prices_slam)

    if direction == "восходящий":
        # Ближайший swing-low под ценой (но не ближе 3%)
        lows_below = [p["price"] for p in swing_points_s1
                      if p.get("type") == "low" and p["price"] < price
                      and (price - p["price"]) / price > MIN_SLAM_DIST_PCT]
        if lows_below:
            slam_price = max(lows_below)
        else:
            below = [t["price"] for t in all_classified
                     if t["price"] < price
                     and (price - t["price"]) / price > MIN_SLAM_DIST_PCT]
            if below:
                slam_price = max(below)

    elif direction == "нисходящий":
        # Ближайший swing-high над ценой (но не ближе 3%)
        # Предпочитаем значимые уровни: совпадающие с пивотами (D.R1, W.P и т.д.)
        highs_above = [p["price"] for p in swing_points_s1
                       if p.get("type") == "high" and p["price"] > price
                       and (p["price"] - price) / price > MIN_SLAM_DIST_PCT]
        # Сортируем по расстоянию от цены (ближайший первый)
        highs_above.sort()
        # Ищем ближайший, совпадающий с пивотом
        slam_price_candidate = None
        for sh in highs_above:
            if _near_pivot(sh):
                slam_price_candidate = sh
                break
        # Если пивот-совпадение не найдено, ищем совпадение с любой известной целью
        if not slam_price_candidate:
            slam_candidates = []
            for sh in highs_above:
                nearby_targets = [tp for tp in all_target_prices
                                  if abs(tp - sh) / max(sh, 0.001) < 0.005]
                slam_candidates.append((sh, len(nearby_targets)))
            # Сортировка: сначала по кол-ву совпадений (больше = лучше),
            # затем по расстоянию от цены (ближайший)
            slam_candidates.sort(key=lambda x: (-x[1], x[0]))
            if slam_candidates:
                slam_price_candidate = slam_candidates[0][0]
        if slam_price_candidate:
            slam_price = slam_price_candidate
        elif highs_above:
            slam_price = min(highs_above)
        else:
            above = [t["price"] for t in all_classified
                     if t["price"] > price
                     and (t["price"] - price) / price > MIN_SLAM_DIST_PCT]
            if above:
                slam_price = min(above)
    else:
        # Боковик — ближайшая граница
        if all_classified:
            slam_price = max(all_classified, key=lambda t: abs(t["price"] - price))["price"]

    # Привязка слома к ближайшей известной цели (если есть в пределах 1% от слома)
    # НО не если это сделает слом ближе MIN_SLAM_DIST_PCT к цене
    if slam_price:
        for tp in sorted(all_target_prices, key=lambda p: abs(p - slam_price)):
            if abs(tp - slam_price) / max(slam_price, 0.001) < 0.01:
                if abs(tp - price) / price > MIN_SLAM_DIST_PCT:
                    slam_price = tp
                break

    # Если слом не найден — fallback: 1.5% от цены в направлении против тренда
    if not slam_price:
        if direction == "восходящий":
            slam_price = price * (1 - 0.015)
        else:
            slam_price = price * (1 + 0.015)

    # Ограничение: слом не дальше чем ±5% от цены
    if slam_price and price > 0:
        max_dist_pct = 0.05
        if abs(slam_price - price) / price > max_dist_pct:
            # Если swing-точка слишком далеко, берём ближайшую target-цель
            if direction == "восходящий":
                cands = sorted([t["price"] for t in all_classified
                               if t["price"] < price
                               and (price - t["price"]) / price > MIN_SLAM_DIST_PCT], reverse=True)
            else:
                cands = sorted([t["price"] for t in all_classified
                               if t["price"] > price
                               and (t["price"] - price) / price > MIN_SLAM_DIST_PCT])
            for c in cands:
                if abs(c - price) / price <= max_dist_pct:
                    slam_price = c
                    break
            else:
                # Совсем fallback: 3% от цены
                if direction == "восходящий":
                    slam_price = price * 0.97
                else:
                    slam_price = price * 1.03

    # ═══════════════════════════════════════════
    # ШАГ 7b: Пересчёт слома по СТАРШЕМУ тренду (direction)
    # ═══════════════════════════════════════════
    # Слом = уровень, пробой которого инвалидирует СТАРШИЙ тренд.
    # Восходящий тренд → слом НИЖЕ цены (пробой вниз ломает up-trend).
    # Нисходящий тренд → слом ВЫШЕ цены (пробой вверх ломает down-trend).
    # Используем direction (старший тренд), а НЕ направление конца маршрута.
    SLAM_OFFSET_PCT = 0.005  # 0.5% запас за экстремумом маршрута
    if len(route) >= 2:
        # Для восходящего тренда слом должен быть НИЖЕ цены
        slam_should_be_below = (direction == "восходящий")
        # Для нисходящего тренда слом должен быть ВЫШЕ цены
        slam_should_be_above = (direction == "нисходящий")

        if slam_should_be_below and slam_price and slam_price > price:
            # Тренд вверх, но слом сверху — неправильно. Слом должен быть снизу.
            lows_for_slam = [rp["price"] for rp in route
                             if rp["price"] < price and rp.get("status") != "start"]
            if lows_for_slam:
                route_min = min(lows_for_slam)
                # Ищем структурный уровень НИЖЕ минимума маршрута
                structural_below = [
                    tp for tp in all_target_prices
                    if tp < route_min and (route_min - tp) / route_min < 0.03
                ]
                if structural_below:
                    slam_price = max(structural_below)  # ближайший ниже
                else:
                    slam_price = route_min * (1 - SLAM_OFFSET_PCT)
            else:
                slam_price = price * (1 - MIN_SLAM_DIST_PCT)
        elif slam_should_be_above and slam_price and slam_price < price:
            # Тренд вниз, но слом снизу — слом должен быть сверху.
            # Ищем ближайший swing-high выше цены (структурную точку)
            highs_for_slam = [p_sw["price"] for p_sw in swing_points_s1
                              if p_sw.get("type") == "high" and p_sw["price"] > price
                              and (p_sw["price"] - price) / price > MIN_SLAM_DIST_PCT]
            if not highs_for_slam:
                # Fallback: любая цель выше цены из маршрута
                highs_for_slam = [rp["price"] for rp in route
                                  if rp["price"] > price and rp.get("status") != "start"]
            if highs_for_slam:
                # Предпочитаем пивот-совпадение
                pivot_slams = [h for h in highs_for_slam if _near_pivot(h)]
                if pivot_slams:
                    slam_price = min(pivot_slams)  # ближайший пивот выше
                else:
                    slam_price = min(highs_for_slam)  # ближайший swing-high выше
            else:
                slam_price = price * (1 + MIN_SLAM_DIST_PCT)

    # ═══════════════════════════════════════════
    # ШАГ 8: Очистка (<0.3% → зона)
    # ═══════════════════════════════════════════
    route = _merge_close_points(route, price)

    # ═══════════════════════════════════════════
    # ШАГ 9: Сроки — ЯДРО v8
    # Формула: ceil(total_dist% / (k × ATR_d% × F))
    # total_dist% = крайняя точка маршрута в основном направлении × (1 + pullback)
    # Основное направление сценария = куда маршрут ушёл дальше от P0
    # (НЕ s01.direction: при манипуляции trend=down, но primary route может идти up
    # как «вынос → возврат → основной ход» либо только вверх)
    # ═══════════════════════════════════════════
    _max_up_pct = max(
        ((rp["price"] - price) / price * 100 for rp in route if rp["price"] > price),
        default=0.0,
    )
    _max_dn_pct = max(
        ((price - rp["price"]) / price * 100 for rp in route if rp["price"] < price),
        default=0.0,
    )
    # Сценарное направление = сторона с большим максимумом
    _scenario_up = _max_up_pct >= _max_dn_pct
    extreme_pct = _max_up_pct if _scenario_up else _max_dn_pct

    # Регламент v4 (раздел IV): Dist% = расстояние крайних целей,
    # без надбавок (пример регламента: 100→106 → Dist=6%)
    total_dist_pct = extreme_pct

    # резервный путь: максимальное отклонение, если extreme не найден
    if total_dist_pct <= 0:
        for rp in route:
            d = abs(rp["price"] - price) / price * 100
            if d > total_dist_pct:
                total_dist_pct = d

    # Коэффициент F — ЯДРО v8: 1.0 тренд / 0.7 коррекция / 1.3 пробой
    F = 1.0
    state_lower = (s1.get("state") or "").lower()
    if "коррекци" in state_lower:
        F = 0.7
    elif depth == "deep":
        F = 0.7
    elif manip["is_manipulation"]:
        F = 0.7
    else:
        # Пробой: ускоренный k и первый шаг против тренда = breakout
        _fs_dir = first_step_dir.get("direction", "вверх")
        _trend_aligns = (
            (direction == "нисходящий" and _fs_dir == "вниз") or
            (direction == "восходящий" and _fs_dir == "вверх")
        )
        if k_tempo >= 1.2 and not _trend_aligns:
            F = 1.3
        elif k_tempo >= 1.2 and _trend_aligns:
            F = 1.0

    # Регламент v4: k_эффект = взвешенное k по ТФ (старший 0.5, рабочий 0.3,
    # младший 0.2). В мульти-ТФ передаётся снаружи; одиночный ТФ — k_tempo.
    k_used = k_effect if (k_effect is not None and k_effect > 0) else k_tempo
    denom = k_used * atr_daily_pct * F
    days_float = (total_dist_pct / denom) if denom > 0 and total_dist_pct > 0 else 1.0
    days = max(math.ceil(days_float), 1)
    hours = max(math.ceil(days_float * 24), 1)  # для интрадей-горизонтов
    farthest_pct = total_dist_pct  # для совместимости ниже

    # ── v8: диапазон сроков (fast F=1.0 / slow F=0.7) для читаемого «min–max дней» ──
    def _days_for(F_val):
        d = k_used * atr_daily_pct * F_val
        if d <= 0 or total_dist_pct <= 0:
            return 1
        return max(math.ceil(total_dist_pct / d), 1)

    # Используем фактический F как центр диапазона; границы — F*1.3 (быстрее) и F*0.7 (медленнее)
    days_fast = _days_for(min(F * 1.3, 1.3))  # быстрая ветка — нижняя граница
    days_slow = _days_for(max(F * 0.7, 0.5))  # медленная ветка — верхняя граница
    days_min = min(days_fast, days_slow, days)
    days_max = max(days_fast, days_slow, days)
    if days_min == days_max:
        # подстраховка: чтобы не было «10–10 дней»
        days_max = days_min + max(1, days_min // 3)

    # ═══════════════════════════════════════════
    # ШАГ 7c (v8): Слом = КЛЮЧЕВАЯ цель, ПРИСУТСТВУЮЩАЯ В МАРШРУТЕ
    # Крайняя в направлении, противоположном финальной.
    # ═══════════════════════════════════════════
    key_prices = {round(t["price"], 4) for t in key_targets}
    # цели маршрута, совпадающие с key (без P0/manip)
    route_key_points = [
        rp for rp in route
        if round(rp["price"], 4) in key_prices
        and rp.get("status") not in ("start", "manip", "manip_return")
    ]
    # финальное направление = последняя значимая точка vs P0
    final_price = route[-1]["price"] if len(route) >= 2 else price
    final_up = final_price >= price

    if route_key_points:
        if final_up:
            # финал вверх → слом = крайняя key-точка НИЖЕ цены
            below_key = [rp for rp in route_key_points if rp["price"] < price]
            if below_key:
                slam_price = min(below_key, key=lambda r: r["price"])["price"]
        else:
            above_key = [rp for rp in route_key_points if rp["price"] > price]
            if above_key:
                slam_price = max(above_key, key=lambda r: r["price"])["price"]

    # ═══════════════════════════════════════════
    # Приостановка и наблюдение (v8): первая граница в направлении слома
    # = ближайший структурный уровень ПРОТИВ финального движения,
    # но БЛИЖЕ к цене, чем слом.
    # ═══════════════════════════════════════════
    pause_price = None
    if slam_price:
        if final_up:
            # финал вверх → pause = ближайшая цель НИЖЕ цены, но ВЫШЕ слома
            cands = [tp for tp in all_target_prices
                     if slam_price < tp < price]
            if cands:
                pause_price = max(cands)  # ближайшая к цене
        else:
            cands = [tp for tp in all_target_prices
                     if price < tp < slam_price]
            if cands:
                pause_price = min(cands)
        # Если соседей нет — pause на полпути к слому
        if pause_price is None:
            pause_price = (price + slam_price) / 2.0

    # Обоснование слома
    slam_reason_parts = []
    if slam_price:
        # Поиск в key_targets → источники/разделы
        for t in key_targets:
            if abs(t["price"] - slam_price) / max(slam_price, 1e-9) < 0.005:
                srcs = t.get("sources", [])
                if srcs:
                    slam_reason_parts.append(
                        "подтверждение " + ", ".join(_SOURCE_NAMES.get(s, s) for s in srcs[:3])
                    )
                break
        slam_reason_parts.append("закрытие дневной свечи за уровнем = слом тренда")
    slam_reason = "; ".join(slam_reason_parts) if slam_reason_parts else "ключевой структурный уровень"

    # ═══════════════════════════════════════════
    # Горизонт v8: 15m интрадей / 1H 1–3 дня / 4H 5–7 / 1D 15–20
    # ═══════════════════════════════════════════
    if tf_hours <= 0.25:
        horizon_label = "интрадей"
    elif tf_hours <= 1:
        horizon_label = "1–3 дня"
    elif tf_hours <= 4:
        horizon_label = "5–7 дней"
    else:
        horizon_label = "15–20 дней"

    # Форматирование маршрута — цена и процент + цветные стрелки направления
    def _route_arrow(prev_price, cur_price):
        """Стрелка между точками: 🔴 падение, 🟢 рост."""
        if cur_price < prev_price:
            return " 🔴 "
        else:
            return " 🟢 "

    def _point_str(rp):
        if rp.get("is_zone"):
            lo_s = _fmt(rp["zone_lo"], price)
            hi_s = _fmt(rp["zone_hi"], price)
            return f"{lo_s}–{hi_s}"
        return _fmt(rp["price"], price)

    route_parts = []
    for i, rp in enumerate(route):
        p_str = _point_str(rp)
        if i == 0:
            route_parts.append(p_str)
        else:
            arrow = _route_arrow(route[i - 1]["price"], rp["price"])
            route_parts.append(f"{arrow}{p_str}")
    route_str = "".join(route_parts)
    slam_str = _fmt(slam_price, price) if slam_price else "—"
    pause_str = _fmt(pause_price, price) if pause_price else "—"

    return {
        "key_targets": [t["str"] for t in key_targets],
        "sec_targets": [t["str"] for t in sec_targets],
        "other_targets": [t["str"] for t in other_targets],
        "key_targets_full": key_targets,
        "sec_targets_full": sec_targets,
        "other_targets_full": other_targets,
        "route": route,
        "route_str": route_str,
        "route_count": len(route),
        "pause_price": pause_price,
        "pause_str": pause_str,
        "slam_price": slam_price,
        "slam_str": slam_str,
        "slam_reason": slam_reason,
        "days": days,
        "hours": hours,
        "days_min": days_min,
        "days_max": days_max,
        "days_detail": f"Total_dist={total_dist_pct:.1f}%, K={k_used:.2f}, ATR_дн={atr_daily_pct:.2f}%, F={F}",
        "total_dist_pct": round(total_dist_pct, 2),
        "F_factor": F,
        "horizon_label": horizon_label,
        "manipulation": manip,
        "first_step": first_step_dir,
    }


def _fmt(p, current):
    if not p or not current:
        return "—"
    pct = (p - current) / current * 100
    # Адаптивный формат: без научной нотации для больших цен
    p_str = f"{p:.2f}" if p >= 1000 else f"{p:.4g}"
    return f"{p_str} ({pct:+.2f}%)"


# Человеческие названия источников целей
_SOURCE_NAMES = {
    "Pivot": "пивот",
    "Swing": "свинг S/R",
    "Fibo": "Фибо",
    "Wave": "цель волны",
    "Pattern": "цель паттерна",
    "POC": "POC",
    "VAH": "VAH",
    "VAL": "VAL",
    "Stop": "зона стопов",
    "FVG": "FVG",
    "Gap": "гэп",
    "ThinVP": "тонкая зона VP",
    "Liquidity": "пул ликвидности",
}


def _fmt_with_sources(p, current, sources=None):
    """Цена(%) [источники]."""
    base = _fmt(p, current)
    if sources:
        names = [_SOURCE_NAMES.get(s, s) for s in sources]
        return f"{base} [{', '.join(names)}]"
    return base


def _determine_first_step(s1, s2, s5, s10, s13, magnets, price, direction):
    """Чек-лист первого шага по регламенту v6 (раздел VI, шаг 1).

    4 пункта проверки → направление первого шага + чек-лист с обоснованиями.
    """
    checklist = []

    # ── 1. Кромка / RSI / ALMA ──
    rsi_val = s5.get("rsi_current")
    rsi_zone = s5.get("rsi_zone", "")
    alma_order = s1.get("alma_order", "")
    linreg = s1.get("linreg", {})
    position_pct = linreg.get("position_pct", 50)

    check1_dir = None
    check1_reason = []

    # У сопротивления / перегрев RSI / замедление ALMA → вниз
    # От поддержки / рост RSI / ускорение ALMA → вверх
    if position_pct > 70:
        check1_reason.append(f"цена у верхней границы канала ({position_pct:.0f}%)")
        check1_dir = "вниз"
    elif position_pct < 30:
        check1_reason.append(f"цена у нижней границы канала ({position_pct:.0f}%)")
        check1_dir = "вверх"

    if rsi_val and rsi_val > 70:
        check1_reason.append(f"RSI {rsi_val:.1f} перекупленность")
        check1_dir = "вниз"
    elif rsi_val and rsi_val < 30:
        check1_reason.append(f"RSI {rsi_val:.1f} перепроданность")
        check1_dir = "вверх"

    if "медвежий" in alma_order:
        check1_reason.append(f"ALMA медвежий")
        if check1_dir is None:
            check1_dir = "вниз"
    elif "бычий" in alma_order:
        check1_reason.append(f"ALMA бычий")
        if check1_dir is None:
            check1_dir = "вверх"

    checklist.append({
        "check": "Кромка / RSI / ALMA",
        "direction": check1_dir or "—",
        "reasons": check1_reason or ["нейтрально"],
    })

    # ── 2. Волны / Вайкофф ──
    check2_dir = None
    check2_reason = []

    # Из нового формата S02 — количество точек коррекции
    corr_count = s2.get("correction_count", 0)
    imp_count = s2.get("impulse_count", 0)

    node = s10.get("current_node", "")
    if "SOS" in node or "LPS" in node or "SPRING" in node or "BU" in node:
        check2_reason.append(f"Вайкофф: {node} -> вверх")
        check2_dir = "вверх"
    elif "SOW" in node or "LPSY" in node or "UT" in node or "UTAD" in node:
        check2_reason.append(f"Вайкофф: {node} -> вниз")
        check2_dir = "вниз"

    # Бэктест: structure_type (Накопление/Распределение) — сильный предиктор (+4.1% WR)
    structure_type = (s10.get("structure_type") or "").lower()
    if "накопление" in structure_type or "accumulation" in structure_type:
        if check2_dir is None:
            check2_dir = "вверх"
        check2_reason.append("структура: Накопление")
    elif "распределение" in structure_type or "distribution" in structure_type:
        if check2_dir is None:
            check2_dir = "вниз"
        check2_reason.append("структура: Распределение")

    # CMF — бэктест: CMF sellers на bull = 69.4% WR (слабый), CMF buyers на bear = 92.3% (сильный)
    cmf_data = s10.get("evidence", {})
    cmf_val = cmf_data.get("cmf", 0)
    if isinstance(cmf_val, (int, float)) and cmf_val != 0:
        if cmf_val < -0.1:
            check2_reason.append(f"CMF {cmf_val:.3f} (отток)")
            if check2_dir is None:
                check2_dir = "вниз"
        elif cmf_val > 0.1:
            check2_reason.append(f"CMF {cmf_val:.3f} (приток)")
            if check2_dir is None:
                check2_dir = "вверх"

    phase = s10.get("phase", "")
    if phase:
        check2_reason.append(f"фаза {phase}")

    checklist.append({
        "check": "Волны / Вайкофф",
        "direction": check2_dir or "—",
        "reasons": check2_reason or ["нейтрально"],
    })

    # ── 3. ALMA динамика / RSI тренд ──
    check3_dir = None
    check3_reason = []

    rsi_trend = s5.get("rsi_trend", "")
    slope = linreg.get("slope_pct_per_bar", 0)

    if rsi_trend == "падающий":
        check3_reason.append("RSI падает")
        check3_dir = "вниз"
    elif rsi_trend == "растущий":
        check3_reason.append("RSI растёт")
        check3_dir = "вверх"

    if slope > 0.05:
        check3_reason.append(f"LinReg ускоряется ↑ ({slope:.3f}%/бар)")
        if check3_dir is None:
            check3_dir = "вверх"
    elif slope < -0.05:
        check3_reason.append(f"LinReg ускоряется ↓ ({slope:.3f}%/бар)")
        if check3_dir is None:
            check3_dir = "вниз"
    else:
        check3_reason.append(f"LinReg нейтральный ({slope:.3f}%/бар)")

    checklist.append({
        "check": "ALMA/RSI динамика",
        "direction": check3_dir or "—",
        "reasons": check3_reason or ["нейтрально"],
    })

    # ── 4. Ближайшая ликвидность / FVG / POC ──
    check4_dir = None
    check4_reason = []

    fs = s13.get("first_step")
    if fs:
        check4_dir = fs.get("direction", "—")
        check4_reason.append(f"ближайший магнит {fs.get('type', '?')} "
                            f"на {fs.get('target', 0):.2f} ({fs.get('distance_atr', 0):.2f} ATR)")

    if magnets:
        nearest = min(magnets, key=lambda m: abs(m["price"] - price))
        if nearest["price"] > price:
            check4_reason.append(f"ближайший магнит сверху {nearest['price']:.2f}")
            if check4_dir is None:
                check4_dir = "вверх"
        else:
            check4_reason.append(f"ближайший магнит снизу {nearest['price']:.2f}")
            if check4_dir is None:
                check4_dir = "вниз"

    checklist.append({
        "check": "Ближайшая ликвидность",
        "direction": check4_dir or "—",
        "reasons": check4_reason or ["нейтрально"],
    })

    # ── ИТОГ: подсчёт голосов 4 пунктов ──
    votes_up = sum(1 for c in checklist if c["direction"] == "вверх")
    votes_down = sum(1 for c in checklist if c["direction"] == "вниз")

    if votes_up > votes_down:
        result_dir = "вверх"
    elif votes_down > votes_up:
        result_dir = "вниз"
    else:
        # Тай-брейк: по основному тренду
        result_dir = "вниз" if direction == "нисходящий" else "вверх"

    return {
        "direction": result_dir,
        "score": f"↑{votes_up} vs ↓{votes_down}",
        "checklist": checklist,
    }


def _check_manipulation(s8, s11, s13, s17, has_divergence, direction,
                        s4=None, s5=None, s10=None, s16=None, s9=None,
                        price=None, atr_daily=None, move_dir=None):
    """Чек-лист манипуляции — Правило 2 Регламента v4 (Дополнение).

    Ровно 4 критерия, порог 3 из 4:
      1. Дивергенция CVD vs цена (движение обеспечено лимитными, не рыночными)
      2. Перегрев осцилляторов БЕЗ экстремума: Stochastic >80/<20 или
         MFI >75/<25, но RSI в нейтральной зоне 30–70
      3. Ликвидность ПО ХОДУ движения: стопы/FVG/тонкие зоны
         в пределах 1.5% от цены в направлении текущего движения
      4. Институциональный интерес ЗА СПИНОЙ: POC / аномальный объём /
         вход крупняка на противоположной стороне от движения

    move_dir — направление ТЕКУЩЕГО движения ("восходящий"/"нисходящий"),
    по умолчанию = direction (старший тренд).
    """
    s4 = s4 or {}
    s5 = s5 or {}
    s10 = s10 or {}
    s16 = s16 or {}
    s9 = s9 or {}
    move = move_dir or direction
    move_up = (move == "восходящий")

    signs = 0
    details = []
    signs_map = {}

    # ── 1. Дивергенция CVD vs цена ──
    cvd_div = s5.get("cvd_divergence")
    sign1 = bool(cvd_div)
    signs_map["cvd_divergence"] = sign1
    if sign1:
        signs += 1
        details.append(f"✅ 1/4 Дивергенция CVD vs цена: {cvd_div}")
    else:
        details.append("❌ 1/4 Дивергенции CVD vs цена нет")

    # ── 2. Перегрев осцилляторов при нейтральном RSI ──
    rsi_val = s5.get("rsi_current")
    stoch = s5.get("stochastic") or {}
    stoch_k = stoch.get("k")
    mfi_val = (s16.get("mfi") or {}).get("value")

    rsi_neutral = rsi_val is not None and 30 <= rsi_val <= 70
    stoch_hot = stoch_k is not None and (stoch_k > 80 or stoch_k < 20)
    mfi_hot = mfi_val is not None and (mfi_val > 75 or mfi_val < 25)
    sign2 = rsi_neutral and (stoch_hot or mfi_hot)
    signs_map["oscillator_overheat"] = sign2
    if sign2:
        signs += 1
        osc = []
        if stoch_hot:
            osc.append(f"Stoch %K={stoch_k:.0f}")
        if mfi_hot:
            osc.append(f"MFI={mfi_val:.0f}")
        details.append(
            f"✅ 2/4 Перегрев без экстремума: {', '.join(osc)} при RSI={rsi_val:.0f} (нейтрален)"
        )
    else:
        details.append(
            f"❌ 2/4 Перегрева без экстремума нет (RSI={rsi_val if rsi_val is not None else '—'}, "
            f"Stoch={stoch_k if stoch_k is not None else '—'}, MFI={mfi_val if mfi_val is not None else '—'})"
        )

    # ── 3. Ликвидность по ходу движения (≤1.5% в направлении move) ──
    LIQ_MAX_PCT = 0.015
    liq_ahead = []
    if price:
        # Стопы (S11)
        for key in ("stops_below", "stops_below_supports", "stops_above", "stops_above_resistances"):
            for st in s11.get(key, []):
                p = st.get("stop_zone") or st.get("level")
                if p:
                    liq_ahead.append(("стопы", p))
        # FVG / гэпы (S13)
        for f in s13.get("open_fvgs", []):
            mid = (f.get("top", 0) + f.get("bottom", 0)) / 2
            if mid > 0:
                liq_ahead.append(("FVG", mid))
        for g in s13.get("open_gaps", []):
            mid = (g.get("gap_top", 0) + g.get("gap_bottom", 0)) / 2
            if mid > 0:
                liq_ahead.append(("гэп", mid))
        # Тонкие зоны VP (S09)
        for pkey in ("profile_a", "profile_b"):
            for ta in s9.get(pkey, {}).get("thin_areas", []):
                if ta.get("price"):
                    liq_ahead.append(("тонкая зона", ta["price"]))

        # Фильтр: по ходу движения и в пределах 1.5%
        liq_ahead = [
            (t, p) for t, p in liq_ahead
            if (p > price if move_up else p < price)
            and abs(p - price) / price <= LIQ_MAX_PCT
        ]
    sign3 = bool(liq_ahead)
    signs_map["liquidity_ahead"] = sign3
    if sign3:
        signs += 1
        nearest = min(liq_ahead, key=lambda x: abs(x[1] - price))
        details.append(
            f"✅ 3/4 Ликвидность по ходу движения: {nearest[0]} {nearest[1]:.4g} "
            f"({len(liq_ahead)} зон ≤1.5%)"
        )
    else:
        details.append("❌ 3/4 Ликвидности по ходу движения (≤1.5%) нет")

    # ── 4. Институциональный интерес за спиной ──
    inst_behind = []
    if price:
        for pkey in ("profile_a", "profile_b"):
            poc_item = s9.get(pkey, {}).get("POC", {})
            if isinstance(poc_item, dict) and poc_item.get("price"):
                inst_behind.append(("POC", poc_item["price"]))
        for anom in (s17 or {}).get("volume_anomalies", []):
            p = anom.get("close_price", 0)
            if p:
                inst_behind.append(("аномальный объём", p))
        # За спиной = противоположная сторона от движения
        inst_behind = [
            (t, p) for t, p in inst_behind
            if (p < price if move_up else p > price)
        ]
    sign4 = bool(inst_behind)
    signs_map["institutional_behind"] = sign4
    if sign4:
        signs += 1
        nearest = min(inst_behind, key=lambda x: abs(x[1] - price))
        details.append(
            f"✅ 4/4 Институциональная зона за спиной: {nearest[0]} {nearest[1]:.4g}"
        )
    else:
        details.append("❌ 4/4 Институциональной зоны за спиной нет")

    # Правило 2: порог 3 из 4
    return {
        "is_manipulation": signs >= 3,
        "signs": signs,
        "total": 4,
        "threshold": 3,
        "signs_map": signs_map,
        "details": details,
    }


def _build_chain(classified, price, first_step, direction, depth, is_manip,
                  magnets, institutional,
                  key_targets=None, sec_targets=None, other_targets=None,
                  swing_points=None, atr_last=None, s10=None, counter_depth=None):
    """Построить маршрут как ОДНУ последовательную ветку (регламент v6).

    Регламент step_4_chain:
      "Фон лонговый -> основная ветка вверх, откаты вниз вспомогательные;
       шортовый -- наоборот."

    Ключевое правило: "фон" (primary direction) определяется по first_step,
    а НЕ по глобальному тренду (direction). Тренд -- это S01 контекст,
    first_step -- куда цена пойдёт ПЕРВЫМ шагом по 4-пунктному чек-листу.

    Алгоритм:
      1. P0 = текущая цена
      2. [Манипуляция] -- если is_manip: короткий вынос ПРОТИВ first_dir,
         затем возврат к кромке/базе и продолжение по основной ветке
      3. Зигзаг по primary целям (в направлении first_dir):
         цель -> откат к counter-цели или midpoint -> цель -> откат -> ...
      4. ВСЕ цели из key/sec/other включаются в маршрут
      5. Цели <0.3% друг от друга = одна зона

    Приоритет целей:
      1. Тонкая ликвидность / FVG / гэпы
      2. Стоп-кластеры
      3. VAH / VAL
      4. POC / HVN
      5. Пивоты / локальные экстремумы

    Глубина (depth):
      - "shallow" (k_tempo >= 1.2): только 1-2 цели со статусом A
      - "normal": несколько A + 1-2 B
      - "deep" (k_tempo < 0.8): допускаем глубокие B цели

    Returns:
      list of {"price": ..., "label": ..., "status": ..., "sources": [...]}
    """
    key_targets = key_targets or []
    sec_targets = sec_targets or []
    other_targets = other_targets or []
    swing_points = swing_points or []
    counter_depth = counter_depth or depth

    # ── Направление основного хода маршрута ──
    # Без манипуляции: основной ход = first_step (реальный ближайший шаг).
    # При манипуляции (Правило 2 v4): задёрг = first_step (к ликвидности),
    # ИМПУЛЬС = старшее направление (от институциональной зоны к целям
    # старшего анализа) — primary строится по импульсу.
    first_dir = first_step.get("direction", "вверх")
    manip_active = bool(is_manip)
    if manip_active and direction in ("восходящий", "нисходящий"):
        impulse_up = (direction == "восходящий")
        jerk_up = (first_dir == "вверх")
        if jerk_up == impulse_up:
            # Первый шаг совпадает с импульсом — манипуляционная вставка не нужна
            manip_active = False
            go_up = (first_dir == "вверх")
        else:
            go_up = impulse_up
    else:
        go_up = (first_dir == "вверх")

    # ── Собрать все цели в единый пул с метаданными ──
    all_targets = []
    for t in key_targets:
        all_targets.append({
            "price": t["price"], "tier": "key",
            "sources": t.get("sources", []),
        })
    for t in sec_targets:
        all_targets.append({
            "price": t["price"], "tier": "sec",
            "sources": t.get("sources", []),
        })
    for t in other_targets:
        all_targets.append({
            "price": t["price"], "tier": "other",
            "sources": t.get("sources", []),
        })

    # Проставляем статус A/B/C из classified (если есть)
    classified_map = {}
    for c in (classified or []):
        classified_map[round(c["price"], 4)] = c.get("status", "C")

    for t in all_targets:
        rp = round(t["price"], 4)
        t["abc"] = classified_map.get(rp, "C")

    # ── Приоритет источников (регламент v6, раздел VI) ──
    _SRC_PRIORITY = {
        "ThinVP": 1, "FVG": 1, "Gap": 1,      # 1. Тонкая ликвидность / FVG / гэпы
        "Liquidity": 1,
        "Stop": 2,                               # 2. Стоп-кластеры
        "VAH": 3, "VAL": 3,                     # 3. VAH / VAL
        "POC": 4,                                # 4. POC / HVN
        "Pivot": 5, "Swing": 5, "Fibo": 5, "Wave": 5,  # 5. Пивоты / локальные экстремумы
        "Pattern": 5,
    }

    def _target_priority(t):
        """Числовой приоритет цели (меньше = важнее)."""
        best = 99
        for s in t.get("sources", []):
            best = min(best, _SRC_PRIORITY.get(s, 99))
        return best

    # ── Разделить на primary (по first_dir) и counter (против first_dir) ──
    # Primary = цели В НАПРАВЛЕНИИ first_step
    # Counter = цели ПРОТИВ first_step (используются для откатов)
    if go_up:
        primary_raw = [t for t in all_targets if t["price"] > price]
        counter_raw = [t for t in all_targets if t["price"] < price]
        # Primary: от ближайшей к дальней (по возрастанию)
        primary_raw.sort(key=lambda t: t["price"])
        # Counter: от ближайшей к дальней (по убыванию)
        counter_raw.sort(key=lambda t: t["price"], reverse=True)
    else:
        primary_raw = [t for t in all_targets if t["price"] < price]
        counter_raw = [t for t in all_targets if t["price"] > price]
        # Primary: от ближайшей к дальней (по убыванию)
        primary_raw.sort(key=lambda t: t["price"], reverse=True)
        # Counter: от ближайшей к дальней (по возрастанию)
        counter_raw.sort(key=lambda t: t["price"])

    # v8: маршрут одно-направленный, 8–15 узлов ВСЕГО (P0 + цели + откаты).
    # Сначала все key-цели в основном направлении, потом дополняем other до
    # MAX_PRIMARY_TARGETS — чтобы маршрут покрывал реальный диапазон цен,
    # а не обрывался на 3-х ближайших key.
    # Counter — только как pullback-точки ВНУТРИ маршрута (не хвостом).
    MAX_PRIMARY_TARGETS = 6  # → итог ~11–13 узлов с pullback'ами и P0
    key_primary = [t for t in primary_raw if t.get("tier") == "key"]
    other_primary = [t for t in primary_raw if t.get("tier") != "key"]
    primary = key_primary + other_primary
    primary = primary[:MAX_PRIMARY_TARGETS]
    if not primary and primary_raw:
        primary = primary_raw[:MAX_PRIMARY_TARGETS]
    counter = list(counter_raw)

    if not primary and primary_raw:
        primary = [primary_raw[0]]
    if not counter and counter_raw:
        counter = [counter_raw[0]]

    # ── Сортировка primary по приоритету внутри одного ценового уровня ──
    # (если две цели рядом, более приоритетная идёт первой)
    primary.sort(key=lambda t: (
        t["price"] if go_up else -t["price"],
        _target_priority(t),
    ))

    # ── P0: стартовая точка ──
    route = [{"price": price, "label": "P0", "status": "start", "sources": []}]

    # ── Манипуляция (Правило 2 Регламента v4) ──
    # «Задёрг в сторону ликвидности → разворот → набор позиции
    #  в институциональной зоне → импульс к целям»
    # Задёрг = ПРОТИВ импульса (по first_step, к ближайшей ликвидности).
    # Разворот = в институциональной зоне (POC/аномальный объём), не у P0.
    MIN_MANIP_DIST_PCT = 0.005   # мин. 0.5% расстояние для манип-цели
    MAX_MANIP_DIST_PCT = 0.03    # макс. 3%

    if manip_active and counter:
        # Задёрг: приоритет — ЛИКВИДНОСТЬ (стопы/FVG/тонкие зоны) против импульса
        _liq_srcs = {"Stop", "Liquidity", "FVG", "Gap", "ThinVP"}
        manip_target = None
        for ct in counter:
            dist = abs(ct["price"] - price) / price
            if (MIN_MANIP_DIST_PCT <= dist <= MAX_MANIP_DIST_PCT
                    and set(ct.get("sources", [])) & _liq_srcs):
                manip_target = ct
                break
        if manip_target is None:
            for ct in counter:
                dist = abs(ct["price"] - price) / price
                if MIN_MANIP_DIST_PCT <= dist <= MAX_MANIP_DIST_PCT:
                    manip_target = ct
                    break

        if manip_target:
            # Шаг 1 манипуляции: задёрг к ликвидности
            route.append({
                "price": manip_target["price"],
                "label": "⚡вынос",
                "status": "manip",
                "sources": manip_target.get("sources", []),
            })
            # Шаг 2: разворот в ИНСТИТУЦИОНАЛЬНОЙ зоне — ближайший POC /
            # аномальный объём на стороне импульса от точки задёрга
            mt_price = manip_target["price"]
            inst_point = None
            inst_cands = []
            for iz in (institutional or []):
                izp = iz.get("price", 0)
                if izp <= 0:
                    continue
                # Зона должна лежать со стороны импульса относительно задёрга
                # и в разумной близости от текущей цены (≤2%)
                on_impulse_side = (izp > mt_price) if go_up else (izp < mt_price)
                if on_impulse_side and abs(izp - price) / price <= 0.02:
                    inst_cands.append(izp)
            if inst_cands:
                inst_point = min(inst_cands, key=lambda p: abs(p - price))
            route.append({
                "price": inst_point if inst_point else price,
                "label": "↩возврат POC" if inst_point else "↩возврат",
                "status": "manip_return",
                "sources": ["POC"] if inst_point else [],
            })

    # ── Вспомогательная функция: найти точку отката ──
    def _find_pullback(current_target_price, next_target_price):
        """Найти точку отката между двумя consecutive primary целями.

        Логика:
        1. Ищем counter-цель между current и P0 (ближайшую к current)
        2. Если нет -- midpoint между current и next
        """
        if go_up:
            # Откат вниз: ищем counter-цель между P0 и current_target
            candidates = [
                c for c in counter
                if price <= c["price"] < current_target_price
            ]
            # Ближайший к current_target (самый высокий counter ниже цели)
            if candidates:
                candidates.sort(key=lambda c: c["price"], reverse=True)
                best = candidates[0]
                return {
                    "price": best["price"],
                    "label": "откат",
                    "status": "pullback",
                    "sources": best.get("sources", []),
                }
        else:
            # Откат вверх: ищем counter-цель между P0 и current_target
            candidates = [
                c for c in counter
                if current_target_price < c["price"] <= price
            ]
            # Ближайший к current_target (самый низкий counter выше цели)
            if candidates:
                candidates.sort(key=lambda c: c["price"])
                best = candidates[0]
                return {
                    "price": best["price"],
                    "label": "откат",
                    "status": "pullback",
                    "sources": best.get("sources", []),
                }

        # Fallback: midpoint между текущей целью и следующей
        if next_target_price is not None:
            mid = (current_target_price + next_target_price) / 2
        else:
            # Нет следующей цели -- откат на 38.2% расстояния от P0 до цели
            dist = current_target_price - price
            mid = current_target_price - dist * 0.382
        return {
            "price": mid,
            "label": "откат",
            "status": "pullback",
            "sources": [],
        }

    # ── Проверка LPSY: Wyckoff phase D/E + Distribution → вставка коррекции вверх ──
    s10 = s10 or {}
    wyckoff_phase = s10.get("phase", "")
    wyckoff_structure = s10.get("structure_type", "")
    wyckoff_node = s10.get("current_node", "")
    has_lpsy = (
        wyckoff_phase in ("D", "E")
        and "Распределение" in wyckoff_structure
    )
    # Также триггер, если текущий узел явно LPSY или SOW
    if "LPSY" in wyckoff_node or "SOW" in wyckoff_node:
        has_lpsy = True

    # Определяем цену LPSY-коррекции: ближайший POC или D.P (из institutional/counter)
    lpsy_target_price = None
    if has_lpsy and not go_up:
        # Ищем POC из institutional
        for inst_zone in institutional:
            if inst_zone.get("type") == "POC" and inst_zone["price"] > price:
                # POC выше текущей цены — подходит для коррекции вверх
                if lpsy_target_price is None or inst_zone["price"] < lpsy_target_price:
                    lpsy_target_price = inst_zone["price"]
        # Fallback: ближайшая counter-цель (сопротивление) для bounce up
        if lpsy_target_price is None and counter:
            lpsy_target_price = counter[0]["price"]
    elif has_lpsy and go_up:
        # Для восходящего LPSY (аккумуляция фаза D/E): коррекция вниз к POC
        for inst_zone in institutional:
            if inst_zone.get("type") == "POC" and inst_zone["price"] < price:
                if lpsy_target_price is None or inst_zone["price"] > lpsy_target_price:
                    lpsy_target_price = inst_zone["price"]
        if lpsy_target_price is None and counter:
            lpsy_target_price = counter[0]["price"]

    # Точка вставки LPSY: после 2-3 primary целей
    lpsy_insert_after = min(2, len(primary) - 1) if len(primary) > 2 else len(primary) - 1
    lpsy_inserted = False

    # ── Основной зигзаг: цель -> откат -> цель -> откат -> ... ──
    for i, t in enumerate(primary):
        # Добавляем primary цель
        route.append({
            "price": t["price"],
            "label": t["tier"],
            "status": t["tier"],
            "sources": t.get("sources", []),
        })

        # LPSY insertion: после 2-3 primary целей, вставляем коррекцию
        if (has_lpsy and not lpsy_inserted and i == lpsy_insert_after
                and lpsy_target_price is not None):
            # Проверяем: LPSY должен быть ПРОТИВ primary (bounce back)
            lpsy_is_valid = (
                (not go_up and lpsy_target_price > t["price"]) or
                (go_up and lpsy_target_price < t["price"])
            )
            if lpsy_is_valid:
                # Не вставлять если слишком близко (<0.5%)
                if abs(lpsy_target_price - t["price"]) / max(t["price"], 0.001) >= 0.005:
                    route.append({
                        "price": lpsy_target_price,
                        "label": "LPSY",
                        "status": "lpsy",
                        "sources": ["Wyckoff"],
                    })
                    lpsy_inserted = True

        # Откат после КАЖДОЙ цели КРОМЕ последней (zigzag per регламент)
        is_last = (i == len(primary) - 1)
        if not is_last:
            next_price = primary[i + 1]["price"]
            # Предыдущая точка маршрута (от которой мы пришли) — для расчёта откатного midpoint
            prev_route_price = route[-2]["price"] if len(route) >= 2 else price
            pb = _find_pullback(t["price"], next_price)
            # Проверяем: откат должен быть ПРОТИВ направления primary (bounce back)
            # Если midpoint-fallback дал точку в том же направлении, пересчитываем
            # как midpoint между текущей целью и предыдущей точкой маршрута
            pb_is_bounce = (go_up and pb["price"] < t["price"]) or (not go_up and pb["price"] > t["price"])
            if not pb_is_bounce:
                mid = (t["price"] + prev_route_price) / 2
                pb = {"price": mid, "label": "откат", "status": "pullback", "sources": []}
            # Не добавляем откат если он слишком близок к цели (<0.3%)
            if abs(pb["price"] - t["price"]) / max(t["price"], 0.001) >= 0.003:
                route.append(pb)

    # v8: маршрут одно-направленный по основному сценарию (slam definir направление).
    # Неиспользованные counter-цели НЕ дописываются хвостом — они живут в other_targets
    # и используются формулой «приостановка и наблюдение» / «слом структуры».
    # Это исправляет баг, когда маршрут шёл вниз через все цели, потом телепорт
    # к ближайшей вверх и шёл вверх через все — нереалистично.

    # ── Дедупликация: убираем точки с одинаковой ценой ──
    # Исключение: manip_return/⚡вынос/P0 — сохраняем даже при совпадении цен,
    # чтобы манипуляционный цикл "P0 → вынос → ↩возврат" не терялся.
    seen = set()
    deduped = []
    PROTECTED = {"start", "manip", "manip_return"}
    for r in route:
        rp = round(r["price"], 4)
        if r.get("status") in PROTECTED:
            # Защищённые точки добавляем всегда, но дубль подряд не плодим
            if deduped and round(deduped[-1]["price"], 4) == rp:
                continue
            deduped.append(r)
            seen.add(rp)
        elif rp not in seen:
            seen.add(rp)
            deduped.append(r)

    return deduped


def _merge_close_points(route, price):
    """v8 шаг 8: цели <0.3% друг от друга → зона X–Y.

    Не сливает: P0 (start) и манипуляционные точки (manip).
    Для слитых точек сохраняется zone_lo/zone_hi для форматирования «X–Y».
    """
    if len(route) <= 2:
        return route
    merged = [route[0]]  # P0 всегда первый
    for r in route[1:]:
        prev = merged[-1]
        is_protected = (
            prev.get("status") in ("start", "manip", "manip_return") or
            r.get("status") in ("start", "manip", "manip_return")
        )
        if is_protected:
            merged.append(r)
            continue
        if abs(r["price"] - prev["price"]) / max(prev["price"], 0.001) < 0.003:
            # Объединяем в зону X–Y
            lo = min(prev.get("zone_lo", prev["price"]), r["price"])
            hi = max(prev.get("zone_hi", prev["price"]), r["price"])
            merged[-1] = {
                "price": (lo + hi) / 2,
                "label": prev.get("label", ""),
                "status": prev.get("status", ""),
                "zone_lo": lo,
                "zone_hi": hi,
                "is_zone": True,
                "sources": list({*(prev.get("sources") or []), *(r.get("sources") or [])}),
            }
        else:
            merged.append(r)
    return merged
