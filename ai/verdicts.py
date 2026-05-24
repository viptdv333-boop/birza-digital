"""
Биржа-цифровой — «Совет директоров».

Каждый раздел (S01-S18) выдаёт структурированный вердикт:
  direction (вверх/вниз/нейтрально), target (цена), summary (1 строка).

AI-модель получает только эти вердикты, а не сырой JSON.
"""


def _pct(val, price):
    if not price or price <= 0 or not val:
        return ""
    p = (val - price) / price * 100
    sign = "+" if p >= 0 else ""
    v_str = f"{val:.2f}" if val >= 1000 else f"{val:.4g}"
    return f"{v_str} ({sign}{p:.2f}%)"


def _dir_arrow(d):
    return {"вверх": "↑", "вниз": "↓"}.get(d, "—")


def extract_verdicts(sections_map: dict, price: float, direction: str) -> dict:
    """Извлечь вердикты из всех 18 разделов.

    Returns:
        {
            "verdicts": [{"section": N, "title": str, "direction": str,
                          "target": float|None, "summary": str}, ...],
            "vote_up": int,
            "vote_down": int,
            "vote_neutral": int,
        }
    """
    def s(sid):
        return sections_map.get(sid, {}).get("data", {})

    verdicts = []

    # ─── S01 ТРЕНДЫ ───
    s1 = s(1)
    d1 = s1.get("direction", "—")
    dir1 = "вниз" if "нисход" in d1 else ("вверх" if "восход" in d1 else "нейтрально")
    stage = s1.get("stage", s1.get("state", "—"))
    alma200 = s1.get("alma_200_position", "—")
    verdicts.append({
        "section": 1, "title": "ТРЕНДЫ", "direction": dir1,
        "target": None,
        "summary": f"{d1}, стадия {stage}, цена {alma200} ALMA 200",
    })

    # ─── S02 ВОЛНЫ ───
    s2 = s(2)
    is_up = s2.get("is_uptrend", True)
    dir2 = "вверх" if is_up else "вниз"
    wave = s2.get("current_wave", "—")
    targets2 = s2.get("wave_targets", [])
    t2 = targets2[0]["price"] if targets2 else None
    t2_all = ", ".join(_pct(t["price"], price) for t in targets2[:3]) if targets2 else "—"
    verdicts.append({
        "section": 2, "title": "ВОЛНЫ", "direction": dir2,
        "target": t2,
        "summary": f"Волна {wave}, цели: {t2_all}",
    })

    # ─── S03 ПАТТЕРНЫ ───
    s3 = s(3)
    patterns = s3.get("patterns", [])
    if patterns:
        p0 = patterns[0]
        dir3_raw = p0.get("direction", "")
        dir3 = "вверх" if "бычий" in dir3_raw or "вверх" in dir3_raw else (
            "вниз" if "медвеж" in dir3_raw or "вниз" in dir3_raw else "нейтрально")
        t3 = p0.get("target")
        name3 = p0.get("name_ru", p0.get("name", "—"))
        status3 = p0.get("status", "—")
        verdicts.append({
            "section": 3, "title": "ПАТТЕРНЫ", "direction": dir3,
            "target": t3,
            "summary": f"{name3} ({status3}), цель {_pct(t3, price) if t3 else '—'}",
        })
    else:
        verdicts.append({
            "section": 3, "title": "ПАТТЕРНЫ", "direction": "нейтрально",
            "target": None, "summary": "Явных паттернов не обнаружено",
        })

    # ─── S04 СВЕЧИ ───
    s4 = s(4)
    dom = s4.get("dominance", "баланс")
    dir4 = "вверх" if "покуп" in dom else ("вниз" if "продав" in dom else "нейтрально")
    char4 = s4.get("character", "—")
    verdicts.append({
        "section": 4, "title": "СВЕЧИ", "direction": dir4,
        "target": None,
        "summary": f"{char4}, доминирование: {dom}",
    })

    # ─── S05 ДИВЕРГЕНЦИИ ───
    s5 = s(5)
    sig5 = s5.get("signal", "нейтрально")
    dir5 = "вверх" if "бычий" in sig5 else ("вниз" if "медвеж" in sig5 else "нейтрально")
    rsi = s5.get("rsi_current")
    rsi_z = s5.get("rsi_zone", "—")
    verdicts.append({
        "section": 5, "title": "ДИВЕРГЕНЦИИ", "direction": dir5,
        "target": None,
        "summary": f"RSI {rsi:.1f} ({rsi_z}), сигнал: {sig5}" if rsi else f"Сигнал: {sig5}",
    })

    # ─── S06 УРОВНИ ───
    s6 = s(6)
    res6 = s6.get("resistances", [])
    sup6 = s6.get("supports", [])
    nearest_r = res6[0]["price"] if res6 else None
    nearest_s = sup6[0]["price"] if sup6 else None
    # Ближайший = вероятное направление первого движения
    dist_r = abs(nearest_r - price) if nearest_r else 999
    dist_s = abs(nearest_s - price) if nearest_s else 999
    dir6 = "вниз" if dist_s < dist_r else ("вверх" if dist_r < dist_s else "нейтрально")
    t6 = nearest_s if dir6 == "вниз" else nearest_r
    verdicts.append({
        "section": 6, "title": "УРОВНИ", "direction": dir6,
        "target": t6,
        "summary": f"Ближайшее сопр. {_pct(nearest_r, price) if nearest_r else '—'}, "
                   f"подд. {_pct(nearest_s, price) if nearest_s else '—'}",
    })

    # ─── S07 ФИБОНАЧЧИ ───
    s7 = s(7)
    fib_nearest = s7.get("nearest_level", {})
    fib_price = fib_nearest.get("price") if isinstance(fib_nearest, dict) else None
    fib_level = fib_nearest.get("level", "—") if isinstance(fib_nearest, dict) else "—"
    # Фибо сам по себе не даёт направления
    verdicts.append({
        "section": 7, "title": "ФИБОНАЧЧИ", "direction": "нейтрально",
        "target": fib_price,
        "summary": f"Ближайший уровень {fib_level}: {_pct(fib_price, price) if fib_price else '—'}",
    })

    # ─── S08 VSA ───
    s8 = s(8)
    obv = s8.get("obv_direction", "—")
    dir8 = "вверх" if "раст" in obv else ("вниз" if "пад" in obv else "нейтрально")
    anom = len(s8.get("volume_anomalies", []))
    verdicts.append({
        "section": 8, "title": "VSA", "direction": dir8,
        "target": None,
        "summary": f"OBV {obv}, аномалий объёма: {anom}",
    })

    # ─── S09 VP ───
    s9 = s(9)
    vp_main = s9.get("profile_a", s9) if "profile_a" in s9 else s9
    poc = vp_main.get("POC", {})
    poc_p = poc.get("price") if isinstance(poc, dict) else None
    vp_pos = vp_main.get("position", "—")
    dir9 = "вниз" if "ниже" in vp_pos else ("вверх" if "выше" in vp_pos else "нейтрально")
    verdicts.append({
        "section": 9, "title": "ОБЪЁМНЫЙ ПРОФИЛЬ", "direction": dir9,
        "target": poc_p,
        "summary": f"POC {_pct(poc_p, price) if poc_p else '—'}, цена {vp_pos}",
    })

    # ─── S10 ВАЙКОФФ ───
    s10 = s(10)
    struct = s10.get("structure_type", "—")
    dir10 = "вверх" if "Накопл" in struct else ("вниз" if "Распред" in struct else "нейтрально")
    phase = s10.get("phase", "—")
    node = s10.get("current_node", "—")
    verdicts.append({
        "section": 10, "title": "ВАЙКОФФ", "direction": dir10,
        "target": None,
        "summary": f"{struct}, фаза {phase}, узел: {node}",
    })

    # ─── S11 СТОПЫ ───
    s11 = s(11)
    stops_below = s11.get("stops_below", s11.get("stops_below_supports", []))
    stops_above = s11.get("stops_above", s11.get("stops_above_resistances", []))
    nearest_stop_below = stops_below[0].get("stop_zone", stops_below[0].get("level")) if stops_below else None
    nearest_stop_above = stops_above[0].get("stop_zone", stops_above[0].get("level")) if stops_above else None
    # Куда ближе кластер стопов = туда крупняк может сходить
    d_below = abs(price - nearest_stop_below) if nearest_stop_below else 999
    d_above = abs(nearest_stop_above - price) if nearest_stop_above else 999
    dir11 = "вниз" if d_below < d_above else "вверх"
    t11 = nearest_stop_below if dir11 == "вниз" else nearest_stop_above
    verdicts.append({
        "section": 11, "title": "СТОПЫ", "direction": dir11,
        "target": t11,
        "summary": f"Стопы снизу: {_pct(nearest_stop_below, price) if nearest_stop_below else '—'}, "
                   f"сверху: {_pct(nearest_stop_above, price) if nearest_stop_above else '—'}",
    })

    # ─── S12 ТЕМП ───
    s12 = s(12)
    k_tempo = s12.get("k_tempo", 1.0)
    tempo_class = s12.get("tempo_class", "—")
    atr_pct = s12.get("atr_daily_pct", s12.get("atr_pct", 0))
    atr_dyn = s12.get("atr_dynamics", "—")
    verdicts.append({
        "section": 12, "title": "ТЕМП", "direction": "нейтрально",
        "target": None,
        "summary": f"K-темпа {k_tempo:.2f} ({tempo_class}), ATR {atr_pct}%, динамика {atr_dyn}",
    })

    # ─── S13 FVG/ГЭПЫ ───
    s13 = s(13)
    fs = s13.get("first_step", {}) or {}
    fs_dir = fs.get("direction", "—")
    dir13 = "вверх" if "вверх" in fs_dir else ("вниз" if "вниз" in fs_dir else "нейтрально")
    fs_target = fs.get("target")
    fvg_count = len(s13.get("open_fvgs", []))
    verdicts.append({
        "section": 13, "title": "FVG/ГЭПЫ", "direction": dir13,
        "target": fs_target,
        "summary": f"Первый шаг: {fs_dir}, цель {_pct(fs_target, price) if fs_target else '—'}, "
                   f"открытых FVG: {fvg_count}",
    })

    # ─── S14 SQUEEZE ───
    s14 = s(14)
    vol_phase = s14.get("vol_phase", "—")
    squeeze = s14.get("squeeze_active", False)
    verdicts.append({
        "section": 14, "title": "SQUEEZE", "direction": "нейтрально",
        "target": None,
        "summary": f"Фаза: {vol_phase}" + (", Squeeze АКТИВЕН — готовность к импульсу" if squeeze else ""),
    })

    # ─── S15 ЭФФЕКТИВНОСТЬ ───
    s15 = s(15)
    er_class = s15.get("er_classification", "—")
    er_val = s15.get("efficiency_ratio", 0)
    confirm = s15.get("confirmation", "—")
    dir15 = "нейтрально"
    if "тренд" in er_class:
        dir15 = direction.replace("нисходящий", "вниз").replace("восходящий", "вверх")
        if dir15 not in ("вверх", "вниз"):
            dir15 = "нейтрально"
    verdicts.append({
        "section": 15, "title": "ЭФФЕКТИВНОСТЬ", "direction": dir15,
        "target": None,
        "summary": f"ER {er_val:.3f} ({er_class}). {confirm}",
    })

    # ─── S16 ПОТОКИ ───
    s16 = s(16)
    cvd16 = s16.get("cvd", {})
    cvd_dir = cvd16.get("direction", "—")
    dir16 = "вверх" if "вверх" in cvd_dir or "покуп" in cvd_dir else (
        "вниз" if "вниз" in cvd_dir or "прод" in cvd_dir else "нейтрально")
    cmf = s16.get("cmf", {})
    cmf_sig = cmf.get("signal", "—")
    verdicts.append({
        "section": 16, "title": "ПОТОКИ", "direction": dir16,
        "target": None,
        "summary": f"CVD: {cvd_dir}, CMF: {cmf_sig}",
    })

    # ─── S17 МИКРОСТРУКТУРА ───
    s17 = s(17)
    inst = s17.get("institutional_signal", "—")
    balance = s17.get("supply_demand_balance", "—")
    dir17 = "вверх" if "покуп" in inst else ("вниз" if "продав" in inst else "нейтрально")
    verdicts.append({
        "section": 17, "title": "МИКРОСТРУКТУРА", "direction": dir17,
        "target": None,
        "summary": f"{inst}, баланс: {balance}",
    })

    # ─── S18 КОРРЕЛЯЦИИ ───
    s18 = s(18)
    fan = s18.get("alma_fan_state", "—")
    regime = s18.get("market_regime", "—")
    dir18 = "вверх" if "бычий" in fan else ("вниз" if "медвеж" in fan else "нейтрально")
    verdicts.append({
        "section": 18, "title": "КОРРЕЛЯЦИИ", "direction": dir18,
        "target": None,
        "summary": f"ALMA веер: {fan}, режим: {regime}",
    })

    # ─── Подсчёт голосов ───
    vote_up = sum(1 for v in verdicts if v["direction"] == "вверх")
    vote_down = sum(1 for v in verdicts if v["direction"] == "вниз")
    vote_neutral = sum(1 for v in verdicts if v["direction"] == "нейтрально")

    return {
        "verdicts": verdicts,
        "vote_up": vote_up,
        "vote_down": vote_down,
        "vote_neutral": vote_neutral,
    }


def format_verdicts_text(result: dict, price: float) -> str:
    """Форматировать вердикты в текст для AI-модели."""
    lines = []
    for v in result["verdicts"]:
        arrow = _dir_arrow(v["direction"])
        tgt = _pct(v["target"], price) if v["target"] else "—"
        lines.append(
            f"{v['section']:2d}. {v['title']}: {arrow} {v['direction']} | "
            f"цель: {tgt} | {v['summary']}"
        )
    lines.append("")
    lines.append(
        f"Голосование: вверх {result['vote_up']}, "
        f"вниз {result['vote_down']}, "
        f"нейтрально {result['vote_neutral']}"
    )
    return "\n".join(lines)
