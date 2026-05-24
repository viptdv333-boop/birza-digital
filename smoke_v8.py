"""Smoke test: end-to-end v8 pipeline on NG 1H CSV, verify new features."""
import os
import sys
import io

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

from pipeline.preprocessor import run_preprocessor
from ai.formatter import format_report
from ai.prompts import _infer_analysis_type


def main():
    csv_path = r"C:\Users\viptd\OneDrive\Изображения\Графики для торговых анализов\NG1\CSV\no_date\NYMEX_NG1!, 60_5d5ca.csv"
    print(f"[INFO] CSV: {csv_path}")

    res = run_preprocessor(csv_path)
    meta = res["meta"]
    sections_data = res["sections_data"]

    # Meta check
    print("\n========== META ==========")
    print(f"ticker: {meta.get('ticker')}")
    print(f"tf: {meta.get('timeframe')}")
    print(f"price: {meta.get('current_price')}")
    inferred = _infer_analysis_type(meta.get("timeframe"))
    print(f"_infer_analysis_type: {inferred}")

    sec_by_id = {s["section_id"]: s for s in sections_data}

    # S01
    s1_data = sec_by_id.get(1, {}).get("data", {})
    print("\n========== S01 ==========")
    st = s1_data.get("senior_trend", {})
    lt = s1_data.get("local_trend", {})
    print(f"senior_trend dir={st.get('direction')} start={st.get('start_price')} window={st.get('analysis_window')}")
    print(f"local_trend dir={lt.get('direction')} start={lt.get('start_price')}")

    # S09
    s9_data = sec_by_id.get(9, {}).get("data", {})
    pa = s9_data.get("profile_a", {})
    pb = s9_data.get("profile_b", {})
    print("\n========== S09 (v8.1 §IV: TF-window VP) ==========")
    print(f"profile_a.description: {pa.get('description')}")
    print(f"profile_a.anchor_reason: {pa.get('anchor_reason')}")
    print(f"profile_a.bars_count: {pa.get('bars_count')} (window={pa.get('window_bars')})")
    print(f"profile_a.phase: {pa.get('phase')}")
    print(f"profile_b.description: {pb.get('description')}")

    # S06 — проверим что уровни в TF-окне
    s6_data = sec_by_id.get(6, {}).get("data", {})
    print("\n========== S06 (v8.1 §IV: levels within window) ==========")
    sup5 = s6_data.get("supports_5", [])
    res5 = s6_data.get("resistances_5", [])
    print(f"supports_5: {[s.get('price') for s in sup5]}")
    print(f"resistances_5: {[r.get('price') for r in res5]}")

    # Full report
    report = format_report(meta, sections_data, include_conclusions=True)

    # Header checks
    print("\n========== HEADER SECTION ==========")
    hdr_lines = report.split("\n")[:10]
    for ln in hdr_lines:
        print(ln)

    checks = []
    checks.append(("ТИП АНАЛИЗА in header", "ТИП АНАЛИЗА" in report))
    checks.append(("ключевые цели bullet", "ключевые цели" in report))
    checks.append(("остальные цели bullet", "остальные цели" in report))
    checks.append(("вероятный маршрут bullet", "вероятный маршрут" in report))
    checks.append(("вероятные сроки bullet", "вероятные сроки" in report))
    checks.append(("отмена маршрута bullet (v8.1)", "отмена маршрута" in report))
    checks.append(("opening 'торгуется по'", "торгуется по" in report))
    checks.append(("no v8 'приостановка и наблюдение'", "приостановка и наблюдение" not in report))
    checks.append(("no v8 'слом структуры и пересмотр'", "слом структуры и пересмотр" not in report))
    checks.append(("no diagnostic K=/ATR= in timing", "Total_dist=" not in report))
    checks.append(("'Нужен ФОМО?' в конце", "Нужен ФОМО?" in report))
    # v8.1: история прихода к цене (case-insensitive)
    rep_low = report.lower()
    checks.append(("история прихода к цене", any(
        w in rep_low for w in ("упали от", "выросли от", "стоим на", "отрабатыва",
                                "удерживается", "после движения от"))))
    # v8.1: характер движения упомянут
    checks.append(("характер движения", any(
        w in report.lower() for w in ("импульс", "манипуляц", "сползан", "тренд", "коррекц"))))

    # Запустить валидатор v8.1 на «вырезанный» вывод
    from ai.formatter import _validate_conclusion_v8
    s19_idx = report.find("Раздел 19 ВЫВОД")
    s19_end = report.find("━━━━━━━", s19_idx)
    s19_text = report[s19_idx:s19_end] if s19_idx >= 0 else report
    v = _validate_conclusion_v8(s19_text)
    print(f"\n========== ВАЛИДАТОР v8.1 ==========")
    print(f"Буллитов: {v['n_bullets']}, слов в абзаце: {v['paragraph_words']}")
    if v['issues']:
        for i in v['issues']:
            print(f"  [ISSUE] {i}")
    else:
        print("  [OK] нет нарушений")

    print("\n========== CHECKS ==========")
    for name, ok in checks:
        tag = "[OK]  " if ok else "[FAIL]"
        print(f"{tag} {name}")

    # Dump vivo section — print s19 conclusion block
    print("\n========== S19 ВЫВОД (from report) ==========")
    if "Раздел 19 ВЫВОД" in report:
        idx = report.find("Раздел 19 ВЫВОД")
        end_idx = report.find("━━━━━━━", idx)
        print(report[idx:end_idx if end_idx > 0 else idx + 4000])
    else:
        print("(нет раздела 19)")

    # Save full report
    out = os.path.join(BASE, "smoke_v8_out.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[INFO] full report → {out}")


if __name__ == "__main__":
    main()
