"""
Биржа-цифровой — Web-интерфейс (Flask).

Запуск:
    python app.py
    -> http://localhost:5001
"""
import os
import sys
import json
import uuid
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, Response)

from db import init_db, save_report, get_report, get_all_reports, delete_report
from pipeline.preprocessor import run_full_pipeline, run_preprocessor
from pipeline.multi_tf import consolidate, format_multi_tf_report
from config import ANALYSIS_TYPES, PROVIDERS, DEFAULT_PROVIDER
from ai.client import get_provider_status, save_env_key, load_env_keys

app = Flask(__name__)
app.secret_key = "birza_digital_v1_secret"

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

tasks = {}


# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────

@app.route("/")
def index():
    providers = get_provider_status()
    return render_template("index.html",
                           active_page="index",
                           analysis_types=ANALYSIS_TYPES,
                           providers=providers,
                           default_provider=DEFAULT_PROVIDER)


@app.route("/settings")
def settings():
    providers = get_provider_status()
    env_keys = load_env_keys()
    return render_template("settings.html",
                           active_page="settings",
                           providers=providers,
                           env_keys=env_keys)


@app.route("/settings/save", methods=["POST"])
def settings_save():
    for pid, prov in PROVIDERS.items():
        key_val = request.form.get(f"key_{pid}", "").strip()
        if key_val:
            save_env_key(prov["env_key"], key_val)
    flash("Ключи сохранены", "success")
    return redirect(url_for("settings"))


# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────

@app.route("/api/providers")
def api_providers():
    """Список провайдеров с моделями и статусом ключей."""
    return jsonify(get_provider_status())


# ──────────────────────────────────────────────
# Run Analysis
# ──────────────────────────────────────────────

@app.route("/run", methods=["POST"])
def run():
    csv_file = request.files.get("csv_file")
    if not csv_file or csv_file.filename == "":
        flash("Выберите CSV файл", "error")
        return redirect(url_for("index"))

    filename = csv_file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    csv_file.save(save_path)

    # Тип анализа определяется автоматически по ТФ в CSV
    analysis_type = "auto"
    skip_ai = request.form.get("skip_ai") == "1"
    provider = request.form.get("provider", DEFAULT_PROVIDER)
    model = request.form.get("model", "")

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "status": "running",
        "step": 0,
        "total": 22,
        "message": "Подготовка...",
        "result": None,
        "error": None,
        "report_id": None,
        "csv_path": save_path,
        "filename": filename,
        "analysis_type": analysis_type,
        "skip_ai": skip_ai,
        "provider": provider,
        "model": model,
    }

    thread = threading.Thread(target=_run_task, args=(task_id,), daemon=True)
    thread.start()

    return jsonify({"task_id": task_id})


def _run_task(task_id):
    task = tasks[task_id]
    start_time = time.time()

    def on_progress(step, total, message):
        task["step"] = step
        task["total"] = total
        task["message"] = message

    try:
        # Автоопределение горизонта по ТФ из CSV (определяется в data_prep)
        # Маппинг: tf_hours → горизонт
        TF_HORIZON = {
            0.25: ("Интрадей-экспресс", "внутри дня"),      # 15m
            1.0:  ("Краткосрок-экспресс", "1–2 сессии"),     # 1H
            4.0:  ("Неделя-экспресс", "5–7 сессий"),         # 4H
            24.0: ("Среднесрок-экспресс", "2–4 недели"),     # 1D
            168.0:("Долгосрок", "несколько месяцев"),         # 1W
        }
        # Пока не знаем tf_hours (определится в pipeline), ставим fallback
        at_name = None
        horizon = None

        result = run_full_pipeline(
            task["csv_path"],
            on_progress=on_progress,
            skip_ai=task["skip_ai"],
            provider=task["provider"],
            model=task["model"] or None,
            horizon=horizon,
            analysis_type_name=at_name,
        )

        elapsed = time.time() - start_time
        meta = result["meta"]

        # Автоопределение горизонта по фактическому ТФ из данных
        tf_hours = meta.get("tf_hours", 4.0)
        best_tf = min(TF_HORIZON.keys(), key=lambda t: abs(t - tf_hours))
        at_name, horizon = TF_HORIZON[best_tf]
        meta["analysis_type_name"] = at_name
        meta["horizon"] = horizon
        ai_response = result.get("ai_report", "")
        if ai_response:
            report_text = ai_response
        else:
            from ai.formatter import format_report as fmt_report
            report_text = fmt_report(meta, result["sections_data"])

        ai_provider = "none"
        if not task["skip_ai"]:
            ai_provider = f"{task['provider']}/{task.get('model', '')}"

        report_id = save_report(
            ticker=meta.get("ticker", ""),
            exchange=meta.get("exchange", ""),
            timeframe=meta.get("timeframe", ""),
            analysis_type=task["analysis_type"],
            ai_provider=ai_provider,
            price=meta.get("current_price", 0),
            n_bars=meta.get("n_bars", 0),
            report_text=report_text,
            report_data=result.get("analysis_json", {}),
            ai_response=ai_response,
            csv_filename=task["filename"],
            elapsed_sec=elapsed,
        )

        task["status"] = "done"
        task["report_id"] = report_id
        task["message"] = "Готово!"
        task["step"] = task["total"]

    except Exception as e:
        task["status"] = "error"
        task["error"] = str(e)
        task["message"] = f"Ошибка: {e}"

    finally:
        if os.path.exists(task["csv_path"]):
            os.remove(task["csv_path"])


# ──────────────────────────────────────────────
# Multi-TF (Elder Triple Screen) Analysis
# ──────────────────────────────────────────────

@app.route("/run_multi", methods=["POST"])
def run_multi():
    """Сводный анализ нескольких ТФ одного тикера по системе Элдера."""
    files = request.files.getlist("csv_files")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "Не выбрано ни одного CSV файла"}), 400
    if len(files) < 2:
        return jsonify({"error": "Для мульти-ТФ анализа нужно минимум 2 файла"}), 400

    analysis_type = request.form.get("analysis_type", "short")
    skip_ai = request.form.get("skip_ai") == "1"
    provider = request.form.get("provider", DEFAULT_PROVIDER)
    model = request.form.get("model", "")

    # Сохраняем все файлы
    saved_files = []
    for f in files:
        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:6]}_{f.filename}")
        f.save(save_path)
        saved_files.append({"path": save_path, "filename": f.filename})

    task_id = "multi-" + str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "status": "running",
        "step": 0,
        "total": len(saved_files) * 22 + 5,  # ~22 шагов на файл + consolidation + AI
        "message": "Подготовка мульти-ТФ анализа...",
        "result": None,
        "error": None,
        "report_id": None,
        "child_report_ids": [],
        "files": saved_files,
        "analysis_type": analysis_type,
        "skip_ai": skip_ai,
        "provider": provider,
        "model": model,
        "is_multi": True,
    }

    thread = threading.Thread(target=_run_multi_task, args=(task_id,), daemon=True)
    thread.start()

    return jsonify({"task_id": task_id})


def _run_multi_task(task_id):
    """Фоновый runner для мульти-ТФ анализа."""
    task = tasks[task_id]
    start_time = time.time()

    at_key = task.get("analysis_type", "")
    at_info = ANALYSIS_TYPES.get(at_key, {})
    horizon = at_info.get("horizon", "—")
    at_name = at_info.get("name", "—")

    # Авто-определение горизонта по ТФ (как в single-mode)
    TF_HORIZON_AUTO = {
        0.25: ("Интрадей-экспресс", "внутри дня"),
        1.0:  ("Краткосрок-экспресс", "1–2 сессии"),
        4.0:  ("Неделя-экспресс", "5–7 сессий"),
        24.0: ("Среднесрок-экспресс", "2–4 недели"),
        168.0:("Долгосрок", "несколько месяцев"),
    }
    need_auto_detect = (at_key == "auto" or not at_info)

    results = []  # {meta, sections_data, report_id, report_text}
    per_tf_reports = {}  # {tf_label: report_text}

    try:
        total_files = len(task["files"])
        step_per_file = 22
        current_base = 0

        # 1. Прогон каждого файла через препроцессор (всегда skip_ai=True,
        #    AI вызывается только на уровне consolidated отчёта)
        for idx, file_info in enumerate(task["files"]):
            task["message"] = f"[{idx+1}/{total_files}] Анализ {file_info['filename']}..."

            def on_progress(step, total, message, _base=current_base, _fname=file_info['filename']):
                task["step"] = _base + step
                task["message"] = f"[{idx+1}/{total_files}] {_fname}: {message}"

            pre = run_preprocessor(file_info["path"], on_progress=on_progress,
                                    original_filename=file_info["filename"])
            meta = pre["meta"]

            from ai.formatter import format_report as fmt_report
            report_text = fmt_report(meta, pre["sections_data"])

            # Сохраняем в БД как дочерний отчёт
            child_id = save_report(
                ticker=meta.get("ticker", ""),
                exchange=meta.get("exchange", ""),
                timeframe=meta.get("timeframe", ""),
                analysis_type=f"[multi] {at_name}",
                ai_provider="none",
                price=meta.get("current_price", 0),
                n_bars=meta.get("n_bars", 0),
                report_text=report_text,
                report_data=pre.get("analysis_json", {}),
                ai_response=None,
                csv_filename=file_info["filename"],
                elapsed_sec=0,
            )
            task["child_report_ids"].append(child_id)

            results.append({
                "meta": meta,
                "sections_data": pre["sections_data"],
                "report_id": child_id,
                "report_text": report_text,
            })
            per_tf_reports[meta.get("timeframe", "?")] = report_text

            current_base += step_per_file

        # Авто-определение типа анализа по старшему ТФ
        if need_auto_detect and results:
            tf_hours_list = [r["meta"].get("tf_hours", 1.0) for r in results]
            senior_tf_hours = max(tf_hours_list)
            best_tf = min(TF_HORIZON_AUTO.keys(), key=lambda t: abs(t - senior_tf_hours))
            at_name, horizon = TF_HORIZON_AUTO[best_tf]

        # Проставить мета в дочерние результаты
        for r in results:
            r["meta"]["analysis_type_name"] = at_name or ""
            r["meta"]["horizon"] = horizon or ""

        # 2. Консолидация
        task["step"] = current_base + 1
        task["message"] = "Консолидация по системе Элдера..."
        consolidated = consolidate(results, horizon=horizon, analysis_type_name=at_name)

        # 3. Pre-report (текстовый)
        pre_report_text = format_multi_tf_report(consolidated)

        # 4. AI (опционально)
        final_text = pre_report_text
        ai_response = ""
        ai_provider_label = "none"

        if not task["skip_ai"]:
            task["step"] = current_base + 3
            task["message"] = f"Вызов AI ({task['provider']}) для сводного вердикта..."
            try:
                from ai.prompts import build_multi_tf_prompt
                from ai.client import create_client
                sys_prompt, user_msg = build_multi_tf_prompt(
                    consolidated, per_tf_reports, horizon=horizon,
                )
                client = create_client(
                    provider_id=task["provider"],
                    model=task["model"] or None,
                )
                ai_response = client.analyze(sys_prompt, user_msg)
                # Финальный отчёт = AI + (свёрнутый preprocessor pre-report для справки)
                final_text = (
                    ai_response
                    + "\n\n\n---\n## 📋 ДЕТАЛЬНЫЕ ДАННЫЕ ПРЕПРОЦЕССОРА\n\n"
                    + pre_report_text
                )
                ai_provider_label = f"{task['provider']}/{task.get('model', '')}"
            except Exception as ai_err:
                final_text = (
                    f"⚠ Ошибка AI: {ai_err}\n\n"
                    + pre_report_text
                )

        # 5. Сохраняем сводный отчёт
        elapsed = time.time() - start_time
        ticker = consolidated.get("ticker", "—")
        exchange = consolidated.get("exchange", "—")
        tfs = consolidated.get("tf_list", [])
        timeframe_label = "+".join(tfs) if tfs else "multi"

        report_data = {
            "multi_tf": True,
            "consolidated": _safe_for_db(consolidated),
            "child_report_ids": task["child_report_ids"],
            "analysis_type": at_key,
            "analysis_type_name": at_name,
            "horizon": horizon,
        }

        report_id = save_report(
            ticker=ticker,
            exchange=exchange,
            timeframe=timeframe_label,
            analysis_type=f"🎯 Мульти-ТФ Элдер ({at_name})",
            ai_provider=ai_provider_label,
            price=consolidated.get("current_price", 0),
            n_bars=sum(r["meta"].get("n_bars", 0) for r in results),
            report_text=final_text,
            report_data=report_data,
            ai_response=ai_response,
            csv_filename=" + ".join(f["filename"] for f in task["files"]),
            elapsed_sec=elapsed,
        )

        task["status"] = "done"
        task["report_id"] = report_id
        task["step"] = task["total"]
        task["message"] = "Готово!"

    except Exception as e:
        import traceback
        task["status"] = "error"
        task["error"] = f"{e}\n{traceback.format_exc()}"
        task["message"] = f"Ошибка: {e}"

    finally:
        for fi in task["files"]:
            try:
                if os.path.exists(fi["path"]):
                    os.remove(fi["path"])
            except Exception:
                pass


def _safe_for_db(obj):
    """Очистить объект от numpy/несериализуемых типов."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _safe_for_db(v) for k, v in obj.items() if not callable(v)}
    elif isinstance(obj, (list, tuple)):
        return [_safe_for_db(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


SUBSCRIPTION_BLOCK = (
    "🟥 Платная аналитика по ГАЗУ и ПЛАТИНЕ: @Siroezhkin_bot\n"
    "💰 Для донатов:\n"
    "💳 https://pay.cloudtips.ru/p/562cbedb\n"
    "💳 2200 7006 2350 2977 (Т-Банк)\n"
    "🔥 Больше инструментов в профиле"
)


def _wrap_with_header_and_subs(body: str, meta: dict) -> str:
    """Добавить шапку в начало и блок подписок в конец.

    Удаляет уже существующие шапки/подписки в теле, чтобы не было дублей.
    """
    import re

    ticker = meta.get("ticker", "—")
    exchange = meta.get("exchange", "—")
    tf = meta.get("tf", meta.get("timeframe", "—"))
    price = meta.get("price", meta.get("current_price", 0))
    from datetime import datetime
    date = datetime.now().strftime("%Y-%m-%d %H:%M")

    header = (
        f"📘ТИКЕР: #{ticker} (#{exchange})\n"
        f"БИРЖА: #{exchange}\n"
        f"ТАЙМФРЕЙМ: {tf}\n"
        f"ЦЕНА: {price}\n"
        f"ДАТА И ВРЕМЯ: {date} UTC+3\n"
        f"❗ НЕ ЯВЛЯЕТСЯ ИИР ❗"
    )

    cleaned = body or ""

    # Убрать уже существующую шапку из тела (если ИИ её добавил)
    # Шапка начинается с 📘ТИКЕР и заканчивается НЕ ЯВЛЯЕТСЯ ИИР
    cleaned = re.sub(
        r"📘\s*ТИКЕР.*?НЕ ЯВЛЯЕТСЯ ИИР\s*❗?",
        "",
        cleaned,
        count=1,
        flags=re.DOTALL,
    ).lstrip()

    # Убрать уже существующий блок подписок
    cleaned = re.sub(
        r"🟥\s*Платная аналитика.*?Больше инструментов в профиле",
        "",
        cleaned,
        flags=re.DOTALL,
    ).rstrip()

    # Убрать разделители-рамки в конце
    cleaned = re.sub(r"[━─=]{3,}\s*$", "", cleaned).rstrip()

    return f"{header}\n\n{cleaned}\n\n━━━━━━━━━━━━━━━━━━━━━━━━\n{SUBSCRIPTION_BLOCK}"


@app.route("/progress/<task_id>")
def progress(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"status": "error", "message": "Задача не найдена"})
    return jsonify({
        "status": task["status"],
        "step": task["step"],
        "total": task["total"],
        "message": task["message"],
        "report_id": task["report_id"],
        "error": task["error"],
        "is_multi": task.get("is_multi", False),
        "child_report_ids": task.get("child_report_ids", []),
    })


@app.route("/report/<int:report_id>")
def view_report(report_id):
    report = get_report(report_id)
    if not report:
        flash("Отчёт не найден", "error")
        return redirect(url_for("history"))
    providers = get_provider_status()
    providers_json = json.dumps({
        pid: {"name": p["name"], "models": p.get("models", []), "default": p.get("default", False)}
        for pid, p in providers.items()
    })
    return render_template("report.html", report=report, active_page="",
                           providers_json=providers_json,
                           default_provider=DEFAULT_PROVIDER)


@app.route("/report/<int:report_id>/download")
def download_report(report_id):
    """Скачать отчёт как .txt файл."""
    report = get_report(report_id)
    if not report:
        flash("Отчёт не найден", "error")
        return redirect(url_for("history"))

    text = report["report_text"] or ""
    ticker = (report["ticker"] or "report").replace("/", "_").replace(" ", "_")
    tf = (report["timeframe"] or "").replace("/", "_")
    filename = f"{ticker}_{tf}_{report_id}.txt".strip("_")

    return Response(
        text,
        mimetype="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.route("/report/<int:report_id>/download_json")
def download_report_json(report_id):
    """Скачать JSON препроцессора."""
    report = get_report(report_id)
    if not report:
        flash("Отчёт не найден", "error")
        return redirect(url_for("history"))

    payload = report["report_json"] or "{}"
    ticker = (report["ticker"] or "report").replace("/", "_").replace(" ", "_")
    tf = (report["timeframe"] or "").replace("/", "_")
    filename = f"{ticker}_{tf}_{report_id}.json".strip("_")

    return Response(
        payload,
        mimetype="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.route("/history")
def history():
    reports = get_all_reports()
    return render_template("history.html", reports=reports, active_page="history")


@app.route("/delete/<int:report_id>", methods=["POST"])
def delete(report_id):
    delete_report(report_id)
    flash("Отчёт удалён", "success")
    return redirect(url_for("history"))


@app.route("/fomo/<int:report_id>", methods=["POST"])
def make_fomo(report_id):
    """Сгенерировать ФОМО по сохранённому отчёту."""
    report = get_report(report_id)
    if not report:
        flash("Отчёт не найден", "error")
        return redirect(url_for("history"))

    provider = request.form.get("provider", DEFAULT_PROVIDER)
    model = request.form.get("model", "")

    try:
        analysis_json = json.loads(report["report_json"]) if report["report_json"] else {}
        pre_report = report["report_text"] or ""

        from ai.prompts import build_fomo_prompt
        from ai.client import create_client

        # Собрать sections_map и meta для «совета директоров»
        sections_map = {}
        aj_sections = analysis_json.get("sections", {})
        for sid_str, sec in aj_sections.items():
            try:
                sections_map[int(sid_str)] = sec
            except (ValueError, TypeError):
                pass

        fomo_meta = analysis_json.get("meta", {})
        fomo_meta.setdefault("ticker", report["ticker"] or "—")
        fomo_meta.setdefault("exchange", report["exchange"] or "—")
        fomo_meta.setdefault("tf", report["timeframe"] or "—")
        fomo_meta.setdefault("price", report["price"] or 0)
        fomo_meta.setdefault("date", "—")
        # Нормализация: current_price → price
        if "current_price" in fomo_meta and "price" not in fomo_meta:
            fomo_meta["price"] = fomo_meta["current_price"]
        elif fomo_meta.get("price") == 0 and fomo_meta.get("current_price"):
            fomo_meta["price"] = fomo_meta["current_price"]

        system_prompt, user_msg = build_fomo_prompt(
            analysis_json, pre_report,
            sections_map=sections_map, meta=fomo_meta,
        )
        client = create_client(provider_id=provider, model=model or None)
        fomo_text = client.analyze(system_prompt, user_msg)

        # Сохранить ФОМО как новый отчёт
        new_id = save_report(
            ticker=report["ticker"],
            exchange=report["exchange"],
            timeframe=report["timeframe"],
            analysis_type=f"ФОМО ({report['analysis_type'] or '—'})",
            ai_provider=f"{provider}/{model}",
            price=report["price"],
            n_bars=report["n_bars"],
            report_text=fomo_text,
            report_data=analysis_json,
            ai_response=fomo_text,
            csv_filename=report["csv_filename"],
            elapsed_sec=0,
        )

        # AJAX или обычный запрос
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": True, "redirect": url_for("view_report", report_id=new_id)})
        flash(f"ФОМО сгенерирован (отчёт #{new_id})", "success")
        return redirect(url_for("view_report", report_id=new_id))

    except Exception as e:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": False, "error": str(e)}), 500
        flash(f"Ошибка ФОМО: {e}", "error")
        return redirect(url_for("view_report", report_id=report_id))


@app.route("/ai_conclusion/<int:report_id>", methods=["POST"])
def make_ai_conclusion(report_id):
    """Сгенерировать ВЫВОД ИИ по данным препроцессора."""
    report = get_report(report_id)
    if not report:
        flash("Отчёт не найден", "error")
        return redirect(url_for("history"))

    provider = request.form.get("provider", DEFAULT_PROVIDER)
    model = request.form.get("model", "")

    try:
        analysis_json = json.loads(report["report_json"]) if report["report_json"] else {}
        pre_report = report["report_text"] or ""

        from ai.prompts import build_ai_conclusion_prompt
        from ai.client import create_client

        sections_map = {}
        aj_sections = analysis_json.get("sections", {})
        for sid_str, sec in aj_sections.items():
            try:
                sections_map[int(sid_str)] = sec
            except (ValueError, TypeError):
                pass

        conclusion_meta = analysis_json.get("meta", {})
        conclusion_meta.setdefault("ticker", report["ticker"] or "—")
        conclusion_meta.setdefault("exchange", report["exchange"] or "—")
        conclusion_meta.setdefault("tf", report["timeframe"] or "—")
        conclusion_meta.setdefault("price", report["price"] or 0)
        conclusion_meta.setdefault("date", "—")
        if "current_price" in conclusion_meta and "price" not in conclusion_meta:
            conclusion_meta["price"] = conclusion_meta["current_price"]
        elif conclusion_meta.get("price") == 0 and conclusion_meta.get("current_price"):
            conclusion_meta["price"] = conclusion_meta["current_price"]

        system_prompt, user_msg = build_ai_conclusion_prompt(
            analysis_json, pre_report,
            sections_map=sections_map, meta=conclusion_meta,
        )
        client = create_client(provider_id=provider, model=model or None)
        conclusion_text = client.analyze(system_prompt, user_msg)

        # v8: валидация вывода + один retry со строгим промптом + fallback-пометкой
        try:
            from ai.formatter import _validate_conclusion_v8, _build_fallback_prose
            validation = _validate_conclusion_v8(conclusion_text)
            if validation.get("issues"):
                strict_suffix = (
                    "\n\nКРИТИЧНО (retry): предыдущий ответ не прошёл валидацию v8.\n"
                    f"Проблемы: {'; '.join(validation['issues'][:5])}.\n"
                    "Требования:\n"
                    "• РОВНО 6 буллетов с маркером 📌 в порядке: "
                    "ключевые цели → второстепенные цели → остальные цели → "
                    "плановый маршрут → вероятные сроки → уровень слома.\n"
                    "• В буллетах — только цены, %, имена уровней. "
                    "НИКАКИХ RSI/ALMA/Wyckoff/FVG/POC/MACD/ADX и прочих ТА-терминов.\n"
                    "• Прозаический абзац перед буллетами — ≥300 слов, связный текст "
                    "со словами «кроме того», «в свою очередь», «наконец»."
                )
                conclusion_text2 = client.analyze(
                    system_prompt, user_msg + strict_suffix
                )
                validation2 = _validate_conclusion_v8(conclusion_text2)
                if not validation2.get("issues"):
                    conclusion_text = conclusion_text2
                else:
                    # fallback: подставляем прозу-пустышку и помечаем
                    try:
                        fb_prose = _build_fallback_prose(
                            analysis_json, sections_map, conclusion_meta
                        )
                        conclusion_text = (
                            "# ⚠ VALIDATION FAILED (v8): "
                            f"{'; '.join(validation2['issues'][:3])}\n\n"
                            + fb_prose + "\n\n" + conclusion_text2
                        )
                    except Exception:
                        conclusion_text = (
                            "# ⚠ VALIDATION FAILED (v8): "
                            f"{'; '.join(validation2['issues'][:3])}\n\n"
                            + conclusion_text2
                        )
        except Exception:
            # валидатор не обязан ломать пайплайн
            pass

        new_id = save_report(
            ticker=report["ticker"],
            exchange=report["exchange"],
            timeframe=report["timeframe"],
            analysis_type=f"Вывод ИИ ({report['analysis_type'] or '—'})",
            ai_provider=f"{provider}/{model}",
            price=report["price"],
            n_bars=report["n_bars"],
            report_text=conclusion_text,
            report_data=analysis_json,
            ai_response=conclusion_text,
            csv_filename=report["csv_filename"],
            elapsed_sec=0,
        )

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": True, "redirect": url_for("view_report", report_id=new_id)})
        flash(f"Вывод ИИ сгенерирован (отчёт #{new_id})", "success")
        return redirect(url_for("view_report", report_id=new_id))

    except Exception as e:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": False, "error": str(e)}), 500
        flash(f"Ошибка генерации вывода ИИ: {e}", "error")
        return redirect(url_for("view_report", report_id=report_id))


if __name__ == "__main__":
    init_db()
    print("\n  Биржа-цифровой v1.0")
    print("  http://localhost:5010\n")
    app.run(host="0.0.0.0", port=5010, debug=True, use_reloader=False, threaded=True)
