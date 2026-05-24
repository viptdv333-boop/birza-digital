"""
Биржа-цифровой — Раздел 19: МАТЕМАТИЧЕСКАЯ МОДЕЛЬ.

Тип: full. Вызывает Z:\math_model\ для скоринга маршрута.
MC-GARCH (вероятности целей), HMM (режим), Bayes (ансамбль),
PDE (коридор P10/P50/P90), Chaos/OGY (предсказуемость).
"""
import sys
import os
from sections.base import SectionProcessor

# Добавить math_model в path для импорта
MATH_MODEL_DIR = r"Z:\math_model"
if MATH_MODEL_DIR not in sys.path:
    sys.path.insert(0, MATH_MODEL_DIR)


class MathModelProcessor(SectionProcessor):
    section_id = 20
    section_emoji = "🔢"
    section_title = "МАТЕМАТИЧЕСКАЯ МОДЕЛЬ"
    section_type = "full"

    def compute(self, df, context: dict) -> dict:
        """Запустить математическую модель на тех же данных."""
        csv_path = context.get("csv_path")
        if not csv_path or not os.path.exists(csv_path):
            return {"error": "CSV файл не найден для мат.модели"}

        try:
            # Временно переключить cwd и sys.path на math_model
            # чтобы его config.py не конфликтовал с birza_digital/config.py
            import importlib
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            os.chdir(MATH_MODEL_DIR)
            # Убрать birza_digital из path, поставить math_model первым
            sys.path = [MATH_MODEL_DIR] + [p for p in sys.path
                                            if "birza_digital" not in p]

            # Перезагрузить config из math_model
            if "config" in sys.modules:
                del sys.modules["config"]

            from main import run_pipeline
            result = run_pipeline(csv_path)

            # Восстановить
            os.chdir(original_cwd)
            sys.path = original_path
        except Exception as e:
            # Восстановить при ошибке
            try:
                os.chdir(original_cwd)
                sys.path = original_path
            except Exception:
                pass
            return {"error": f"Ошибка мат.модели: {str(e)[:200]}"}

        current_price = float(context["close"][-1])
        atr_last = context["atr_last"]

        def _safe(obj, key, default=None):
            """Safe .get() — если obj не dict, вернуть default."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default if default is not None else {}

        # ── MC-GARCH: сценарии и вероятности ──
        mc = _safe(result, "mc", {})
        scenarios = _safe(mc, "scenarios", [])
        mc_scenarios = []
        for s in (scenarios if isinstance(scenarios, list) else []):
            mc_scenarios.append({
                "center": round(_safe(s, "center", 0), 2),
                "center_pct": round((_safe(s, "center", 0) / current_price - 1) * 100, 2),
                "share": round(_safe(s, "share", 0) * 100, 1),
                "range_10": round(_safe(s, "p10", 0), 2),
                "range_90": round(_safe(s, "p90", 0), 2),
            })

        mc_targets = _safe(mc, "targets", {})
        mc_t1 = _safe(mc_targets, "T1", {})
        mc_t2 = _safe(mc_targets, "T2", {})
        mc_t3 = _safe(mc_targets, "T3", {})

        # ── HMM: режим рынка ──
        hmm = _safe(result, "hmm", {})
        hmm_data = {
            "current_regime": _safe(hmm, "current_regime", "—"),
            "regime_probs": _safe(hmm, "pi_T", []),
            "confidence": round(_safe(hmm, "confidence", 0), 3),
            "projected_price": round(_safe(hmm, "S_proj", current_price), 2),
            "projected_pct": round((_safe(hmm, "S_proj", current_price) / current_price - 1) * 100, 2),
        }

        # ── PDE Ремизов: коридор ──
        pde = _safe(result, "pde", {})
        pde_data = {
            "P10": round(_safe(pde, "P10", 0), 2),
            "P50": round(_safe(pde, "P50", 0), 2),
            "P90": round(_safe(pde, "P90", 0), 2),
            "E_S": round(_safe(pde, "E_S", 0), 2),
            "P10_pct": round((_safe(pde, "P10", current_price) / current_price - 1) * 100, 2),
            "P50_pct": round((_safe(pde, "P50", current_price) / current_price - 1) * 100, 2),
            "P90_pct": round((_safe(pde, "P90", current_price) / current_price - 1) * 100, 2),
            "confidence": round(_safe(pde, "confidence", 0), 3),
        }

        # ── Chaos / OGY ──
        chaos = _safe(result, "chaos", {})
        chaos_forecast = _safe(chaos, "forecast", {})
        chaos_data = {
            "lyapunov": round(_safe(chaos, "lambda1", 0), 4),
            "T_lyapunov": round(_safe(chaos, "T_lyap", 0), 1),
            "D2": round(_safe(chaos, "D2", 0), 2),
            "predictability": _safe(chaos, "predictability", "—"),
            "n_upo": len(_safe(chaos, "upos", [])),
            "on_orbit": _safe(chaos_forecast, "on_orbit", False),
            "ogy_forecast": round(_safe(chaos_forecast, "forecast", [current_price])[-1], 2)
                if _safe(chaos_forecast, "forecast") else None,
        }

        # ── Bayes: ансамбль ──
        bayes = _safe(result, "bayes", {})
        bayes_data = {
            "posterior": _safe(bayes, "posterior", []),
            "best_scenario": _safe(bayes, "best_scenario", 0),
            "weights": _safe(bayes, "weights", {}),
        }

        # ── Model Probs: P(up/down/flat) по каждой модели ──
        model_probs = _safe(result, "model_probs", {})

        # ── Route Racing: согласие моделей ──
        racing = _safe(result, "racing", {})
        racing_data = {
            "dominant": _safe(racing, "dominant", "—"),
            "agreement": _safe(racing, "agreement", {}),
            "conflicts": _safe(racing, "conflicts", []),
            "targets": _safe(racing, "targets", {}),
        }

        # ── Уровни мат.модели с P(touch) ──
        math_levels = _safe(result, "levels", [])
        if not isinstance(math_levels, list):
            math_levels = []
        math_levels_out = []
        for lv in math_levels:
            if not isinstance(lv, dict):
                continue
            math_levels_out.append({
                "name": _safe(lv, "name", "?"),
                "price": round(_safe(lv, "price", 0), 2),
                "pct": round((_safe(lv, "price", current_price) / current_price - 1) * 100, 2),
                "p_touch": round(_safe(lv, "p_touch", 0), 3),
                "p_touch_pct": round(_safe(lv, "p_touch", 0) * 100, 1),
                "t_first": _safe(lv, "t_first_human", "N/A"),
                "source": _safe(lv, "source", ""),
            })

        # ── Скоринг целей birza_digital по MC-путям ──
        # v8: если route_targets не переданы, собираем напрямую из sections_map
        route_targets = context.get("route_targets", [])
        if not route_targets:
            sections_map = context.get("sections_map") or {}
            if sections_map:
                try:
                    from pipeline.route_engine import collect_all_targets
                    route_targets = collect_all_targets(sections_map, current_price)
                except Exception:
                    route_targets = []
        scored_targets = []
        mc_paths = _safe(mc, "paths", None)
        # DEBUG: track why scored might be empty
        _dbg_rt = len(route_targets)
        _dbg_paths = mc_paths is not None
        _dbg_paths_shape = ""
        if mc_paths is not None:
            import numpy as np
            try:
                _dbg_paths_shape = str(np.array(mc_paths).shape)
            except Exception:
                _dbg_paths_shape = f"type={type(mc_paths).__name__}"
        if mc_paths is not None and len(route_targets) > 0:
            import numpy as np
            paths = np.array(mc_paths)
            for t in route_targets:
                tp = t.get("price", 0)
                if tp <= 0:
                    continue
                if tp < current_price:
                    touched = np.any(paths <= tp, axis=1)
                else:
                    touched = np.any(paths >= tp, axis=1)
                p_touch = float(np.mean(touched))
                scored_targets.append({
                    "price": round(tp, 2),
                    "pct": round((tp / current_price - 1) * 100, 2),
                    "p_touch": round(p_touch, 3),
                    "p_touch_pct": round(p_touch * 100, 1),
                    "source": t.get("source", ""),
                    "label": t.get("label", ""),
                })

        # ── v8: сводка по маршруту для AI (классификация, k-темпа, F, манипуляция) ──
        route_summary = {}
        try:
            sections_map = context.get("sections_map") or {}
            if sections_map:
                from pipeline.route_engine import build_route
                # Направление из S01
                s1d = sections_map.get(1, {}).get("data", {})
                direction = s1d.get("current_direction", s1d.get("direction", "боковик"))
                tf_h = context.get("tf_hours", 4)
                route_full = build_route(sections_map, current_price, direction, tf_hours=tf_h)
                route_summary = {
                    "key_targets_count": len(route_full.get("key_targets", [])),
                    "sec_targets_count": len(route_full.get("sec_targets", [])),
                    "other_targets_count": len(route_full.get("other_targets", [])),
                    "route_count": route_full.get("route_count", 0),
                    "total_dist_pct": route_full.get("total_dist_pct", 0),
                    "F_factor": route_full.get("F_factor", 1.0),
                    "horizon_label": route_full.get("horizon_label", ""),
                    "manipulation_signs": route_full.get("manipulation", {}).get("signs", 0),
                    "manipulation_threshold": route_full.get("manipulation", {}).get("threshold", 3),
                    "is_manipulation": route_full.get("manipulation", {}).get("is_manipulation", False),
                    "slam_reason": route_full.get("slam_reason", ""),
                }
        except Exception as _e:
            route_summary = {"error": str(_e)[:100]}

        return {
            "mc_scenarios": mc_scenarios,
            "mc_targets": {
                "T1": {"price": round(_safe(mc_t1, "price", 0), 2),
                       "pct": round((_safe(mc_t1, "price", current_price) / current_price - 1) * 100, 2)},
                "T2": {"price": round(_safe(mc_t2, "price", 0), 2),
                       "pct": round((_safe(mc_t2, "price", current_price) / current_price - 1) * 100, 2)},
                "T3": {"price": round(_safe(mc_t3, "price", 0), 2),
                       "pct": round((_safe(mc_t3, "price", current_price) / current_price - 1) * 100, 2)},
            },
            "hmm": hmm_data,
            "pde": pde_data,
            "chaos": chaos_data,
            "bayes": bayes_data,
            "model_probs": model_probs,
            "racing": racing_data,
            "math_levels": math_levels_out,
            "scored_targets": scored_targets,
            "route_v8": route_summary,
            # v2.6.3 — полные данные из мат.модели
            "manipulation": _safe(result, "manipulation", {}) if isinstance(_safe(result, "manipulation", {}), dict) else {},
            "route": _safe(result, "route", {}) if isinstance(_safe(result, "route", {}), dict) else {},
            "route_main": _safe(_safe(result, "route", {}), "main_route", []),
            "route_alt": _safe(_safe(result, "route", {}), "alt_route", []),
            "invalidation": _safe(result, "invalidation", {}) if isinstance(_safe(result, "invalidation", {}), dict) else {},
            "levels_with_p": math_levels_out,
            "horizon": _safe(result, "horizon", 0),
            "elapsed_sec": round(_safe(result, "elapsed", 0), 1),
        }
