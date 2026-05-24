"""
Биржа-цифровой — Конфигурация.
"""

# ──────────────────────────────────────────────
# Типы анализа
# ──────────────────────────────────────────────

ANALYSIS_TYPES = {
    "intraday_express": {
        "name": "Интрадей-экспресс",
        "timeframes": ["15m"],
        "horizon": "внутри дня",
    },
    "short": {
        "name": "Краткосрок",
        "timeframes": ["15m", "1H", "4H"],
        "horizon": "1–2 сессии",
    },
    "short_express": {
        "name": "Краткосрок-экспресс",
        "timeframes": ["1H"],
        "horizon": "1–2 сессии",
    },
    "week": {
        "name": "Неделя",
        "timeframes": ["1H", "4H", "1D"],
        "horizon": "5–7 сессий",
    },
    "week_express": {
        "name": "Неделя-экспресс",
        "timeframes": ["4H"],
        "horizon": "5–7 сессий",
    },
    "medium": {
        "name": "Среднесрок",
        "timeframes": ["4H", "1D", "1W"],
        "horizon": "2–4 недели",
    },
    "medium_express": {
        "name": "Среднесрок-экспресс",
        "timeframes": ["1D"],
        "horizon": "2–4 недели",
    },
    "main": {
        "name": "Основной",
        "timeframes": ["15m", "1H", "4H", "1D", "1W"],
        "horizon": "максимально широкий",
    },
    "long": {
        "name": "Долгосрок",
        "timeframes": ["1W", "1M", "3M"],
        "horizon": "несколько месяцев",
    },
}

# ──────────────────────────────────────────────
# Маппинг ТФ-меток → часы
# ──────────────────────────────────────────────

TF_HOURS = {
    "1m": 1 / 60,
    "5m": 5 / 60,
    "15m": 0.25,
    "30m": 0.5,
    "1H": 1.0,
    "2H": 2.0,
    "4H": 4.0,
    "1D": 24.0,
    "1W": 168.0,
    "1M": 720.0,
    "3M": 2160.0,
}

# ──────────────────────────────────────────────
# ZigZag
# ──────────────────────────────────────────────

ZIGZAG_DEV_MAJOR = 5.0   # % для основных волн (базовый, масштабируется по ТФ)
ZIGZAG_DEV_MINOR = 1.0   # % для подволн   (базовый, масштабируется по ТФ)

# Множитель девиации ZigZag по ТФ.
# Логика: дневной тренд — крупные развороты (×2.5), 15m — ближайшие swing (×0.4).
# Базовая девиация (5%) × множитель = реальная.
# 1D: 5% × 2.5 = 12.5% — только крупные тренды
# 4H: 5% × 1.5 = 7.5%  — среднесрочные
# 1H: 5% × 1.0 = 5.0%  — базовый
# 15m:5% × 0.6 = 3.0%  — мелкие swing
ZIGZAG_TF_SCALE = {
    0.0167:  0.3,   # 1m
    0.0833:  0.4,   # 5m
    0.25:    0.6,   # 15m
    0.5:     0.8,   # 30m
    1.0:     1.0,   # 1H  (базовый)
    2.0:     1.2,   # 2H
    4.0:     1.5,   # 4H
    8.0:     1.8,   # 8H
    24.0:    2.5,   # 1D
    168.0:   3.5,   # 1W
    720.0:   4.5,   # 1M
}

# ──────────────────────────────────────────────
# Williams Fractal
# ──────────────────────────────────────────────

FRACTAL_PERIOD = 5

# ──────────────────────────────────────────────
# Linear Regression
# ──────────────────────────────────────────────

LINREG_PERIOD = 100

# ──────────────────────────────────────────────
# Volume Profile
# ──────────────────────────────────────────────

VP_WINDOW = 350
VP_BINS = 100
VP_VAH_VAL_THRESHOLD = 0.7  # 70% зона

# TPO (Time Price Opportunity)
TPO_PERIOD_MINUTES = 30   # длина одного TPO-периода (букв)
TPO_TICK_DIVISOR = 50     # tick_size = (high-low) / TPO_TICK_DIVISOR
TPO_VA_THRESHOLD = 0.7    # 70% зона по TPO

# ──────────────────────────────────────────────
# Уровни
# ──────────────────────────────────────────────

LEVELS_MERGE_ATR_MULT = 0.3  # объединение уровней ближе 0.3×ATR

# ──────────────────────────────────────────────
# Fibonacci
# ──────────────────────────────────────────────

FIBO_RETRACEMENT = [0.236, 0.382, 0.500, 0.618, 0.786]
FIBO_EXTENSION = [1.272, 1.618, 2.0, 2.618]

# ──────────────────────────────────────────────
# Bollinger / Keltner / Squeeze
# ──────────────────────────────────────────────

BB_PERIOD = 20
BB_STD = 2.0
KC_PERIOD = 20
KC_ATR_PERIOD = 10
KC_MULT = 1.5

# ──────────────────────────────────────────────
# Efficiency Ratio
# ──────────────────────────────────────────────

ER_PERIOD = 20

# ──────────────────────────────────────────────
# FVG / Gaps
# ──────────────────────────────────────────────

FVG_MIN_SIZE_ATR_MULT = 0.1  # минимальный размер FVG в ATR

# ──────────────────────────────────────────────
# Microstructure
# ──────────────────────────────────────────────

VOL_ANOMALY_ZSCORE = 2.0  # порог z-score для аномалий объёма

# ──────────────────────────────────────────────
# Correlations
# ──────────────────────────────────────────────

CORR_WINDOW = 50

# ──────────────────────────────────────────────
# AI Providers & Models
# ──────────────────────────────────────────────

AI_MAX_TOKENS = 16000

PROVIDERS = {
    "anthropic": {
        "name": "Anthropic",
        "api_type": "anthropic",
        "base_url": None,  # SDK default
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            {"id": "claude-opus-4-20250514",           "name": "Claude Opus 4.6"},
            {"id": "claude-opus-4-20250514:extended",  "name": "Claude Opus 4.6 Extended (1M)", "extended": True},
            {"id": "claude-sonnet-4-20250514",         "name": "Claude Sonnet 4"},
            {"id": "claude-sonnet-4-5-20241022",       "name": "Claude Sonnet 3.5 v2"},
            {"id": "claude-haiku-4-5-20251001",        "name": "Claude Haiku 3.5"},
        ],
        "default_model": "claude-opus-4-20250514:extended",
    },
    "openrouter": {
        "name": "OpenRouter",
        "api_type": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            # ── Бесплатные модели (актуально 2026-04) ──
            {"id": "qwen/qwen3-coder:free",                       "name": "🆓 Qwen3 Coder 480B (free)"},
            {"id": "nousresearch/hermes-3-llama-3.1-405b:free",   "name": "🆓 Hermes 3 405B (free)"},
            {"id": "nvidia/nemotron-3-super-120b-a12b:free",      "name": "🆓 Nemotron 3 Super 120B (free)"},
            {"id": "openai/gpt-oss-120b:free",                    "name": "🆓 GPT-OSS 120B (free)"},
            {"id": "qwen/qwen3-next-80b-a3b-instruct:free",      "name": "🆓 Qwen3 Next 80B (free)"},
            {"id": "meta-llama/llama-3.3-70b-instruct:free",      "name": "🆓 Llama 3.3 70B (free)"},
            {"id": "minimax/minimax-m2.5:free",                   "name": "🆓 MiniMax M2.5 (free)"},
            {"id": "google/gemma-4-31b-it:free",                  "name": "🆓 Gemma 4 31B (free)"},
            {"id": "google/gemma-3-27b-it:free",                  "name": "🆓 Gemma 3 27B (free)"},
            {"id": "nvidia/nemotron-3-nano-30b-a3b:free",         "name": "🆓 Nemotron 3 Nano 30B (free)"},
            {"id": "z-ai/glm-4.5-air:free",                      "name": "🆓 GLM 4.5 Air (free)"},
            {"id": "openai/gpt-oss-20b:free",                    "name": "🆓 GPT-OSS 20B (free)"},
            {"id": "arcee-ai/trinity-large-preview:free",         "name": "🆓 Arcee Trinity Large (free)"},
            # ── Платные модели ──
            {"id": "anthropic/claude-sonnet-4",      "name": "Claude Sonnet 4"},
            {"id": "anthropic/claude-haiku-4",       "name": "Claude Haiku 4"},
            {"id": "openai/gpt-4o",                  "name": "GPT-4o"},
            {"id": "openai/gpt-4o-mini",             "name": "GPT-4o Mini"},
            {"id": "google/gemini-2.5-pro",          "name": "Gemini 2.5 Pro"},
            {"id": "google/gemini-2.5-flash",        "name": "Gemini 2.5 Flash"},
            {"id": "deepseek/deepseek-r1",           "name": "DeepSeek R1"},
            {"id": "deepseek/deepseek-chat-v3-0324", "name": "DeepSeek V3"},
            {"id": "meta-llama/llama-4-maverick",    "name": "Llama 4 Maverick"},
            {"id": "qwen/qwen3-235b-a22b",          "name": "Qwen 3 235B"},
            {"id": "moonshotai/kimi-k2",             "name": "Kimi K2"},
        ],
        "default_model": "qwen/qwen3-coder:free",
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "api_type": "openai",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "models": [
            {"id": "moonshotai/kimi-k2-instruct",        "name": "Kimi K2 Instruct (2.5)"},
            {"id": "meta/llama-3.3-70b-instruct",       "name": "Llama 3.3 70B"},
            {"id": "meta/llama-3.1-405b-instruct",      "name": "Llama 3.1 405B"},
            {"id": "nvidia/llama-3.1-nemotron-ultra-253b-v1", "name": "Nemotron Ultra 253B"},
            {"id": "deepseek/deepseek-r1",               "name": "DeepSeek R1"},
            {"id": "qwen/qwen2.5-72b-instruct",         "name": "Qwen 2.5 72B"},
            {"id": "google/gemma-2-27b-it",              "name": "Gemma 2 27B"},
        ],
        "default_model": "moonshotai/kimi-k2-instruct",
    },
}

DEFAULT_PROVIDER = "anthropic"

# ──────────────────────────────────────────────
# Реестр разделов
# ──────────────────────────────────────────────

SECTIONS = [
    {"id": 1,  "emoji": "📈", "title": "ТРЕНДЫ",                          "type": "partial"},
    {"id": 2,  "emoji": "🌊", "title": "ВОЛНОВОЙ АНАЛИЗ",                 "type": "ai_only"},
    {"id": 3,  "emoji": "🔺", "title": "ГРАФИЧЕСКИЕ ПАТТЕРНЫ",            "type": "partial"},
    {"id": 4,  "emoji": "🕯", "title": "СВЕЧНЫЕ ПАТТЕРНЫ",                "type": "partial"},
    {"id": 5,  "emoji": "📉", "title": "ДИВЕРГЕНЦИИ",                     "type": "full"},
    {"id": 6,  "emoji": "📊", "title": "УРОВНИ",                          "type": "full"},
    {"id": 7,  "emoji": "📐", "title": "ФИБОНАЧЧИ",                       "type": "full"},
    {"id": 8,  "emoji": "🔊", "title": "VSA",                             "type": "partial"},
    {"id": 9,  "emoji": "📊", "title": "ОБЪЁМНЫЕ ЗОНЫ",                   "type": "full"},
    {"id": 10, "emoji": "⚖",  "title": "ВАЙКОФФ",                         "type": "ai_only"},
    {"id": 11, "emoji": "📍", "title": "ЗОНЫ СБОРА СТОПОВ",               "type": "full"},
    {"id": 12, "emoji": "🔄", "title": "ТЕМП РЫНКА",                      "type": "full"},
    {"id": 13, "emoji": "💧", "title": "ИМБАЛАНСЫ / ЛИКВИДНОСТЬ / ГЭПЫ / FVG", "type": "full"},
    {"id": 14, "emoji": "📊", "title": "BOLLINGER / KELTNER / SQUEEZE",   "type": "full"},
    {"id": 15, "emoji": "⚡", "title": "ЭФФЕКТИВНОСТЬ ДВИЖЕНИЙ",          "type": "full"},
    {"id": 16, "emoji": "💰", "title": "ПОТОКОВЫЕ ИНДИКАТОРЫ",            "type": "full"},
    {"id": 17, "emoji": "🔬", "title": "МИКРОСТРУКТУРА",                  "type": "full"},
    {"id": 18, "emoji": "🔗", "title": "КОРРЕЛЯЦИИ И КОНВЕРГЕНЦИЯ",       "type": "full"},
]
