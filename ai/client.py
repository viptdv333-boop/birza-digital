"""
Биржа-цифровой — Мульти-провайдер AI клиент.

Поддержка: Anthropic, OpenRouter, NVIDIA NIM.
"""
import os
import time

from config import PROVIDERS, DEFAULT_PROVIDER, AI_MAX_TOKENS

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")


def load_env_keys() -> dict[str, str]:
    """Загрузить все API ключи из .env файла."""
    keys = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    keys[k.strip()] = v.strip().strip("\"'")
    return keys


def save_env_key(key_name: str, value: str):
    """Сохранить или обновить ключ в .env файле."""
    lines = []
    found = False
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key_name}="):
            new_lines.append(f"{key_name}={value}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key_name}={value}\n")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def get_api_key(provider_id: str) -> str | None:
    """Получить API ключ для провайдера (env → .env)."""
    prov = PROVIDERS.get(provider_id)
    if not prov:
        return None
    env_var = prov["env_key"]
    # Сначала из env
    key = os.environ.get(env_var)
    if key:
        return key
    # Затем из .env
    keys = load_env_keys()
    return keys.get(env_var)


def get_provider_status() -> dict:
    """Статус каждого провайдера: есть ли ключ."""
    result = {}
    env_keys = load_env_keys()
    for pid, prov in PROVIDERS.items():
        env_var = prov["env_key"]
        has_key = bool(os.environ.get(env_var) or env_keys.get(env_var))
        result[pid] = {
            "name": prov["name"],
            "has_key": has_key,
            "models": prov["models"],
            "default_model": prov["default_model"],
        }
    return result


# ──────────────────────────────────────────────
# Anthropic Client
# ──────────────────────────────────────────────

class AnthropicClient:
    """Клиент Anthropic API (нативный SDK)."""

    def __init__(self, api_key: str, model: str, max_tokens: int = AI_MAX_TOKENS,
                 extended: bool = False):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.extended = extended

    def analyze(self, system_prompt: str, user_message: str,
                max_retries: int = 3) -> str:
        import anthropic
        for attempt in range(max_retries):
            try:
                if self.extended:
                    # Extended thinking — streaming обязателен для долгих запросов
                    result_text = ""
                    with self.client.messages.stream(
                        model=self.model,
                        max_tokens=16000,
                        temperature=1,
                        thinking={
                            "type": "enabled",
                            "budget_tokens": 10000,
                        },
                        system=[{
                            "type": "text",
                            "text": system_prompt,
                        }],
                        messages=[{"role": "user", "content": user_message}],
                    ) as stream:
                        response = stream.get_final_message()
                    for block in response.content:
                        if block.type == "text":
                            result_text += block.text
                    return result_text
                else:
                    # Streaming обязателен для Claude Opus 4.6
                    # (non-streaming вызовы могут упасть по таймауту)
                    result_text = ""
                    with self.client.messages.stream(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=[{
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=[{"role": "user", "content": user_message}],
                    ) as stream:
                        response = stream.get_final_message()
                    for block in response.content:
                        if block.type == "text":
                            result_text += block.text
                    return result_text
            except anthropic.RateLimitError:
                time.sleep(2 ** attempt * 5)
            except anthropic.APIError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
        raise RuntimeError("Anthropic API: превышено количество попыток")


# ──────────────────────────────────────────────
# OpenAI-Compatible Client (OpenRouter, NVIDIA)
# ──────────────────────────────────────────────

class OpenAICompatibleClient:
    """Клиент для OpenAI-совместимых API (OpenRouter, NVIDIA NIM)."""

    def __init__(self, api_key: str, model: str, base_url: str,
                 max_tokens: int = AI_MAX_TOKENS,
                 extra_headers: dict | None = None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}

    def analyze(self, system_prompt: str, user_message: str,
                max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    extra_headers=self.extra_headers if self.extra_headers else None,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt * 2)
        raise RuntimeError("API: превышено количество попыток")


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def create_client(provider_id: str = None, model: str = None,
                  api_key: str = None):
    """Создать AI клиент по провайдеру и модели.

    Args:
        provider_id: "anthropic" / "openrouter" / "nvidia"
        model: model ID (или default для провайдера)
        api_key: ключ (или автозагрузка из env/.env)

    Returns:
        AnthropicClient | OpenAICompatibleClient
    """
    provider_id = provider_id or DEFAULT_PROVIDER
    prov = PROVIDERS.get(provider_id)
    if not prov:
        raise ValueError(f"Неизвестный провайдер: {provider_id}")

    if not api_key:
        api_key = get_api_key(provider_id)
    if not api_key:
        raise RuntimeError(
            f"API ключ для {prov['name']} не найден. "
            f"Установите {prov['env_key']} в настройках или .env"
        )

    model = model or prov["default_model"]

    # Extended thinking: model id с суффиксом :extended
    extended = False
    if model.endswith(":extended"):
        extended = True
        model = model.replace(":extended", "")

    if prov["api_type"] == "anthropic":
        return AnthropicClient(api_key=api_key, model=model, extended=extended)
    else:
        extra_headers = {}
        if provider_id == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://birza-digital.local",
                "X-Title": "Birza Digital TA",
            }
        return OpenAICompatibleClient(
            api_key=api_key,
            model=model,
            base_url=prov["base_url"],
            extra_headers=extra_headers,
        )
