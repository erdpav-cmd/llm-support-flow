"""
llm.py — единственный модуль, который ходит в сеть.

Задача: отдать модели системный промпт + обращение и вернуть СЫРОЙ текст ответа.
Валидацией занимается schemas.py, решением об эскалации — pipeline.py.
Любая проблема (401 / 429 / таймаут / пустой ответ) -> LLMError, а не падение.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from app.config import settings
from app.prompts import SYSTEM_PROMPT, build_user_prompt


class LLMError(RuntimeError):
    """Ошибка обращения к модели. kind пойдёт в audit_log.status/error."""

    def __init__(self, message: str, kind: str = "unknown", status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


@dataclass(frozen=True)
class LLMResponse:
    raw: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    model: str
    used_json_mode: bool


_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Ленивая инициализация: клиент создаётся один раз на процесс."""
    global _client
    if _client is None:
        if not settings.llm_ready:
            raise LLMError("не задан LLM_API_KEY в .env", kind="config")
        try:
            _client = OpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
                max_retries=0,  # повторами управляем сами, а не SDK
            )
        except TypeError:
            # мажорные версии SDK могут менять набор параметров конструктора
            _client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    return _client


def _classify(exc: Exception) -> LLMError:
    """Приводим любое исключение к нашему LLMError с понятным kind."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    text = str(exc)[:300]
    low = text.lower()

    if status in (401, 403):
        kind = "auth"
    elif status == 404:
        kind = "not_found"
    elif status == 429:
        kind = "rate_limit"
    elif isinstance(status, int) and status >= 500:
        kind = "upstream"
    elif "timeout" in name or "timeout" in low or "timed out" in low:
        kind = "timeout"
    elif "connection" in name or "connection" in low:
        kind = "network"
    elif status == 400:
        kind = "bad_request"
    else:
        kind = "unknown"
    return LLMError(f"{type(exc).__name__}: {text}", kind=kind, status_code=status)


def _call(client: OpenAI, model: str, messages: list[dict], json_mode: bool):
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": settings.temperature,  # 0.2 — извлечение, не творчество
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs)


def analyze(text: str, *, model: str | None = None) -> LLMResponse:
    """Отправляем обращение в LLM и возвращаем сырой JSON-текст + метрики."""
    model_name = model or settings.model
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]
    client = get_client()
    started = time.perf_counter()
    json_mode = True

    try:
        resp = _call(client, model_name, messages, json_mode=True)
    except Exception as exc:
        err = _classify(exc)
        # если прокси не поддерживает response_format — повторяем без него
        if err.kind == "bad_request" and "response_format" in str(exc).lower():
            try:
                resp = _call(client, model_name, messages, json_mode=False)
                json_mode = False
            except Exception as exc2:
                raise _classify(exc2) from exc2
        else:
            raise err from exc

    latency_ms = int((time.perf_counter() - started) * 1000)

    try:
        raw = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMError("неожиданный формат ответа модели", kind="empty") from exc

    if not raw.strip():
        raise LLMError("модель вернула пустой ответ", kind="empty")

    usage = getattr(resp, "usage", None)
    return LLMResponse(
        raw=raw,
        latency_ms=latency_ms,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        model=model_name,
        used_json_mode=json_mode,
    )


if __name__ == "__main__":
    from app.schemas import SchemaError, parse_llm_json

    sample = (
        "Здравствуйте! Заказ 1042 не приходит уже неделю. Я Иван, "
        "телефон +79001234567. Верните деньги или привезите товар."
    )

    print("=" * 60)
    print("1. ЖИВОЙ РАЗБОР ОБРАЩЕНИЯ")
    try:
        result = analyze(sample)
        print("СЫРОЙ ОТВЕТ LLM:")
        print(result.raw)
        print(f"\nlatency={result.latency_ms} ms | "
              f"tokens in/out={result.prompt_tokens}/{result.completion_tokens} | "
              f"json_mode={result.used_json_mode}")
        parsed = parse_llm_json(result.raw)
        print("\nПОСЛЕ ВАЛИДАЦИИ:")
        print(f"category={parsed.category} | priority={parsed.priority} | "
              f"confidence={parsed.confidence} | escalate={parsed.escalate}")
        print(f"summary: {parsed.summary}")
        print(f"contact: {parsed.fields.contact} | order_id: {parsed.fields.order_id}")
    except (LLMError, SchemaError) as exc:
        print("ОШИБКА:", type(exc).__name__, exc)

    print("=" * 60)
    print("2. ПРОВЕРКА FALLBACK (несуществующая модель, денег не тратит)")
    try:
        analyze(sample, model="no-such-model-xyz")
        print("FAIL: ошибка не возникла")
    except LLMError as exc:
        print(f"OK | kind={exc.kind} | status={exc.status_code}")
        print(f"Сообщение: {str(exc)[:200]}")