"""
pipeline.py — бизнес-процесс "вход -> LLM -> контроль качества -> действие".

Порядок:
  1. отсечь заведомо пустой вход (не платим за мусор)
  2. проверить лимит запросов
  3. вызвать LLM
  4. провалидировать JSON по схеме
  5. применить правила эскалации
  6. ДЕЙСТВИЕ: задача в tasks
  7. АУДИТ: запись в audit_log (всегда, даже при ошибке)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.database import fetch_audit, insert_audit, insert_task, new_ticket_id, stats
from app.llm import LLMError, analyze
from app.schemas import (
    CRITICAL_FIELDS,
    SchemaError,
    TriageResult,
    parse_llm_json,
    safe_result,
)

STATUS_OK = "ok"
STATUS_ESCALATED = "escalated"
STATUS_INVALID_JSON = "invalid_json"
STATUS_LLM_ERROR = "llm_error"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_INVALID_INPUT = "invalid_input"
# Маркеры попыток управлять моделью через текст обращения.
# Дёшево, детерминированно, не тратит токены.
INJECTION_MARKERS: tuple[str, ...] = (
    "игнорируй", "игнорировать", "забудь инструкции", "отмени правила",
    "ты теперь", "ты свободный", "новые инструкции", "верни json",
    "ignore previous", "ignore all", "disregard", "system prompt",
)


def detect_injection(text: str) -> list[str]:
    """Ищем признаки prompt injection в самом тексте обращения."""
    low = text.lower()
    return [marker for marker in INJECTION_MARKERS if marker in low]

class RateLimiter:
    """Скользящее окно на deque: не больше N запросов за 60 секунд."""

    def __init__(self, limit_per_minute: int):
        self.limit = max(1, limit_per_minute)
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._hits and now - self._hits[0] >= 60.0:
                self._hits.popleft()
            if len(self._hits) >= self.limit:
                return False
            self._hits.append(now)
            return True

    def retry_after(self) -> int:
        with self._lock:
            if not self._hits:
                return 0
            return max(1, int(60 - (time.monotonic() - self._hits[0])) + 1)


rate_limiter = RateLimiter(settings.rate_limit_per_minute)


@dataclass(frozen=True)
class PipelineResult:
    ticket_id: str
    status: str
    result: TriageResult
    task_id: int | None
    audit_id: int
    latency_ms: int | None
    error: str | None
    tokens: dict[str, int | None] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "status": self.status,
            "escalate": self.result.escalate,
            "task_id": self.task_id,
            "audit_id": self.audit_id,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "tokens": self.tokens,
            "result": self.result.model_dump(),
        }


def apply_escalation_rules(result: TriageResult) -> tuple[TriageResult, list[str]]:
    """
    Бизнес-правила поверх извлечённых фактов.

    ВАЖНО: решение об эскалации принимает КОД, а не модель.
    Флаг escalate от LLM — рекомендация: на одном и том же тексте модель
    может вернуть и true, и false. Детерминированные правила стабильны всегда.
    """
    reasons: list[str] = []

    if result.confidence == "low":
        reasons.append("низкая уверенность модели")

    missing_critical = [
        name for name in CRITICAL_FIELDS if not getattr(result.fields, name, None)
    ]
    if missing_critical:
        reasons.append("нет данных для ответа: " + ", ".join(missing_critical))

    if result.category == "other" and result.confidence != "high":
        reasons.append("категория обращения не определена")

    if not reasons:
        return result.model_copy(update={"escalate": False}), []

    # model_copy не запускает валидацию — длину режем сами
    next_action = ("передать оператору: " + "; ".join(reasons))[:200]
    updated = result.model_copy(
        update={
            "escalate": True,
            "next_action": next_action,
            "missing_fields": sorted(set(result.missing_fields) | set(missing_critical)),
        }
    )
    return updated, reasons


def _finalize(
    ticket_id: str,
    raw_input: str,
    result: TriageResult,
    *,
    status: str,
    source: str,
    error: str | None = None,
    llm_raw: str | None = None,
    latency_ms: int | None = None,
    tokens: dict[str, int | None] | None = None,
    create_task: bool = True,
) -> PipelineResult:
    """Действие + аудит. Единая точка выхода из процесса."""
    payload = result.model_dump()
    task_id = insert_task(ticket_id, payload) if create_task else None
    audit_id = insert_audit(
        raw_input=raw_input,
        status=status,
        ticket_id=ticket_id,
        source=source,
        llm_raw=llm_raw,
        result=payload,
        error=error,
        latency_ms=latency_ms,
    )
    return PipelineResult(
        ticket_id=ticket_id,
        status=status,
        result=result,
        task_id=task_id,
        audit_id=audit_id,
        latency_ms=latency_ms,
        error=error,
        tokens=tokens,
    )


def process(
    text: str,
    *,
    source: str = "api",
    enforce_rate_limit: bool = True,
) -> PipelineResult:
    """Главная функция процесса. Никогда не бросает исключение наружу."""
    ticket_id = new_ticket_id()
    raw_input = text or ""
    cleaned = raw_input.strip()

    # 1. пустой вход отсекаем до платного вызова
    if len(cleaned) < 3:
        return _finalize(
            ticket_id, raw_input,
            safe_result("пустой или слишком короткий текст"),
            status=STATUS_INVALID_INPUT, source=source,
            error="вход короче 3 символов",
        )
    # 1b. prompt injection блокируем до вызова LLM: и безопаснее, и бесплатно
    markers = detect_injection(cleaned)
    if markers:
        blocked = safe_result("обнаружены признаки prompt injection")
        blocked = blocked.model_copy(update={
            "priority": "high",
            "next_action": "передать оператору: подозрение на prompt injection",
        })
        return _finalize(
            ticket_id, raw_input, blocked,
            status=STATUS_ESCALATED, source=source,
            error="prompt injection: " + ", ".join(markers),
        )
        
    # 2. лимит запросов
    if enforce_rate_limit and not rate_limiter.allow():
        limited = safe_result(f"превышен лимит {rate_limiter.limit} запросов/мин")
        limited = limited.model_copy(
            update={"next_action": f"повторить через {rate_limiter.retry_after()} c"}
        )
        return _finalize(
            ticket_id, raw_input, limited,
            status=STATUS_RATE_LIMITED, source=source,
            error="rate limit exceeded", create_task=False,
        )

    # 3. вызов LLM
    try:
        llm_resp = analyze(cleaned)
    except LLMError as exc:
        return _finalize(
            ticket_id, raw_input,
            safe_result(f"LLM недоступна ({exc.kind})"),
            status=STATUS_LLM_ERROR, source=source,
            error=f"{exc.kind}: {str(exc)[:400]}",
        )

    tokens = {"prompt": llm_resp.prompt_tokens, "completion": llm_resp.completion_tokens}

    # 4. валидация схемы
    try:
        parsed = parse_llm_json(llm_resp.raw)
    except SchemaError as exc:
        return _finalize(
            ticket_id, raw_input,
            safe_result("ответ модели не прошёл проверку схемы"),
            status=STATUS_INVALID_JSON, source=source,
            error=str(exc)[:500], llm_raw=llm_resp.raw,
            latency_ms=llm_resp.latency_ms, tokens=tokens,
        )

    # 5. бизнес-правила эскалации
    final, reasons = apply_escalation_rules(parsed)
    status = STATUS_ESCALATED if final.escalate else STATUS_OK

    # 6-7. действие + аудит
    return _finalize(
        ticket_id, raw_input, final,
        status=status, source=source,
        error="; ".join(reasons) or None,
        llm_raw=llm_resp.raw, latency_ms=llm_resp.latency_ms, tokens=tokens,
    )


if __name__ == "__main__":
    cases = [
        ("нормальное обращение",
         "Здравствуйте! Заказ 1042 не приходит неделю. Я Иван, +79001234567. Верните деньги."),
        ("нет контакта -> эскалация",
         "У вас на сайте ничего не работает, кнопки не нажимаются, исправьте срочно."),
        ("бессмыслица -> низкая уверенность",
         "ывафы вафыва 123"),
        ("пустой вход -> без вызова LLM",
         "  "),
    ]

    print("ПРОГОН ПРОЦЕССА (3 платных запроса, ~0.12 руб)")
    for name, text in cases:
        res = process(text, source="selftest")
        print("-" * 70)
        print(f"[{name}]")
        print(f"  ticket={res.ticket_id} | status={res.status} | escalate={res.result.escalate}")
        print(f"  category={res.result.category} | priority={res.result.priority} | "
              f"confidence={res.result.confidence}")
        print(f"  next_action: {res.result.next_action}")
        print(f"  task_id={res.task_id} | audit_id={res.audit_id} | latency={res.latency_ms} ms")
        if res.error:
            print(f"  причина: {res.error}")

    print("=" * 70)
    print("СЧЁТЧИКИ БАЗЫ:", stats())
    print("\nПОСЛЕДНИЕ 3 ЗАПИСИ АУДИТА:")
    for row in fetch_audit(3):
        print(f"  #{row['id']} {row['created_at']} | {row['status']} | "
              f"ticket={row['ticket_id']} | latency={row['latency_ms']}")