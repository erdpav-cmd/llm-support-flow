"""
schemas.py — контракт данных (Pydantic v2).

Здесь живёт строгая схема JSON, в которую LLM обязана уложиться.
Всё, что не прошло проверку, превращается в SchemaError -> эскалация.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Ограниченные словари значений (enum) — модель не сможет выдумать категорию
Category = Literal["billing", "technical", "complaint", "delivery", "other"]
Priority = Literal["low", "medium", "high"]
Confidence = Literal["high", "medium", "low"]

# Поля, без которых бизнес не может закрыть обращение (нужен канал для ответа)
CRITICAL_FIELDS: tuple[str, ...] = ("contact",)


class SchemaError(ValueError):
    """Ответ LLM не соответствует схеме."""


class ExtractedFields(BaseModel):
    """Извлечённые из текста факты. Только то, что реально было во входе."""

    model_config = ConfigDict(extra="ignore")

    client_name: str | None = None
    contact: str | None = None
    order_id: str | None = None
    amount: float | None = None
    deadline: str | None = None

    @field_validator("client_name", "contact", "order_id", "deadline", mode="before")
    @classmethod
    def _clean_text(cls, value):
        """Пустые строки и мусор приводим к None."""
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        return value or None

    @field_validator("amount", mode="before")
    @classmethod
    def _clean_amount(cls, value):
        """Модель часто пишет '12 000 руб.' вместо числа — аккуратно вытаскиваем число."""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = re.sub(r"[^\d.,]", "", value).replace(",", ".")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None


class TriageResult(BaseModel):
    """Результат разбора обращения — то, что уходит в задачу менеджеру."""

    model_config = ConfigDict(extra="ignore")  # лишний безобидный ключ не ломает процесс

    category: Category
    summary: str = Field(min_length=3, max_length=300)
    priority: Priority
    next_action: str = Field(min_length=3, max_length=200)
    fields: ExtractedFields = Field(default_factory=ExtractedFields)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: Confidence
    escalate: bool = False

    @field_validator("summary", "next_action", mode="before")
    @classmethod
    def _strip_and_cut(cls, value):
        if not isinstance(value, str):
            value = str(value)
        return value.strip()[:300]

    @field_validator("fields", mode="before")
    @classmethod
    def _fields_not_null(cls, value):
        return {} if value is None else value

    @field_validator("missing_fields", mode="before")
    @classmethod
    def _missing_not_null(cls, value):
        return [] if value is None else value


def _extract_json(raw: str) -> str:
    """Достаём JSON-объект из ответа: срезаем ``` и болтовню вокруг."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise SchemaError("в ответе нет JSON-объекта")
    return text[start : end + 1]


def _format_errors(exc: ValidationError) -> str:
    """Человекочитаемое описание проблем — пойдёт в audit_log.error."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(корень)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)[:500]


def parse_llm_json(raw: str) -> TriageResult:
    """Главная функция контроля качества: строка от LLM -> валидный объект или SchemaError."""
    payload = _extract_json(raw)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"битый JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise SchemaError("ожидался JSON-объект, пришёл другой тип")
    try:
        return TriageResult.model_validate(data)
    except ValidationError as exc:
        raise SchemaError(_format_errors(exc)) from exc


def safe_result(reason: str = "недостаточно данных") -> TriageResult:
    """
    Безопасный результат для эскалации.
    Отдаём человеку, а не выдумываем ответ клиенту.
    """
    return TriageResult(
        category="other",
        summary=f"Требует ручной проверки: {reason}",
        priority="medium",
        next_action="передать оператору",
        fields=ExtractedFields(),
        missing_fields=["manual_review"],
        confidence="low",
        escalate=True,
    )


if __name__ == "__main__":
    good = (
        '{"category":"billing","summary":"Двойное списание по заказу 1042",'
        '"priority":"high","next_action":"Проверить платёж и оформить возврат",'
        '"fields":{"client_name":"Иван","contact":"+79001234567","order_id":"1042",'
        '"amount":"12 000 руб."},"missing_fields":[],"confidence":"high","escalate":false}'
    )
    cases = [
        ("валидный JSON", good, True),
        ("JSON в обёртке ```", f"```json\n{good}\n```", True),
        ("битый JSON", '{"category":"billing",', False),
        ("неверный enum", good.replace('"billing"', '"finance"'), False),
        ("нет обязательных ключей", '{"category":"billing"}', False),
        ("пустой ответ", "", False),
    ]
    print("САМОПРОВЕРКА СХЕМЫ (без обращения к LLM, 0 руб.)")
    for name, raw, should_pass in cases:
        try:
            res = parse_llm_json(raw)
            passed, detail = True, f"category={res.category}, amount={res.fields.amount}"
        except SchemaError as exc:
            passed, detail = False, f"SchemaError: {exc}"
        mark = "OK  " if passed == should_pass else "FAIL"
        print(f"{mark} | {name} | {detail}")

    print("\nБезопасный результат при эскалации:")
    print(json.dumps(safe_result("LLM вернула битый JSON").model_dump(),
                     ensure_ascii=False, indent=2))