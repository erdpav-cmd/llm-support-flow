"""
main.py — HTTP-обёртка бизнес-процесса (FastAPI).

Запуск из корня проекта:
    uvicorn app.main:app --reload --reload-dir app --port 8000
Документация: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.database import fetch_audit, fetch_tasks, init_db, stats
from app.pipeline import STATUS_RATE_LIMITED, process, rate_limiter
from app.schemas import TriageResult


class IngestRequest(BaseModel):
    """Вход процесса. Пустую строку пропускаем в pipeline — она попадёт в аудит."""

    text: str = Field(
        max_length=2000,
        description="Текст обращения клиента (до 2000 символов)",
        examples=["Заказ 1042 не приходит неделю. Я Иван, +79001234567. Верните деньги."],
    )
    source: str = Field(
        default="api",
        max_length=32,
        description="Источник входа: api / file / script",
    )


class IngestResponse(BaseModel):
    """Выход процесса: результат + следы обработки (для аудита и отладки)."""

    ticket_id: str
    status: str
    escalate: bool
    task_id: int | None
    audit_id: int
    latency_ms: int | None
    error: str | None
    tokens: dict[str, int | None] | None
    result: TriageResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Перед первым запросом создаём таблицы, если их нет."""
    init_db()
    yield


app = FastAPI(
    title="LLM Support Flow",
    version="1.0.0",
    description=(
        "Автоматизация обработки обращений: вход → LLM (строгий JSON) → "
        "контроль качества и эскалация → задача менеджеру + журнал аудита."
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Процесс"],
    summary="Обработать обращение",
)
def ingest(payload: IngestRequest):
    """
    Полный цикл: LLM → валидация схемы → правила эскалации → задача + аудит.

    Коды ответа: 200 — обработано (в т.ч. с эскалацией), 429 — превышен лимит,
    422 — вход не прошёл валидацию (например, текст длиннее 2000 символов).
    """
    outcome = process(payload.text, source=payload.source)
    if outcome.status == STATUS_RATE_LIMITED:
        return JSONResponse(status_code=429, content=outcome.to_dict())
    return outcome.to_dict()


@app.get("/tasks", tags=["Результат"], summary="Задачи менеджеру и оператору")
def list_tasks(limit: int = Query(20, ge=1, le=100, description="Сколько последних задач")):
    items = fetch_tasks(limit)
    return {"count": len(items), "items": items}


@app.get("/audit", tags=["Аудит"], summary="Журнал обработки (вход + LLM + статус)")
def list_audit(limit: int = Query(20, ge=1, le=100, description="Сколько последних записей")):
    items = fetch_audit(limit)
    return {"count": len(items), "items": items}


@app.get("/health", tags=["Служебное"], summary="Состояние сервиса и счётчики")
def health():
    return {
        "status": "ok",
        "llm_ready": settings.llm_ready,
        "model": settings.model,
        "temperature": settings.temperature,
        "rate_limit_per_minute": rate_limiter.limit,
        "db_path": str(settings.db_path),
        "counts": stats(),
    }