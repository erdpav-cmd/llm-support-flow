"""
database.py — работа с SQLite (чистый sqlite3, без ORM).

Две таблицы:
  tasks     — ДЕЙСТВИЕ: задача менеджеру/оператору (то, что получает бизнес).
  audit_log — АУДИТ: вход + сырой ответ LLM + результат + статус + ошибка.

Правило: аудит пишется ВСЕГДА, даже при падении LLM.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL,
    category    TEXT NOT NULL,
    priority    TEXT NOT NULL,
    summary     TEXT NOT NULL,
    next_action TEXT NOT NULL,
    fields_json TEXT NOT NULL DEFAULT '{}',
    confidence  TEXT NOT NULL,
    escalate    INTEGER NOT NULL DEFAULT 0,
    assignee    TEXT NOT NULL DEFAULT 'manager',
    status      TEXT NOT NULL DEFAULT 'new'
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    ticket_id   TEXT,
    source      TEXT NOT NULL DEFAULT 'api',
    raw_input   TEXT NOT NULL,
    llm_raw     TEXT,
    result_json TEXT,
    status      TEXT NOT NULL,
    error       TEXT,
    latency_ms  INTEGER,
    model       TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_status  ON audit_log(status);
"""


def utc_now() -> str:
    """Единый формат времени: UTC, ISO-8601 (2026-11-17T09:49:00+00:00)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_ticket_id() -> str:
    """Читаемый номер тикета: TCK-20261117-A1B2C3."""
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"TCK-{day}-{uuid.uuid4().hex[:6].upper()}"


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """
    Соединение на одну операцию: открыли -> сделали -> закрыли.
    Так безопасно работать из разных потоков FastAPI.
    """
    conn = sqlite3.connect(settings.db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Создаёт папку data/ и таблицы, если их ещё нет."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_task(ticket_id: str, result: dict[str, Any]) -> int:
    """
    ДЕЙСТВИЕ: кладём задачу в работу.
    result — уже провалидированный словарь (category, summary, priority, ...).
    """
    escalate = bool(result.get("escalate", False))
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO tasks (
                ticket_id, created_at, category, priority, summary,
                next_action, fields_json, confidence, escalate, assignee, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                utc_now(),
                result.get("category", "other"),
                result.get("priority", "medium"),
                result.get("summary", ""),
                result.get("next_action", "передать оператору"),
                json.dumps(result.get("fields", {}), ensure_ascii=False),
                result.get("confidence", "low"),
                int(escalate),
                "operator" if escalate else "manager",
                "needs_human" if escalate else "new",
            ),
        )
        return int(cur.lastrowid)


def insert_audit(
    *,
    raw_input: str,
    status: str,
    ticket_id: str | None = None,
    source: str = "api",
    llm_raw: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    latency_ms: int | None = None,
) -> int:
    """
    АУДИТ: пишем всегда — ok / escalated / invalid_json / llm_error / rate_limited.
    Храним и исходник, и сырой ответ модели: без этого ошибку не разобрать.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO audit_log (
                created_at, ticket_id, source, raw_input, llm_raw,
                result_json, status, error, latency_ms, model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                ticket_id,
                source,
                raw_input,
                llm_raw,
                json.dumps(result, ensure_ascii=False) if result is not None else None,
                status,
                error,
                latency_ms,
                settings.model,
            ),
        )
        return int(cur.lastrowid)


def fetch_tasks(limit: int = 20) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_audit(limit: int = 20) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def stats() -> dict[str, int]:
    with get_conn() as conn:
        tasks_n = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        audit_n = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        esc_n = conn.execute("SELECT COUNT(*) FROM tasks WHERE escalate = 1").fetchone()[0]
    return {"tasks": tasks_n, "audit_log": audit_n, "escalated": esc_n}


if __name__ == "__main__":
    init_db()
    print(f"База готова: {settings.db_path}")
    with get_conn() as conn:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        print(f"Таблицы: {', '.join(tables)}")
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            print(f"  {table}: {len(cols)} колонок -> {', '.join(c[1] for c in cols)}")
    print(f"Записей: {stats()}")