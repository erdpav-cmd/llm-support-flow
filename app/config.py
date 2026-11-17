"""
config.py — единая точка настроек приложения.

Читает .env из корня проекта и отдаёт настройки остальным модулям.
Секреты наружу не печатаем: ключ показываем замаскированным.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта: .../llm-support-flow
BASE_DIR = Path(__file__).resolve().parent.parent

# Загружаем .env один раз при импорте модуля
load_dotenv(BASE_DIR / ".env")


def _as_int(name: str, default: int) -> int:
    """Безопасное чтение целого числа из .env (кривое значение -> дефолт)."""
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _as_float(name: str, default: float) -> float:
    """Безопасное чтение дробного числа из .env."""
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _resolve_db_path(raw: str) -> Path:
    """Относительный путь из .env превращаем в абсолютный от корня проекта."""
    path = Path(raw)
    return path if path.is_absolute() else (BASE_DIR / path)


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    db_path: Path
    api_key: str
    base_url: str
    model: str
    temperature: float
    timeout_seconds: float
    rate_limit_per_minute: int

    @property
    def llm_ready(self) -> bool:
        """Достаточно ли настроек, чтобы дёргать LLM."""
        key = self.api_key.strip()
        return bool(key) and not key.lower().startswith("вставь")

    def masked_key(self) -> str:
        """Маска для логов: показываем только края ключа."""
        key = self.api_key.strip()
        if not key:
            return "(пусто)"
        if len(key) <= 10:
            return key[:2] + "***"
        return f"{key[:5]}...{key[-4:]}"

    def describe(self) -> str:
        return "\n".join([
            f"BASE_DIR    : {self.base_dir}",
            f"DB_PATH     : {self.db_path}",
            f"LLM_BASE_URL: {self.base_url}",
            f"LLM_MODEL   : {self.model}",
            f"TEMPERATURE : {self.temperature}",
            f"TIMEOUT     : {self.timeout_seconds} c",
            f"RATE_LIMIT  : {self.rate_limit_per_minute} запросов/мин",
            f"API_KEY     : {self.masked_key()}",
            f"LLM_READY   : {'да' if self.llm_ready else 'НЕТ - проверь .env'}",
        ])


settings = Settings(
    base_dir=BASE_DIR,
    db_path=_resolve_db_path(os.getenv("DB_PATH", "data/app.db")),
    api_key=os.getenv("LLM_API_KEY", ""),
    base_url=os.getenv("LLM_BASE_URL", "https://api.proxyapi.ru/openai/v1"),
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=_as_float("LLM_TEMPERATURE", 0.2),
    timeout_seconds=_as_float("LLM_TIMEOUT_SECONDS", 30.0),
    rate_limit_per_minute=_as_int("RATE_LIMIT_PER_MINUTE", 5),
)


if __name__ == "__main__":
    print(settings.describe())