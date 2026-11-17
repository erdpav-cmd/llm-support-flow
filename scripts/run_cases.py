"""
scripts/run_cases.py — прогон 10 тестовых обращений через бизнес-процесс.

Запуск из корня проекта:
    python scripts/run_cases.py

Что делает:
  * прогоняет tests/cases.json через pipeline (лимит отключён — локальный тест)
  * сверяет результат с ожиданиями
  * считает токены, задержку и себестоимость
  * сохраняет outputs/results.json и outputs/results.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings          # noqa: E402
from app.database import init_db, stats  # noqa: E402
from app.pipeline import process         # noqa: E402

CASES_PATH = ROOT / "tests" / "cases.json"
OUT_DIR = ROOT / "outputs"

# Ориентир по наблюдениям: ~0.04 руб за запрос.
# Точную цифру сверим по дашборду ProxyApi: баланс до и после прогона.
COST_PER_REQUEST_RUB = 0.04


def check(case: dict, outcome) -> tuple[bool, str]:
    expect = case.get("expect") or {}
    problems = []
    if "status" in expect and outcome.status != expect["status"]:
        problems.append(f"ждали status={expect['status']}, получили {outcome.status}")
    if "escalate" in expect and outcome.result.escalate != expect["escalate"]:
        problems.append(f"ждали escalate={expect['escalate']}, получили {outcome.result.escalate}")
    if "category" in expect and outcome.result.category != expect["category"]:
        problems.append(f"ждали category={expect['category']}, получили {outcome.result.category}")
    return (not problems), "; ".join(problems)


def main() -> None:
    init_db()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Модель: {settings.model} | temperature: {settings.temperature}")
    print(f"Кейсов: {len(cases)} | лимит запросов отключён (локальный прогон)")
    print("=" * 104)
    print(f"{'N':<3}{'Кейс':<34}{'Статус':<15}{'Esc':<5}{'Катег.':<11}"
          f"{'Приор.':<8}{'Conf':<8}{'мс':<7}{'Токены':<10}{'Итог'}")
    print("-" * 104)

    rows: list[dict] = []
    details: list[dict] = []
    statuses: Counter = Counter()
    total_prompt = total_completion = paid = passed = 0
    latencies: list[int] = []
    started = time.perf_counter()

    for case in cases:
        outcome = process(case["text"], source="testrun", enforce_rate_limit=False)
        ok, problem = check(case, outcome)
        passed += int(ok)
        statuses[outcome.status] += 1

        tokens = outcome.tokens or {}
        pt, ct = tokens.get("prompt"), tokens.get("completion")
        if pt is not None:
            paid += 1
            total_prompt += pt
            total_completion += ct or 0
        if outcome.latency_ms is not None:
            latencies.append(outcome.latency_ms)

        token_str = f"{pt}/{ct}" if pt is not None else "-"
        print(f"{case['id']:<3}{case['name'][:33]:<34}{outcome.status:<15}"
              f"{'да' if outcome.result.escalate else 'нет':<5}"
              f"{outcome.result.category:<11}{outcome.result.priority:<8}"
              f"{outcome.result.confidence:<8}{str(outcome.latency_ms or '-'):<7}"
              f"{token_str:<10}{'PASS' if ok else 'CHECK'}")
        if problem:
            print(f"    -> {problem}")

        rows.append({
            "id": case["id"],
            "name": case["name"],
            "status": outcome.status,
            "escalate": outcome.result.escalate,
            "category": outcome.result.category,
            "priority": outcome.result.priority,
            "confidence": outcome.result.confidence,
            "summary": outcome.result.summary,
            "next_action": outcome.result.next_action,
            "contact": outcome.result.fields.contact or "",
            "latency_ms": outcome.latency_ms or "",
            "prompt_tokens": pt or "",
            "completion_tokens": ct or "",
            "error": outcome.error or "",
            "verdict": "PASS" if ok else "CHECK",
        })
        details.append({"case": case, "outcome": outcome.to_dict()})

    elapsed = time.perf_counter() - started
    avg_latency = sum(latencies) // len(latencies) if latencies else 0
    auto = statuses.get("ok", 0)

    print("=" * 104)
    print(f"Совпало с ожиданиями: {passed}/{len(cases)}")
    print(f"Статусы: {dict(statuses)}")
    print(f"Закрыто автоматически: {auto}/{len(cases)} ({auto / len(cases) * 100:.0f}%)")
    print(f"Платных запросов: {paid} | токены: {total_prompt} вход / {total_completion} выход")
    print(f"Средняя задержка LLM: {avg_latency} мс | время прогона: {elapsed:.1f} c")
    if paid:
        run_cost = paid * COST_PER_REQUEST_RUB
        per_request = run_cost / paid
        print(f"Себестоимость прогона: ~{run_cost:.2f} руб | 100 обработок: ~{per_request * 100:.2f} руб")
        print("(точную цифру сверь по балансу ProxyApi до и после прогона)")
    print(f"Счётчики базы: {stats()}")

   

    json_path = OUT_DIR / "results.json"
    json_path.write_text(
        json.dumps({
            "model": settings.model,
            "temperature": settings.temperature,
            "summary": {
                "cases": len(cases),
                "passed": passed,
                "statuses": dict(statuses),
                "auto_closed_percent": round(auto / len(cases) * 100),
                "paid_requests": paid,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "avg_latency_ms": avg_latency,
                "elapsed_sec": round(elapsed, 1),
            },
            "results": details,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nСохранено: {json_path}")

    csv_path = OUT_DIR / "results.csv"
    try:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Сохранено: {csv_path}")
    except PermissionError:
        print(f"\nВНИМАНИЕ: {csv_path} открыт в Excel — CSV не записан.")
        print("Закрой файл и повтори запуск. results.json сохранён.")


if __name__ == "__main__":
    main()