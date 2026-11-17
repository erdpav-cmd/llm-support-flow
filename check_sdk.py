"""
check_sdk.py — разовая диагностика подключения к LLM.
Запуск из корня проекта: python check_sdk.py
Стоимость: один мини-запрос (~0.03 руб).
"""
from app.config import settings

print("=" * 60)
print("0. НАСТРОЙКИ")
print("model:", settings.model)
print("base_url:", settings.base_url)
print("key:", settings.masked_key(), "| llm_ready:", settings.llm_ready)

print("=" * 60)
print("1. ВЕРСИИ БИБЛИОТЕК")
try:
    import openai
    print("openai:", openai.__version__, "| есть класс OpenAI:", hasattr(openai, "OpenAI"))
except Exception as exc:
    print("openai недоступен:", type(exc).__name__, exc)

try:
    import httpx
    print("httpx:", httpx.__version__, "| есть Client:", hasattr(httpx, "Client"))
except Exception as exc:
    print("httpx недоступен:", type(exc).__name__, exc)

print("=" * 60)
print("2. ИНТЕРФЕЙС SDK")
client = None
try:
    from openai import OpenAI
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    print("клиент создан:", type(client).__name__)
    print("есть chat.completions.create:", hasattr(client.chat.completions, "create"))
except Exception as exc:
    print("ОШИБКА создания клиента:", type(exc).__name__, str(exc)[:500])

print("=" * 60)
print("3. ЖИВОЙ МИНИ-ЗАПРОС")
if client is None:
    print("Клиент не создан - запрос пропускаем")
else:
    try:
        resp = client.chat.completions.create(
            model=settings.model,
            temperature=0,
            max_tokens=5,
            messages=[{"role": "user", "content": "Ответь одним словом: OK"}],
        )
        print("ОТВЕТ МОДЕЛИ:", resp.choices[0].message.content)
        print("USAGE:", getattr(resp, "usage", None))
        print("ВЫВОД: SDK работает -> llm.py пишем на openai SDK")
    except Exception as exc:
        print("ОШИБКА запроса:", type(exc).__name__)
        print(str(exc)[:800])
        print("ВЫВОД: нужен другой бэкенд -> дам вариант на httpx или stdlib")