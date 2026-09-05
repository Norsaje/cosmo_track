"""Фикстуры backend-тестов.

Два инварианта этого файла.

1. **Изоляция от окружения.** `Settings` читает переменные с префиксом `COSMO_`,
   а `get_settings` закэширован. Без очистки окружения и сброса кэша переменная
   вроде `COSMO_API_PREFIX`, выставленная в CI-раннере, меняет пути роутов, и
   прогон перестаёт быть детерминированным — а §8.13 ТЗ требует обратного.
2. **Работоспособность в лёгком окружении.** Разработчик 4 поднимает окружение
   командой `uv sync --extra dev` и запускает `uv run pytest -q`. FastAPI и
   pydantic-settings живут в extras `web`/`core`, поэтому здесь нет ни одного
   импорта на уровне модуля: иначе сбор падал бы с ModuleNotFoundError и exit 2,
   и он решил бы, что сломан backend. Модули, которым нужен `web`, объявляют
   `importorskip` сами; core-тест обязан выполняться и без него.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith("COSMO_"):
            monkeypatch.delenv(name, raising=False)
    try:
        from apps.api.settings import get_settings
    except ModuleNotFoundError:
        # Лёгкое окружение без extra `core`: сбрасывать нечего, тесты, которым
        # нужны настройки, всё равно пропускаются своим importorskip.
        yield
        return
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    pytest.importorskip("fastapi", reason="нужен extra `web`")
    from fastapi.testclient import TestClient

    from apps.api.main import app

    # raise_server_exceptions=False: иначе TestClient пробрасывает исключение вместо
    # ответа, и обработчик 500 — единственное место, где формируется безопасное
    # сообщение, — остаётся непроверяемым.
    return TestClient(app, raise_server_exceptions=False)
