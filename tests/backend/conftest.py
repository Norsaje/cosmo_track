"""Фикстуры backend-тестов.

Два инварианта этого файла.

1. **Изоляция от окружения.** `Settings` читает переменные с префиксом `COSMO_`,
   а `get_settings` закэширован. Без очистки окружения и сброса кэша переменная
   вроде `COSMO_API_PREFIX`, выставленная в CI-раннере, меняет пути роутов, и
   прогон перестаёт быть детерминированным — а §8.13 ТЗ требует обратного.
2. **Работоспособность в лёгком окружении.** При запуске
   `uv sync --extra dev` и `uv run pytest -q` FastAPI и
   pydantic-settings живут в extras `web`/`core`, поэтому здесь нет ни одного
   импорта на уровне модуля: иначе сбор падал бы с ModuleNotFoundError и exit 2,
   и создавал бы ложное впечатление поломки backend. Модули, которым нужен `web`, объявляют
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


@pytest.fixture
def authed_client(client):
    """Клиент с действующей сессией.

    Сессия создаётся прямо в базе, минуя Braining ID: тесты не должны зависеть
    от внешнего сервиса и от доставки писем. Проверяется здесь наша половина —
    что вход открывает доступ и что данные разделены по владельцам.
    """
    pytest.importorskip("sqlalchemy", reason="нужен extra `web`")
    from datetime import UTC, datetime, timedelta

    from apps.auth.sessions import SESSION_COOKIE, _hash
    from apps.db.base import session_scope
    from apps.db.models import Session, User

    token = "test-session-token"
    try:
        with session_scope() as db:
            user = User(braining_user_id="test-user", email="test@example.com")
            db.add(user)
            db.flush()
            db.add(Session(
                token_hash=_hash(token), user_id=user.id,
                expires_at=datetime.now(UTC) + timedelta(days=1),
            ))
            user_id = user.id
    except Exception:
        pytest.skip("БД недоступна: тест требует поднятого PostgreSQL")

    client.cookies.set(SESSION_COOKIE, token)
    client.test_user_id = user_id
    yield client

    # За собой убираем: тестовый пользователь и его поля не должны копиться
    # в базе стенда, на которой потом показывают демо.
    with session_scope() as db:
        stale = db.get(User, user_id)
        if stale is not None:
            db.delete(stale)
