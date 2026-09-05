"""Зависимости FastAPI (BE-004).

Сессия открывается на запрос и закрывается вместе с ним. Долгие операции сюда
не попадают: растровый fetch и инференс идут в воркере (инвариант 1), поэтому
транзакция обработчика короткая и не держит соединение во время анализа.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from apps.api.settings import get_settings
from apps.auth import SESSION_COOKIE, resolve
from apps.db.base import get_session_factory
from apps.db.models import User


def get_db() -> Iterator[Session]:
    # Гостевой запрос не должен падать только из-за отсутствующей конфигурации БД:
    # проверка cookie не выполняет запросов, если токена нет. Непривязанная Session
    # позволяет зависимости авторизации вернуть штатные 200/401; первая реальная
    # операция с БД по-прежнему завершится явной ошибкой конфигурации.
    session = get_session_factory()() if get_settings().database_url else Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    """Пользователь текущей сессии или None.

    Не бросает 401: есть роуты, которым важно различать гостя и сбой, — им нужен
    ответ, а не исключение.
    """
    return resolve(db, request.cookies.get(SESSION_COOKIE))


def require_user(user: User | None = Depends(current_user)) -> User:
    """Обязательная авторизация.

    Отдельная зависимость, а не проверка внутри каждого обработчика: забыть
    проверку в одном месте легко, а цена ошибки — чужие поля в чужих руках.
    """
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="нужен вход: откройте меню и войдите по коду из письма",
        )
    return user
