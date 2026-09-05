"""Подключение к PostgreSQL и базовый класс моделей (BE-002).

`create_all` здесь намеренно отсутствует: схему в любом контуре создаёт только
Alembic (инвариант 12). Автосоздание таблиц из моделей молча расходится с
миграциями, и расхождение всплывает на чужой машине, а не у автора.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from apps.api.settings import get_settings


class Base(DeclarativeBase):
    """Общий декларативный базовый класс всех таблиц сервиса."""


@lru_cache
def get_engine() -> Engine:
    """Один engine на процесс.

    `pool_pre_ping` обязателен: воркер живёт долго и держит соединения между
    задачами, а PostgreSQL в Compose может быть перезапущен — без пинга первая
    задача после рестарта падает на протухшем соединении.
    """
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("COSMO_DATABASE_URL не задан — подключение к БД не настроено")
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Транзакция с явным commit/rollback.

    FSM воркера переводит стадию отдельной короткой транзакцией, а не держит одну
    на весь анализ: иначе прогресс не виден снаружи до самого конца, и `GET /jobs/{id}`
    показывает QUEUED там, где работа уже идёт.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
