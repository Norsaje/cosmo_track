"""Зависимости FastAPI (BE-004).

Сессия открывается на запрос и закрывается вместе с ним. Долгие операции сюда
не попадают: растровый fetch и инференс идут в воркере (инвариант 1), поэтому
транзакция обработчика короткая и не держит соединение во время анализа.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from apps.db.base import get_session_factory


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
