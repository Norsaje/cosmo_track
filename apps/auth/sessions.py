"""Сессии входа: выдача, проверка, завершение.

Сессия своя, а не заимствованная у IdP. Причина простая: срок жизни входа —
наше продуктовое решение (здесь 30 дней), и привязывать его к чужим настройкам
токенов значит согласиться на любые их изменения без предупреждения.

В cookie уходит случайный токен, в базе хранится только его SHA256. Дамп базы
тогда не даёт войти ни в один аккаунт: восстановить токен по хэшу нельзя.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from apps.api.settings import get_settings
from apps.auth.braining_id import Identity
from apps.db.models import Session, User

#: Имя cookie. Префикс `__Host-` не используется намеренно: он требует Secure и
#: обязателен к домену без пути, а стенд поднимается и по http на localhost.
SESSION_COOKIE = "cosmo_session"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def upsert_user(db: DbSession, identity: Identity) -> User:
    """Найти или завести пользователя по идентификатору из IdP.

    Ключ — `braining_user_id`, а не почта: почту человек может сменить у себя
    в профиле, и привязка по ней однажды подсунула бы его поля другому.
    """
    user = db.execute(
        select(User).where(User.braining_user_id == identity.user_id)
    ).scalar_one_or_none()
    if user is None:
        user = User(braining_user_id=identity.user_id, email=identity.email)
        db.add(user)
        db.flush()
    elif identity.email and user.email != identity.email:
        # Почта могла измениться на стороне IdP — обновляем, она у нас только
        # для показа.
        user.email = identity.email
    user.last_seen_at = datetime.now(UTC)
    return user


def issue(db: DbSession, user: User) -> tuple[str, datetime]:
    """Создать сессию. Возвращает открытый токен — он существует только здесь."""
    settings = get_settings()
    token = secrets.token_urlsafe(48)
    expires = datetime.now(UTC) + timedelta(days=settings.session_ttl_days)
    db.add(Session(token_hash=_hash(token), user_id=user.id, expires_at=expires))
    db.flush()
    return token, expires


def resolve(db: DbSession, token: str | None) -> User | None:
    """Пользователь по токену сессии, если она жива."""
    if not token:
        return None
    session = db.execute(
        select(Session).where(Session.token_hash == _hash(token))
    ).scalar_one_or_none()
    if session is None:
        return None
    # Сравнение в UTC: наивная дата из базы и aware-датой now() иначе дают
    # TypeError, а «истёкшая» сессия при этом выглядела бы как ошибка сервера.
    expires = session.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    if expires <= datetime.now(UTC):
        # Просроченную запись сразу убираем: иначе таблица растёт мусором,
        # а по индексу токена он всё равно никогда не найдётся полезным.
        db.delete(session)
        return None
    return db.get(User, session.user_id)


def revoke(db: DbSession, token: str | None) -> None:
    """Завершить одну сессию. Остальные устройства пользователя не трогаем."""
    if not token:
        return
    session = db.execute(
        select(Session).where(Session.token_hash == _hash(token))
    ).scalar_one_or_none()
    if session is not None:
        db.delete(session)


def purge_expired(db: DbSession) -> int:
    """Убрать просроченные сессии. Вызывается при входе — отдельного планировщика
    ради этого заводить незачем."""
    stale = db.execute(
        select(Session).where(Session.expires_at <= datetime.now(UTC))
    ).scalars().all()
    for session in stale:
        db.delete(session)
    return len(stale)
