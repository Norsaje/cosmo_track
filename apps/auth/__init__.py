"""Аутентификация: внешняя личность (Braining ID) и собственные сессии.

Разделение намеренное: `braining_id` знает, как подтвердить, что человек владеет
почтой, а `sessions` — сколько живёт вход и как он проверяется у нас. Сменить
поставщика личности можно, не трогая второе.
"""

from apps.auth.braining_id import (
    Challenge,
    Identity,
    IdentityRejected,
    IdentityUnavailable,
    request_code,
    verify_code,
)
from apps.auth.sessions import SESSION_COOKIE, issue, purge_expired, resolve, revoke, upsert_user

__all__ = [
    "SESSION_COOKIE",
    "Challenge",
    "Identity",
    "IdentityRejected",
    "IdentityUnavailable",
    "issue",
    "purge_expired",
    "request_code",
    "resolve",
    "revoke",
    "upsert_user",
    "verify_code",
]
