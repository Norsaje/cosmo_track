"""Роуты входа (BE-auth). Личность подтверждает Braining ID, сессия — наша.

Три операции и ничего лишнего: попросить код, обменять код на сессию, выйти.
Смены почты и личного кабинета здесь нет намеренно — профиль живёт в Braining ID,
и дублировать его редактирование значит завести второй источник правды.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session as DbSession

from apps.api.deps import current_user, get_db
from apps.api.schemas import CurrentUser, EmailCodeRequest, EmailCodeSent, SessionCreate
from apps.api.settings import get_settings
from apps.auth import (
    SESSION_COOKIE,
    IdentityRejected,
    IdentityUnavailable,
    issue,
    purge_expired,
    request_code,
    revoke,
    upsert_user,
    verify_code,
)
from apps.db.models import User

router = APIRouter(tags=["auth"])


def _login_configured() -> bool:
    settings = get_settings()
    return bool(settings.braining_id_client_id and settings.braining_id_client_secret)


@router.post("/auth/email-code", response_model=EmailCodeSent)
async def send_email_code(payload: EmailCodeRequest) -> EmailCodeSent:
    """Отправить код подтверждения на почту.

    Ответ одинаков и для существующей почты, и для незнакомой: разные ответы
    превратили бы этот роут в проверку «зарегистрирован ли такой человек».
    """
    try:
        challenge = request_code(str(payload.email))
    except IdentityUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None
    except IdentityRejected as exc:
        headers = {"retry-after": str(exc.retry_after)} if exc.retry_after else None
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc), headers=headers
        ) from None
    return EmailCodeSent(
        challenge_id=challenge.challenge_id, retry_after_seconds=challenge.retry_after_seconds
    )


@router.post("/auth/session", response_model=CurrentUser)
async def create_session(
    payload: SessionCreate, response: Response, db: DbSession = Depends(get_db)
) -> CurrentUser:
    """Обменять код на сессию и поставить cookie."""
    try:
        identity = verify_code(payload.challenge_id, payload.code)
    except IdentityUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None
    except IdentityRejected as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from None

    purge_expired(db)
    user = upsert_user(db, identity)
    token, expires = issue(db, user)
    settings = get_settings()

    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_days * 24 * 3600,
        # httponly: скрипту токен не нужен, а XSS без него не уносит сессию.
        httponly=True,
        # lax: cookie не уходит на сторонние POST-запросы, но переход по обычной
        # ссылке из письма вход не теряет.
        samesite="lax",
        secure=settings.session_cookie_secure,
        path="/",
    )
    return CurrentUser(authenticated=True, id=str(user.id), email=user.email)


@router.delete("/auth/session", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    request: Request, response: Response, db: DbSession = Depends(get_db)
) -> None:
    """Выйти. Завершается только текущая сессия — другие устройства не трогаем."""
    revoke(db, request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")


@router.get("/auth/me", response_model=CurrentUser)
async def read_me(user: User | None = Depends(current_user)) -> CurrentUser:
    """Кто сейчас вошёл. Для гостя это не ошибка, а обычный ответ."""
    if user is None:
        return CurrentUser(authenticated=False, login_available=_login_configured())
    return CurrentUser(
        authenticated=True, id=str(user.id), email=user.email, login_available=True
    )
