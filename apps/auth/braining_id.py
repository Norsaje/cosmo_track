"""Клиент Braining ID — внешнего поставщика личности (IdP).

Своей аутентификации у сервиса нет намеренно: пароли, их сброс, защита от
перебора и рассылка писем — отдельная система со своими способами ошибиться,
и заводить её ради одного демо значит взять на себя риск без выигрыша.

Наружу отдаётся ровно две операции: запросить код на почту и обменять код на
личность. Токены Braining ID дальше этого модуля не уходят — сервис выдаёт
собственную сессию, чтобы срок жизни входа не зависел от чужих настроек.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from apps.api.settings import get_settings


class IdentityUnavailable(RuntimeError):
    """IdP недоступен или не настроен. Отличается от неверного кода."""


class IdentityRejected(RuntimeError):
    """IdP отказал: неверный или просроченный код, слишком частые попытки."""

    def __init__(self, message: str, *, retry_after: int | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)


@dataclass(frozen=True)
class Challenge:
    """Выданный запрос кода: по нему потом проверяется введённое значение."""

    challenge_id: str
    retry_after_seconds: int


@dataclass(frozen=True)
class Identity:
    """Подтверждённая личность. Ровно то, что нам нужно, и ничего больше."""

    user_id: str
    email: str | None
    is_new_user: bool


def _client() -> httpx.Client:
    settings = get_settings()
    if not (settings.braining_id_client_id and settings.braining_id_client_secret):
        raise IdentityUnavailable("вход не настроен: нет client_id/secret для Braining ID")
    return httpx.Client(
        base_url=settings.braining_id_url,
        auth=(settings.braining_id_client_id, settings.braining_id_client_secret),
        timeout=httpx.Timeout(10.0, connect=5.0),
        headers={"user-agent": "cosmo_track/0.1"},
    )


def request_code(email: str) -> Challenge:
    """Попросить IdP отправить код на почту."""
    try:
        with _client() as client:
            response = client.post("/api/v1/email-code/challenges", json={"email": email})
    except httpx.HTTPError as exc:
        raise IdentityUnavailable("сервис входа недоступен") from exc

    if response.status_code == 429:
        # Ограничение частоты — это не ошибка пользователя и не сбой: сообщаем,
        # сколько ждать, вместо общего «попробуйте позже».
        retry = response.headers.get("retry-after")
        raise IdentityRejected(
            "код уже отправлен, подождите перед повторной отправкой",
            retry_after=int(retry) if retry and retry.isdigit() else None,
        )
    if response.status_code >= 400:
        # Текст ошибки IdP наружу не пробрасывается: он может содержать детали
        # чужой системы, а пользователю нужно действие, а не диагностика.
        raise IdentityRejected("не удалось отправить код на эту почту")

    body = response.json()
    return Challenge(
        challenge_id=str(body["challenge_id"]),
        retry_after_seconds=int(body.get("retry_after_seconds", 60)),
    )


def verify_code(challenge_id: str, code: str) -> Identity:
    """Обменять код на личность.

    Возвращается только идентификатор и почта: access/refresh-токены IdP нам не
    нужны — сервис не ходит от имени пользователя в чужие API, а собственная
    сессия живёт по нашим правилам.
    """
    try:
        with _client() as client:
            response = client.post(
                f"/api/v1/email-code/challenges/{challenge_id}/verify", json={"code": code}
            )
    except httpx.HTTPError as exc:
        raise IdentityUnavailable("сервис входа недоступен") from exc

    if response.status_code >= 400:
        raise IdentityRejected("код неверный или устарел")

    body = response.json()
    profile = body.get("profile") or {}
    return Identity(
        user_id=str(body["user_id"]),
        email=profile.get("email") if isinstance(profile, dict) else None,
        is_new_user=bool(body.get("is_new_user", False)),
    )
