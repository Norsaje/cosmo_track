"""Схемы входа (часть контракта C-07).

Пароля в этих схемах нет и не появится: личность подтверждает Braining ID
кодом на почту, а сервис хранит только результат.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class EmailCodeRequest(BaseModel):
    """Тело `POST /api/v1/auth/email-code`."""

    model_config = ConfigDict(extra="forbid")

    #: EmailStr, а не str: опечатка вроде «ivan@mail» должна отсекаться до
    #: обращения к IdP, иначе на каждую такую попытку тратится его лимит.
    email: EmailStr


class EmailCodeSent(BaseModel):
    """Ответ на запрос кода."""

    challenge_id: str
    #: Через сколько секунд можно просить код заново. Показывается человеку,
    #: чтобы «отправить ещё раз» не выглядело сломанной кнопкой.
    retry_after_seconds: int


class SessionCreate(BaseModel):
    """Тело `POST /api/v1/auth/session` — обмен кода на сессию."""

    model_config = ConfigDict(extra="forbid")

    challenge_id: str = Field(min_length=1, max_length=128)
    #: Код фиксированной длины приходит от IdP; ограничение здесь только от
    #: явного мусора, точную проверку делает他 сам.
    code: str = Field(min_length=1, max_length=32)


class CurrentUser(BaseModel):
    """Ответ `GET /api/v1/auth/me`.

    Возвращается всегда, в том числе для неавторизованного гостя: интерфейсу
    нужно различать «не вошёл» и «сервис недоступен», а 401 на этом роуте
    смешивал бы эти состояния.
    """

    authenticated: bool
    id: str | None = None
    email: str | None = None
    #: Настроен ли вход вообще. Без учётных данных Braining ID кнопка входа
    #: должна честно сказать, что войти пока некуда, а не молча не работать.
    login_available: bool = True
