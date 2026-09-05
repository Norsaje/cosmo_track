"""Единый формат ошибки API — часть контракта C-07 v0.1.

Наружу отдаём только машиночитаемый код и безопасный текст. Stack trace, SQL,
URL провайдера и любые секреты в ответ не попадают никогда (инвариант 11).
"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    #: Идентификатор запроса для сопоставления с логами. Сам лог наружу не отдаётся.
    request_id: str | None = None
