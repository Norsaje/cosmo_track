"""Схемы health-эндпоинтов — часть контракта C-07 v0.1.

`/health/live` отвечает всегда, пока процесс жив. `/health/ready` проверяет БД,
Redis и model bundle. `/health/providers` — диагностический: он показывает
недоступность внешних источников, но **не** блокирует cached demo (§8.4 ТЗ).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ComponentStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
    #: Компонент намеренно не сконфигурирован — например, нет credentials провайдера.
    #: Это не отказ: демо обязано продолжать работать на кэше.
    NOT_CONFIGURED = "not_configured"


class LiveResponse(BaseModel):
    status: ComponentStatus = ComponentStatus.OK


class ReadyResponse(BaseModel):
    status: ComponentStatus
    database: ComponentStatus
    redis: ComponentStatus
    model_bundle: ComponentStatus


class ProviderHealth(BaseModel):
    name: str
    status: ComponentStatus
    detail: str | None = None


class ProvidersResponse(BaseModel):
    #: Агрегированный статус. DEGRADED допустим и не должен ронять демо.
    status: ComponentStatus
    providers: list[ProviderHealth]
