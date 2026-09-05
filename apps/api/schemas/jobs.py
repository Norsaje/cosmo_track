"""Состояния джобы анализа и её представление в API — часть контракта C-07 v0.1.

Потребители контракта — фронтенд `apps/web/` и offline smoke-тесты.
Любое изменение состава `JobState` требует одновременного обновления
smoke/demo-проверок, использующих этот словарь.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

CONTRACT_ID = "C-07"
CONTRACT_VERSION = "0.1"


class JobState(StrEnum):
    """Стадии FSM джобы анализа.

    Основной путь: QUEUED → FETCHING → PREPROCESSING → RECONSTRUCTING → ANALYZING → COMPLETED.
    Два отклонения: FETCHING → PARTIAL → PREPROCESSING (провайдер отдал часть данных)
    и FAILED из любой рабочей стадии.
    """

    QUEUED = "QUEUED"
    FETCHING = "FETCHING"
    PREPROCESSING = "PREPROCESSING"
    RECONSTRUCTING = "RECONSTRUCTING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


#: Разрешённые переходы FSM. Ключ — текущая стадия, значение — куда из неё можно уйти.
#: Таблица объявлена рядом со схемой намеренно: и воркер (BE-006), и contract-тест
#: обязаны читать один и тот же источник, иначе FSM разъедется с тем, что видит фронт.
ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.FETCHING, JobState.FAILED}),
    JobState.FETCHING: frozenset({JobState.PREPROCESSING, JobState.PARTIAL, JobState.FAILED}),
    # PARTIAL — не терминальная стадия: провайдер отдал неполный набор наблюдений,
    # но анализ продолжается на том, что есть, и результат честно помечается как частичный.
    JobState.PARTIAL: frozenset({JobState.PREPROCESSING, JobState.FAILED}),
    JobState.PREPROCESSING: frozenset({JobState.RECONSTRUCTING, JobState.FAILED}),
    JobState.RECONSTRUCTING: frozenset({JobState.ANALYZING, JobState.FAILED}),
    JobState.ANALYZING: frozenset({JobState.COMPLETED, JobState.FAILED}),
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset(),
}

#: Терминальные стадии: из них переходов нет, опрос `GET /jobs/{id}` можно прекращать.
TERMINAL_STATES: frozenset[JobState] = frozenset({JobState.COMPLETED, JobState.FAILED})


class JobStatus(BaseModel):
    """Ответ `GET /api/v1/jobs/{job_id}`."""

    job_id: str
    analysis_id: str | None = None
    state: JobState
    #: Прогресс 0..100. Считается по стадиям, а не по времени: фронт рисует шкалу стадий.
    progress: int = Field(ge=0, le=100)
    #: True, если джоба проходила через PARTIAL. Фронт опрашивает только этот роут,
    #: поэтому без флага «завершено» и «завершено на неполных данных» неразличимы.
    partial: bool = False
    stage_started_at: datetime | None = None
    retry_count: int = Field(default=0, ge=0)
    #: Машиночитаемый код ошибки (например MODEL_SCHEMA_MISMATCH, PROVIDER_UNAVAILABLE).
    error_code: str | None = None
    #: Безопасное сообщение для пользователя. Ни секретов, ни stack traces (инвариант 11).
    error_message_safe: str | None = None
    pipeline_version: str | None = None
    model_version: str | None = None
