"""Роут статуса джобы (BE-006). Схема зафиксирована в C-07.

Это единственный роут, который фронт опрашивает в цикле, поэтому он обязан быть
дешёвым: одна строка по первичному ключу, без джойнов к ряду и событиям.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas import ErrorResponse, JobStatus
from apps.api.schemas.jobs import JobState
from apps.db.models import Analysis, AnalysisJob

#: Единый конверт ошибки C-07 в OpenAPI (см. комментарий в routes/polygons.py).
ERRORS = {
    404: {"model": ErrorResponse, "description": "Объект не найден"},
    422: {"model": ErrorResponse, "description": "Некорректное тело запроса"},
}

router = APIRouter(tags=["jobs"], responses=ERRORS)


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, db: Session = Depends(get_db)) -> JobStatus:
    """Статус выполнения анализа.

    `partial` отдаётся отдельно от `state`: фронт опрашивает только этот роут, и без
    флага «завершено» и «завершено на неполных данных» были бы неразличимы.
    """
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="джоба не найдена")

    analysis = db.get(Analysis, job.analysis_id)
    return JobStatus(
        job_id=str(job.id),
        analysis_id=str(job.analysis_id),
        state=JobState(job.state),
        progress=job.progress,
        partial=job.partial,
        stage_started_at=job.stage_started_at,
        retry_count=job.retry_count,
        error_code=job.error_code,
        error_message_safe=job.error_message_safe,
        pipeline_version=analysis.pipeline_version if analysis else None,
        model_version=analysis.model_version if analysis else None,
    )
