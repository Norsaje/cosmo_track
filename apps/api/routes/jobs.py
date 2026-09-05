"""Роут статуса джобы. Схема зафиксирована в C-07, durable FSM — BE-006."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from apps.api.schemas import ErrorResponse, JobStatus

#: Единый конверт ошибки C-07 в OpenAPI (см. комментарий в routes/polygons.py).
ERRORS = {
    422: {"model": ErrorResponse, "description": "Некорректное тело запроса"},
    501: {"model": ErrorResponse, "description": "Ещё не реализовано"},
}

router = APIRouter(tags=["jobs"], responses=ERRORS)


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail="Реализуется в BE-006 (durable job state machine).",
    )
