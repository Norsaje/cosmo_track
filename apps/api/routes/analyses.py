"""Роуты анализа. Схемы зафиксированы в C-07; оркестрация — BE-006/BE-008.

`POST /analyses` обязан отвечать 202 и не держать HTTP-запрос открытым во время
спутникового анализа (инварианты 1 и 7).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import PlainTextResponse

from apps.api.schemas import (
    AnalysisAccepted,
    AnalysisCreate,
    AnalysisOut,
    AnomaliesResponse,
    ErrorResponse,
    ProvenanceResponse,
    SeriesResponse,
)

#: Единый конверт ошибки C-07 в OpenAPI (см. комментарий в routes/polygons.py).
ERRORS = {
    422: {"model": ErrorResponse, "description": "Некорректное тело запроса"},
    501: {"model": ErrorResponse, "description": "Ещё не реализовано"},
}

router = APIRouter(tags=["analyses"], responses=ERRORS)

_NOT_IMPLEMENTED = "Реализуется в BE-006 (FSM) и BE-008 (offline fixture E2E)."


@router.post(
    "/analyses",
    response_model=AnalysisAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_analysis(payload: AnalysisCreate) -> AnalysisAccepted:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)


@router.get("/analyses/{analysis_id}", response_model=AnalysisOut)
async def get_analysis(analysis_id: str) -> AnalysisOut:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)


@router.get("/analyses/{analysis_id}/series", response_model=SeriesResponse)
async def get_series(analysis_id: str) -> SeriesResponse:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)


@router.get("/analyses/{analysis_id}/anomalies", response_model=AnomaliesResponse)
async def get_anomalies(analysis_id: str) -> AnomaliesResponse:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)


@router.get("/analyses/{analysis_id}/provenance", response_model=ProvenanceResponse)
async def get_provenance(analysis_id: str) -> ProvenanceResponse:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)


@router.get(
    "/analyses/{analysis_id}/export.csv",
    response_class=PlainTextResponse,
    responses={200: {"content": {"text/csv": {}}, "description": "Ряд в формате CSV"}},
)
async def export_csv(analysis_id: str) -> Response:
    """Экспорт ряда. Тип содержимого объявлен явно: по умолчанию FastAPI объявил бы
    application/json, и сгенерированный клиент попытался бы распарсить CSV как JSON."""
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)
