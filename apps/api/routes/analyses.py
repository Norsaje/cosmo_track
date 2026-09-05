"""Роуты анализа (BE-006/BE-008). Схемы зафиксированы в C-07.

`POST /analyses` не считает ничего сам: он ставит задачу и отвечает 202. Инвариант 1
запрещает долгую работу внутри HTTP-запроса, а инвариант 7 требует идемпотентности
по `geometry_hash + date_range + pipeline_version`.
"""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas import (
    AnalysisAccepted,
    AnalysisCreate,
    AnalysisOut,
    AnomaliesResponse,
    AnomalyOut,
    ErrorResponse,
    ProvenanceEntry,
    ProvenanceResponse,
    SeriesPoint,
    SeriesResponse,
)
from apps.api.schemas.jobs import JobState
from apps.api.settings import get_settings
from apps.db.models import Analysis, AnalysisJob, AnomalyEvent, Polygon, Provenance, Reconstruction

#: Единый конверт ошибки C-07 в OpenAPI (см. комментарий в routes/polygons.py).
ERRORS = {
    404: {"model": ErrorResponse, "description": "Объект не найден"},
    422: {"model": ErrorResponse, "description": "Некорректное тело запроса"},
    501: {"model": ErrorResponse, "description": "Ещё не реализовано"},
}

router = APIRouter(tags=["analyses"], responses=ERRORS)


def _get_analysis(db: Session, analysis_id: str) -> Analysis:
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="анализ не найден")
    return analysis


def _analysis_out(analysis: Analysis) -> AnalysisOut:
    return AnalysisOut(
        id=str(analysis.id),
        polygon_id=str(analysis.polygon_id),
        date_from=analysis.date_from,
        date_to=analysis.date_to,
        state=JobState(analysis.state),
        pipeline_version=analysis.pipeline_version,
        model_version=analysis.model_version,
        created_at=analysis.created_at,
        started_at=analysis.started_at,
        finished_at=analysis.finished_at,
        partial=analysis.partial,
        cached=analysis.cached,
        data_retrieved_at=analysis.data_retrieved_at,
    )


@router.post(
    "/analyses",
    response_model=AnalysisAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_analysis(
    payload: AnalysisCreate, response: Response, db: Session = Depends(get_db)
) -> AnalysisAccepted:
    """Поставить анализ в очередь. Отвечает 202 и не считает ничего синхронно.

    Идемпотентность (инвариант 7): повторный запрос с тем же `geometry_hash`,
    диапазоном и версией пайплайна возвращает существующий анализ и его последнюю
    джобу, а не плодит дубликаты. Ключ построен на хэше геометрии, а не на
    `polygon_id`: два одинаковых контура — это один и тот же анализ.
    """
    polygon = db.get(Polygon, payload.polygon_id)
    if polygon is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="полигон не найден")

    # Ранняя проверка источника данных. Раньше полигон без привязки к ряду
    # проходил дальше, джоба создавалась и падала NO_DATA_SOURCE через несколько
    # секунд — пользователь узнавал о невозможности анализа уже постфактум, а в
    # БД оставалась мёртвая запись. Живых провайдеров ещё нет (BE-007/BE-009),
    # поэтому единственный источник — offline-ряд, и его отсутствие видно сразу.
    if not polygon.anon_polygon_id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "NO_DATA_SOURCE: у поля нет привязки к ряду наблюдений. "
                "Спутниковые провайдеры ещё не подключены (BE-007/BE-009), поэтому "
                "данные берутся из конкурсных рядов: выберите ряд в списке "
                "GET /api/v1/reference-polygons и укажите его в properties.anon_polygon_id."
            ),
        )

    settings = get_settings()
    existing = db.execute(
        select(Analysis).where(
            Analysis.geometry_hash == polygon.geometry_hash,
            Analysis.date_from == payload.date_from,
            Analysis.date_to == payload.date_to,
            Analysis.pipeline_version == settings.pipeline_version,
        )
    ).scalar_one_or_none()

    if existing is not None:
        job = db.execute(
            select(AnalysisJob)
            .where(AnalysisJob.analysis_id == existing.id)
            .order_by(AnalysisJob.created_at.desc())
        ).scalars().first()
        if job is not None:
            # 200 вместо 202: ничего нового не создано, и клиенту важно это различать.
            response.status_code = status.HTTP_200_OK
            return AnalysisAccepted(
                job_id=str(job.id), analysis_id=str(existing.id), state=JobState(job.state)
            )

    analysis = Analysis(
        polygon_id=polygon.id,
        geometry_hash=polygon.geometry_hash,
        date_from=payload.date_from,
        date_to=payload.date_to,
        pipeline_version=settings.pipeline_version,
        state=JobState.QUEUED.value,
    )
    db.add(analysis)
    db.flush()
    job = AnalysisJob(analysis_id=analysis.id, state=JobState.QUEUED.value, progress=0)
    db.add(job)
    db.flush()
    job_id, analysis_id = str(job.id), str(analysis.id)

    # Коммит до постановки в очередь: иначе воркер может выхватить задачу раньше,
    # чем строка станет видимой другой транзакции, и не найдёт джобу.
    db.commit()

    from apps.worker.tasks import run_analysis

    run_analysis.delay(job_id)
    return AnalysisAccepted(job_id=job_id, analysis_id=analysis_id, state=JobState.QUEUED)


@router.get("/analyses/{analysis_id}", response_model=AnalysisOut)
async def get_analysis(analysis_id: str, db: Session = Depends(get_db)) -> AnalysisOut:
    return _analysis_out(_get_analysis(db, analysis_id))


@router.get("/analyses/{analysis_id}/series", response_model=SeriesResponse)
async def get_series(analysis_id: str, db: Session = Depends(get_db)) -> SeriesResponse:
    """Итоговый ряд для графика.

    `primary_ndvi` и `ndvi_harmonized` отдаются раздельно (решение D-003): сырой
    конкурсный ряд и продуктовый гармонизированный — разные величины, и склеивать
    их в одну линию запрещено отдельным пунктом red-team checklist.
    """
    analysis = _get_analysis(db, analysis_id)
    rows = db.execute(
        select(Reconstruction)
        .where(Reconstruction.analysis_id == analysis_id)
        .order_by(Reconstruction.date)
    ).scalars()

    points = [
        SeriesPoint(
            date=row.date,
            primary_ndvi=row.primary_ndvi_raw,
            primary_ndvi_reconstructed=row.primary_ndvi_reconstructed,
            ndvi_harmonized=row.ndvi_harmonized,
            is_observed=row.is_observed,
            is_reconstructed=row.is_reconstructed,
            lower=row.lower,
            upper=row.upper,
            interval_coverage=row.interval_level,
            method=row.method,
            selected_source=row.selected_source,
            qa_flags={"quality_flags": row.quality_flags} if row.quality_flags else None,
        )
        for row in rows
    ]
    return SeriesResponse(
        analysis_id=analysis_id,
        points=points,
        cached=analysis.cached,
        data_retrieved_at=analysis.data_retrieved_at,
    )


@router.get("/analyses/{analysis_id}/anomalies", response_model=AnomaliesResponse)
async def get_anomalies(analysis_id: str, db: Session = Depends(get_db)) -> AnomaliesResponse:
    """События аномалий.

    Пустой `items` — валидный результат «событий не найдено», но он **не** равен
    «поле в норме»: при нехватке истории детектор возвращает предупреждение,
    и потребитель обязан различать эти два состояния.
    """
    _get_analysis(db, analysis_id)
    rows = db.execute(
        select(AnomalyEvent)
        .where(AnomalyEvent.analysis_id == analysis_id)
        .order_by(AnomalyEvent.start_date)
    ).scalars()

    warnings: list[str] = []
    provenance = db.execute(
        select(Provenance).where(Provenance.analysis_id == analysis_id)
    ).scalars().first()
    if provenance is not None and provenance.processing_params:
        warnings = list(provenance.processing_params.get("warnings", []))

    items = [
        AnomalyOut(
            id=str(row.id),
            start_date=row.start_date,
            end_date=row.end_date,
            severity=row.severity,
            score=row.score,
            confidence=row.confidence,
            min_robust_z=row.min_robust_z,
            negative_area=row.negative_area,
            observed_points=row.observed_points,
            reconstructed_points=row.reconstructed_points,
            reason_codes=list(row.reason_codes or []),
            explanation_ru=row.explanation_ru,
            algorithm_version=row.algorithm_version,
            duration_days=row.duration_days,
            schema_version=row.schema_version,
        )
        for row in rows
    ]
    return AnomaliesResponse(
        analysis_id=analysis_id,
        items=items,
        warnings=warnings,
        # Семантика владельца C-09: confidence — эвристическая поддержка, а не
        # калиброванная вероятность. UI не имеет права рисовать её процентом.
        confidence_semantics="heuristic_support_not_calibrated_probability",
    )


@router.get("/analyses/{analysis_id}/provenance", response_model=ProvenanceResponse)
async def get_provenance(analysis_id: str, db: Session = Depends(get_db)) -> ProvenanceResponse:
    _get_analysis(db, analysis_id)
    rows = db.execute(
        select(Provenance).where(Provenance.analysis_id == analysis_id)
    ).scalars()
    entries = [
        ProvenanceEntry(
            provider=row.provider,
            collection_id=row.collection_id,
            collection_version=row.collection_version,
            item_ids=list(row.item_ids or []),
            queried_at=row.queried_at,
            crs=row.crs,
            resolution_m=row.resolution_m,
            qa_definition=row.qa_definition,
            processing_params=row.processing_params or {},
            license_url=row.license_url,
            cached=row.cached,
        )
        for row in rows
    ]
    return ProvenanceResponse(analysis_id=analysis_id, entries=entries)


@router.get(
    "/analyses/{analysis_id}/export.csv",
    response_class=PlainTextResponse,
    # Тип объявляется явно: PlainTextResponse сам по себе описывает ответ как
    # text/plain, и сгенерированный клиент решил бы, что это не таблица.
    responses={200: {"content": {"text/csv": {"schema": {"type": "string"}}}}},
)
async def export_csv(analysis_id: str, db: Session = Depends(get_db)) -> PlainTextResponse:
    """Выгрузка ряда.

    Колонки raw и harmonized разнесены явно, а `is_observed`/`is_reconstructed`
    едут вместе со значениями: без них выгрузка не отличает наблюдение от
    восстановления, и результат легко принять за измерение.
    """
    _get_analysis(db, analysis_id)
    rows = db.execute(
        select(Reconstruction)
        .where(Reconstruction.analysis_id == analysis_id)
        .order_by(Reconstruction.date)
    ).scalars()

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "date",
            "primary_ndvi_raw",
            "primary_ndvi_reconstructed",
            "ndvi_harmonized",
            "is_observed",
            "is_reconstructed",
            "lower",
            "upper",
            "method",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.date.isoformat(),
                "" if row.primary_ndvi_raw is None else row.primary_ndvi_raw,
                "" if row.primary_ndvi_reconstructed is None else row.primary_ndvi_reconstructed,
                "" if row.ndvi_harmonized is None else row.ndvi_harmonized,
                int(row.is_observed),
                int(row.is_reconstructed),
                "" if row.lower is None else row.lower,
                "" if row.upper is None else row.upper,
                row.method or "",
            ]
        )
    return PlainTextResponse(
        buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"content-disposition": f'attachment; filename="analysis_{analysis_id}.csv"'},
    )
