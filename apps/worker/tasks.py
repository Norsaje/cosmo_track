"""Задачи воркера: durable FSM анализа (BE-006, решение D-004).

Почему не BackgroundTasks: анализ переживает перезапуск процесса. Стадия хранится
в БД, а не в памяти, поэтому упавший контейнер не превращает выполняющуюся джобу
в вечный QUEUED, и `GET /jobs/{id}` показывает реальное положение дел.

Каждая смена стадии — отдельная короткая транзакция. Одна длинная на весь анализ
означала бы, что прогресс не виден снаружи до самого конца.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from functools import lru_cache
from typing import Any

import pandas as pd

from apps.api.schemas.jobs import ALLOWED_TRANSITIONS, JobState
from apps.api.settings import get_settings
from apps.db.base import session_scope
from apps.db.models import (
    Analysis,
    AnalysisJob,
    AnomalyEvent,
    Observation,
    Provenance,
    Reconstruction,
)
from apps.worker.celery_app import celery_app
from veg_recovery.providers.fixture import FixtureUnavailable, load_series
from veg_recovery.service import LoadedReconstructor, ModelUnavailable, build_reconstructor
from veg_recovery.service.orchestrator import AnalysisOutcome, run_reconstruction

#: Прогресс по стадиям. Шкала стадийная, а не временная: фронт рисует этапы,
#: и «47 %» в середине FETCHING было бы выдумкой.
STAGE_PROGRESS: dict[JobState, int] = {
    JobState.QUEUED: 0,
    JobState.FETCHING: 15,
    JobState.PARTIAL: 30,
    JobState.PREPROCESSING: 45,
    JobState.RECONSTRUCTING: 70,
    JobState.ANALYZING: 90,
    JobState.COMPLETED: 100,
    JobState.FAILED: 100,
}

#: Сколько лет истории нужно baseline-детектору, чтобы вообще что-то сказать.
#: Меньше — не «аномалий нет», а «сравнивать не с чем» (семантика C-09).
MIN_REFERENCE_YEARS = 3


@lru_cache(maxsize=1)
def get_reconstructor() -> LoadedReconstructor:
    """Модель загружается один раз на процесс воркера.

    Не оптимизация, а необходимость: при загрузке считаются SHA256 обоих CSV
    поставки, разбираются 149 145 строк набора и поднимается pickle смеси на
    17 МБ. Делать это на каждой задаче значит тратить секунды впустую.
    """
    settings = get_settings()
    return build_reconstructor(
        settings.model_package_path,
        run_name=settings.model_run_name,
        allow_stub=settings.allow_model_stub,
        trusted=settings.model_bundle_trusted,
        environment=settings.environment,
    )


def _advance(job_id: str, state: JobState, *, partial: bool = False) -> None:
    """Перевести джобу в следующую стадию с проверкой таблицы переходов.

    Переходы валидируются по тому же словарю, который читает contract-тест C-07:
    один источник истины, иначе FSM воркера тихо разъедется с тем, что видит фронт.
    """
    with session_scope() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            raise RuntimeError("джоба исчезла из БД")
        current = JobState(job.state)
        if state not in ALLOWED_TRANSITIONS[current]:
            raise RuntimeError(f"недопустимый переход {current} → {state}")
        job.state = state.value
        job.progress = STAGE_PROGRESS[state]
        job.stage_started_at = datetime.now(UTC)
        if partial:
            job.partial = True
        analysis = session.get(Analysis, job.analysis_id)
        if analysis is not None:
            analysis.state = state.value
            if partial:
                analysis.partial = True
            if state is JobState.FETCHING and analysis.started_at is None:
                analysis.started_at = datetime.now(UTC)
            if state in (JobState.COMPLETED, JobState.FAILED):
                analysis.finished_at = datetime.now(UTC)


def _fail(job_id: str, code: str, message: str) -> None:
    """Пометить джобу упавшей. В сообщение попадает только безопасный текст."""
    with session_scope() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            return
        job.state = JobState.FAILED.value
        job.progress = 100
        job.error_code = code
        job.error_message_safe = message
        job.stage_started_at = datetime.now(UTC)
        analysis = session.get(Analysis, job.analysis_id)
        if analysis is not None:
            analysis.state = JobState.FAILED.value
            analysis.finished_at = datetime.now(UTC)


def _store_observations(session, polygon_id: str, series) -> None:
    """Сохранить исходные наблюдения.

    `ndvi_raw` пишется как есть, а отсутствие значения остаётся NULL: подменять
    его нулём запрещено (инвариант 3) — ноль в NDVI это реальное значение.
    """
    session.query(Observation).filter(Observation.polygon_id == polygon_id).delete()
    frame = series.frame
    for row in frame.to_dict("records"):
        value = row.get("primary_ndvi")
        finite = value is not None and pd.notna(value)
        session.add(
            Observation(
                polygon_id=polygon_id,
                date=pd.Timestamp(row["date"]).date(),
                source="fixture",
                processing_version="v1",
                ndvi_raw=float(value) if finite else None,
                ndvi_cleaned=float(value) if finite else None,
                ndvi_invalid_flag=not finite,
                temp_c=_optional_float(row.get("era5_temp_c")),
                temp_available=pd.notna(row.get("era5_temp_c")),
                precip_mm=_optional_float(row.get("era5_precip_mm")),
                precip_available=pd.notna(row.get("era5_precip_mm")),
            )
        )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None and pd.notna(value) else None


def _store_outcome(session, analysis_id: str, outcome: AnalysisOutcome) -> None:
    session.query(Reconstruction).filter(Reconstruction.analysis_id == analysis_id).delete()
    for point in outcome.points:
        diagnostics = point.diagnostics
        session.add(
            Reconstruction(
                analysis_id=analysis_id,
                date=point.date.date(),
                primary_ndvi_raw=point.primary_ndvi_raw,
                primary_ndvi_reconstructed=point.primary_ndvi_reconstructed,
                ndvi_harmonized=point.ndvi_harmonized,
                is_observed=point.is_observed,
                is_reconstructed=point.is_reconstructed,
                lower=point.lower,
                upper=point.upper,
                method=point.method,
                p_s2=diagnostics.get("p_s2"),
                p_landsat=diagnostics.get("p_landsat"),
                p_modis=diagnostics.get("p_modis"),
                p_unknown=diagnostics.get("p_unknown"),
                left_distance_days=diagnostics.get("left_distance_days"),
                right_distance_days=diagnostics.get("right_distance_days"),
                model_disagreement=diagnostics.get("model_disagreement"),
                fallback_reason=diagnostics.get("fallback_reason"),
                context_quality=diagnostics.get("context_quality"),
                source_confidence=diagnostics.get("source_confidence"),
                quality_flags=diagnostics.get("quality_flags"),
                interval_status=diagnostics.get("interval_status"),
                interval_level=diagnostics.get("interval_level"),
                harmonization_status=diagnostics.get("harmonization_status"),
                climatology_mean=point.climatology_mean,
                climatology_std=point.climatology_std,
            )
        )


def _detect_anomalies(session, analysis: Analysis, series, outcome: AnalysisOutcome) -> list[str]:
    """Аномалии baseline-детектором ML — разрешённый fallback до прихода C-09 от DL.

    Возвращает предупреждения. Отсутствие событий при нехватке истории — **не**
    подтверждённая норма: это отдельное состояние, и оно проговаривается явно,
    как требует семантика C-09.
    """
    from veg_recovery.anomalies.baseline import detect_anomalies

    session.query(AnomalyEvent).filter(AnomalyEvent.analysis_id == analysis.id).delete()

    harmonized = {point.date: point.ndvi_harmonized for point in outcome.points}
    query = series.frame.copy()
    query["date"] = pd.to_datetime(query["date"]).dt.normalize()
    query["ndvi_harmonized"] = [
        harmonized.get(moment) if harmonized.get(moment) is not None else raw
        for moment, raw in zip(query["date"], query["primary_ndvi"], strict=True)
    ]

    # Референс — только предыдущие годы того же полигона: сравнение с текущим годом
    # было бы утечкой и превратило бы аномалию в самоподтверждение.
    reference = load_series(
        analysis.polygon.anon_polygon_id,
        date(1900, 1, 1),
        date(analysis.date_from.year - 1, 12, 31),
        data_dir=get_settings().model_data_dir,
    )
    reference_frame = reference.frame.copy()
    reference_frame["date"] = pd.to_datetime(reference_frame["date"]).dt.normalize()
    reference_frame["ndvi_harmonized"] = reference_frame["primary_ndvi"]
    reference_frame = reference_frame[reference_frame["ndvi_harmonized"].notna()]

    years = reference_frame["date"].dt.year.nunique()
    if years < MIN_REFERENCE_YEARS:
        return [
            "INSUFFICIENT_REFERENCE_YEARS: истории меньше "
            f"{MIN_REFERENCE_YEARS} лет ({years}); отсутствие событий не означает норму"
        ]

    result = detect_anomalies(query, reference_frame=reference_frame)
    events = result.events
    warnings = [
        "ANOMALY_SOURCE_IS_BASELINE: события получены baseline-детектором ML, "
        "а не контрактом C-09 от DL; категории severity (normal/biomass_suppression/"
        "critical) им не присваиваются"
    ]
    if events.empty:
        return warnings

    # Схема baseline-детектора ML НЕ совпадает с C-09: у него `event_id`, `magnitude`,
    # числовой `severity` (magnitude × duration × confidence) и `interpretation`.
    # Поэтому поля переносятся явно, а не «как получится»: наивный маппинг положил бы
    # число 0.89 в колонку severity, где C-09 ожидает категорию.
    # Чего у baseline нет — того мы не выдумываем: min_robust_z и negative_area
    # остаются нулями, а не подставленными «правдоподобными» величинами.
    observed_dates = {point.date.date() for point in outcome.points if point.is_observed}
    reconstructed_dates = {point.date.date() for point in outcome.points if point.is_reconstructed}

    for row in events.to_dict("records"):
        start = pd.Timestamp(row["start_date"]).date()
        end = pd.Timestamp(row["end_date"]).date()
        span = pd.date_range(start, end, freq="D").date
        session.add(
            AnomalyEvent(
                analysis_id=analysis.id,
                start_date=start,
                end_date=end,
                # Явная строка вместо категории C-09: baseline её не определяет,
                # и подставить сюда «critical» значило бы придумать оценку тяжести.
                severity="baseline_unranked",
                score=float(row.get("severity", 0.0)),
                confidence=float(row.get("confidence", 0.0)),
                # magnitude — средняя величина отрицательного остатка. Это не робастный
                # z-счёт C-09, поэтому со знаком минус кладётся именно как «насколько
                # ниже ожидания», а не выдаётся за min_robust_z другого алгоритма.
                min_robust_z=-abs(float(row.get("magnitude", 0.0))),
                negative_area=0.0,
                observed_points=sum(1 for day in span if day in observed_dates),
                reconstructed_points=sum(1 for day in span if day in reconstructed_dates),
                reason_codes=list(row.get("reason_codes", []) or []),
                explanation_ru=str(row.get("interpretation", "")),
                algorithm_version="ml-baseline-anomalies",
                schema_version="baseline",
                duration_days=int(row.get("duration_days", (end - start).days + 1)),
            )
        )
    return warnings


@celery_app.task(name="cosmo_track.ping")
def ping() -> str:
    """Смоук-задача: подтверждает, что брокер доступен и воркер забирает задачи."""
    return "pong"


@celery_app.task(name="cosmo_track.run_analysis", bind=True, max_retries=2)
def run_analysis(self, job_id: str) -> dict[str, Any]:  # noqa: ANN001 - сигнатура Celery
    """Выполнить анализ по джобе, проводя её через стадии FSM."""
    with session_scope() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            return {"job_id": job_id, "state": "MISSING"}
        analysis_id = job.analysis_id
        if JobState(job.state) in (JobState.COMPLETED, JobState.FAILED):
            # Идемпотентность по стадии: повторная доставка задачи не переигрывает
            # уже завершённый анализ.
            return {"job_id": job_id, "state": job.state, "repeated": True}

    try:
        _advance(job_id, JobState.FETCHING)
        with session_scope() as session:
            analysis = session.get(Analysis, analysis_id)
            polygon = analysis.polygon
            anon_id = polygon.anon_polygon_id
            date_from, date_to = analysis.date_from, analysis.date_to
            polygon_id = polygon.id

        if not anon_id:
            _fail(
                job_id,
                "NO_DATA_SOURCE",
                "у полигона нет привязки к конкурсному ряду, а живые провайдеры "
                "появятся в BE-007/BE-009",
            )
            return {"job_id": job_id, "state": JobState.FAILED.value}

        series = load_series(anon_id, date_from, date_to, data_dir=get_settings().model_data_dir)

        _advance(job_id, JobState.PREPROCESSING)
        with session_scope() as session:
            _store_observations(session, polygon_id, series)

        _advance(job_id, JobState.RECONSTRUCTING)
        outcome = run_reconstruction(series.frame, get_reconstructor(), context=series.context)

        _advance(job_id, JobState.ANALYZING)
        warnings = list(outcome.warnings)
        with session_scope() as session:
            analysis = session.get(Analysis, analysis_id)
            _store_outcome(session, analysis_id, outcome)
            try:
                warnings += _detect_anomalies(session, analysis, series, outcome)
            except (ValueError, FixtureUnavailable) as exc:
                # Детектор — не критический путь: без него анализ остаётся полезным.
                warnings.append(f"ANOMALY_DETECTION_SKIPPED: {exc}")
            analysis.model_version = outcome.model_version
            analysis.data_retrieved_at = datetime.now(UTC)
            session.add(
                Provenance(
                    analysis_id=analysis_id,
                    provider="fixture",
                    collection_id=series.source_path,
                    collection_version="competition-csv",
                    queried_at=datetime.now(UTC),
                    processing_params={
                        "model_version": outcome.model_version,
                        "observed_rows": outcome.observed_count,
                        "reconstructed_rows": outcome.reconstructed_count,
                        "warnings": warnings,
                    },
                    # Ряд читается из локального CSV, а не запрашивается у провайдера:
                    # это честно помечено как cached, иначе демо выдаёт офлайн за live
                    # (инвариант 10, решение D-005).
                    cached=True,
                )
            )
            analysis.cached = True

        _advance(job_id, JobState.COMPLETED)
        return {
            "job_id": job_id,
            "state": JobState.COMPLETED.value,
            "model_version": outcome.model_version,
            "reconstructed": outcome.reconstructed_count,
            "observed": outcome.observed_count,
            "warnings": warnings,
        }

    except FixtureUnavailable as exc:
        _fail(job_id, "DATA_NOT_AVAILABLE", str(exc))
    except ModelUnavailable as exc:
        # Единый код отказа модели (инвариант 6); в сообщении — безопасный текст.
        _fail(job_id, exc.error_code, exc.safe_message)
    except Exception:
        # Наружу не уходит ни трейс, ни текст исключения: он может содержать пути
        # и строку подключения (инвариант 11). Подробности остаются в логах воркера.
        _fail(job_id, "INTERNAL_ERROR", "анализ не выполнен из-за внутренней ошибки")
        raise
    return {"job_id": job_id, "state": JobState.FAILED.value}
