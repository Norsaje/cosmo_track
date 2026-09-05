"""Схема PostGIS сервиса (BE-002). Состав таблиц задан `docs/BACKEND.md` §8.3.

Три правила, которые эта схема обязана соблюдать и которые легко нарушить молча:

1. **Никаких CHECK на диапазон NDVI.** Реальный target выходит за `[-1, 1]`:
   train до −2.1304, видимый test до 1.8429 (`reports/data_contract_issues.md`).
   Констрейнт превратил бы штатные данные в ошибку записи (инвариант 4).
2. **`reason_codes` — jsonb без CHECK и без enum-типа.** У производителя C-09
   сейчас девять кодов, но контракт версии 0.1 не объявлен закрытым; расширение
   списка у DL не должно ронять нашу вставку (§4.1).
3. **Missing никогда не подменяется нулём.** Для каждого числового источника
   держим `raw`, `cleaned` и `invalid_flag`, для погоды — флаг доступности
   (инвариант 3). Ноль в NDVI — это физическое значение, а не «нет данных».
"""

from __future__ import annotations

import uuid
from datetime import date as date_type
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.schemas.jobs import JobState
from apps.db.base import Base

#: Источники контура поля (§8.3). Хранится строкой, а не PostgreSQL enum: добавление
#: нового провайдера в BE-010 не должно требовать миграции типа.
POLYGON_SOURCES = ("manual", "fields_world", "osm", "worldcereal", "demo")


def _uuid() -> str:
    return str(uuid.uuid4())


class Polygon(Base):
    """Контур поля. Геометрия хранится в EPSG:4326 — это граница API (инвариант 2).

    Площадь считается **не** здесь: `area_ha` заполняет `geospatial.geometry`,
    проецируя контур в метрическую CRS. Площадь в градусах — отдельный пункт
    red-team checklist перед CP-4.
    """

    __tablename__ = "polygons"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name: Mapped[str | None] = mapped_column(String(200))
    geometry: Mapped[object] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=False), nullable=False
    )
    #: SHA256 нормализованной геометрии. Половина ключа идемпотентности (инвариант 7).
    geometry_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    area_ha: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    #: Тип культуры. Значения русскоязычные («озимая пшеница», «пастбища/зерновые»),
    #: и это не наша вольность, а факт данных. Колонка обязательна для модели:
    #: `validate_request` у ML требует `crop_type` даже на веб-пути.
    crop_type: Mapped[str | None] = mapped_column(String(120))
    #: Идентификатор контура во внешнем источнике (§8.3). Без него повторный поиск
    #: по тому же bbox создаст дубликат того же поля.
    source_id: Mapped[str | None] = mapped_column(String(200))
    #: Произвольные атрибуты источника (§8.3, `properties jsonb`).
    properties: Mapped[dict | None] = mapped_column(JSONB)
    #: Диагностика приведения геометрии. `make_valid` разрешён только с ней (§8.4):
    #: пользователь обязан узнать, что сервер сохранил не в точности его контур.
    validation_notes: Mapped[list | None] = mapped_column(JSONB)
    #: Ключ конкурсного полигона, если контур соответствует строке датасета.
    #: Нужен offline-пути BE-008: фикстура читает ряд из `data/*.csv` по этому id.
    anon_polygon_id: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    analyses: Mapped[list[Analysis]] = relationship(back_populates="polygon")

    __table_args__ = (
        # Список источников закрыт CHECK'ом, а не enum-типом: добавить провайдера
        # в BE-010 — это правка констрейнта, а не ALTER TYPE с блокировкой таблицы.
        CheckConstraint(
            "source IN (" + ", ".join(f"'{value}'" for value in POLYGON_SOURCES) + ")",
            name="ck_polygon_source",
        ),
        Index("ix_polygons_geometry", "geometry", postgresql_using="gist"),
        Index("ix_polygons_geometry_hash", "geometry_hash"),
    )


class Analysis(Base):
    """Запрошенный анализ полигона за период.

    Идемпотентность (инвариант 7) обеспечивает уникальный индекс по
    `geometry_hash + date_from + date_to + pipeline_version`: повторный
    `POST /analyses` обязан вернуть тот же `analysis_id`, а не создать дубликат джобы.
    """

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    polygon_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("polygons.id", ondelete="CASCADE"), nullable=False
    )
    #: Дублируется из полигона намеренно: ключ идемпотентности обязан переживать
    #: удаление и повторное создание контура с той же геометрией.
    geometry_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    date_from: Mapped[date_type] = mapped_column(Date, nullable=False)
    date_to: Mapped[date_type] = mapped_column(Date, nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(120))
    state: Mapped[str] = mapped_column(String(20), nullable=False, default=JobState.QUEUED.value)
    #: Провайдер отдал неполные данные. Отдельно от state: анализ мог завершиться,
    #: но на частичных наблюдениях, и потребитель обязан это видеть.
    partial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: Результат собран из кэша. Выдавать кэш за live запрещено (инвариант 10, D-005).
    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    data_retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    polygon: Mapped[Polygon] = relationship(back_populates="analyses")
    jobs: Mapped[list[AnalysisJob]] = relationship(back_populates="analysis")

    __table_args__ = (
        UniqueConstraint(
            "geometry_hash",
            "date_from",
            "date_to",
            "pipeline_version",
            name="uq_analysis_idempotency",
        ),
    )


class AnalysisJob(Base):
    """Состояние выполнения анализа — durable FSM (BE-006, решение D-004).

    Стадия живёт в БД, а не в памяти воркера: перезапуск контейнера не должен
    превращать выполняющийся анализ в вечный QUEUED.
    """

    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    state: Mapped[str] = mapped_column(String(20), nullable=False, default=JobState.QUEUED.value)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    partial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    stage_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Машиночитаемый код (MODEL_SCHEMA_MISMATCH, PROVIDER_UNAVAILABLE, ...).
    error_code: Mapped[str | None] = mapped_column(String(64))
    #: Безопасный текст для пользователя: ни секретов, ни stack trace (инвариант 11).
    error_message_safe: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    analysis: Mapped[Analysis] = relationship(back_populates="jobs")

    __table_args__ = (
        CheckConstraint("progress BETWEEN 0 AND 100", name="ck_job_progress_range"),
        Index("ix_analysis_jobs_analysis", "analysis_id"),
    )


class Observation(Base):
    """Суточное наблюдение полигона от одного источника.

    Для каждого числового ряда держим три колонки: `*_raw` — как пришло,
    `*_cleaned` — после QA, `*_invalid_flag` — признак, что значение отбраковано.
    Это и есть запрет «заменять missing нулём»: отбракованное значение видно как
    отбракованное, а не как ноль (инвариант 3).
    """

    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    polygon_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("polygons.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date_type] = mapped_column(Date, nullable=False)
    #: s2 / landsat / modis / era5 / fixture.
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    processing_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")

    ndvi_raw: Mapped[float | None] = mapped_column(Float)
    ndvi_cleaned: Mapped[float | None] = mapped_column(Float)
    ndvi_invalid_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    evi_raw: Mapped[float | None] = mapped_column(Float)
    evi_cleaned: Mapped[float | None] = mapped_column(Float)
    evi_invalid_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ndwi_raw: Mapped[float | None] = mapped_column(Float)
    ndwi_cleaned: Mapped[float | None] = mapped_column(Float)
    ndwi_invalid_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: Погода — с флагом доступности: отсутствие ряда ERA5 не то же самое, что 0 °C.
    temp_c: Mapped[float | None] = mapped_column(Float)
    temp_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    precip_mm: Mapped[float | None] = mapped_column(Float)
    precip_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: Доля и абсолютное число валидных пикселей. Инвариант 3 требует оба: 3 пикселя
    #: и 3000 при одинаковой доле — разной достоверности наблюдения.
    valid_pixel_fraction: Mapped[float | None] = mapped_column(Float)
    pixel_count: Mapped[int | None] = mapped_column(Integer)
    qa_flags: Mapped[dict | None] = mapped_column(JSONB)
    asset_ids: Mapped[list | None] = mapped_column(JSONB)

    __table_args__ = (
        UniqueConstraint(
            "polygon_id", "date", "source", "processing_version", name="uq_observation_key"
        ),
        Index("ix_observations_polygon_date", "polygon_id", "date"),
    )


class Reconstruction(Base):
    """Точка итогового ряда: наблюдение или восстановление, плюс диагностика C-05.

    `primary_ndvi_raw` и `ndvi_harmonized` — **разные** ряды (решение D-003), и путать
    их запрещено отдельным пунктом red-team checklist. `is_observed` выводится из
    исходного конечного значения, а не из факта реконструкции: естественный NaN
    не является reconstructed (требование DL из consumer review).
    """

    __tablename__ = "reconstructions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date_type] = mapped_column(Date, nullable=False)

    primary_ndvi_raw: Mapped[float | None] = mapped_column(Float)
    primary_ndvi_reconstructed: Mapped[float | None] = mapped_column(Float)
    ndvi_harmonized: Mapped[float | None] = mapped_column(Float)
    is_observed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_reconstructed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    lower: Mapped[float | None] = mapped_column(Float)
    upper: Mapped[float | None] = mapped_column(Float)
    selected_source: Mapped[str | None] = mapped_column(String(32))
    method: Mapped[str | None] = mapped_column(String(64))

    # --- диагностика C-05: 16 полей DiagnosticRow ---
    p_s2: Mapped[float | None] = mapped_column(Float)
    p_landsat: Mapped[float | None] = mapped_column(Float)
    p_modis: Mapped[float | None] = mapped_column(Float)
    p_unknown: Mapped[float | None] = mapped_column(Float)
    #: Расстояния до контекста. У производителя `float | None` и в JSON — null,
    #: не бесконечность; nullable-колонка сохраняет ровно эту семантику.
    left_distance_days: Mapped[float | None] = mapped_column(Float)
    right_distance_days: Mapped[float | None] = mapped_column(Float)
    model_disagreement: Mapped[float | None] = mapped_column(Float)
    fallback_reason: Mapped[str | None] = mapped_column(String(64))
    context_quality: Mapped[float | None] = mapped_column(Float)
    source_confidence: Mapped[float | None] = mapped_column(Float)
    #: Открытый список строк — jsonb, без CHECK: набор флагов у ML может расшириться.
    quality_flags: Mapped[list | None] = mapped_column(JSONB)
    interval_status: Mapped[str | None] = mapped_column(String(64))
    interval_level: Mapped[float | None] = mapped_column(Float)
    harmonization_status: Mapped[str | None] = mapped_column(String(64))

    #: Климатическая норма для этой даты. ТЗ §8.9 требует её на графике как
    #: нейтральную линию: без неё отклонение не с чем сопоставить глазом, а
    #: `min_robust_z` из события аномалии остаётся числом без визуального смысла.
    #: Модели она не передаётся — на реальной gap-строке её не существует.
    climatology_mean: Mapped[float | None] = mapped_column(Float)
    climatology_std: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("analysis_id", "date", name="uq_reconstruction_key"),
        Index("ix_reconstructions_analysis_date", "analysis_id", "date"),
    )


class AnomalyEvent(Base):
    """Событие C-09. Двенадцать полей производителя плюс наши служебные.

    `severity` — строка, хотя у производителя множество из трёх значений: закрытый
    тип в БД потребовал бы миграции при любом расширении. `reason_codes` — jsonb
    без CHECK по той же причине (§4.1).
    """

    __tablename__ = "anomaly_events"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    start_date: Mapped[date_type] = mapped_column(Date, nullable=False)
    end_date: Mapped[date_type] = mapped_column(Date, nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    #: Heuristic support, а НЕ калиброванная вероятность аномалии. Хранится как есть;
    #: запрет показывать это значение процентом уверенности — на стороне UI (BE-013).
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    min_robust_z: Mapped[float] = mapped_column(Float, nullable=False)
    negative_area: Mapped[float] = mapped_column(Float, nullable=False)
    observed_points: Mapped[int] = mapped_column(Integer, nullable=False)
    reconstructed_points: Mapped[int] = mapped_column(Integer, nullable=False)
    reason_codes: Mapped[list] = mapped_column(JSONB, nullable=False)
    explanation_ru: Mapped[str] = mapped_column(Text, nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str | None] = mapped_column(String(16))
    duration_days: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("ix_anomaly_events_analysis", "analysis_id"),)


class Provenance(Base):
    """Происхождение данных анализа и запись кэша.

    Токены и credentials здесь не хранятся никогда (§8.3, инвариант 11) — только
    идентификаторы коллекций и параметры обработки.
    """

    __tablename__ = "provenance"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    collection_version: Mapped[str | None] = mapped_column(String(64))
    item_ids: Mapped[list | None] = mapped_column(JSONB)
    queried_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    crs: Mapped[str | None] = mapped_column(String(32))
    resolution_m: Mapped[float | None] = mapped_column(Float)
    qa_definition: Mapped[str | None] = mapped_column(Text)
    processing_params: Mapped[dict | None] = mapped_column(JSONB)
    license_url: Mapped[str | None] = mapped_column(String(400))
    #: Ключ кэша провайдера и срок годности записи (BE-014).
    fingerprint: Mapped[str | None] = mapped_column(String(64), index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Этот источник отдан из кэша. Ровно этот флаг UI показывает как «cached».
    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (Index("ix_provenance_analysis", "analysis_id"),)
