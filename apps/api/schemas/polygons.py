"""Схемы полигонов — часть контракта C-07 v0.1.

Геометрия на границе API — только GeoJSON в EPSG:4326 (инвариант 2). Площадь
считается сервером в подходящей projected CRS и отдаётся в гектарах; клиент
никогда не присылает площадь сам.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PolygonSource(StrEnum):
    """Откуда взялся контур. Источник виден в UI в легенде слоя (BE-005/BE-010)."""

    MANUAL = "manual"
    FIELDS_WORLD = "fields_world"
    OSM = "osm"
    WORLDCEREAL = "worldcereal"
    DEMO = "demo"


class GeoJSONGeometry(BaseModel):
    """Минимальный GeoJSON. Полная валидация (self-intersection, antimeridian,
    лимит вершин, диапазон площади) выполняется в BE-004 и здесь не дублируется."""

    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any]


class PolygonCreate(BaseModel):
    #: extra="forbid": молча проглоченное поле — это потерянные данные пользователя.
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    geometry: GeoJSONGeometry
    source: PolygonSource = PolygonSource.MANUAL
    #: Идентификатор контура во внешнем источнике (§8.3, `source_id text nullable`).
    #: Без него повторный поиск в том же bbox создаст дубликат того же поля.
    source_id: str | None = None
    #: Культура. Доступна даже на скрытой gap-строке и используется feature builder'ом
    #: ML, поэтому web-путь обязан её передавать (docs/01_ml_developer.md:24, :331).
    #: В данных значения русскоязычные: зерновые, озимая пшеница, подсолнечник.
    crop_type: str | None = None
    #: Произвольные атрибуты источника (§8.3, `properties jsonb`).
    properties: dict[str, Any] | None = None


class PolygonOut(BaseModel):
    id: str
    name: str
    geometry: GeoJSONGeometry
    source: PolygonSource
    source_id: str | None = None
    crop_type: str | None = None
    properties: dict[str, Any] | None = None
    #: Площадь в гектарах, посчитанная в projected CRS. В EPSG:4326 не считается никогда.
    area_ha: float
    #: Хэш геометрии — часть ключа идемпотентности анализа (инвариант 7).
    geometry_hash: str
    #: Диагностика починки геометрии: `make_valid` разрешён только с ней (§8.4 ТЗ),
    #: иначе пользователь не узнает, что сервер изменил его контур.
    validation_notes: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime | None = None


class PolygonList(BaseModel):
    items: list[PolygonOut]
    total: int


class FieldSearchRequest(BaseModel):
    """Тело `POST /api/v1/field-search` (BE-010).

    ТЗ §8.4 требует принимать `bbox/point`: поиск по клику на карте — основной
    сценарий. Ровно один из двух обязан быть задан; синтезировать псевдо-bbox
    вокруг точки на фронте нельзя — это вынесло бы геометрию в браузер вопреки
    инварианту 2.
    """

    model_config = ConfigDict(extra="forbid")

    bbox: tuple[float, float, float, float] | None = None
    #: Точка [lon, lat] в EPSG:4326.
    point: tuple[float, float] | None = None
    limit: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def _exactly_one_area(self) -> FieldSearchRequest:
        if (self.bbox is None) == (self.point is None):
            raise ValueError("задайте ровно одно из полей: bbox или point")
        return self


class FieldCandidate(BaseModel):
    geometry: GeoJSONGeometry
    source: PolygonSource
    #: Идентификатор во внешнем источнике — переносится в PolygonCreate.source_id
    #: при сохранении, иначе дедупликация по source/source_id невозможна.
    source_id: str | None = None
    name: str | None = None
    crop_type: str | None = None
    #: Уверенность источника в том, что это сельхозконтур. Не «точность» и не вероятность.
    confidence: float = Field(ge=0.0, le=1.0)
    area_ha: float


class FieldSearchResponse(BaseModel):
    items: list[FieldCandidate]
    #: Источники, которые фактически ответили. Пустой список — не ошибка,
    #: а валидное состояние «в этом bbox контуров не найдено» (UI обязан его показать).
    sources_queried: list[PolygonSource]


class ReferenceSeries(BaseModel):
    """Ряд, доступный offline-источнику данных (BE-008).

    Появился из практики: нарисованный на карте контур сам по себе данных не имеет.
    Геометрий в конкурсных CSV нет, сопоставить полигон с рядом автоматически
    невозможно, а живых провайдеров ещё нет (BE-007/BE-009). Значит выбор ряда
    делает человек — и обязан видеть, из чего выбирает, ДО запуска анализа.
    """

    anon_polygon_id: str
    #: Сколько дней в ряду реально наблюдалось. Полигон с 900 наблюдениями и
    #: полигон с 70 дают очень разное демо, и это должно быть видно в списке.
    observations: int
    first_date: date
    last_date: date
    crop_type: str | None = None
    #: Из какого файла взят ряд. Часть провенанса: потребитель должен знать,
    #: что источник офлайновый, а не живой провайдер.
    dataset: str


class ReferenceSeriesList(BaseModel):
    items: list[ReferenceSeries]
    total: int
