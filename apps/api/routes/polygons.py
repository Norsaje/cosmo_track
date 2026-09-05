"""Роуты полигонов (BE-004). Схемы зафиксированы в C-07.

Геометрия пересекает границу API только как GeoJSON в EPSG:4326 (инвариант 2);
всё метрическое — площадь, пределы — считает `geospatial.geometry` в projected CRS.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from geoalchemy2.shape import from_shape, to_shape
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from apps.api.deps import get_db, require_user
from apps.api.schemas import (
    ErrorResponse,
    FieldSearchRequest,
    FieldSearchResponse,
    GeoJSONGeometry,
    PolygonCreate,
    PolygonList,
    PolygonOut,
    PolygonSource,
    ReferenceSeries,
    ReferenceSeriesList,
)
from apps.api.settings import get_settings
from apps.db.models import Polygon, User
from veg_recovery.geospatial.geometry import GeometryError, normalize

#: Единый конверт ошибки объявляется в OpenAPI, иначе сгенерированный клиент
#: не узнает про error_code и будет читать несуществующее поле detail.
ERRORS = {
    401: {"model": ErrorResponse, "description": "Нужен вход"},
    404: {"model": ErrorResponse, "description": "Объект не найден"},
    422: {"model": ErrorResponse, "description": "Некорректное тело запроса"},
    501: {"model": ErrorResponse, "description": "Ещё не реализовано"},
}

router = APIRouter(tags=["polygons"], responses=ERRORS)


def _parse_bbox(raw: str) -> tuple[float, float, float, float]:
    """Разбор `minx,miny,maxx,maxy` в EPSG:4326 с проверкой диапазонов."""
    parts = raw.split(",")
    if len(parts) != 4:
        raise HTTPException(422, detail="bbox должен содержать ровно четыре числа")
    try:
        minx, miny, maxx, maxy = (float(p) for p in parts)
    except ValueError:
        raise HTTPException(422, detail="bbox содержит нечисловые значения") from None
    lon_ok = -180 <= minx <= 180 and -180 <= maxx <= 180
    lat_ok = -90 <= miny <= 90 and -90 <= maxy <= 90
    if not (lon_ok and lat_ok):
        raise HTTPException(422, detail="bbox вне допустимых координат EPSG:4326")
    if minx >= maxx or miny >= maxy:
        raise HTTPException(422, detail="bbox вырожден: minx<maxx и miny<maxy обязательны")
    return minx, miny, maxx, maxy


def _to_out(row: Polygon) -> PolygonOut:
    """Модель БД → схема C-07. Геометрия отдаётся как GeoJSON, не как WKB."""
    shapely_geometry = to_shape(row.geometry)
    return PolygonOut(
        id=str(row.id),
        name=row.name or "",
        geometry=GeoJSONGeometry(**json.loads(json.dumps(shapely_geometry.__geo_interface__))),
        source=PolygonSource(row.source),
        source_id=row.source_id,
        crop_type=row.crop_type,
        properties=row.properties,
        area_ha=row.area_ha or 0.0,
        geometry_hash=row.geometry_hash,
        validation_notes=row.validation_notes or [],
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("/polygons", response_model=PolygonList)
async def list_polygons(
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
    bbox: str | None = Query(default=None, description="minx,miny,maxx,maxy в EPSG:4326"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PolygonList:
    """Список контуров, при необходимости — в пределах bbox.

    Фильтр выполняется в PostGIS через `ST_Intersects` по GIST-индексу, а не
    вычитыванием всех строк в Python: иначе демо на нескольких тысячах полей
    начнёт тянуть всю таблицу на каждый сдвиг карты.
    """
    # Выборка сразу ограничена владельцем. Фильтровать после — значит однажды
    # забыть про новый роут и показать чужие поля.
    statement = select(Polygon).where(Polygon.user_id == user.id)
    counter = select(func.count()).select_from(Polygon).where(Polygon.user_id == user.id)
    if bbox is not None:
        minx, miny, maxx, maxy = _parse_bbox(bbox)
        envelope = func.ST_MakeEnvelope(minx, miny, maxx, maxy, 4326)
        statement = statement.where(func.ST_Intersects(Polygon.geometry, envelope))
        counter = counter.where(func.ST_Intersects(Polygon.geometry, envelope))
    total = db.execute(counter).scalar_one()
    rows = db.execute(statement.order_by(Polygon.created_at.desc()).limit(limit).offset(offset))
    return PolygonList(items=[_to_out(row) for row in rows.scalars()], total=int(total))


@router.post("/polygons", response_model=PolygonOut, status_code=status.HTTP_201_CREATED)
async def create_polygon(
    payload: PolygonCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> PolygonOut:
    """Создать контур.

    Геометрия проверяется до записи: тип, диапазон координат, antimeridian, число
    вершин, валидность и площадь в гектарах. `make_valid` разрешён, но только с
    отметкой в `validation_notes` — молча подменять контур пользователя нельзя (§8.4).
    """
    try:
        normalized = normalize(payload.geometry.model_dump())
    except GeometryError as exc:
        # Текст GeometryError сформирован для показа пользователю: ни путей, ни трейсов.
        raise HTTPException(422, detail=f"{exc.code}: {exc}") from None

    notes: list[str] = []
    if normalized.repaired:
        notes.append(
            "GEOMETRY_REPAIRED: контур был невалиден и восстановлен make_valid; "
            "сохранена исправленная геометрия"
        )

    row = Polygon(
        name=payload.name,
        geometry=from_shape(normalized.geometry, srid=4326),
        geometry_hash=normalized.geometry_hash,
        area_ha=normalized.area_ha,
        source=payload.source.value,
        source_id=payload.source_id,
        crop_type=payload.crop_type,
        properties=payload.properties,
        validation_notes=notes,
        # Конкурсный идентификатор приходит через properties: он нужен offline-пути
        # BE-008, чтобы найти ряд полигона в data/*.csv, но в C-07 отдельного поля нет.
        anon_polygon_id=(payload.properties or {}).get("anon_polygon_id"),
        user_id=user.id,
    )
    db.add(row)
    db.flush()
    db.refresh(row)
    return _to_out(row)


@router.get("/polygons/{polygon_id}", response_model=PolygonOut)
async def get_polygon(
    polygon_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> PolygonOut:
    row = db.get(Polygon, polygon_id)
    # Чужой полигон отдаёт 404, а не 403: 403 подтвердил бы, что объект с таким
    # идентификатором существует, и превратил бы роут в способ их перебирать.
    if row is None or row.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="полигон не найден")
    return _to_out(row)


@router.delete("/polygons/{polygon_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_polygon(
    polygon_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
) -> None:
    """Удалить контур вместе с его анализами (каскад в схеме)."""
    row = db.get(Polygon, polygon_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="полигон не найден")
    db.delete(row)


@router.post("/field-search", response_model=FieldSearchResponse)
async def field_search(payload: FieldSearchRequest) -> FieldSearchResponse:
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail="Реализуется в BE-010 (fields_world → OSM → WorldCereal → ручной контур).",
    )


@router.get("/reference-polygons", response_model=ReferenceSeriesList)
async def list_reference_polygons() -> ReferenceSeriesList:
    """Ряды, по которым сейчас возможен анализ.

    Временный роут на время отсутствия живых провайдеров. Нарисованный контур
    данных не несёт: геометрий в конкурсных CSV нет, и связать полигон с рядом
    автоматически нельзя. Поэтому интерфейс предлагает выбрать ряд явно, а не
    выясняет это через упавшую джобу спустя несколько секунд.

    Когда появятся BE-007/BE-009, роут останется как список офлайн-источников
    для демо без сети, но перестанет быть единственным способом получить данные.
    """
    from veg_recovery.providers.fixture import list_available_series

    try:
        # Каталог берётся из настроек, а не строкой: ряды обязаны быть теми же,
        # что видит модель, иначе выбранный полигон окажется вне её контекста.
        series = list_available_series(get_settings().model_data_dir)
    except (OSError, ValueError):
        # Отсутствие CSV — не отказ сервиса: пустой список честнее ошибки, а
        # интерфейс сам объяснит, что источников данных нет.
        series = ()
    items = [
        ReferenceSeries(
            anon_polygon_id=descriptor.anon_polygon_id,
            observations=descriptor.observations,
            first_date=descriptor.first_date,
            last_date=descriptor.last_date,
            crop_type=descriptor.crop_type,
            dataset=descriptor.dataset,
        )
        for descriptor in series
    ]
    return ReferenceSeriesList(items=items, total=len(items))
