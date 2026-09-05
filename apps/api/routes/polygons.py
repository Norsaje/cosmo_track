"""Роуты полигонов. Схемы зафиксированы в C-07, реализация — BE-004."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from apps.api.schemas import (
    ErrorResponse,
    FieldSearchRequest,
    FieldSearchResponse,
    PolygonCreate,
    PolygonList,
    PolygonOut,
    PolygonSource,
)

#: Единый конверт ошибки объявляется в OpenAPI, иначе сгенерированный клиент
#: не узнает про error_code и будет читать несуществующее поле detail.
ERRORS = {
    422: {"model": ErrorResponse, "description": "Некорректное тело запроса"},
    501: {"model": ErrorResponse, "description": "Ещё не реализовано"},
}

router = APIRouter(tags=["polygons"], responses=ERRORS)

_NOT_IMPLEMENTED = "Реализуется в BE-004 (polygon CRUD) и BE-010 (field search)."


def _parse_bbox(raw: str) -> tuple[float, float, float, float]:
    """Разбор `minx,miny,maxx,maxy` в EPSG:4326 с проверкой диапазонов."""
    parts = raw.split(",")
    if len(parts) != 4:
        raise HTTPException(422, detail="bbox должен содержать ровно четыре числа")
    try:
        minx, miny, maxx, maxy = (float(p) for p in parts)
    except ValueError:
        raise HTTPException(422, detail="bbox содержит нечисловые значения") from None
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise HTTPException(422, detail="bbox вне допустимых координат EPSG:4326")
    return minx, miny, maxx, maxy


@router.get("/polygons", response_model=PolygonList)
async def list_polygons(
    bbox: str | None = Query(default=None, description="minx,miny,maxx,maxy в EPSG:4326"),
    source: PolygonSource | None = Query(default=None, description="Фильтр по источнику контура"),
) -> PolygonList:
    """Список полигонов. Пустой список — валидное состояние, а не ошибка.

    bbox валидируется здесь, а не в BE-004: иначе мусорное значение молча
    возвращало бы 200 и регрессия всплыла бы только после реализации фильтрации.
    """
    if bbox is not None:
        _parse_bbox(bbox)
    return PolygonList(items=[], total=0)


@router.post("/polygons", response_model=PolygonOut, status_code=status.HTTP_201_CREATED)
async def create_polygon(payload: PolygonCreate) -> PolygonOut:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)


@router.get("/polygons/{polygon_id}", response_model=PolygonOut)
async def get_polygon(polygon_id: str) -> PolygonOut:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)


@router.delete("/polygons/{polygon_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_polygon(polygon_id: str) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)


@router.post("/field-search", response_model=FieldSearchResponse)
async def field_search(payload: FieldSearchRequest) -> FieldSearchResponse:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_IMPLEMENTED)
