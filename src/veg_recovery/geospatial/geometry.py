"""Валидация контура, площадь и ключ идемпотентности (BE-004).

Главное правило модуля — инвариант 2: **площадь и буферы считаются в projected CRS**.
EPSG:4326 — это градусы, и «площадь» в них не имеет физического смысла: одна и та же
фигура у экватора и на широте 60° даст одинаковое число квадратных градусов, но разное
число гектаров. «Площадь полигона не считается в градусах» — отдельный пункт red-team
checklist перед CP-4, и здесь он выполняется явным перепроецированием.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pyproj import CRS, Transformer
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform

#: Границы допустимой площади поля в гектарах (§8.4 ТЗ). Значения настраиваемые:
#: 0.1 га отсекает случайный клик по карте, 50 000 га — обведённую область размером
#: с район, для которой один анализ бессмысленен.
MIN_AREA_HA = 0.1
MAX_AREA_HA = 50_000.0

#: Лимит вершин: контур из десятков тысяч точек — это результат импорта чужого
#: шейпфайла целиком, а не поле. Такой запрос кладёт и провайдера, и растровую обрезку.
MAX_VERTICES = 10_000

WGS84 = CRS.from_epsg(4326)


class GeometryError(ValueError):
    """Геометрия непригодна. Текст безопасен для показа пользователю."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class NormalizedGeometry:
    """Приведённый контур: то, что кладём в БД, вместе с производными."""

    geometry: MultiPolygon
    geometry_hash: str
    area_ha: float
    vertices: int
    #: Геометрия была невалидной и восстановлена `make_valid`. Инвариант: чинить
    #: молча нельзя — потребитель обязан узнать, что контур изменился.
    repaired: bool


def utm_crs_for(geometry: MultiPolygon | Polygon) -> CRS:
    """Подобрать метрическую CRS по центроиду контура.

    UTM-зона, а не единая проекция на весь мир: у Web Mercator (EPSG:3857) искажение
    площади растёт как квадрат секанса широты — на 60° это уже вчетверо, и «поле
    в 100 га» превратилось бы в 400. Для полей внутри одной зоны UTM ошибка площади
    остаётся долями процента.
    """
    centroid = geometry.centroid
    zone = int((centroid.x + 180.0) // 6.0) + 1
    # 326xx — северное полушарие, 327xx — южное.
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def area_hectares(geometry: MultiPolygon | Polygon) -> float:
    """Площадь в гектарах через перепроецирование в метрическую CRS."""
    transformer = Transformer.from_crs(WGS84, utm_crs_for(geometry), always_xy=True)
    projected = transform(transformer.transform, geometry)
    return projected.area / 10_000.0


def geometry_hash(geometry: MultiPolygon) -> str:
    """Стабильный ключ геометрии — половина ключа идемпотентности (инвариант 7).

    Хешируется GeoJSON с округлением координат до 7 знаков (примерно 1 см на экваторе):
    без округления двойная точность float даёт разный хеш для геометрически одного
    и того же контура, пришедшего разными путями, и идемпотентность перестаёт работать.
    """

    def _round(coords):
        return [
            [[round(x, 7), round(y, 7)] for x, y in ring] for ring in coords
        ]

    payload = {
        "type": "MultiPolygon",
        "coordinates": [
            _round(polygon.__geo_interface__["coordinates"]) for polygon in geometry.geoms
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _count_vertices(geometry: MultiPolygon) -> int:
    total = 0
    for polygon in geometry.geoms:
        total += len(polygon.exterior.coords)
        total += sum(len(ring.coords) for ring in polygon.interiors)
    return total


def normalize(geojson: dict) -> NormalizedGeometry:
    """Проверить и привести контур из тела API к тому, что можно хранить.

    Порядок проверок важен: сначала дешёвые (тип, координаты, число вершин), потом
    восстановление, и только в конце площадь — она требует перепроецирования.
    """
    try:
        geometry = shape(geojson)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise GeometryError("INVALID_GEOMETRY", "не удалось разобрать geometry") from exc

    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise GeometryError(
            "INVALID_GEOMETRY_TYPE",
            f"ожидался Polygon или MultiPolygon, получен {geometry.geom_type}",
        )
    if geometry.is_empty:
        raise GeometryError("EMPTY_GEOMETRY", "геометрия пуста")

    minx, miny, maxx, maxy = geometry.bounds
    if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0):
        raise GeometryError("COORDINATES_OUT_OF_RANGE", "долгота вне диапазона EPSG:4326")
    if not (-90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0):
        raise GeometryError("COORDINATES_OUT_OF_RANGE", "широта вне диапазона EPSG:4326")
    # Контур, растянутый почти на полмира по долготе, — это почти всегда полигон,
    # пересекающий 180-й меридиан и записанный без переноса. Считать по нему площадь
    # бессмысленно, а провайдер получит запрос на пол-планеты.
    if maxx - minx > 180.0:
        raise GeometryError(
            "ANTIMERIDIAN_NOT_SUPPORTED",
            "контур пересекает 180-й меридиан; разделите его на части",
        )

    repaired = False
    if not geometry.is_valid:
        # make_valid, но обязательно с диагностикой: пользователь должен знать,
        # что сохранён не в точности тот контур, который он прислал (§8.4 ТЗ).
        geometry = make_valid(geometry)
        repaired = True
        if geometry.geom_type == "GeometryCollection":
            parts = [g for g in geometry.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if not parts:
                raise GeometryError("INVALID_GEOMETRY", "после восстановления не осталось площади")
            flattened = [
                piece
                for part in parts
                for piece in (part.geoms if part.geom_type == "MultiPolygon" else [part])
            ]
            geometry = MultiPolygon(flattened)
        if geometry.is_empty or not geometry.is_valid:
            raise GeometryError("INVALID_GEOMETRY", "геометрию не удалось восстановить")

    multi = geometry if geometry.geom_type == "MultiPolygon" else MultiPolygon([geometry])

    vertices = _count_vertices(multi)
    if vertices > MAX_VERTICES:
        raise GeometryError(
            "TOO_MANY_VERTICES", f"вершин {vertices}, допустимо не больше {MAX_VERTICES}"
        )

    area = area_hectares(multi)
    if area < MIN_AREA_HA:
        raise GeometryError("AREA_TOO_SMALL", f"площадь {area:.4f} га меньше {MIN_AREA_HA} га")
    if area > MAX_AREA_HA:
        raise GeometryError("AREA_TOO_LARGE", f"площадь {area:.1f} га больше {MAX_AREA_HA} га")

    return NormalizedGeometry(
        geometry=multi,
        geometry_hash=geometry_hash(multi),
        area_ha=area,
        vertices=vertices,
        repaired=repaired,
    )
