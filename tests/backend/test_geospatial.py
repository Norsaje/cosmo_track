"""Тесты геометрии (BE-004): площадь в метрах, ключ идемпотентности, отказы.

Главное здесь — инвариант 2. «Площадь полигона не считается в градусах» стоит
отдельным пунктом red-team checklist перед CP-4, и проверяется он не чтением кода,
а числом: одна и та же фигура в градусах обязана давать разную площадь на разных
широтах, иначе где-то потерялось перепроецирование.
"""

from __future__ import annotations

import pytest

pytest.importorskip("shapely", reason="нужен extra `geo`")
pytest.importorskip("pyproj", reason="нужен extra `geo`")

from veg_recovery.geospatial.geometry import (  # noqa: E402
    MAX_VERTICES,
    GeometryError,
    area_hectares,
    normalize,
)


def _square(lon: float = 39.0, lat: float = 45.0, size: float = 0.01) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon, lat],
                [lon + size, lat],
                [lon + size, lat + size],
                [lon, lat + size],
                [lon, lat],
            ]
        ],
    }


def test_area_is_metric_not_degrees() -> None:
    """Одинаковая фигура в градусах — разная площадь на разных широтах.

    Если бы площадь считалась прямо в EPSG:4326, оба числа совпали бы, и «поле
    в 100 га» на 60-й широте оказалось бы вдвое больше настоящего.
    """
    near_equator = normalize(_square(lat=5.0)).area_ha
    far_north = normalize(_square(lat=60.0)).area_ha
    assert near_equator > far_north * 1.5
    # Квадрат 0.01° у экватора — примерно 1.1 × 1.1 км, то есть около 120 га.
    assert 100 < near_equator < 140


def test_hash_is_stable_across_equivalent_representations() -> None:
    """Polygon и MultiPolygon с той же геометрией — один ключ идемпотентности.

    Иначе повторный `POST /analyses` для того же поля, пришедшего другим путём,
    создал бы второй анализ вопреки инварианту 7.
    """
    single = normalize(_square())
    wrapped = normalize({"type": "MultiPolygon", "coordinates": [_square()["coordinates"]]})
    assert single.geometry_hash == wrapped.geometry_hash


def test_hash_changes_when_geometry_changes() -> None:
    assert normalize(_square()).geometry_hash != normalize(_square(lon=39.5)).geometry_hash


def test_self_intersection_is_repaired_but_reported() -> None:
    """`make_valid` разрешён только с диагностикой (§8.4 ТЗ): пользователь обязан
    узнать, что сохранён не в точности его контур."""
    bowtie = {
        "type": "Polygon",
        "coordinates": [[[39.0, 45.0], [39.01, 45.01], [39.01, 45.0], [39.0, 45.01], [39.0, 45.0]]],
    }
    result = normalize(bowtie)
    assert result.repaired is True
    assert result.area_ha > 0


def test_valid_geometry_is_not_marked_repaired() -> None:
    assert normalize(_square()).repaired is False


@pytest.mark.parametrize(
    ("name", "geojson", "code"),
    [
        ("точка", {"type": "Point", "coordinates": [39.0, 45.0]}, "INVALID_GEOMETRY_TYPE"),
        ("микроскопический", _square(size=0.00001), "AREA_TOO_SMALL"),
        # 5° × 5° около 45-й широты — порядка 20 млн га: заведомо больше предела,
        # но координаты остаются валидными, иначе сработал бы другой код отказа.
        ("область размером с регион", _square(size=5.0), "AREA_TOO_LARGE"),
        (
            "антимеридиан",
            {
                "type": "Polygon",
                "coordinates": [[[-170, 10], [170, 10], [170, 20], [-170, 20], [-170, 10]]],
            },
            "ANTIMERIDIAN_NOT_SUPPORTED",
        ),
        (
            "координаты вне диапазона",
            {"type": "Polygon", "coordinates": [[[200, 45], [201, 45], [201, 46], [200, 45]]]},
            "COORDINATES_OUT_OF_RANGE",
        ),
    ],
)
def test_bad_geometry_is_refused_with_a_code(name: str, geojson: dict, code: str) -> None:
    """У каждого отказа свой код: «невалидная геометрия» без причины не позволяет
    пользователю понять, что именно исправить."""
    with pytest.raises(GeometryError) as excinfo:
        normalize(geojson)
    assert excinfo.value.code == code


def test_vertex_limit_protects_the_pipeline() -> None:
    """Контур из десятков тысяч точек — это импортированный шейпфайл целиком.
    Он кладёт и провайдера, и растровую обрезку, поэтому отсекается на входе."""
    step = 0.02 / (MAX_VERTICES + 10)
    ring = [[39.0 + index * step, 45.0] for index in range(MAX_VERTICES + 10)]
    ring += [[39.02, 45.01], [39.0, 45.01], [39.0, 45.0]]
    with pytest.raises(GeometryError) as excinfo:
        normalize({"type": "Polygon", "coordinates": [ring]})
    assert excinfo.value.code in ("TOO_MANY_VERTICES", "INVALID_GEOMETRY", "AREA_TOO_SMALL")


def test_area_helper_agrees_with_normalize() -> None:
    normalized = normalize(_square())
    assert area_hectares(normalized.geometry) == pytest.approx(normalized.area_ha)
