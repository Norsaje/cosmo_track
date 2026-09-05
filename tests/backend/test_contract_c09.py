"""Потребление контракта C-09 (владелец — Разработчик 2, DL).

Мы событие только считаем чужим детектором, храним и отдаём. Здесь проверяется
граница: что мы кладём на вход, что переносим в БД и чем честно помечаем откат.
Алгоритм DL проверяют их собственные тесты `tests/anomalies` — дублировать их
здесь значило бы отвечать за чужую логику.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import pandas as pd
import pytest

pytest.importorskip("numpy")

from apps.worker.tasks import (  # noqa: E402
    MIN_REFERENCE_YEARS,
    _c09_frame,
    _detect_with_baseline,
    _detect_with_c09,
    _selected_source,
)
from veg_recovery.anomalies.advanced import AnomalyConfig  # noqa: E402
from veg_recovery.anomalies.events import ALGORITHM_VERSION, SCHEMA_VERSION  # noqa: E402
from veg_recovery.service.orchestrator import SeriesPointRecord  # noqa: E402

SEVERITIES = {"normal", "biomass_suppression", "critical"}


@dataclass
class _Polygon:
    anon_polygon_id: str = "TOY-0001"


@dataclass
class _Analysis:
    id: str = "00000000-0000-0000-0000-000000000001"
    polygon: _Polygon = None  # type: ignore[assignment]


@dataclass
class _Series:
    frame: pd.DataFrame


@dataclass
class _Outcome:
    points: list


def _season(day_of_year: int) -> float:
    """Грубая сезонная кривая: зимой около нуля, в июле максимум."""
    import math

    return 0.15 + 0.55 * max(0.0, math.sin(math.pi * (day_of_year - 60) / 210))


def _frame(year_from: int, year_to: int, *, dip: tuple[int, int] | None = None) -> pd.DataFrame:
    rows = []
    for year in range(year_from, year_to + 1):
        for day in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="5D"):
            value = _season(int(day.dayofyear))
            if dip and day.year == year_to and dip[0] <= int(day.dayofyear) <= dip[1]:
                value -= 0.30
            rows.append(
                {
                    "anon_polygon_id": "TOY-0001",
                    "date": day,
                    "crop_type": "зерновые",
                    "primary_ndvi": round(value, 6),
                    "s2_ndvi": round(value, 6),
                }
            )
    return pd.DataFrame(rows)


def _outcome_for(frame: pd.DataFrame) -> _Outcome:
    return _Outcome(
        points=[
            SeriesPointRecord(
                date=pd.Timestamp(row["date"]),
                primary_ndvi_raw=float(row["primary_ndvi"]),
                primary_ndvi_reconstructed=None,
                ndvi_harmonized=None,
                is_observed=bool(pd.notna(row["primary_ndvi"])),
                is_reconstructed=False,
            )
            for row in frame.to_dict("records")
        ]
    )


def test_reference_years_guard_matches_the_detector() -> None:
    """Наш порог истории обязан совпадать с порогом детектора.

    Разойдись они — мы бы либо звали детектор там, где он сам объявляет историю
    недостаточной, либо молча отказывались считать то, что он посчитать может.
    """
    assert MIN_REFERENCE_YEARS == AnomalyConfig().min_reference_years


def test_selected_source_is_matched_by_value_not_by_order() -> None:
    """Источник определяется сверкой значения, а не правилом «первый непустой».

    Иерархия S2 → Landsat → MODIS подтверждена эмпирически, но подставлять
    сенсор, которого в строке нет, значит придумывать провенанс: гармонизатор
    DL по этому полю выбирает калибровку.
    """
    assert _selected_source(0.5, 0.5, 0.4, 0.3) == "s2"
    assert _selected_source(0.4, None, 0.4, 0.3) == "landsat"
    assert _selected_source(0.3, None, None, 0.3) == "modis"
    # Ни один сенсор не совпал — источник неизвестен, и выдумывать его нельзя.
    assert _selected_source(0.9, 0.5, 0.4, 0.3) is None
    # Гэп: значения нет, источника тоже.
    assert _selected_source(None, 0.5, 0.4, 0.3) is None


def test_c09_frame_carries_exactly_what_the_validator_demands() -> None:
    """`_validate` DL отвергает кадр целиком при пустом ключе, дате или культуре."""
    frame = _frame(2020, 2020)
    prepared = _c09_frame(frame)
    for column in ("anon_polygon_id", "date", "crop_type", "primary_ndvi", "selected_source"):
        assert column in prepared, column
    assert not prepared[["anon_polygon_id", "date", "crop_type"]].isna().any().any()
    # Дата — календарный день строкой, без таймзоны: этого требует их валидатор.
    assert prepared["date"].iloc[0] == "2020-01-01"


def test_events_come_from_the_producer_with_its_own_categories() -> None:
    """Подключён именно C-09, а не наш пересказ: категория и версия — от DL."""
    reference = _frame(2018, 2023)
    query = _frame(2024, 2024, dip=(170, 220))
    analysis = _Analysis(polygon=_Polygon())

    rows, warnings = _detect_with_c09(
        analysis, _Series(query), _outcome_for(query), reference
    )

    assert rows, f"детектор не нашёл события на явном провале; warnings={warnings}"
    for row in rows:
        assert row.severity in SEVERITIES, row.severity
        # Версия алгоритма хранится: по ней различаются 0.1.0 и 0.1.1.
        assert row.algorithm_version == ALGORITHM_VERSION
        assert row.schema_version == SCHEMA_VERSION
        assert row.explanation_ru.strip()
        assert isinstance(row.reason_codes, list)
        assert row.observed_points + row.reconstructed_points > 0
        assert row.duration_days >= 1


def test_baseline_fallback_names_itself_and_never_invents_a_category() -> None:
    """Откат разрешён, молчаливый откат — нет.

    Baseline тяжесть не оценивает, поэтому категория C-09 ему не присваивается,
    а причина отката попадает в предупреждение дословно.
    """
    reference = _frame(2018, 2023)
    query = _frame(2024, 2024, dip=(170, 220))
    analysis = _Analysis(polygon=_Polygon())

    rows, warnings = _detect_with_baseline(
        analysis, _Series(query), _outcome_for(query), reference, reason="ValueError: проверка"
    )

    assert any("ANOMALY_SOURCE_IS_BASELINE" in w for w in warnings)
    assert any("ValueError: проверка" in w for w in warnings)
    for row in rows:
        assert row.severity == "baseline_unranked"
        assert row.severity not in SEVERITIES
        assert row.algorithm_version == "ml-baseline-anomalies"


def test_reconstructed_points_are_not_silently_harmonized() -> None:
    """Восстановленная точка не получает чужую шкалу молча.

    У предсказания нет сенсора, а гармонизатор переводит значение в шкалу S2.
    Сырой `primary_ndvi` — S2 лишь на 36.8 % ряда, поэтому объявить предсказание
    «шкалой S2» значило бы исказить две трети данных. Мы этого не делаем и
    говорим об этом вслух.
    """
    reference = _frame(2018, 2023)
    query = _frame(2024, 2024, dip=(170, 220))
    outcome = _outcome_for(query)
    # Одну точку объявляем восстановленной моделью.
    outcome.points[10] = SeriesPointRecord(
        date=outcome.points[10].date,
        primary_ndvi_raw=None,
        primary_ndvi_reconstructed=0.4,
        ndvi_harmonized=0.4,
        is_observed=False,
        is_reconstructed=True,
    )

    _rows, warnings = _detect_with_c09(
        _Analysis(polygon=_Polygon()), _Series(query), outcome, reference
    )
    assert any("RECONSTRUCTED_POINTS_NOT_HARMONIZED" in w for w in warnings)
