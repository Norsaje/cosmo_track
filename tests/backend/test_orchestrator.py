"""Тесты оркестратора (C-08) и offline-источника наблюдений (BE-008).

Оркестратор проверяется без БД, брокера и FastAPI — ровно за этим он и вынесен
из воркера. Модель подменяется заглушкой там, где важна логика сборки ряда, и
берётся настоящая там, где важен реальный ответ.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veg_recovery.service import ModelStub, build_reconstructor
from veg_recovery.service.orchestrator import build_gap_mask, run_reconstruction

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/ml/ndvi_backend_handoff_v1/bundle"

pytestmark = pytest.mark.skipif(
    not (ROOT / "src/veg_recovery/contracts.py").is_file(),
    reason="код ML недоступен: ветка models не смержена",
)


def _frame(values: list[float], polygon: str = "AOI-TEST") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anon_polygon_id": polygon,
            "date": pd.date_range("2024-05-01", periods=len(values), freq="D"),
            "primary_ndvi": values,
            "crop_type": "зерновые",
        }
    )


class _StubHandle:
    """Минимальный держатель модели: оркестратору достаточно `predict` и флагов."""

    def __init__(self, *, is_stub: bool = True) -> None:
        self.reconstructor = ModelStub()
        self.model_version = ModelStub.model_version
        self.is_stub = is_stub

    def predict(self, request):
        return self.reconstructor.predict(request)


def test_gap_mask_matches_the_frame_index() -> None:
    """Маска обязана быть Series с индексом кадра и строго bool.

    `validate_request` владельца контракта отвергает numpy-массив и чужой индекс,
    поэтому ошибка здесь всплыла бы уже внутри модели, дальше от причины.
    """
    frame = _frame([0.3, np.nan, 0.5])
    mask = build_gap_mask(frame)
    assert isinstance(mask, pd.Series)
    assert mask.index.equals(frame.index)
    assert mask.dtype == bool
    assert mask.tolist() == [False, True, False]


def test_series_keeps_observed_and_reconstructed_apart() -> None:
    """Наблюдение и восстановление не смешиваются в одном поле (решение D-003).

    В точке-наблюдении `primary_ndvi_reconstructed` пуст, в восстановленной — пуст
    `primary_ndvi_raw`. Склеенная линия на графике выдала бы модель за измерение.
    """
    frame = _frame([0.30, 0.33, np.nan, 0.42, 0.45])
    outcome = run_reconstruction(frame, _StubHandle())

    assert outcome.observed_count == 4
    assert outcome.reconstructed_count == 1

    observed = [p for p in outcome.points if p.is_observed]
    reconstructed = [p for p in outcome.points if p.is_reconstructed]
    assert all(p.primary_ndvi_reconstructed is None for p in observed)
    assert all(p.primary_ndvi_raw is None for p in reconstructed)
    assert all(not p.is_reconstructed for p in observed)


def test_stub_usage_is_announced() -> None:
    """«Ответ получен заглушкой» — свойство результата, а не деталь реализации.
    Без этого предупреждения демо не отличит модель от подстановки."""
    outcome = run_reconstruction(_frame([0.3, np.nan, 0.5]), _StubHandle(is_stub=True))
    assert any(w.startswith("MODEL_STUB_USED") for w in outcome.warnings)


def test_no_gaps_is_a_valid_state_not_an_error() -> None:
    outcome = run_reconstruction(_frame([0.3, 0.4, 0.5]), _StubHandle())
    assert outcome.reconstructed_count == 0
    assert any(w.startswith("NO_GAPS") for w in outcome.warnings)


def test_series_without_observations_is_not_invented() -> None:
    """Ряд без единого наблюдения не восстанавливается.

    Придумать значения тут технически можно, но это была бы выдача выдумки за
    результат: пустой ряд с предупреждением честнее.
    """
    outcome = run_reconstruction(_frame([np.nan, np.nan, np.nan]), _StubHandle())
    assert outcome.reconstructed_count == 0
    assert any(w.startswith("NO_OBSERVATIONS") for w in outcome.warnings)


def test_diagnostics_keep_null_instead_of_zero() -> None:
    """Отсутствующее расстояние до контекста остаётся None, а не нулём.

    Ноль дней означал бы «сосед в тот же день», то есть прямо противоположное
    отсутствию контекста.
    """
    frame = _frame([np.nan, 0.33, 0.36])
    outcome = run_reconstruction(frame, _StubHandle())
    point = next(p for p in outcome.points if p.is_reconstructed)
    assert point.diagnostics["left_distance_days"] is None
    assert point.diagnostics["right_distance_days"] == 1.0


@pytest.mark.skipif(not (BUNDLE / "manifest.json").is_file(), reason="нет bundle C-04")
def test_real_model_fills_the_series() -> None:
    """Тот же оркестратор с настоящей обученной моделью."""
    frame = _frame(list(np.where(np.arange(11) == 5, np.nan, np.linspace(0.3, 0.6, 11))))
    outcome = run_reconstruction(frame, build_reconstructor(BUNDLE, trusted=True))
    assert outcome.model_version == "p0-catboost-gpu-v1"
    assert outcome.reconstructed_count == 1
    assert not any(w.startswith("MODEL_STUB_USED") for w in outcome.warnings)
    point = next(p for p in outcome.points if p.is_reconstructed)
    assert point.method != "mean_neighbors"
    assert point.ndvi_harmonized is not None


# --------------------------------------------------------------- offline-источник


DATA_DIR = ROOT / "data"

fixture_available = pytest.mark.skipif(
    not (DATA_DIR / "train_dataset.csv").is_file(), reason="нет конкурсных CSV"
)


@fixture_available
def test_fixture_returns_only_allowed_columns() -> None:
    """В кадр модели не попадают поля, которых нет на скрытой строке.

    `ndvi_zscore`, `ndvi_climatology_*`, `status` и `n_reference_years` в проде
    на gap-строке отсутствуют; подать их значило бы проверять веб-путь на данных,
    которых он никогда не увидит.
    """
    from veg_recovery.providers.fixture import load_series

    series = load_series("AOI-0002", date(2024, 4, 1), date(2024, 6, 30), data_dir=DATA_DIR)
    forbidden = {"ndvi_zscore", "ndvi_climatology_mean", "status", "n_reference_years"}
    assert forbidden.isdisjoint(series.frame.columns)
    assert {"anon_polygon_id", "date", "primary_ndvi", "crop_type"} <= set(series.frame.columns)


@fixture_available
def test_fixture_dates_are_naive_calendar_days() -> None:
    from veg_recovery.providers.fixture import load_series

    series = load_series("AOI-0002", date(2024, 4, 1), date(2024, 6, 30), data_dir=DATA_DIR)
    dates = series.frame["date"]
    assert dates.dt.tz is None
    assert dates.equals(dates.dt.normalize())
    assert not series.frame.duplicated(["anon_polygon_id", "date"]).any()


@fixture_available
def test_fixture_reports_missing_polygon() -> None:
    from veg_recovery.providers.fixture import FixtureUnavailable, load_series

    with pytest.raises(FixtureUnavailable):
        load_series("AOI-НЕТ-ТАКОГО", date(2024, 1, 1), date(2024, 12, 31), data_dir=DATA_DIR)
