"""Оркестрация одного анализа (контракт C-08).

Модуль намеренно **не знает** ни о FastAPI, ни о БД, ни о Celery: по §8.6 ТЗ этим
управляет воркер. Сюда приходит готовый суточный кадр, отсюда уходит собранный ряд
и диагностика. Такую логику можно проверить без PostgreSQL и без брокера.

Модель здесь не дублируется и не «доводится»: вызывается один общий
`NDVIReconstructor` (инвариант 6, решение D-002), результат сохраняется как есть.
Переписывать ML-логику в сервисном слое запрещено.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from veg_recovery.service.reconstructor import LoadedReconstructor, contract_module


@dataclass(frozen=True)
class SeriesPointRecord:
    """Одна суточная точка итогового ряда — то, что ляжет в `reconstructions`."""

    date: pd.Timestamp
    primary_ndvi_raw: float | None
    primary_ndvi_reconstructed: float | None
    ndvi_harmonized: float | None
    is_observed: bool
    is_reconstructed: bool
    lower: float | None = None
    upper: float | None = None
    method: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Климатическая норма даты. Модели не передаётся (на скрытой строке её нет),
    #: но UI без неё не может показать, насколько значение отклонилось от обычного.
    climatology_mean: float | None = None
    climatology_std: float | None = None


@dataclass(frozen=True)
class AnalysisOutcome:
    """Результат анализа: ряд, версия модели и честные предупреждения."""

    points: list[SeriesPointRecord]
    model_version: str
    reconstructed_count: int
    observed_count: int
    #: Диагностические предупреждения. Пустой список не означает «всё хорошо» —
    #: он означает, что предупреждений не было; отсутствие данных фиксируется
    #: отдельным предупреждением, а не молчанием.
    warnings: list[str] = field(default_factory=list)


def build_gap_mask(frame: pd.DataFrame) -> pd.Series:
    """Какие строки восстанавливать: там, где `primary_ndvi` не конечен.

    Маска обязана быть `pd.Series` **с тем же индексом**, что и кадр, и строго
    булевой: `validate_request` владельца контракта отвергает numpy-массив,
    чужой индекс и любой не-bool dtype, и делает это до вызова модели.
    """
    values = pd.to_numeric(frame["primary_ndvi"], errors="coerce").to_numpy(dtype=float)
    return pd.Series(~np.isfinite(values), index=frame.index, dtype=bool)


def run_reconstruction(
    frame: pd.DataFrame,
    reconstructor: LoadedReconstructor,
    context: pd.DataFrame | None = None,
) -> AnalysisOutcome:
    """Восстановить пропуски суточного ряда и собрать итоговую серию.

    Веб-путь всегда передаёт `context_mode="web"`: режим `competition` требует
    колонку `is_synthetic_gap`, точно равную маске, и предназначен для batch.
    """
    contracts = contract_module()
    warnings: list[str] = []

    work = frame.reset_index(drop=True).copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    raw = pd.to_numeric(work["primary_ndvi"], errors="coerce").to_numpy(dtype=float)
    observed = np.isfinite(raw)
    mask = build_gap_mask(work)

    predictions: dict[pd.Timestamp, dict[str, Any]] = {}
    diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
    model_version = reconstructor.model_version

    if int(mask.sum()) == 0:
        # Валидное состояние: восстанавливать нечего. Это не ошибка и не «нет данных».
        warnings.append("NO_GAPS: в запрошенном периоде нет пропусков для восстановления")
    elif not observed.any():
        # Ни одного наблюдения — восстанавливать не из чего. Придумывать значения
        # в такой ситуации запрещено: пустой ряд честнее выдуманного.
        warnings.append("NO_OBSERVATIONS: в периоде нет ни одного наблюдения, ряд не восстановлен")
    else:
        request = contracts.ReconstructionRequest(work, mask, "web")
        result = reconstructor.predict(request)
        model_version = result.model_version
        for row in result.predictions.to_dict("records"):
            predictions[pd.Timestamp(row["date"]).normalize()] = row
        for row in result.diagnostics.to_dict("records"):
            diagnostics[pd.Timestamp(row["date"]).normalize()] = row

        # Модель строит признаки из своего набора и точку вне него восстановить не
        # может. Молча вернуть на графике разрыв — значит выдать пробел за отсутствие
        # данных, хотя причина другая и она известна.
        unresolved = int(mask.sum()) - len(predictions)
        if unresolved > 0:
            warnings.append(
                f"POINTS_OUTSIDE_MODEL_DATASET: {unresolved} из {int(mask.sum())} "
                "пропусков не восстановлены — этих точек нет в наборе модели"
            )

    # Климатология берётся из отдельного кадра: в кадр модели она не входит
    # намеренно, но графику нужна. Индексируется по дате, а не по позиции —
    # контекст может прийти в другом порядке или с иным набором строк.
    climatology: dict[pd.Timestamp, tuple[float | None, float | None]] = {}
    if context is not None and not context.empty and "date" in context.columns:
        for row in context.to_dict("records"):
            moment = pd.Timestamp(row["date"]).normalize()
            climatology[moment] = (
                _finite(row.get("ndvi_climatology_mean")),
                _finite(row.get("ndvi_climatology_std")),
            )

    points: list[SeriesPointRecord] = []
    for position in range(len(work)):
        moment = pd.Timestamp(work.loc[position, "date"]).normalize()
        prediction = predictions.get(moment)
        diagnostic = diagnostics.get(moment, {})
        is_observed = bool(observed[position])
        # Требование DL из consumer review: `is_reconstructed` не равно отрицанию
        # `is_observed`. Строка без наблюдения, которую модель не восстановила
        # (её не было в запросе или модель не вызывалась), остаётся просто пустой.
        is_reconstructed = prediction is not None
        points.append(
            SeriesPointRecord(
                date=moment,
                primary_ndvi_raw=float(raw[position]) if is_observed else None,
                primary_ndvi_reconstructed=(
                    float(prediction["primary_ndvi_reconstructed"]) if is_reconstructed else None
                ),
                ndvi_harmonized=(
                    float(prediction["ndvi_harmonized"]) if is_reconstructed else None
                ),
                is_observed=is_observed,
                is_reconstructed=is_reconstructed,
                lower=float(prediction["lower"]) if is_reconstructed else None,
                upper=float(prediction["upper"]) if is_reconstructed else None,
                method=str(prediction["method"]) if is_reconstructed else None,
                diagnostics=_clean_diagnostics(diagnostic),
                climatology_mean=climatology.get(moment, (None, None))[0],
                climatology_std=climatology.get(moment, (None, None))[1],
            )
        )

    if reconstructor.is_stub:
        # Заглушка обязана быть видна снаружи: «результат получен заглушкой» —
        # это не деталь реализации, а свойство ответа (red-team перед CP-3).
        warnings.append("MODEL_STUB_USED: ответ получен заглушкой, а не обученной моделью")

    return AnalysisOutcome(
        points=points,
        model_version=model_version,
        reconstructed_count=len(predictions),
        observed_count=int(observed.sum()),
        warnings=warnings,
    )


def _finite(value: Any) -> float | None:
    """Число или None. NaN климатологии — это «нормы нет», а не ноль."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _clean_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    """Диагностика в JSON-безопасном виде.

    NaN превращается в `None`, а не в ноль: расстояние до контекста, которого нет,
    и расстояние в ноль дней — это разные вещи, и владелец контракта отдаёт здесь
    именно null.
    """
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if key in ("anon_polygon_id", "date"):
            continue
        if isinstance(value, float) and not np.isfinite(value):
            cleaned[key] = None
        elif isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned
