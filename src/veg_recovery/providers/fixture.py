"""Offline-источник наблюдений из конкурсных CSV (BE-008).

Это не провайдер спутниковых данных: он ничего не скачивает и не режет растры.
Его задача — дать сквозному пути реальный суточный ряд, пока живых провайдеров
нет (BE-007/BE-009), чтобы E2E проверялся на настоящих данных, а не на синтетике.

Читается только то, что разрешено на границе модели. `ndvi_zscore`,
`ndvi_climatology_*`, `n_reference_years` и `status` в запрос к модели **не**
попадают: на скрытой строке их нет, и подавать их значило бы обучать веб-путь на
том, чего в проде не будет. Здесь они остаются в наблюдениях для графика,
но в кадр реконструкции не переносятся.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

#: Колонки, которые обязан содержать кадр для `validate_request` владельца C-02.
REQUIRED_COLUMNS = ("anon_polygon_id", "date", "primary_ndvi", "crop_type")

#: Признаки, которые модель использует и которые есть в датасете. README ML:
#: «для качества предсказаний следует передавать доступные s2_*, landsat_*,
#: modis_*, era5_temp_c, era5_precip_mm».
FEATURE_COLUMNS = (
    "s2_ndvi",
    "s2_evi",
    "s2_ndwi",
    "landsat_ndvi",
    "landsat_evi",
    "landsat_ndwi",
    "modis_ndvi",
    "modis_evi",
    "era5_temp_c",
    "era5_precip_mm",
)

#: Контекст для графика: климатология нужна UI как нейтральная линия, но модели
#: она не передаётся — на реальной gap-строке её не существует.
CONTEXT_COLUMNS = ("ndvi_climatology_mean", "ndvi_climatology_std", "n_reference_years")


class FixtureUnavailable(RuntimeError):
    """Ряд для полигона не найден. Отдельный тип, чтобы воркер отличил это от сбоя."""


@dataclass(frozen=True)
class FixtureSeries:
    """Суточный ряд одного полигона за период."""

    #: Кадр для модели: ключи, target, crop_type и разрешённые признаки.
    frame: pd.DataFrame
    #: Полный кадр с контекстом для графика и таблицы observations.
    context: pd.DataFrame
    source_path: str
    #: Сколько строк реально наблюдалось (конечный primary_ndvi).
    observed_rows: int


@lru_cache(maxsize=4)
def _load(path: str) -> pd.DataFrame:
    """CSV читается один раз на процесс: train — сто тысяч строк, и перечитывать
    его на каждый анализ значит тратить секунды на каждой джобе."""
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["anon_polygon_id"] = frame["anon_polygon_id"].astype(str)
    return frame


def available_datasets(data_dir: str | Path = "data") -> list[Path]:
    root = Path(data_dir)
    return [path for path in (root / "train_dataset.csv", root / "test_data.csv") if path.is_file()]


@dataclass(frozen=True)
class SeriesDescriptor:
    """Что offline-источник знает о полигоне до того, как его запросили."""

    anon_polygon_id: str
    observations: int
    first_date: date
    last_date: date
    crop_type: str | None
    dataset: str


@lru_cache(maxsize=2)
def list_available_series(data_dir: str = "data") -> tuple[SeriesDescriptor, ...]:
    """Перечислить ряды, доступные offline-источнику.

    Нужен интерфейсу: нарисованный на карте контур сам по себе не имеет данных —
    геометрий в конкурсных CSV нет, сопоставить полигон с рядом автоматически
    невозможно. Пока живых провайдеров нет (BE-007/BE-009), выбор ряда делает
    человек, и он должен видеть, из чего выбирать.

    Полигоны из train идут раньше: у них на порядок больше наблюдений (медиана
    767 против 72), и демо на них выглядит осмысленнее.
    """
    seen: dict[str, SeriesDescriptor] = {}
    for path in available_datasets(data_dir):
        table = _load(str(path))
        values = pd.to_numeric(table["primary_ndvi"], errors="coerce")
        observed = table[values.notna()]
        if observed.empty:
            continue
        grouped = observed.groupby("anon_polygon_id")
        for polygon_id, group in grouped:
            if polygon_id in seen:
                # Полигон встречается и в train, и в test. Оставляем первую находку:
                # порядок файлов задаёт `available_datasets`, train идёт первым.
                continue
            crop = group["crop_type"].dropna()
            seen[polygon_id] = SeriesDescriptor(
                anon_polygon_id=str(polygon_id),
                observations=int(len(group)),
                first_date=group["date"].min().date(),
                last_date=group["date"].max().date(),
                crop_type=str(crop.iloc[0]) if not crop.empty else None,
                dataset=path.name,
            )
    return tuple(sorted(seen.values(), key=lambda item: (-item.observations, item.anon_polygon_id)))


def load_series(
    anon_polygon_id: str,
    date_from: date,
    date_to: date,
    *,
    data_dir: str | Path = "data",
) -> FixtureSeries:
    """Собрать ряд полигона за период из конкурсных CSV.

    Полигон ищется в обоих файлах: часть полигонов есть только в test. Если он
    нашёлся в обоих, берётся train — там больше видимых значений, а строки test
    для того же ключа не добавляют наблюдений, только гэпы.
    """
    datasets = available_datasets(data_dir)
    if not datasets:
        raise FixtureUnavailable(f"нет конкурсных CSV в каталоге {data_dir}")

    for path in datasets:
        table = _load(str(path))
        subset = table[table["anon_polygon_id"] == anon_polygon_id]
        if subset.empty:
            continue

        mask = (subset["date"].dt.date >= date_from) & (subset["date"].dt.date <= date_to)
        window = subset.loc[mask].sort_values("date").reset_index(drop=True)
        if window.empty:
            raise FixtureUnavailable(
                f"у полигона {anon_polygon_id} нет строк в диапазоне {date_from}…{date_to}"
            )

        # Дата обязана быть tz-naive календарным днём: `validate_request` владельца
        # C-02 отвергает и таймзону, и время внутри дня.
        window["date"] = pd.to_datetime(window["date"]).dt.normalize()

        columns = [column for column in REQUIRED_COLUMNS if column in window.columns]
        columns += [column for column in FEATURE_COLUMNS if column in window.columns]
        frame = window[columns].copy()
        if "crop_type" in frame.columns:
            # crop_type обязателен и не должен быть NaN: модель использует его как
            # категорию, а пустое значение ломает валидацию входа.
            frame["crop_type"] = frame["crop_type"].fillna("unknown").astype(str)
        else:
            frame["crop_type"] = "unknown"

        context_columns = ["date", *(c for c in CONTEXT_COLUMNS if c in window.columns)]
        observed = int(pd.to_numeric(window["primary_ndvi"], errors="coerce").notna().sum())
        return FixtureSeries(
            frame=frame,
            context=window[context_columns].copy(),
            source_path=str(path),
            observed_rows=observed,
        )

    raise FixtureUnavailable(f"полигон {anon_polygon_id} не найден в конкурсных CSV")
