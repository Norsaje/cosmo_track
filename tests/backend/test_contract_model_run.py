"""Contract-тест работающей модели: поставка `model/` через интерфейс C-02 (BE-011R).

Проверяется наш путь, а не качество ML: сервис обязан открыть поставку, отказаться
от недоверенного pickle, вернуть полную схему C-02 и **совпасть с пакетным
submission поставки** на тех же ключах. Последнее — красный пункт перед CP-4
(«Batch и worker predictions совпадают»): расхождение здесь означает, что демо
показывает не то, что отправлено организаторам.

Поставка целиком (данные + смесь) весит 49 МБ и в offline CI может отсутствовать,
поэтому весь модуль пропускается, если каталога нет.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veg_recovery.service import ModelUnavailable, build_reconstructor

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
RUN = MODEL / "runs/local"
SUBMISSION = RUN / "submission.csv"

pytestmark = pytest.mark.skipif(
    not (RUN / "artifacts/model_bundle.pkl").is_file(),
    reason="нет поставки модели в model/",
)


@pytest.fixture(scope="module")
def contracts():
    return pytest.importorskip("veg_recovery.contracts", reason="код ML недоступен")


@pytest.fixture(scope="module")
def model():
    return build_reconstructor(MODEL, trusted=True)


@pytest.fixture(scope="module")
def submission() -> pd.DataFrame:
    frame = pd.read_csv(SUBMISSION, parse_dates=["date"])
    frame["anon_polygon_id"] = frame["anon_polygon_id"].astype(str)
    return frame


def _polygon_frame(polygon: str) -> pd.DataFrame:
    """Ряд одного полигона в том виде, в каком его отдаёт offline-источник."""
    from veg_recovery.providers.fixture import load_series

    series = load_series(
        polygon, pd.Timestamp("1900-01-01").date(), pd.Timestamp("2100-01-01").date(),
        data_dir=MODEL / "data",
    )
    return series.frame


def _gap_mask(frame: pd.DataFrame, keys: set[tuple[str, pd.Timestamp]]) -> pd.Series:
    marked = [
        (str(polygon), pd.Timestamp(moment).normalize()) in keys
        for polygon, moment in zip(frame["anon_polygon_id"], frame["date"], strict=True)
    ]
    return pd.Series(marked, index=frame.index, dtype=bool)


def test_model_version_names_the_run(model) -> None:
    """Версия обязана отличать эту модель от прежнего бандла ML.

    В БД, в API и на экране лежит одна строка версии; если она не меняется при
    смене модели, провенанс перестаёт что-либо значить.
    """
    assert model.is_stub is False
    assert model.model_version.startswith("ndvi-blend-")
    assert model.model_version != "p0-catboost-gpu-v1"
    config = json.loads((RUN / "artifacts/run_config.json").read_text(encoding="utf-8"))
    assert config["feature_version"] in model.model_version
    assert config["source_hash"][:8] in model.model_version


def test_pickle_requires_explicit_trust() -> None:
    """Pickle исполняет код при загрузке, поэтому доверие включается осознанно."""
    with pytest.raises(ModelUnavailable) as excinfo:
        build_reconstructor(MODEL, trusted=False)
    assert excinfo.value.kind == "untrusted"
    assert excinfo.value.error_code == "MODEL_SCHEMA_MISMATCH"


def test_web_prediction_returns_the_full_schema(model, contracts, submission) -> None:
    """Обе таблицы C-02 приходят целиком и проходят валидацию владельца контракта.

    Проверяется не наличие колонок, а то, что каждая строка собирается в
    `PredictionRow`/`DiagnosticRow`: у них `extra="forbid"` и `allow_inf_nan=False`,
    то есть NaN или лишнее поле здесь падает, а не доезжает до БД.
    """
    polygon = str(submission["anon_polygon_id"].iloc[0])
    frame = _polygon_frame(polygon)
    keys = {
        (polygon, pd.Timestamp(moment).normalize())
        for moment in submission.loc[submission["anon_polygon_id"] == polygon, "date"]
    }
    mask = _gap_mask(frame, keys)
    result = model.predict(contracts.ReconstructionRequest(frame, mask, "web"))

    assert len(result.predictions) == int(mask.sum())
    assert len(result.diagnostics) == int(mask.sum())
    for row in result.predictions.to_dict("records"):
        contracts.PredictionRow(**row)
    for row in result.diagnostics.to_dict("records"):
        contracts.DiagnosticRow(**{
            key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in row.items()
        })


def test_web_predictions_match_the_batch_submission(model, contracts, submission) -> None:
    """Главный пункт: на тех же ключах веб-путь даёт ровно пакетный ответ.

    Скрываются именно контрольные пропуски — тот же набор запросов, что у
    `python -m ndvi.inference`. Их динамические каналы в наборе и так пусты,
    поэтому контекст обеих сторон совпадает и расхождению взяться неоткуда.

    Допуск 1e-10 — это не «примерно совпало», а точность самого файла: поставка
    пишет submission с `float_format="%.10f"`, то есть округляет до десятого
    знака (`ndvi/pipeline.py:validate_submission`). Арифметика при этом одна и
    та же; сверка байт в байт делается отдельно, прогоном самой поставки.
    """
    polygon = str(submission["anon_polygon_id"].value_counts().index[0])
    expected = submission[submission["anon_polygon_id"] == polygon]
    frame = _polygon_frame(polygon)
    keys = {(polygon, pd.Timestamp(moment).normalize()) for moment in expected["date"]}
    mask = _gap_mask(frame, keys)
    assert int(mask.sum()) == len(expected)

    result = model.predict(contracts.ReconstructionRequest(frame, mask, "web"))
    merged = result.predictions.merge(
        expected, on=["anon_polygon_id", "date"], how="inner", validate="one_to_one"
    )
    assert len(merged) == len(expected)
    difference = (merged["primary_ndvi_pred_x"] - merged["primary_ndvi_pred_y"]).abs()
    assert float(difference.max()) < 1e-10


def test_source_probabilities_are_a_distribution(model, contracts, submission) -> None:
    """Сумма вероятностей источника равна единице, как требует C-05.

    Значения берутся у классификатора самой смеси; выдумывать распределение,
    в котором «примерно единица», нельзя — на нём считается source_confidence.
    """
    polygon = str(submission["anon_polygon_id"].iloc[0])
    frame = _polygon_frame(polygon)
    keys = {
        (polygon, pd.Timestamp(moment).normalize())
        for moment in submission.loc[submission["anon_polygon_id"] == polygon, "date"]
    }
    result = model.predict(
        contracts.ReconstructionRequest(frame, _gap_mask(frame, keys), "web")
    )
    columns = ["p_s2", "p_landsat", "p_modis", "p_unknown"]
    total = result.diagnostics[columns].sum(axis=1).to_numpy(dtype=float)
    assert np.allclose(total, 1.0, atol=1e-6)
    assert (result.diagnostics["source_confidence"].to_numpy() <= 1.0).all()


def test_interval_is_the_measured_one_and_says_it_is_not_certified(
    model, contracts, submission
) -> None:
    """Лента — эмпирический q90 модуля ошибки, а не доверительный интервал.

    Покрытие на audit 89.25 %, гарантий для зависимых рядов нет. Подписать её
    процентом уверенности значит заявить точность, которой никто не измерял.
    """
    report = json.loads((RUN / "reports/uncertainty.json").read_text(encoding="utf-8"))
    half_width = float(report["development_absolute_error_q90"])
    polygon = str(submission["anon_polygon_id"].iloc[0])
    frame = _polygon_frame(polygon)
    keys = {
        (polygon, pd.Timestamp(moment).normalize())
        for moment in submission.loc[submission["anon_polygon_id"] == polygon, "date"]
    }
    result = model.predict(
        contracts.ReconstructionRequest(frame, _gap_mask(frame, keys), "web")
    )
    predictions, diagnostics = result.predictions, result.diagnostics
    width = (predictions["upper"] - predictions["lower"]).to_numpy(dtype=float)
    assert np.allclose(width, 2 * half_width)
    assert (diagnostics["interval_level"] == 0.90).all()
    assert "conformal" not in diagnostics["interval_status"].iloc[0].replace("not_conformal", "")
    assert all(
        "interval_empirical_not_conformal" in flags for flags in diagnostics["quality_flags"]
    )


def test_harmonization_is_not_claimed(model, contracts, submission) -> None:
    """Модель восстанавливает сырой ряд и гармонизацию не выполняет.

    Колонка `ndvi_harmonized` заполняется значением как есть, а статус говорит
    об этом прямо: выдать сырой ряд за гармонизированный — подмена шкалы
    (инвариант 4 модельного контракта).
    """
    polygon = str(submission["anon_polygon_id"].iloc[0])
    frame = _polygon_frame(polygon)
    keys = {
        (polygon, pd.Timestamp(moment).normalize())
        for moment in submission.loc[submission["anon_polygon_id"] == polygon, "date"]
    }
    result = model.predict(
        contracts.ReconstructionRequest(frame, _gap_mask(frame, keys), "web")
    )
    assert (result.diagnostics["harmonization_status"] == "not_applied").all()
    assert all(
        "harmonization_not_applied" in flags for flags in result.diagnostics["quality_flags"]
    )
    assert result.predictions["ndvi_harmonized"].equals(
        result.predictions["primary_ndvi_reconstructed"]
    )


def test_context_distances_are_real_days_or_null(model, contracts, submission) -> None:
    """Расстояния до контекста приходят из признаков модели, а не из нуля.

    Ноль дней означал бы «наблюдение в тот же день» — прямо противоположное
    отсутствию контекста, поэтому отсутствие остаётся NaN и превращается в null.
    """
    polygon = str(submission["anon_polygon_id"].iloc[0])
    frame = _polygon_frame(polygon)
    keys = {
        (polygon, pd.Timestamp(moment).normalize())
        for moment in submission.loc[submission["anon_polygon_id"] == polygon, "date"]
    }
    result = model.predict(
        contracts.ReconstructionRequest(frame, _gap_mask(frame, keys), "web")
    )
    left = result.diagnostics["left_distance_days"].to_numpy(dtype=float)
    right = result.diagnostics["right_distance_days"].to_numpy(dtype=float)
    finite = np.isfinite(left)
    assert finite.any()
    assert (left[finite] > 0).all()
    assert (right[np.isfinite(right)] > 0).all()


def test_points_outside_the_dataset_are_not_invented(model, contracts) -> None:
    """Полигона нет в наборе — значит предсказания нет.

    Признаки строятся из сезонной нормы полигона, синхронных рядов и наблюдений
    других AOI. Для незнакомого ключа этого контекста не существует, и любое
    значение здесь было бы выдумкой.
    """
    frame = pd.DataFrame(
        {
            "anon_polygon_id": "AOI-НЕТ-В-НАБОРЕ",
            "date": pd.date_range("2024-05-01", periods=5, freq="D"),
            "primary_ndvi": [0.3, np.nan, 0.35, np.nan, 0.4],
            "crop_type": "зерновые",
        }
    )
    mask = pd.Series(frame["primary_ndvi"].isna().to_numpy(), index=frame.index)
    result = model.predict(contracts.ReconstructionRequest(frame, mask, "web"))
    assert result.predictions.empty
    assert result.diagnostics.empty
    assert result.model_version == model.model_version


def test_natural_gaps_are_reconstructed_too(model, contracts, submission) -> None:
    """Продуктовый путь восстанавливает не только конкурсные пропуски.

    Пользователь смотрит на ряд поля, а не на контрольные ключи: пропуск без
    наблюдения обязан заполняться так же. Совпадения с пакетным submission здесь
    не требуется и не заявляется — набор запросов другой, а вместе с ним другой
    и скрытый контекст.
    """
    polygon = str(submission["anon_polygon_id"].iloc[0])
    frame = _polygon_frame(polygon)
    frame = frame[frame["date"].dt.year == 2024].reset_index(drop=True)
    mask = pd.Series(frame["primary_ndvi"].isna().to_numpy(), index=frame.index, dtype=bool)
    control = set(
        pd.Timestamp(moment).normalize()
        for moment in submission.loc[submission["anon_polygon_id"] == polygon, "date"]
    )
    natural = int(mask.sum()) - sum(1 for moment in frame.loc[mask, "date"] if moment in control)
    assert natural > 0

    result = model.predict(contracts.ReconstructionRequest(frame, mask, "web"))
    assert len(result.predictions) == int(mask.sum())
    assert np.isfinite(result.predictions["primary_ndvi_pred"].to_numpy(dtype=float)).all()
