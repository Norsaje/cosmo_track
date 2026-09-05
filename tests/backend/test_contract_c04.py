"""Contract-тест C-04: обученный bundle `p0-catboost-gpu-v1` из ветки `models`.

**Это исторический тест.** С BE-011R веб-путь обслуживает модель из каталога `model/`,
i `build_reconstructor` этот bundle больше не открывает. Тест проверяет обратную
совместимость: `load_ml_bundle`
обязан открыть bundle, отказать без явного доверия и вернуть полную схему C-02.
Ни одно утверждение отсюда не описывает работающую модель сервиса — её ограничения
проверяет `test_contract_model_run.py`.

Полное сравнение с эталоном на 3 112 гэпов помечено маркером `slow` — оно занимает
около 33 секунд и не место ему в каждом прогоне. Запуск:

    uv run pytest -q -m slow tests/backend/test_contract_c04.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veg_recovery.service import ModelUnavailable, load_ml_bundle

HANDOFF = Path(__file__).resolve().parents[2] / "artifacts/ml/ndvi_backend_handoff_v1"
BUNDLE = HANDOFF / "bundle"

pytestmark = pytest.mark.skipif(
    not (BUNDLE / "manifest.json").is_file(),
    reason="C-04 недоступен: нет artifacts/ml/ndvi_backend_handoff_v1/bundle",
)


@pytest.fixture(scope="module")
def contracts():
    return pytest.importorskip("veg_recovery.contracts", reason="код ML недоступен")


@pytest.fixture(scope="module")
def handoff() -> dict:
    return json.loads((HANDOFF / "handoff.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    """Trained-бандл грузится один раз на модуль — ровно как в воркере."""
    return load_ml_bundle(BUNDLE, trusted=True)


def test_trained_bundle_requires_explicit_trust() -> None:
    """`estimators.joblib` исполняет код при десериализации, поэтому доверие обязано
    быть осознанным. Проверяем, что путь по умолчанию — отказ, а не тихая загрузка."""
    with pytest.raises(ModelUnavailable) as excinfo:
        load_ml_bundle(BUNDLE, trusted=False)
    assert excinfo.value.kind == "untrusted"


def test_bundle_identity_matches_the_handoff(model, handoff) -> None:
    assert model.is_stub is False
    assert model.model_version == handoff["model_version"] == "p0-catboost-gpu-v1"
    manifest = model.reconstructor.bundle.manifest
    assert manifest.bundle_kind == "trained"
    assert manifest.schema_version == "1.0"
    assert manifest.feature_version == "ndvi-context-v1"


def test_declared_metrics_are_the_ones_we_may_quote(handoff) -> None:
    """Числа для демо берём из `handoff.json`, а не из чужих отчётов.

    Ансамбль лучше baseline по composite, но это **не** значит «лучше везде»:
    ограничение по temporal CV проверяется отдельно ниже.
    """
    metrics = handoff["metrics"]
    assert metrics["ensemble_composite_rmse"] < metrics["baseline_composite_rmse"]
    assert metrics["ensemble_composite_rmse"] == pytest.approx(0.09741183557217524)
    weights = metrics["ensemble_weights"]
    assert weights["baseline"] + weights["catboost"] == pytest.approx(1.0)


def test_temporal_transfer_is_worse_than_baseline(model) -> None:
    """ML предупреждает прямым текстом: «в temporal CV ансамбль хуже простого baseline».

    Проверяем это по `cv_summary` самого манифеста, чтобы предупреждение не осталось
    только словами: режим C — перенос на будущий сезон, и там ансамбль проигрывает.
    UI и демо не имеют права подавать модель как безусловно лучшую.
    """
    metrics = model.reconstructor.bundle.manifest.cv_summary["metrics"]
    by_mode = {(row["model"], row["mode"]): row["rmse"] for row in metrics}
    baseline_c = by_mode.get(("baseline", "C"))
    ensemble_c = by_mode.get(("ensemble", "C")) or by_mode.get(("catboost", "C"))
    assert baseline_c is not None and ensemble_c is not None
    assert ensemble_c > baseline_c, "ожидали худший перенос на будущий сезон"


def test_web_prediction_returns_the_full_schema(model, contracts) -> None:
    """Веб-путь: один полигон, `context_mode="web"`, полная схема на выходе."""
    dates = pd.date_range("2024-05-01", periods=11, freq="D")
    values = np.linspace(0.30, 0.60, 11)
    values[5] = np.nan
    frame = pd.DataFrame(
        {
            "anon_polygon_id": "AOI-0002",
            "date": dates,
            "primary_ndvi": values,
            "crop_type": "зерновые",
        }
    )
    mask = pd.Series(frame["primary_ndvi"].isna().to_numpy(), index=frame.index)
    result = model.predict(contracts.ReconstructionRequest(frame, mask, "web"))

    assert list(result.predictions.columns) == list(contracts.PredictionRow.model_fields)
    assert list(result.diagnostics.columns) == list(contracts.DiagnosticRow.model_fields)
    assert result.model_version == "p0-catboost-gpu-v1"
    for row in result.predictions.to_dict("records"):
        contracts.PredictionRow.model_validate(row)
    for row in result.diagnostics.to_dict("records"):
        contracts.DiagnosticRow.model_validate(row)
    # Обученная модель обязана называться иначе, чем baseline: по `method` мы отличаем,
    # что именно ответило, и без этого «cached/live» в UI не различить.
    assert result.predictions["method"].iloc[0] != "mean_neighbors"
    payload = contracts.ReconstructionPayload.from_result(result)
    assert payload.schema_version == "1.0"


def test_quality_flags_reach_the_consumer(model, contracts) -> None:
    """Backend должен сохранять diagnostics и показывать `quality_flags`, особенно
    для новых полигонов и слабого контекста. Полигон ниже модели незнаком,
    поэтому флаг обязан появиться — иначе показывать нечего."""
    dates = pd.date_range("2024-06-01", periods=7, freq="D")
    values = np.array([0.40, 0.42, np.nan, 0.46, 0.48, 0.50, 0.52])
    frame = pd.DataFrame(
        {
            "anon_polygon_id": "AOI-UNSEEN-XYZ",
            "date": dates,
            "primary_ndvi": values,
            "crop_type": "подсолнечник",
        }
    )
    mask = pd.Series(frame["primary_ndvi"].isna().to_numpy(), index=frame.index)
    result = model.predict(contracts.ReconstructionRequest(frame, mask, "web"))
    flags = result.diagnostics["quality_flags"].iloc[0]
    assert "unseen_polygon" in flags
    assert result.diagnostics["interval_status"].iloc[0] != "empirical_oof_subgroups_passed"


def test_prediction_depends_on_the_frame_composition(model, contracts) -> None:
    """**Ограничение интеграции, найденное измерением.**

    Один и тот же полигон, поданный отдельно и в составе полного test-фрейма, даёт
    разные предсказания: max_abs_delta около 0.125 против допуска эталона 1e-10.
    Значит feature builder использует контекст за пределами полигона, и «batch и web
    дают одинаковый ответ» верно только при одинаковом составе входа.

    Тест закрепляет факт, а не желаемое: пункт CP-4 «Batch/API prediction parity»
    выполним лишь при совпадающем фрейме, и формулировку нужно согласовать с ML.
    Если однажды это перестанет быть правдой — тест упадёт, и это будет хорошей новостью.
    """
    fixture = HANDOFF / "examples/test_data.csv"
    if not fixture.is_file():
        pytest.skip("нет examples/test_data.csv")
    from veg_recovery.data import read_dataset

    frame = read_dataset(fixture, kind="test", strict_current=True)
    mask = frame["is_synthetic_gap"].astype(bool)
    polygon = frame.loc[mask, "anon_polygon_id"].value_counts().index[0]

    alone = frame[frame["anon_polygon_id"] == polygon].reset_index(drop=True)
    alone_mask = alone["is_synthetic_gap"].astype(bool)
    part = model.predict(contracts.ReconstructionRequest(alone, alone_mask, "competition"))

    expected = pd.read_csv(HANDOFF / "examples/expected_submission.csv", parse_dates=["date"])
    merged = part.predictions[["anon_polygon_id", "date", "primary_ndvi_pred"]].merge(
        expected, on=["anon_polygon_id", "date"], suffixes=("_alone", "_batch")
    )
    assert not merged.empty
    delta = float(
        np.max(np.abs(merged["primary_ndvi_pred_alone"] - merged["primary_ndvi_pred_batch"]))
    )
    assert delta > 1e-10, (
        "предсказание перестало зависеть от состава фрейма — проверьте, "
        "не изменил ли ML feature builder, и снимите ограничение по parity"
    )


@pytest.mark.slow
def test_full_parity_with_the_published_expectation(model, contracts, handoff) -> None:
    """Полное воспроизведение эталона ML: 3 112 гэпов, допуск 1e-10.

    Это наша собственная проверка C-04 через `build_reconstructor`, а не запуск
    `verify_handoff.py` владельца: подтверждаем, что модель считает то же самое
    именно на нашем пути загрузки.
    """
    from veg_recovery.data import read_dataset

    frame = read_dataset(HANDOFF / "examples/test_data.csv", kind="test", strict_current=True)
    mask = frame["is_synthetic_gap"].astype(bool)
    assert int(mask.sum()) == handoff["input_fixture"]["synthetic_gaps"] == 3112

    result = model.predict(contracts.ReconstructionRequest(frame, mask, "competition"))
    expected = pd.read_csv(HANDOFF / "examples/expected_submission.csv", parse_dates=["date"])
    actual = result.predictions[["anon_polygon_id", "date", "primary_ndvi_pred"]].reset_index(
        drop=True
    )
    assert actual[["anon_polygon_id", "date"]].equals(expected[["anon_polygon_id", "date"]])
    got = actual["primary_ndvi_pred"].to_numpy()
    want = expected["primary_ndvi_pred"].to_numpy()
    delta = float(np.max(np.abs(got - want)))
    assert delta <= 1e-10, f"max_abs_delta={delta}"
