"""Contract-тест C-02 (BE-003): заглушка и загрузка бандла Разработчика 1.

Тест устроен в два слоя.

**Слой 1 — всегда.** Арифметика заглушки, классификация отказов бандла и правила
подмены проверяются без кода Разработчика 1: он опубликован в ветке `ML` и в `main`
ещё не смержен, а offline CI обязан оставаться зелёным (инвариант 13).

**Слой 2 — когда контракт доступен.** Настоящие `ReconstructionRequest`/`Result`,
`validate_request` и реальный bundle. Включается сам, как только `veg_recovery.contracts`
появится в окружении, а до тех пор — по переменной `ML_CONTRACT_SRC` с путём к `src`
ветки `ML`. Префикс `COSMO_` здесь намеренно не используется: `conftest.py` вычищает
все `COSMO_*`, чтобы окружение CI-раннера не меняло настройки приложения, и переменная
теста с таким префиксом исчезала бы до его тела:

    ML_CONTRACT_SRC=/path/to/ml/src uv run pytest -q tests/backend/test_contract_c02.py

Проверить полностью, включая реальный bundle, можно так:

    git worktree add --detach /tmp/ml 39a8f73
    ML_CONTRACT_SRC=/tmp/ml/src ML_CONTRACT_BUNDLE=/tmp/ml/artifacts/ml/baseline_v1/bundle \\
        uv run pytest -q tests/backend/test_contract_c02.py
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys

import numpy as np
import pandas as pd
import pytest

from veg_recovery.service import (
    MODEL_ERROR_CODE,
    STUB_MODEL_VERSION,
    ModelContractMissing,
    ModelStub,
    ModelUnavailable,
    build_reconstructor,
    classify_bundle_failure,
)
from veg_recovery.service.reconstructor import neighbour_context


def _frame(polygon: str = "AOI-TEST", values: list[float] | None = None) -> pd.DataFrame:
    """Суточный ряд одного полигона. `crop_type` обязателен: без него настоящий
    `validate_request` отвергает запрос, и заглушка обязана вести себя так же."""
    values = [0.30, 0.33, 0.36, np.nan, 0.42, 0.45, 0.48] if values is None else values
    return pd.DataFrame(
        {
            "anon_polygon_id": polygon,
            "date": pd.date_range("2024-05-01", periods=len(values), freq="D"),
            "primary_ndvi": values,
            "crop_type": "зерновые",
        }
    )


def _mask(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(frame["primary_ndvi"].isna().to_numpy(), index=frame.index)


# --------------------------------------------------------------------------- слой 1


def test_stub_version_is_recognisable() -> None:
    """По префиксу `stub-` и API, и red-team-проверка перед CP-3 отличают заглушку
    от настоящей модели. Без этого «выключен ли стаб в проде» непроверяемо."""
    assert STUB_MODEL_VERSION.startswith("stub-")
    assert ModelStub.model_version == STUB_MODEL_VERSION


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("Incompatible schema_version: '0.9'", "schema"),
        ("Incompatible feature_version: ndvi-context-v0", "schema"),
        ("Manifest file set does not match bundle kind", "schema"),
        ("Model file format does not match its role", "schema"),
        ("Trained bundle requires trusted=True after provenance review", "untrusted"),
        ("SHA256 mismatch: feature_state.json", "transport"),
        ("Manifest symlinks are forbidden", "transport"),
        ("Invalid bundle file: config.json", "transport"),
        ("что-то совсем новое", "unknown"),
    ],
)
def test_every_documented_bundle_failure_is_classified(message: str, kind: str) -> None:
    """Тексты взяты дословно из `models/bundle.py` Разработчика 1.

    Ключевой случай — `SHA256 mismatch`: это блокер B-DL-003, где Git переписал EOL
    уже захешированных файлов. Назвать порчу транспорта «несовместимой схемой»
    значит отправить владельца чинить не то.
    """
    assert classify_bundle_failure(ValueError(message)) == kind


def test_missing_bundle_is_not_a_schema_problem() -> None:
    assert classify_bundle_failure(FileNotFoundError("manifest.json")) == "missing"


def test_absent_bundle_falls_back_to_the_stub() -> None:
    handle = build_reconstructor(None)
    assert handle.is_stub is True
    assert handle.model_version == STUB_MODEL_VERSION
    assert handle.bundle_path is None


def test_stub_can_be_forbidden_explicitly() -> None:
    with pytest.raises(ModelUnavailable) as excinfo:
        build_reconstructor(None, allow_stub=False)
    assert excinfo.value.kind == "missing"
    assert excinfo.value.error_code == MODEL_ERROR_CODE


def test_stub_is_refused_in_production() -> None:
    """Red-team checklist перед CP-3: «ModelStub выключен в production».

    Проверяется и текст: у отказа своя причина `stub_forbidden`. Прежняя редакция
    переиспользовала `untrusted` и сообщала «trained bundle требует подтверждения
    происхождения» — правдоподобно и неверно, диагностика уходила не туда.
    """
    with pytest.raises(ModelUnavailable) as excinfo:
        build_reconstructor(None, allow_stub=True, environment="production")
    assert excinfo.value.kind == "stub_forbidden"
    assert "trained bundle" not in str(excinfo.value)


def test_broken_bundle_never_degrades_to_the_stub(tmp_path) -> None:
    """Заглушка заменяет отсутствующий bundle и никогда — сломанный.

    Молчаливый переход на заглушку при испорченной модели — это выдача заглушки
    за модель, ровно то, что запрещает инвариант 10 и red-team-проверка.
    """
    broken = tmp_path / "bundle"
    broken.mkdir()
    (broken / "manifest.json").write_text('{"schema_version": "0.0"}', encoding="utf-8")
    # Без кода Разработчика 1 сюда прилетает ModelContractMissing, с ним —
    # ModelUnavailable("schema"). Важно ровно одно: не тихий возврат заглушки.
    with pytest.raises((ModelUnavailable, ModelContractMissing)):
        build_reconstructor(broken, allow_stub=True)


def test_mean_of_two_neighbours() -> None:
    frame = _frame()
    values, left, right = neighbour_context(frame, _mask(frame))
    # Соседи 0.36 и 0.42 — ориентир ТЗ «mean two neighbors».
    assert values[3] == pytest.approx(0.39)
    assert left[3] == 1.0 and right[3] == 1.0


def test_one_sided_context_uses_the_single_neighbour() -> None:
    frame = _frame(values=[np.nan, 0.33, 0.36])
    values, left, right = neighbour_context(frame, _mask(frame))
    assert values[0] == pytest.approx(0.33)
    assert np.isnan(left[0]) and right[0] == 1.0


def test_masked_value_is_not_used_as_its_own_context() -> None:
    """В режиме `competition` скрытая строка физически содержит значение.
    Взять его — прямая утечка ответа, поэтому наблюдением она не считается.
    """
    frame = _frame(values=[0.30, 0.33, 0.36, 0.99, 0.42, 0.45, 0.48])
    mask = pd.Series([False, False, False, True, False, False, False], index=frame.index)
    values, _, _ = neighbour_context(frame, mask)
    assert values[3] == pytest.approx(0.39)


def test_context_never_crosses_polygons() -> None:
    """Ряд соседнего поля контекстом не является."""
    other = _frame(polygon="AOI-OTHER", values=[0.90, 0.91, 0.92])
    frame = pd.concat([_frame(), other], ignore_index=True)
    values, _, _ = neighbour_context(frame, _mask(frame))
    assert values[3] == pytest.approx(0.39)


def test_unsorted_input_is_ordered_by_date() -> None:
    """Фрейм приходит из провайдера или CSV и отсортированным не гарантирован."""
    frame = _frame().iloc[::-1].reset_index(drop=True)
    values, _, _ = neighbour_context(frame, _mask(frame))
    gap = int(np.flatnonzero(frame["primary_ndvi"].isna().to_numpy())[0])
    assert values[gap] == pytest.approx(0.39)


def test_distances_are_measured_in_days_not_rows() -> None:
    frame = pd.DataFrame(
        {
            "anon_polygon_id": "AOI-TEST",
            "date": pd.to_datetime(["2024-05-01", "2024-05-10", "2024-06-01"]),
            "primary_ndvi": [0.30, np.nan, 0.60],
            "crop_type": "зерновые",
        }
    )
    _, left, right = neighbour_context(frame, _mask(frame))
    assert left[1] == 9.0 and right[1] == 22.0


def test_polygon_without_observations_is_refused() -> None:
    """Отказ честнее выдуманного числа: инвариант «fallback не возвращает NaN»
    не означает «fallback придумывает значение из воздуха»."""
    frame = _frame(values=[np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="ни одного наблюдения"):
        neighbour_context(frame, _mask(frame))


# --------------------------------------------------------------------------- слой 2


@pytest.fixture(scope="module")
def contracts():
    """Настоящий C-02, если он доступен в окружении или указан `ML_CONTRACT_SRC`."""
    try:
        return importlib.import_module("veg_recovery.contracts")
    except ModuleNotFoundError:
        pass
    ml_src = os.environ.get("ML_CONTRACT_SRC")
    if not ml_src or not os.path.isdir(ml_src):
        pytest.skip("контракт C-02 недоступен: код ML не в main; задайте ML_CONTRACT_SRC")
    sys.path.insert(0, ml_src)
    # Наш `veg_recovery` — обычный пакет с __init__.py, у Разработчика 1 — namespace
    # без него, поэтому наш каталог затеняет его модули. extend_path объединяет оба
    # каталога в один пакет; это локальный приём теста, общий пакет мы не меняем.
    package = importlib.import_module("veg_recovery")
    package.__path__ = pkgutil.extend_path(package.__path__, package.__name__)
    importlib.invalidate_caches()
    return importlib.import_module("veg_recovery.contracts")


def test_stub_satisfies_the_reconstructor_protocol(contracts) -> None:
    assert isinstance(ModelStub(), contracts.NDVIReconstructor)


def test_stub_returns_the_full_published_schema(contracts) -> None:
    """8 полей предсказаний и 16 полей диагностики — ровно то, что публикует владелец."""
    frame = _frame()
    result = ModelStub().predict(
        contracts.ReconstructionRequest(frame, _mask(frame), "web")
    )
    assert list(result.predictions.columns) == list(contracts.PredictionRow.model_fields)
    assert list(result.diagnostics.columns) == list(contracts.DiagnosticRow.model_fields)
    assert result.model_version == STUB_MODEL_VERSION
    # Строгая валидация владельца: extra="forbid" и allow_inf_nan=False.
    for row in result.predictions.to_dict("records"):
        contracts.PredictionRow.model_validate(row)
    for row in result.diagnostics.to_dict("records"):
        contracts.DiagnosticRow.model_validate(row)


def test_stub_output_survives_the_json_boundary(contracts) -> None:
    """`ReconstructionPayload` — объявленная владельцем граница между воркером и API.
    DataFrame через HTTP не отдаём, своей DTO не заводим."""
    frame = _frame()
    result = ModelStub().predict(
        contracts.ReconstructionRequest(frame, _mask(frame), "web")
    )
    payload = contracts.ReconstructionPayload.from_result(result)
    assert payload.schema_version == "1.0"
    assert len(payload.predictions) == 1


def test_analysis_without_gaps_is_a_valid_result(contracts) -> None:
    """Маска целиком False — это «восстанавливать нечего», а не ошибка."""
    frame = _frame(values=[0.30, 0.33, 0.36])
    empty = pd.Series(False, index=frame.index)
    result = ModelStub().predict(contracts.ReconstructionRequest(frame, empty, "web"))
    assert len(result.predictions) == 0


@pytest.mark.parametrize(
    ("name", "broken"),
    [
        ("нет crop_type", lambda f: f.drop(columns=["crop_type"])),
        ("даты с таймзоной", lambda f: f.assign(date=f["date"].dt.tz_localize("UTC"))),
        ("дата со временем", lambda f: f.assign(date=f["date"] + pd.Timedelta(hours=3))),
    ],
)
def test_stub_rejects_what_the_real_model_rejects(contracts, name, broken) -> None:
    """Проверку входа выполняет `validate_request` владельца, а не наша копия,
    поэтому переход со стаба на bundle не вскрывает новых ошибок на границе."""
    frame = broken(_frame())
    with pytest.raises(ValueError):
        ModelStub().predict(
            contracts.ReconstructionRequest(frame, _mask(_frame()), "web")
        )


def test_mask_index_must_match_the_frame(contracts) -> None:
    frame = _frame()
    foreign = pd.Series(_mask(frame).to_numpy(), index=range(100, 100 + len(frame)))
    with pytest.raises(ValueError, match="exact frame index"):
        ModelStub().predict(contracts.ReconstructionRequest(frame, foreign, "web"))


def test_competition_mode_requires_the_synthetic_gap_flag(contracts) -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="is_synthetic_gap"):
        ModelStub().predict(
            contracts.ReconstructionRequest(frame, _mask(frame), "competition")
        )


def test_stub_and_real_bundle_agree_on_the_baseline(contracts) -> None:
    """Parity: пока опубликован baseline `mean_neighbors`, заглушка и настоящий bundle
    обязаны давать одно и то же значение — иначе подмена одного другим меняла бы ответ.

    Это не проверка ML-качества, а проверка того, что мы правильно поняли ориентир.
    """
    bundle = os.environ.get("ML_CONTRACT_BUNDLE")
    if not bundle or not os.path.isdir(bundle):
        pytest.skip("не задан ML_CONTRACT_BUNDLE с путём к bundle Разработчика 1")
    frame = _frame()
    mask = _mask(frame)
    handle = build_reconstructor(bundle)
    assert handle.is_stub is False
    real = handle.predict(contracts.ReconstructionRequest(frame, mask, "web"))
    stub = ModelStub().predict(contracts.ReconstructionRequest(frame, mask, "web"))
    assert real.predictions["primary_ndvi_pred"].to_numpy() == pytest.approx(
        stub.predictions["primary_ndvi_pred"].to_numpy()
    )
    assert real.predictions["method"].tolist() == stub.predictions["method"].tolist()
