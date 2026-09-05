"""Адаптер C-02 поверх поставленного пакета `model/` (замена модели, BE-011R).

Что здесь происходит. Сервис перешёл с бандла C-04 `p0-catboost-gpu-v1` на пакет,
лежащий в каталоге `model/`: смесь LightGBM, LightGBM-без-донорских-признаков и
трёх спутниковых экспертов CatBoost, 324 признака, обучение на новом test
(49 190 строк, 20 полигонов, 2 323 контрольных пропуска). Никакой ML-логики
здесь нет и быть не может: предсказание целиком считает `ndvi.inference.
predict_queries` из поставки, а этот модуль только переводит запрос C-02 в её
интерфейс и обратно.

Почему пакет не копируется в `src/`. Поставка самодостаточна и хеширована
(`model/PACKAGE_MANIFEST.json`), а `ndvi.data.read_data` при каждом старте
сверяет SHA256 обоих CSV. Копия немедленно рассинхронизировалась бы с этими
хешами, и проверка целостности превратилась бы в проверку копии. Поэтому
каталог монтируется как есть, а импорт идёт по пути (см. `_ndvi_modules`).

Чем этот путь отличается от прежнего бандла:

* модель нельзя вызвать «на произвольном кадре». Признаки строятся из **всего**
  набора: сезонная норма полигона, наблюдения других AOI на ту же дату и
  донорские ряды. Поэтому набор грузится один раз на процесс, а запрос C-02
  сопоставляется с ним по ключу `(anon_polygon_id, date)`;
* точка, которой в наборе нет, не восстанавливается. Придумать ей контекст
  нельзя, а вернуть значение «примерно такое» — это выдумка, а не прогноз;
* `predict_queries` сам скрывает запрошенные строки во всех динамических
  каналах контекста (`hide`) и отказывается работать, если скрытие не сработало.
  Это его встроенный барьер от утечки, и мы его не обходим.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from veg_recovery.service.reconstructor import (
    BundleFailureKind,
    ModelUnavailable,
    contract_module,
)

#: Файлы поставки, без которых модель не работает. Пути относительны корню пакета.
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test_features.csv"
BUNDLE_FILE = "artifacts/model_bundle.pkl"
RUN_CONFIG_FILE = "artifacts/run_config.json"
UNCERTAINTY_FILE = "reports/uncertainty.json"

#: Метод восстановления в терминах C-02. Не `oof_ensemble` прежнего владельца:
#: это другая смесь, и называть её его именем значит путать провенанс.
METHOD = "weighted_blend"

#: Интервал эмпирический: q90 модуля ошибки на development, покрытие на audit
#: 89.25 %. Это не conformal-интервал для зависимых рядов, и статус говорит об
#: этом прямо — подписывать ленту процентом уверенности запрещено.
INTERVAL_LEVEL = 0.90
INTERVAL_STATUS = "empirical_q90_not_conformal"
#: Запасная полуширина на случай, если отчёт о неопределённости не доехал.
#: Значение из результатов поставки; расхождение с файлом помечается флагом.
FALLBACK_INTERVAL_HALF_WIDTH = 0.07804636720864672

#: Гармонизацию эта модель не выполняет: она восстанавливает сырой конкурсный
#: ряд. Колонка `ndvi_harmonized` заполняется значением как есть, а статус
#: честно говорит, что преобразования не было (инвариант 4).
HARMONIZATION_STATUS = "not_applied"

#: Порядок классов классификатора источника внутри поставки: 0=s2, 1=landsat, 2=modis.
SENSOR_ORDER = ("s2", "landsat", "modis")


def _ndvi_modules(package: Path) -> SimpleNamespace:
    """Импортировать пакет поставки по пути, не устанавливая его.

    `sys.path` правится один раз и только на корень поставки. Пакет называется
    `ndvi` — имя общее, поэтому корень добавляется в начало пути осознанно:
    другой `ndvi` в окружении означал бы, что мы считаем чужой моделью.
    """
    root = str(package.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from ndvi import data as ndvi_data
        from ndvi import features as ndvi_features
        from ndvi import inference as ndvi_inference
        from ndvi import models as ndvi_models
        from ndvi import pipeline as ndvi_pipeline
    except ModuleNotFoundError as exc:
        raise ModelUnavailable(
            "missing", f"пакет модели не найден в {root}: {exc}"
        ) from exc
    return SimpleNamespace(
        data=ndvi_data,
        features=ndvi_features,
        inference=ndvi_inference,
        models=ndvi_models,
        pipeline=ndvi_pipeline,
    )


def _interval_half_width(run_dir: Path) -> tuple[float, bool]:
    """Полуширина эмпирического интервала и признак того, что она из отчёта."""
    path = run_dir / UNCERTAINTY_FILE
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        value = float(report["development_absolute_error_q90"])
    except (OSError, ValueError, KeyError, TypeError):
        return FALLBACK_INTERVAL_HALF_WIDTH, False
    if not np.isfinite(value) or value <= 0:
        return FALLBACK_INTERVAL_HALF_WIDTH, False
    return value, True


@dataclass(frozen=True)
class _Sources:
    """Вероятности источника скрытой точки и уверенность классификатора."""

    probabilities: np.ndarray  # (n, 3) в порядке SENSOR_ORDER
    confidence: np.ndarray  # (n,)


class NdviRunReconstructor:
    """Реализация протокола `NDVIReconstructor` поверх обученного run поставки.

    Набор и веса читаются один раз на процесс: `read_data` сверяет SHA256 обоих
    CSV, а pickle смеси весит 17 МБ. Делать это на каждый запрос значит тратить
    секунду на то, что не менялось.
    """

    def __init__(self, package_path: str | Path, run_name: str, *, trusted: bool) -> None:
        package = Path(package_path)
        run_dir = package / "runs" / run_name
        if not (run_dir / BUNDLE_FILE).is_file():
            raise ModelUnavailable(
                "missing", f"нет обученного запуска {run_name} в {package}"
            )
        if not trusted:
            # Pickle исполняет код при загрузке. SHA256 из PACKAGE_MANIFEST.json
            # ловит порчу файла, но не подлог автора, поэтому доверие включается
            # осознанно, а не «чтобы заработало» (то же правило, что и у C-04).
            raise ModelUnavailable(
                "untrusted",
                "pickle обученной смеси загружается только после проверки "
                "происхождения: включите COSMO_MODEL_BUNDLE_TRUSTED",
            )

        self._package = package
        self._run_dir = run_dir
        self._ndvi = _ndvi_modules(package)

        try:
            self._df = self._ndvi.data.read_data(
                package / TRAIN_CSV, package / TEST_CSV, strict=True
            )
            self._bundle = self._ndvi.pipeline.load(run_dir / BUNDLE_FILE)
        except FileNotFoundError as exc:
            raise ModelUnavailable("missing", str(exc)) from exc
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelUnavailable(_classify(exc), str(exc)) from exc

        # Индекс ключей строится один раз: сопоставление запроса с набором идёт
        # по (полигон, дата), а не по позиции — кадр приходит из БД и из CSV
        # в разном порядке.
        keys = self._df[["anon_polygon_id", "date"]]
        self._row_by_key = {
            (str(polygon), pd.Timestamp(moment).normalize()): int(row_id)
            for polygon, moment, row_id in zip(
                keys["anon_polygon_id"], keys["date"], self._df["row_id"], strict=True
            )
        }

        config = _read_json(run_dir / RUN_CONFIG_FILE)
        feature_version = str(config.get("feature_version", self._bundle.get("version", "?")))
        source_hash = str(config.get("source_hash", ""))[:8]
        #: Версия попадает в БД, в API и на экран: по ней отличают этот прогон
        #: от прежнего бандла ML и от будущего Kaggle-запуска.
        self.model_version = f"ndvi-blend-{feature_version}" + (
            f"+{source_hash}" if source_hash else ""
        )
        self._half_width, self._half_width_measured = _interval_half_width(run_dir)
        self._weights = {
            name: float(weight)
            for name, weight in self._bundle.get("weights", {}).items()
            if float(weight) > 0
        }

    # ------------------------------------------------------------------ C-02
    def predict(self, request: Any) -> Any:
        """Восстановить точки под маской. Интерфейс — ровно C-02."""
        contracts = contract_module()
        frame, mask = contracts.validate_request(request)
        keys = frame.loc[mask, contracts.KEY_COLUMNS].reset_index(drop=True)
        if keys.empty:
            return self._empty(contracts)

        row_ids = [
            self._row_by_key.get((str(polygon), pd.Timestamp(moment).normalize()))
            for polygon, moment in zip(keys["anon_polygon_id"], keys["date"], strict=True)
        ]
        known = [(position, row) for position, row in enumerate(row_ids) if row is not None]
        if not known:
            # Ни одной точки из запроса нет в наборе модели. Это не ошибка
            # сервиса: полигон может быть вне поставки. Пустой результат честнее
            # выдуманных значений, а воркер об этом предупредит.
            return self._empty(contracts)

        positions = [position for position, _ in known]
        queries = self._df.loc[[row for _, row in known]]
        predicted, features = self._ndvi.inference.predict_queries(
            self._df, queries, self._run_dir
        )
        predicted = np.asarray(predicted, dtype=float).reshape(-1)
        if not np.isfinite(predicted).all():
            # Дублирует проверку поставки намеренно: NaN не имеет права дойти
            # ни до БД, ни до графика.
            raise ValueError("модель вернула неконечное предсказание")

        found = keys.iloc[positions].reset_index(drop=True)
        left = _column(features, "primary_prev1_days")
        right = _column(features, "primary_next1_days")
        both = np.isfinite(left) & np.isfinite(right)
        neither = ~np.isfinite(left) & ~np.isfinite(right)
        sources = self._source_probabilities(features)
        disagreement, disagreement_measured = self._disagreement(features)

        predictions = found.copy()
        predictions["primary_ndvi_pred"] = predicted
        predictions["lower"] = predicted - self._half_width
        predictions["upper"] = predicted + self._half_width
        predictions["method"] = METHOD
        predictions["primary_ndvi_reconstructed"] = predicted
        # Гармонизации не было: колонка заполняется значением как есть, статус
        # это фиксирует. Выдать сырой ряд за гармонизированный нельзя.
        predictions["ndvi_harmonized"] = predicted

        diagnostics = found.copy()
        diagnostics["p_s2"] = sources.probabilities[:, 0]
        diagnostics["p_landsat"] = sources.probabilities[:, 1]
        diagnostics["p_modis"] = sources.probabilities[:, 2]
        diagnostics["p_unknown"] = np.clip(1.0 - sources.probabilities.sum(axis=1), 0.0, 1.0)
        diagnostics["left_distance_days"] = left
        diagnostics["right_distance_days"] = right
        diagnostics["model_disagreement"] = disagreement
        diagnostics["fallback_reason"] = ""
        # Контекст: обе стороны — 1.0, одна — 0.5, ни одной (только сезонная
        # норма и доноры) — 0.25. Шкала грубая и намеренно не выдаёт себя за
        # измеренную вероятность.
        diagnostics["context_quality"] = np.where(both, 1.0, np.where(neither, 0.25, 0.5))
        diagnostics["source_confidence"] = sources.confidence
        diagnostics["quality_flags"] = _quality_flags(
            predicted,
            both=both,
            neither=neither,
            interval_measured=self._half_width_measured,
            disagreement_measured=disagreement_measured,
        )
        diagnostics["interval_status"] = INTERVAL_STATUS
        diagnostics["interval_level"] = INTERVAL_LEVEL
        diagnostics["harmonization_status"] = HARMONIZATION_STATUS

        return contracts.ReconstructionResult(
            predictions[list(contracts.PredictionRow.model_fields)],
            diagnostics[list(contracts.DiagnosticRow.model_fields)],
            self.model_version,
        )

    # ------------------------------------------------------------- внутреннее
    def _empty(self, contracts: Any) -> Any:
        return contracts.ReconstructionResult(
            pd.DataFrame(columns=list(contracts.PredictionRow.model_fields)),
            pd.DataFrame(columns=list(contracts.DiagnosticRow.model_fields)),
            self.model_version,
        )

    def _source_probabilities(self, features: pd.DataFrame) -> _Sources:
        """Вероятности источника — из классификатора самой смеси.

        Это не наша модель и не наша эвристика: тот же объект, которым смесь
        взвешивает трёх спутниковых экспертов внутри `predict_fitted`. Локально
        измеренная точность по трём классам — 95.6–98.2 %.
        Если экспертов в бандле нет, вся масса честно уходит в `p_unknown`.
        """
        size = len(features)
        experts = self._bundle.get("models", {}).get("source_experts")
        if not experts or "classifier" not in experts:
            return _Sources(np.zeros((size, 3)), np.zeros(size))
        classifier = experts["classifier"]
        probabilities = np.zeros((size, 3))
        columns = classifier.classes_.astype(int)
        # Порядок колонок задаёт сам классификатор: подставлять свой порядок
        # значит перепутать спутники местами и не заметить этого.
        valid = (columns >= 0) & (columns < 3)
        probabilities[:, columns[valid]] = np.asarray(
            classifier.predict_proba(features), dtype=float
        )[:, valid]
        return _Sources(probabilities, probabilities.max(axis=1))

    def _disagreement(self, features: pd.DataFrame) -> tuple[np.ndarray, bool]:
        """Разброс между компонентами смеси, взвешенный их же весами.

        Компоненты считаются публичной функцией поставки `predict_fitted` — той
        самой, которой пользуется `predict_queries`. Собственного смешивания
        здесь нет: итоговое значение всегда приходит из поставки, а компоненты
        нужны только чтобы сказать, насколько модели разошлись.
        """
        size = len(features)
        columns: list[np.ndarray] = []
        weights: list[float] = []
        for name, weight in self._weights.items():
            try:
                if name in {"linear", "midpoint"}:
                    values = self._ndvi.features.baseline(features, name)
                else:
                    values = self._ndvi.models.predict_fitted(
                        name, self._bundle["models"][name], features
                    )
            except (KeyError, ValueError, TypeError, AttributeError):
                return np.zeros(size), False
            values = np.asarray(values, dtype=float).reshape(-1)
            if values.shape[0] != size or not np.isfinite(values).all():
                return np.zeros(size), False
            columns.append(values)
            weights.append(weight)
        if len(columns) < 2:
            return np.zeros(size), False
        matrix = np.vstack(columns)
        share = np.asarray(weights, dtype=float)
        share = share / share.sum()
        mean = (matrix * share[:, None]).sum(axis=0)
        variance = (((matrix - mean) ** 2) * share[:, None]).sum(axis=0)
        return np.sqrt(np.clip(variance, 0.0, None)), True


def _quality_flags(
    predicted: np.ndarray,
    *,
    both: np.ndarray,
    neither: np.ndarray,
    interval_measured: bool,
    disagreement_measured: bool,
) -> list[list[str]]:
    """Флаги качества по точкам. Каждый флаг — про то, чего модель не знает."""
    shared = ["interval_empirical_not_conformal", "harmonization_not_applied"]
    if not interval_measured:
        shared.append("interval_width_from_documentation")
    if not disagreement_measured:
        shared.append("model_disagreement_unavailable")
    flags: list[list[str]] = []
    for value, has_both, has_neither in zip(predicted, both, neither, strict=True):
        row = list(shared)
        if has_neither:
            row.append("no_same_season_context")
        elif not has_both:
            row.append("one_sided_context")
        # Физический диапазон не является правилом валидации: цель конкурса не
        # клипается, и значение вне [-1, 1] помечается флагом, а не отвергается
        # (инвариант 4).
        if not -1.0 <= float(value) <= 1.0:
            row.append("prediction_outside_physical_range")
        flags.append(row)
    return flags


def _column(features: pd.DataFrame, name: str) -> np.ndarray:
    if name not in features.columns:
        return np.full(len(features), np.nan)
    return features[name].to_numpy(dtype=float)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _classify(error: BaseException) -> BundleFailureKind:
    """Отказы поставки в наши виды. Тексты у неё русские и свои."""
    text = str(error).lower()
    if "sha256" in text:
        return "transport"
    if "схема" in text or "ключ" in text or "бесконечные" in text:
        return "schema"
    return "unknown"
