"""Единая точка доступа к предсказателю C-02 для веб-пути и воркера (BE-003).

Модель не дублируется: и batch Разработчика 1, и наш веб-путь зовут один и тот же
`NDVIReconstructor` (инвариант 6, решение D-002). Здесь нет ни одной строки ML-логики —
только загрузка бандла, классификация отказов и заглушка на время, пока бандла нет.

Почему импорты `veg_recovery.contracts`/`inference` ленивые. Код Разработчика 1
опубликован в ветке `ML` и ещё не смержен в `main`, поэтому в нашем окружении модулей
может не быть. Жёсткий импорт на уровне модуля уронил бы сбор тестов и `import
veg_recovery` целиком, а SH-005 требует обратного: базовый импорт обязан работать
в лёгком окружении. Свои копии контракта мы при этом не заводим — дублировать
`ReconstructionRequest`/`Result` запрещено, лучше честно сказать «контракта нет».
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

#: Контракт, который реализует этот модуль, и версия схемы, с которой мы совместимы.
#: Расхождение `schema_version` бандла с этим значением — отказ, а не предупреждение.
CONTRACT_ID = "C-02"
CONTRACT_SCHEMA_VERSION = "1.0"

#: Версия заглушки. Начинается со `stub-`, и по этому префиксу и API, и red-team-проверка
#: перед CP-3 отличают ответ заглушки от ответа настоящей модели. Значение попадает
#: в `model_version` результата, то есть доезжает до пользователя и до БД.
STUB_MODEL_VERSION = "stub-mean-two-neighbours-0.1"

#: Единый машиночитаемый код отказа модели (инвариант 6). Один код на все причины
#: намеренно: потребителю API незачем разбирать внутренние детали бандла.
MODEL_ERROR_CODE = "MODEL_SCHEMA_MISMATCH"

BundleFailureKind = Literal[
    "missing", "schema", "transport", "untrusted", "stub_forbidden", "unknown"
]

#: Классификация отказов `load_bundle` по тексту исключения. Разработчик 1 поднимает
#: голый `ValueError` без кода, поэтому разбираем сообщение — других признаков нет.
#: Совпадение ищется по подстроке, регистр приводится: так правка формулировки
#: у владельца не ломает нас молча, а меняет разве что kind на "unknown".
#:
#: Зачем вообще различать. `SHA256 mismatch` — это не «модель несовместима», а
#: «файл доехал побитым»: ровно в этом состоит блокер B-DL-003, где Git переписал
#: EOL уже захешированных JSON. Сказать в такой ситуации «несовместимая схема»
#: значит послать владельца чинить не то. Код ответа один, текст — разный.
_FAILURE_PATTERNS: tuple[tuple[str, BundleFailureKind], ...] = (
    ("incompatible schema_version", "schema"),
    ("incompatible feature_version", "schema"),
    ("manifest file set does not match", "schema"),
    ("model file format does not match", "schema"),
    ("requires trusted=true", "untrusted"),
    ("sha256 mismatch", "transport"),
    ("symlink", "transport"),
    ("invalid bundle file", "transport"),
)

#: Человекочитаемое объяснение каждого вида отказа. Уходит в лог и в поле
#: `error_message_safe` схемы C-07 — без путей, версий и stack trace (инвариант 11).
_FAILURE_HINTS: dict[BundleFailureKind, str] = {
    "missing": "model bundle не смонтирован",
    "schema": "model bundle несовместим по схеме или версии признаков",
    "transport": "файлы model bundle повреждены при переносе (не совпал SHA256)",
    "untrusted": "trained bundle требует явного подтверждения происхождения",
    "stub_forbidden": "заглушка модели запрещена в этом контуре",
    "unknown": "model bundle не загружается",
}


class ModelContractMissing(RuntimeError):
    """Пакет `veg_recovery.contracts` недоступен — код Разработчика 1 ещё не в `main`."""


class ModelUnavailable(RuntimeError):
    """Бандл есть, но использовать его нельзя. `kind` объясняет, кому чинить."""

    def __init__(self, kind: BundleFailureKind, detail: str = "") -> None:
        self.kind: BundleFailureKind = kind
        self.error_code = MODEL_ERROR_CODE
        self.safe_message = _FAILURE_HINTS[kind]
        super().__init__(f"{self.safe_message}{f': {detail}' if detail else ''}")


def classify_bundle_failure(error: BaseException) -> BundleFailureKind:
    """Определить причину отказа загрузки бандла."""
    if isinstance(error, OSError):
        # Нет каталога, нет manifest.json, нет прав — бандл просто не смонтирован.
        return "missing"
    text = str(error).lower()
    for needle, kind in _FAILURE_PATTERNS:
        if needle in text:
            return kind
    return "unknown"


def contract_module() -> Any:
    """`veg_recovery.contracts` или понятная ошибка вместо ModuleNotFoundError."""
    try:
        import veg_recovery.contracts as contracts
    except ModuleNotFoundError as exc:  # pragma: no cover - зависит от состояния main
        raise ModelContractMissing(
            "контракт C-02 недоступен: модуль veg_recovery.contracts не установлен "
            "(код Разработчика 1 опубликован в ветке ML и ещё не смержен в main)"
        ) from exc
    return contracts


def contract_is_available() -> bool:
    """Есть ли в окружении реальный C-02. Используется тестами и health-проверками."""
    try:
        contract_module()
    except ModelContractMissing:
        return False
    return True


def neighbour_context(
    frame: pd.DataFrame, gap_mask: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ближайшие наблюдения слева и справа от каждой скрытой точки.

    Возвращает три массива длины `len(frame)`: предсказание среднего двух соседей и
    расстояния в днях влево и вправо. Значения заполнены только в строках под маской.

    Наблюдением считается строка с конечным `primary_ndvi`, **не попавшая под маску**.
    Это не перестраховка: в режиме `competition` скрытая строка может физически
    содержать значение, и брать его — прямая утечка ответа в предсказание.

    Соседи ищутся внутри одного полигона: ряд соседнего поля не является контекстом.
    """
    values = frame["primary_ndvi"].to_numpy(dtype=float)
    mask = gap_mask.to_numpy(dtype=bool)
    # Календарные дни как целые: разница дат сразу получается в сутках.
    days = frame["date"].to_numpy("datetime64[D]").astype("int64")
    polygons = frame["anon_polygon_id"].to_numpy()
    observed = np.isfinite(values) & ~mask

    size = len(frame)
    predictions = np.full(size, np.nan)
    left_days = np.full(size, np.nan)
    right_days = np.full(size, np.nan)

    for polygon in pd.unique(polygons[mask]):
        rows = np.flatnonzero(polygons == polygon)
        # Ряд упорядочен по дате, а не по порядку строк: входной фрейм приходит
        # из провайдера или из CSV и отсортированным не гарантирован.
        order = rows[np.argsort(days[rows], kind="stable")]
        observed_rows = order[observed[order]]
        if observed_rows.size == 0:
            raise ValueError(
                f"полигон {polygon} не содержит ни одного наблюдения: "
                "восстанавливать не из чего"
            )
        observed_days = days[observed_rows]
        gap_rows = order[mask[order]]
        # Позиция вставки даты гэпа в отсортированный ряд наблюдений: слева от неё
        # ближайший левый сосед, справа — ближайший правый.
        insert = np.searchsorted(observed_days, days[gap_rows])
        for row, position in zip(gap_rows, insert, strict=True):
            neighbours = []
            if position > 0:
                left = observed_rows[position - 1]
                left_days[row] = float(days[row] - days[left])
                neighbours.append(values[left])
            if position < observed_rows.size:
                right = observed_rows[position]
                right_days[row] = float(days[right] - days[row])
                neighbours.append(values[right])
            # Хотя бы один сосед есть всегда: полигон без наблюдений отсеян выше.
            predictions[row] = float(np.mean(neighbours))
    return predictions, left_days, right_days


class ModelStub:
    """Заглушка интерфейса C-02 на время, пока нет бандла.

    Содержимое — официальный ориентир ТЗ Разработчика 1: «mean two neighbors»
    (`docs/01_ml_developer.md:302`). Ничего сверх этого заглушка не делает: её задача —
    разблокировать BE-006/BE-008, а не притворяться моделью.

    Проверку входа выполняет `validate_request` владельца контракта, а не наша копия.
    Так заглушка отвергает ровно те же запросы, что и настоящая модель, и переход
    со стаба на бандл не вскрывает новых ошибок на границе.
    """

    model_version = STUB_MODEL_VERSION

    def predict(self, request: Any) -> Any:
        contracts = contract_module()
        frame, mask = contracts.validate_request(request)
        keys = frame.loc[mask, contracts.KEY_COLUMNS].reset_index(drop=True)
        if keys.empty:
            # Анализ без единого гэпа — валидный результат, а не ошибка.
            return contracts.ReconstructionResult(
                pd.DataFrame(columns=list(contracts.PredictionRow.model_fields)),
                pd.DataFrame(columns=list(contracts.DiagnosticRow.model_fields)),
                self.model_version,
            )

        values, left, right = neighbour_context(frame, mask)
        selected = mask.to_numpy(dtype=bool)
        predicted = values[selected]
        if not np.isfinite(predicted).all():
            # Инвариант: никакой fallback не имеет права вернуть NaN
            # (`docs/01_ml_developer.md:315`). Лучше отказ, чем молчаливый мусор.
            raise ValueError("заглушка получила неконечное предсказание")
        left_days, right_days = left[selected], right[selected]
        both_sides = np.isfinite(left_days) & np.isfinite(right_days)

        predictions = keys.copy()
        predictions["primary_ndvi_pred"] = predicted
        # Интервал не откалиброван, и это заявлено явно: полуширина и статус взяты
        # такими же, как у владельца контракта в ветке без uncertainty-конфигурации.
        # Узкая красивая лента здесь была бы враньём о точности заглушки.
        predictions["lower"] = predicted - 1.0
        predictions["upper"] = predicted + 1.0
        predictions["method"] = "mean_neighbors"
        predictions["primary_ndvi_reconstructed"] = predicted
        # Гармонизацию производит не заглушка и вообще не мы: колонка заполняется
        # значением как есть, а `harmonization_status` честно говорит, что её не было.
        predictions["ndvi_harmonized"] = predicted

        diagnostics = keys.copy()
        # Источник скрытой точки заглушке неизвестен, поэтому вся вероятностная масса
        # лежит на `unknown`. Приписать её S2 было бы выдумкой, а сумма обязана быть 1.
        diagnostics["p_s2"] = 0.0
        diagnostics["p_landsat"] = 0.0
        diagnostics["p_modis"] = 0.0
        diagnostics["p_unknown"] = 1.0
        diagnostics["left_distance_days"] = left_days
        diagnostics["right_distance_days"] = right_days
        diagnostics["model_disagreement"] = 0.0
        diagnostics["fallback_reason"] = "model_stub"
        diagnostics["context_quality"] = np.where(both_sides, 1.0, 0.5)
        diagnostics["source_confidence"] = 1.0
        diagnostics["quality_flags"] = [
            ["model_stub", "interval_not_coverage_certified", "harmonization_incomplete"]
            + ([] if flag else ["one_sided_or_no_context"])
            for flag in both_sides
        ]
        diagnostics["interval_status"] = "uncalibrated_default"
        diagnostics["interval_level"] = 0.95
        diagnostics["harmonization_status"] = "partial_or_identity"

        return contracts.ReconstructionResult(
            predictions[list(contracts.PredictionRow.model_fields)],
            diagnostics[list(contracts.DiagnosticRow.model_fields)],
            self.model_version,
        )


@dataclass(frozen=True)
class LoadedReconstructor:
    """Предсказатель вместе с провенансом: чем считали и можно ли этому верить."""

    reconstructor: Any
    model_version: str
    is_stub: bool
    #: Путь к бандлу; у заглушки его нет.
    bundle_path: str | None = None

    def predict(self, request: Any) -> Any:
        return self.reconstructor.predict(request)


def load_bundle_reconstructor(bundle_path: str | Path, *, trusted: bool = False) -> Any:
    """Загрузить бандл владельца контракта, переведя его отказы в `ModelUnavailable`."""
    try:
        from veg_recovery.inference import load_reconstructor
    except ModuleNotFoundError as exc:
        raise ModelContractMissing(
            "контракт C-02 недоступен: модуль veg_recovery.inference не установлен"
        ) from exc
    try:
        return load_reconstructor(bundle_path, trusted=trusted)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelUnavailable(classify_bundle_failure(exc), str(exc)) from exc


def load_ml_bundle(bundle_path: str | Path, *, trusted: bool = False) -> LoadedReconstructor:
    """Прежний бандл Разработчика 1 (C-03 baseline_v1 или C-04 p0-catboost-gpu-v1).

    Веб-путь его больше не зовёт: с BE-011R модель приходит из каталога `model/`.
    Функция остаётся точкой, которой contract-тесты подтверждают, что принятый
    handoff H-003 по-прежнему загружается нашим кодом. В `build_reconstructor`
    она не участвует намеренно — двух моделей в горячем пути быть не должно.
    """
    reconstructor = load_bundle_reconstructor(bundle_path, trusted=trusted)
    return LoadedReconstructor(
        reconstructor=reconstructor,
        model_version=reconstructor.bundle.manifest.model_version,
        is_stub=False,
        bundle_path=str(bundle_path),
    )


def build_reconstructor(
    package_path: str | Path | None,
    *,
    run_name: str = "local",
    allow_stub: bool = True,
    trusted: bool = False,
    environment: str = "development",
) -> LoadedReconstructor:
    """Собрать предсказатель для процесса. Вызывается один раз на старте.

    С BE-011R это пакет из каталога `model/`: смесь LightGBM и спутниковых
    экспертов CatBoost на 324 признаках. Прежний бандл C-04 остаётся в дереве
    как принятый handoff H-003, но веб-путь его больше не зовёт — двух моделей
    в сервисе быть не должно (инвариант 6).

    Загрузка на старте — не оптимизация: `read_data` сверяет SHA256 обоих CSV
    поставки и разбирает 149 145 строк, а pickle смеси весит 17 МБ. Делать это
    внутри обработки HTTP значит платить секунду на каждом запросе.

    Правило подмены ровно одно: заглушка заменяет **отсутствующую** поставку и
    никогда — сломанную. Битые файлы, не совпавший хеш и недоверенный pickle
    обязаны валить старт. Молча съехать на заглушку при сломанной модели — это
    выдать заглушку за модель, ровно тот обман, который запрещает
    red-team-проверка перед CP-3.
    """
    if environment == "production" and allow_stub:
        # Отдельная проверка до всякой загрузки: конфигурация, разрешающая стаб
        # в проде, ошибочна сама по себе, даже если модель сейчас на месте.
        raise ModelUnavailable(
            "stub_forbidden",
            "выключите COSMO_ALLOW_MODEL_STUB или смонтируйте реальную модель "
            "(red-team checklist перед CP-3)",
        )

    if package_path:
        # Путь задан — значит модель обязана загрузиться. Пустой каталог на этом
        # месте (Docker создаёт его сам, если host-пути нет) обязан валить старт,
        # а не тихо превращаться в заглушку.
        # Импорт ленивый: пакет поставки лежит вне нашего дерева, и модуль
        # `veg_recovery.service` обязан импортироваться без него (SH-005).
        from veg_recovery.service.ndvi_run import NdviRunReconstructor

        reconstructor = NdviRunReconstructor(package_path, run_name, trusted=trusted)
        return LoadedReconstructor(
            reconstructor=reconstructor,
            model_version=reconstructor.model_version,
            is_stub=False,
            bundle_path=str(Path(package_path) / "runs" / run_name),
        )

    if not allow_stub:
        raise ModelUnavailable("missing", "каталог модели не смонтирован")
    return LoadedReconstructor(
        reconstructor=ModelStub(), model_version=STUB_MODEL_VERSION, is_stub=True
    )
