# H-003 — acknowledgement Backend: C-02 и C-04 приняты

Заполнено по шаблону §15 `00_team_coordination.md`. Канонический журнал handoff живёт
в координационном файле — он в зоне teamlead, мы его не редактируем. Этот документ
готов к переносу в §15 и в описание PR в `main`.

Закрывает два ожидания, стоявших **на нас**: `C-02 · REVIEW: Backend integration pending`
и подключение `C-04`. Producer просил об этом прямо: «Integrators should approve the DTO
schema when connecting their own routes» (`artifacts/ml/CONTRACT_CHANGELOG.md`).

---

- **Producer:** ML (Разработчик 1)
- **Consumer:** Backend (Разработчик 3)
- **Task ID:** BE-003 (адаптер), BE-011 (подключение C-04), SH-002 (зависимости)
- **Contract ID/version:** C-01/C-02 `producer 1.0`; C-04 `p0-catboost-gpu-v1`;
  C-05 фактически реализован как `DiagnosticRow` (16 полей)
- **Status:** **ACCEPTED** консьюмером. Статус самих контрактов остаётся за владельцем:
  мы подтверждаем интеграцию, а не объявляем чужую задачу DONE.
- **Artifact path:** `artifacts/ml/ndvi_backend_handoff_v1/` (ветка `models`),
  `src/veg_recovery/{contracts,inference}.py`
- **Git commit:** producer — `models@3384de9` (ветвится от `ML@39a8f73`);
  consumer — `backend@51b2ef9`, мерж `5ad7f65`
- **Data/model fingerprint:** `MANIFEST.sha256`, 41 файл, проверены полностью;
  `model_version = p0-catboost-gpu-v1`, `bundle_kind = trained`,
  `feature_version = ndvi-context-v1`, `schema_version = 1.0`
- **Breaking change:** нет
- **Input schema:** `anon_polygon_id`, `date`, `crop_type`, `primary_ndvi` — обязательны;
  `gap_mask` — `pd.Series` с индексом фрейма; даты — tz-naive календарные дни
- **Output schema:** `PredictionRow` 8 полей, `DiagnosticRow` 16 полей,
  JSON-граница — `ReconstructionPayload.from_result()`, `schema_version 1.0`

## Validation command и Test result

Все три проверки выполнены на Linux, Python 3.11, CPU.

~~~bash
# 1. Проверка владельца: целостность и полный инференс
cd artifacts/ml/ndvi_backend_handoff_v1
PYTHONPATH=runtime/src python verify_handoff.py --full-inference
# integrity OK: 41 files; model=p0-catboost-gpu-v1
# full inference OK: rows=3112; max_abs_delta=1.11e-16

# 2. Наш собственный путь загрузки (build_reconstructor), тот же эталон
uv run pytest -q -m slow tests/backend/test_contract_c04.py     # 1 passed

# 3. Contract-тесты Backend
uv run pytest -q tests/backend                                  # 73 passed
~~~

`max_abs_delta = 1.11e-16` — машинная точность при допуске эталона `1e-10`.
Проверено и внутри развёрнутого стенда: api и worker поднимают
`p0-catboost-gpu-v1`, `/health/ready` отдаёт `model_bundle: ok`,
веб-путь возвращает `method = oof_ensemble` с интервалом и `quality_flags`.

## Consumer acknowledgement

**accepted**, 2026-09-05. Приняты: публичный контракт C-02, DTO
`ReconstructionPayload`, диагностика C-05, обученный bundle C-04.

## Known limitations — зафиксировано, не обойдено

1. **Предсказание зависит от состава фрейма.** Тот же полигон, поданный отдельно и
   в составе полного test-фрейма, расходится на `max_abs_delta ≈ 0.125` — при том что
   полный фрейм воспроизводит эталон с точностью `1e-16`. Значит feature builder
   использует контекст за пределами полигона. Следствие: пункт CP-4
   «Batch/API prediction parity» выполним **только при совпадающем составе входа**,
   а веб-путь с одним полигоном даёт другой ответ, чем batch. Закреплено тестом
   `test_prediction_depends_on_the_frame_composition`. **Нужна формулировка parity
   от ML:** сравнивать одинаковые фреймы или зафиксировать допустимое расхождение.
2. **Верхние границы версий поставлены по воспроизведённому отказу.** На
   `scikit-learn 1.9.0` загрузка `estimators.joblib` падает
   `AttributeError: Can't get attribute '_RemainderColsList'`. Закреплено
   `scikit-learn>=1.6,<1.7` (ML обучал на 1.6.1) и `pandas>=2.2,<3` (ML на 2.3.3).
   `numpy` в нашем lock — 2.4.6 против 2.0.2 у ML; расхождение есть, но полный
   инференс воспроизводится, поэтому границу не сужаем.
3. **Temporal transfer.** README ML: «в temporal CV ансамбль хуже простого baseline».
   Проверяется тестом по `cv_summary` манифеста, чтобы предупреждение не осталось
   словами. UI и демо не имеют права подавать модель как безусловно лучшую.
4. **Интервалы не сертифицированы.** `interval_level = 0.95`, но `interval_status`
   говорит, что интервалы эмпирические. Процентом уверенности не подписываем.
5. **`trusted=True` включено осознанно** после проверки `MANIFEST.sha256`: joblib
   исполняет код при загрузке, SHA256 ловит порчу файла, но не подлог автора.
6. **B-DL-003 на Linux не воспроизводится** — оба бандла грузятся без SHA256 mismatch.
   Проблема остаётся Windows/EOL-специфичной.

## Дефект, найденный у producer — не чинили, сообщаем

`tests/ml/test_ml.py::OfflineTrainingContractTests::test_gpu_factory_contract_without_fitting`
падает в чистом checkout:

~~~text
FileNotFoundError: 'artifacts/ml/kaggle/train_ndvi.ipynb'
~~~

Файл исключён **вашим же** `.gitignore` (`/artifacts/ml/kaggle/*.ipynb`) и отсутствует
и в `ML@39a8f73`, и в `models@3384de9`. Это делает набор ML красным у любого консьюмера
и ломает пункт SH-005 «Offline CI». Чужой тест мы не правим (§0). Варианты — за вами:
пометить тест как требующий локальной сборки Kaggle-пакета, либо снять зависимость
от игнорируемого файла.

## Migration/fallback

Откат — снять монтирование bundle: `MODEL_BUNDLE_HOST_PATH=./infra/model_bundle`.
Сервис вернётся на `ModelStub` (`mean two neighbors`), API продолжит работать.
Сломанный bundle заглушкой **не** подменяется никогда — только отсутствующий.

## Reviewer

ML — как владелец C-01/C-02/C-04. Ожидаем от вас ответа по пункту 1
(формулировка parity) и по дефекту `tests/ml`.

- **Updated at UTC:** 2026-09-05
