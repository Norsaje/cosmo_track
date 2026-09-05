# DL consumer review новых веток — 2026-09-05

Проверены `origin/ML@39a8f73` и `origin/backend@350faec`, опубликованные после первого
DL handoff. В `main@4f40520` их код ещё не интегрирован. Статусы других разработчиков
не объявляются DONE по факту наличия файлов. Этот отчёт уточняет ранее записанное
«C-01/C-03/SH-002 отсутствуют»: теперь часть входов доступна в ветках.

## C-01/C-03: реально проверено

ML `read_dataset(strict_current=True)`, `MaskSpec.from_test`, `load_folds` и `split_fold`
исполнены непосредственно из изолированного worktree ML. Копии ML-реализации нет.
DL adapter принимает nullable pandas dtypes. 16 folds, 13 317 OOF оценок; во всех
проверены точные ключи/labels, отсутствие outer targets и excluded polygons в fit,
полная маска target-строки. На первых 8 ключах каждого fold построены конечные
окна, в центре нет наблюдаемых каналов и loss labels. Это 128 probes, не обучение DL.

Mapping: A=matched, B=unseen, C=temporal, D=hard. `repeat/fold` становятся
`r{repeat}_f{fold}`: повторное оценивание физического ключа не теряется.
Фактическая temporal policy — весь год 2024 скрыт, fit только до 2024;
hard policy — один конечный левый сосед дальше 15 дней, целый polygon исключён из fit.

Из сохранённых предсказаний `mean_neighbors` пересчитаны и сверены с опубликованными
CSV/JSON (tolerance 1e-12) RMSE по режимам и composite:

| Метрика | Значение |
|---|---:|
| A RMSE | 0.088454630981 |
| B RMSE | 0.095591471674 |
| C RMSE | 0.178772374310 |
| D RMSE | 0.117576046148 |
| Composite | 0.106698644170 |
| Overall row-weighted RMSE | 0.103167745464 |

Это воспроизведение **метрик сохранённого ML baseline OOF**, не повторный fit/predict
baseline и не результат DL. Byte hashes и проверки: `artifacts/dl/ml_handoff_39a8f73.json`.

## Что ещё требуется до DL CV

1. ML подтвердить C-03 handoff в main. Frozen outer keys уже есть, но DL train-target
   masks и отдельный inner early-stop protocol не опубликованы. Альтернатива — заранее
   согласованный fixed epoch budget без early stopping; нельзя выбирать epoch по outer OOF.
2. DL runner должен потреблять именно `split_fold.context_frame`: простая фильтрация
   raw keys не воспроизводит цензурирование C/D. При передаче уже masked frame природные
   пропуски нужно отличать от policy censoring по raw provenance, не восстанавливая значения.
   Read-only audit это учитывает; полный training runner ещё не адаптирован.
3. Нужны сами final ensemble OOF и manifest. `reports/ml_ablation.md` заявляет CatBoost
   GPU и замену HGB/ExtraTrees по решению пользователя, но `trained_gpu_v1/` и его OOF
   отсутствуют в дереве опубликованного `39a8f73`. Числа из отчёта не заменяют predictions.
   Требование входа теперь — **фактический сильный ML baseline/final OOF**, а не непременно HGB.

## Блокирующий дефект транспорта ML bundle

Фактический `load_reconstructor(.../baseline_v1/bundle)` в чистом Windows worktree
падает: `ValueError: SHA256 mismatch: feature_state.json`.

Диагностика показала: SHA256 обоих JSON после CRLF→LF **точно совпадает** с manifest;
байты рабочего checkout — нет. Hash validation не отключалась, файлы ML не правились.
ML нужно зафиксировать EOL для уже LF-хешированных файлов (`text eol=lf` в owned
artifact directory) либо публиковать byte-preserving архив и проверить свежую загрузку
на Windows/Linux. Fingerprints исходных CSV также требуют явной политики EOL при переносе.

DL research artifacts защищены собственным `artifacts/dl/.gitattributes` (`-text`):
Git не должен переписывать уже захешированные JSON/CSV/weights. Это локальное правило
в DL-owned path, не изменение shared dependency/ML policy.

## C-09 и ML product series

У ML `reconstruct_product_series` есть `is_reconstructed`, а C-09 требует `is_observed`.
Backend consumer должен выводить observed из исходного конечного primary и отсутствия
reconstruction; естественные NaN не являются reconstructed. Не передавать product
`confidence` как cloud QA: это разные величины. Lower/upper ML primary-шкалы нельзя
выдавать за uncertainty harmonized-шкалы без соответствующего преобразования.

Reference и query должны иметь одну и ту же calibration/version. Нельзя смешивать
ML median/IQR affine с DL IRLS/pooling mapping в одном detector call. Gap source mixture
не является фактически известным source label. Пока uncertainty/source provenance
не согласованы, reconstructed events консервативно ослабляются и сопровождаются warning.

## SH-002: dependency proposal Backend

Получен `docs/handoffs/H-SH-002.md` из backend. Его текст ещё говорит «не закоммичено»,
но `pyproject.toml`, `uv.lock` уже опубликованы в `350faec`. Python 3.11, core numpy/pandas
и lazy torch extra соответствуют структуре DL. Shared файлы DL не изменял.

- Для нынешних TCN training/inference достаточно `dl = [torch]`; PyPOTS пока не добавлять:
  реальный matched-mask training bridge ещё не реализован.
- Для графиков дополнительно нужен matplotlib (локально проверен 3.10.0); добавить
  в согласованный reporting/dev extra, не в runtime core. Проверки требуют pytest.
- Локально тестирован torch 2.5.1 на CPU; Backend lock с другой версией не объявляем
  проверенным нами. `uv sync --extra dl` этого lock DL consumer ещё не выполнял.
- Коллизия Backend/ED pyproject/lock остаётся shared интеграционной задачей владельца;
  DL поддерживает Backend ownership, но не делает самовольный merge чужих зависимостей.
- `acknowledgement=pending` до clean-install и smoke на общем lock. Отсутствие файлов
  уже не blocker; остаётся проверка совместимости конкретной установки.

## Повторение read-only consumer review

```powershell
git worktree add --detach tmp/dl-review-ml 39a8f73
$env:PYTHONPATH='src;tmp/dl-review-ml/src'
python -m veg_recovery.dl.ml_handoff --ml-root tmp/dl-review-ml --output artifacts/dl/new_ml_review.json
```

Повторно создавать уже существующий worktree не требуется. `ml_handoff` не импортирует
torch, не генерирует folds, не обучает модель и не исправляет producer artifacts.

---

# Обновление после реального DL CV — 2026-09-05, DL@4fdfc60

## Утечка, найденная в нашем же runner

`--fold-manifest` брал inference-контекст срезом сырого кадра, а маскировал только
inner и outer ключи. Producer censoring при этом терялось: в CV-C модель увидела бы
наблюдения 2024 года, скрытые `past_only`, а в CV-D — все точки полигона, кроме
единственного разрешённого соседа. Любые метрики C и D до этого исправления были бы
недействительны.

Дефект был в DL runner, артефакты ML корректны. Исправление: manifest 0.2 несёт
`censored_context_keys`, а `c03_bridge` выводит их разностью

```
censored = raw[MaskSpec.columns].notna() & context[MaskSpec.columns].isna()
```

и обязан пройти round-trip: `apply_mask(train, censored_keys)` должен побайтово
воспроизвести `split_fold(...).context_frame`, иначе сборка входов падает. Проверка
не зависит от того, какие поля есть у естественно отсутствующих наблюдений.

Причинность-проверка causal-фолдов переписана: раньше она смотрела на наличие ключа
позже первой query date и потому ложно падала на CV-D; теперь ищет **незацензурированные**
наблюдения позже первой outer query date конкретного полигона.

## Как разблокировано обучение без подписи ML

`veg_recovery.dl.c03_bridge` исполняет producer-код ML read-only и сохраняет его
outer keys, MaskSpec, censoring и baseline OOF. Мост падает, если импортированный
`split_fold` лежит вне указанного `--ml-root`. Собственных folds DL не создаёт.

`review_status = derived_from_producer_artifacts` — отдельный статус, не `accepted`.
Runner требует `producer_evidence` (commit ML и SHA256 входов) и печатает статус
в preflight, в каждый checkpoint и в отчёт прогона.

Inner keys ML не передал, поэтому по умолчанию действует `--epoch-policy fixed`:
epoch не выбирается ни по outer OOF, ни по DL-inner. Бюджет 16 зафиксирован заранее
по inner-кривой пилотного фолда (плато на 12–17 эпохе во всех трёх проверенных
режимах). `early_stop` остаётся вариантом проверки чувствительности.

## B-DL-004: у adoption gate нет знаменателя

`artifacts/ml/baseline_v1/oof_predictions.csv.gz` содержит 39 колонок; из
предсказаний только `pred_nearest_left`, `pred_nearest_right`, `pred_mean_neighbors`,
`pred_linear`, `pred_seasonal_crop`, `pred_pchip`, `pred_akima`. Колонок
CatBoost или ансамбля нет.

Причина найдена: в `.gitignore` ветки ML есть строка `/artifacts/ml/trained_gpu_v1/`,
поэтому обученный bundle и его OOF намеренно не попадают в репозиторий, хотя
`artifacts/ml/delivery_verification.json` сообщает `gpu_training_executed: true`
и `ensemble_composite_rmse: 0.09741183557217524`.

Следствие: порог gate «улучшение composite на 0.002 относительно финального ML»
вычислить нельзя. DL сравнивается с опубликованным `mean_neighbors` на тех же
ключах, а сравнение с CatBoost остаётся арифметикой по отчёту ML.

**Запрос узкий:** добавить `pred_catboost` и `pred_ensemble` в тот же OOF-файл или
опубликовать отдельный OOF CSV с ключом `mode,repeat,fold,anon_polygon_id,date` и
колонками `primary_ndvi,pred_*`. Публиковать сам bundle и обходить `.gitignore`
не требуется. Это одна колонка данных, а не смена политики артефактов.

## Наблюдение по CV-C для владельца ML

По опубликованным числам ML принятый глобальный blend (30.955% baseline +
69.045% CatBoost) в режиме C даёт 0.220068 против 0.178772 у их же baseline.
Composite ансамбля 0.097412; если в C оставить baseline — 0.091218; если вдобавок
взять чистый CatBoost в A и B — 0.089373.

Оговорки: CV-C это один фолд на 1200 строк из одного repeat, а выбор весов по
режимам — отбор на тех же OOF, на которых считается метрика. Это предложение
проверить, решение принимает владелец C-04. Для DL последствие прямое: принятие
per-mode blend сдвигает планку adoption gate примерно к 0.0874 composite.

## Повторение сборки входов и Kaggle bundle

```powershell
git worktree add --detach tmp/dl-review-ml origin/ML
$env:PYTHONPATH='src;tmp/dl-review-ml/src'
python -m veg_recovery.dl.c03_bridge --ml-root tmp/dl-review-ml --output artifacts/dl/c03_derived
python -m veg_recovery.dl.train --fold-manifest artifacts/dl/c03_derived/dl_c03.json --preflight-only
python configs/dl/kaggle/package_dataset.py --ml-root tmp/dl-review-ml
```

Сборка входов детерминирована: gzip пишется с `mtime=0`, поэтому SHA256 в
manifest воспроизводятся при повторном запуске. `artifacts/dl/.gitattributes`
помечает `*.json`, `*.csv`, `*.gz`, `*.zip`, `*.pt` как `-text`, чтобы Git не
переписал EOL уже захешированным файлам — тот же класс дефекта, что B-DL-003.
