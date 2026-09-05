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
