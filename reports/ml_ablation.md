# NDVI ML: результаты и передача

Дата: 2026-09-05. Реализованы и измерены baseline-контур и CatBoost GPU.
**Локально ML-модели не обучались.** Первый фиксированный запуск успешно выполнен
на Kaggle: CatBoost, CPU source classifier, OOF-ансамбль, bundle и batch inference
завершились без ошибки. На основном CV-A CatBoost получил RMSE 0.066256, а
принятый четырёхрежимный ансамбль улучшил composite RMSE baseline на 8.70%.
Исходная постановка, сырая иерархическая цель и файлы CSV сохранены.

## Изменённые файлы

- `src/veg_recovery/contracts.py`: frozen dataclasses, Protocol, Pydantic DTO.
- `src/veg_recovery/data/{__init__,io}.py`: реальные dtype, schema, ключи, сводки, SHA256.
- `src/veg_recovery/validation/{__init__,masking,folds,metrics}.py`: маскирование,
  четыре CV, разбиения по ключам, официальная метрика и подгруппы.
- `src/veg_recovery/features/{__init__,builder}.py`: единая функция признаков,
  fit-only priors, видимый временной контекст, source hierarchy, sentinel guard.
- `src/veg_recovery/models/{__init__,baselines,manifest,bundle,estimators,calibration,training}.py`:
  baselines, проверенный bundle, P0/P1 factories, OOF blend/gates/intervals,
  отдельный CLI обучения с обязательным `--allow-training`, явным устройством и
  ранней проверкой доступности CUDA для GPU-режима.
- `src/veg_recovery/inference.py`, `src/veg_recovery/cli/batch.py`: единый API и batch.
- `src/veg_recovery/anomalies/baseline.py`: отдельная sensor-wise harmonization,
  прошлогодняя median/MAD норма и отрицательные persistent events.
- `configs/ml/`: зависимости, `training_p0.json`, сохранённый `folds_v1.csv`.
- `tests/ml/test_ml.py`: 20 тестов, включая toy series, leakage, bundle и API/CLI.
- `artifacts/ml/`: скрипт baseline-оценки, результаты, notebook, архив и changelog.
- `reports/experiments.csv`, этот отчёт, `reports/data_contract_issues.md`.

Frontend, provider adapters и backend routes не менялись. Backend-кода в
исходном workspace нет. Уже имевшееся удаление `docs/01_ml_developer.md` не изменялось.

## Команды

В Python-окружении с зависимостями из `configs/ml/requirements.txt`:

```bash
python -m pip install -r configs/ml/requirements.txt
PYTHONPATH=src python -m unittest discover -s tests/ml -v
PYTHONPATH=src python artifacts/ml/evaluate_baselines.py
PYTHONPATH=src python -m veg_recovery.cli.batch \
  --input data/test_data.csv --bundle artifacts/ml/baseline_v1/bundle \
  --output artifacts/ml/submission.csv --diagnostics artifacts/ml/diagnostics.csv \
  --expected-count 3112
python artifacts/ml/package_kaggle.py
```

Для проверки в этой рабочей среде использован уже установленный интерпретатор
`/Users/matvey/Desktop/ml/.venv/bin/python` с
`PYTHONPATH=src:/opt/homebrew/lib/python3.14/site-packages`. Другие проекты не менялись.
Фактические версии: Python 3.14.5, pandas 3.0.3, numpy 2.4.6, scipy 1.18.0,
scikit-learn 1.9.0, Pydantic 2.13.4, joblib 1.5.3. Версии сохранены в manifest.

Обучение запускается **на Kaggle**, инструкции:
`artifacts/ml/kaggle/README_RU.md`, notebook `artifacts/ml/kaggle/train_ndvi.ipynb`.
Фиксированный запуск CatBoost выполнен с
`--device gpu --models catboost --trials 0 --max-train-gaps 6000`. Kaggle видел
две CUDA-карты; CatBoost использовал GPU, feature preparation и RandomForest
source cross-fit работали на CPU. Отдельный tuning — 30 GPU trials, MedianPruner,
неизменные folds/seeds. Он ещё не запускался.

## Результат первого CatBoost GPU-запуска

| Режим | Baseline RMSE | CatBoost RMSE | Ансамбль RMSE | CatBoost GapScore |
|---|---:|---:|---:|---:|
| A matched-mask | 0.088455 | **0.066256** | 0.068649 | 10.12 |
| B unseen-polygon | 0.095591 | **0.071849** | 0.074441 | 8.45 |
| C past-only temporal | **0.178772** | 0.246606 | 0.220068 | 0.00 |
| D hard | 0.117576 | 0.114809 | **0.114671** | 0.00 |

CatBoost заметно улучшил главные interpolation-подобные режимы A и B, но
ухудшил forecasting-режим C. Поэтому по заранее зафиксированному composite
принят глобальный неотрицательный blend: **30.955% baseline + 69.045% CatBoost**.
Его composite RMSE **0.097412** против 0.106699 у baseline и 0.099562 у чистого
CatBoost. Это OOF-оценка, а не leaderboard score.

Все шесть проверенных gates отклонены: ни один одновременно не улучшил медиану
повторов, composite и unseen RMSE в разрешённых пределах. Clip не применялся.
Фиксированный запуск обучил 16 CatBoost-моделей OOF и одну финальную модель;
суммарное чистое время зарегистрированных OOF fit — 1 398 секунд. Полный Kaggle
run, включая CPU-подготовку признаков, source cross-fit, importance, финальный fit
и batch inference, завершился примерно за 25 минут.

## CV: официальный baseline, среднее соседей

| Режим | Количество оценённых строк | RMSE | GapScore | Known RMSE | Unseen RMSE |
|---|---:|---:|---:|---:|---:|
| A matched-mask, 5 repeats | 6 000 | 0.088455 | 3.46 | 0.083478 | 0.089297 |
| B unseen-polygon, 5 folds | 6 000 | 0.095591 | 1.32 | — | 0.095591 |
| C past-only forecasting, holdout 2024 | 1 200 | 0.178772 | 0.00 | 0.178772 | — |
| D hard, 5 folds | 117 | 0.117576 | 0.00 | — | 0.117576 |

В A повторное попадание физического ключа в разные repeats возможно; N — число
OOF-оценок, а не уникальных строк данных. Каждый fold/key уникален. Агрегированный
RMSE вычислен по всем остаткам режима, без усреднения округлённых RMSE.
CV-C намеренно прогнозирует поздний год **без текущего года с обеих сторон**,
поэтому его нельзя выдавать за оценку интерполяции. CV-D оставляет ровно одного
соседа >15 дней, скрывает серии длиной 2–4 и полностью исключает полигоны из fit.
В D представлены все культуры, включая редкие `пастбища/зерновые` (5 точек),
и все три источника, включая MODIS у подсолнечника (2 точки).

**Composite decision = 0.50 × RMSE_A + 0.25 × RMSE_B + 0.15 × RMSE_C + 0.10 × RMSE_D**.
Весов не меняли, меньше — лучше. Для принятого baseline: **0.106699**.
Это не конкурсный RMSE на test и не leaderboard-результат.

CV-A: доля known 0.149000 против test 0.149100. Total variation distance по
gap length = 0.017912, по месяцам = 0.022339. Распределения приближены, а не
тождественны; календарный 2025 отсутствует в train. Proxy source использует
видимые test строки, истинный источник скрытой строки неизвестен.

| Baseline | Composite RMSE | Решение |
|---|---:|---|
| Среднее соседей | **0.106699** | keep |
| Time-weighted linear | 0.107668 | reject |
| PCHIP | 0.110616 | reject |
| Akima | 0.113545 | reject |
| Nearest right | 0.120761 | reject |
| Nearest left | 0.121105 | reject |
| Crop seasonal climatology | 0.173387 | reject |

На краю все интерполяционные baselines используют одинаковый fallback:
один сосед + seasonal prior; затем polygon×crop, crop, global DOY/global median.
При совершенно пустом fit/context явно маркируется неподтверждённый default 0.5.
Ни один fallback не возвращает NaN. Финальный clip отсутствует.
Финальный ML-ансамбль использует веса baseline 0.309550 и CatBoost 0.690450.
Выбор сделан только по OOF с неотрицательными весами и суммой 1. Все gates
проверены с ограничением unseen Δ≤0.003 и отклонены по заданным критериям.

Полные per-fold таблицы crop/source/gap length/distances/year/confidence/known:
`artifacts/ml/baseline_v1/subgroup_metrics.csv`. Предсказание и ошибка каждого
baseline на каждом OOF-ключе: `oof_predictions.csv.gz`. Негативные варианты
и повторные измерения сохранены в `reports/experiments.csv`.

## Источник и неопределённость

Все 30 520 train и 17 641 видимых test target сопоставлены по абсолютному tolerance
1e-7 с иерархией S2 → Landsat → MODIS; unknown = 0. Train: S2 11 235,
Landsat 13 284, MODIS 6 001. Test visible: S2 7 815, Landsat 6 804, MODIS 3 022.
В финальном bundle source probabilities выдаёт обученный RandomForest;
GroupKFold source cross-fitting выполнен без hidden-row значений. Source OOF,
accuracy и reliability bins сохранены. В A accuracy: Landsat 75.46%, MODIS
97.48%, S2 91.84%; temporal C показывает сильный сдвиг источников и остаётся
открытым риском. Вероятности не названы откалиброванными.

Интервалы: эмпирические абсолютные OOF-остатки по context/source bins. Для
coverage оценяемый fold и все повторения его физических ключей исключены из
calibration residual pool. Ниже pooled coverage, с весами по числу точек:

| Подгруппа | Coverage 80% | Средняя ширина 80% | Coverage 95% | Средняя ширина 95% |
|---|---:|---:|---:|---:|
| Overall | 79.09% | 0.2063 | 93.94% | 0.3898 |
| Unseen | 80.06% | 0.1986 | 95.07% | 0.3814 |
| Hard | 85.47% | 0.2905 | 94.87% | 0.5321 |

В отдельных folds nominal coverage провален (минимум 95%-coverage ≈81.7%).
Поэтому `interval_status=empirical_oof_not_certified`; калиброванность не заявляется.
MAPIE/EnbPI не запускались. Выбор baseline использует тот же OOF, поэтому
selection bias остаётся даже при раздельной cross-fold оценке остатков.

## Гипотезы: наблюдение → гипотеза → одинаковый тест → результат → решение

| Наблюдение | Гипотеза | Одинаковый тест | Результат | Решение |
|---|---|---|---|---|
| Нерегулярные интервалы | Linear улучшит mean neighbors | Все A/B/C/D keys | 0.107668 > 0.106699 composite | reject |
| Форма сезона нелинейна | PCHIP/Akima улучшат интерполяцию | Те же 13 317 OOF оценок | 0.110616 / 0.113545 | reject |
| Датчики меняются по годам | Одна сторона надёжнее среднего | Те же folds и fallback | nearest 0.120761–0.121105 | reject |
| Есть сезонность культуры | Seasonal prior заменит локальную историю | Те же folds | 0.173387 | reject как основная модель; keep как fallback |
| Есть выбросы raw target | Clip может помочь | OOF clip-сравнение пока не выполнено | Нет доказательства выигрыша | Не применять clip |
| 85.1% test gaps на новых полигонах | Random row split оптимистичен | CV-A имитирует unseen; CV-B исключает polygon | A unseen 0.089297, B 0.095591 | keep grouped CV |
| Gap скрывает dynamic/derived row | Старая климатология может давать leakage | Poison target/dynamic, затем exact mask | Признаки побитово равны | keep полное маскирование |
| pandas может хранить даты в разных единицах | int64 timestamp деление может исказить days | Toy расстояние ровно 5 дней | Явное datetime64[D] проходит | keep явные единицы |
| Фильтрация test gaps сохраняет исходный index | Смешение loc и позиции ломает диагностику | Fixture с непоследовательными индексами | Исправлено reset_index, тест проходит | keep |
| Baselines не достигают 0.06 | CatBoost GPU + source probabilities дадут выигрыш | Те же A/B/C/D keys на Kaggle GPU | Composite 0.099562 против 0.106699; A RMSE 0.066256 | keep CatBoost |
| Модели дополняют друг друга | Неотрицательный OOF blend уменьшит ошибку | Те же folds и заранее заданный composite | 0.097412 против 0.099562 у CatBoost | keep blend 0.309550/0.690450 |
| Ошибки зависят от контекста | Conservative gates улучшат устойчивость | Медиана repeats, composite и unseen Δ≤0.003 | Все 6 gates нарушили хотя бы один критерий | reject все gates |
| CatBoost переносит сезонную динамику | Модель улучшит past-only C | Holdout 2024 без будущего контекста | 0.246606 против 0.178772 baseline | reject CatBoost как forecasting-only модель; сохранить blend по composite |

Whittaker/HANTS, географические кластеры, MAPIE и продвинутый детектор не
добавлялись как непроверенные улучшения. В CSV отсутствует геометрия/region.

## Проверки и leakage evidence

20 unittest tests прошли. Тесты выполняют schema/key checks, текущие контрольные
числа, exact mask с status/z-score, target/source hierarchy, poisoned target и
dynamic fields sentinel, исключение групп из priors, leave-one-polygon-out
same-date median, детерминизм folds после shuffle, past-only C, remote-neighbor D,
train/inference feature parity, finite fallback, toy interior/edge/length-4/empty,
детерминизм predict, schema/hash bundle rejection, строгий submission, CLI exit
code, API/CLI equality и Pydantic smoke. Кроме unit-тестов, реальный Kaggle run
загрузил сохранённый CatBoost bundle, выполнил batch inference и post-validation.

Доказательство в пределах проверяемых инвариантов:

1. `test_sentinel_all_dynamic`: прибавление/подмена validation target и всех
   числовых dynamic/derived значений на 1e9 до mask не меняет итоговые features;
   даже прямой build_features самостоятельно маскирует gap keys.
2. State хранит fit keys и отказывается строить признаки при пересечении с
   requested gaps. В grouped test изменение всех targets held-out полигона на
   1e9 не меняет сериализованные fitted priors.
3. Сам текущий polygon исключён из same-date cross-polygon статистик; все
   реальные synthetic gaps и selected pseudo gaps скрыты до context features.
4. Temporal state/context не содержит данных holdout года. D оставляет ровно
   один конечный сосед >15 дней и не допускает этот polygon в fit.
5. Batch и Python API вызывают один build_features, один bundle и одно predict;
   совпадение на fixture проверено до 1e-15. Реальный trained bundle сформировал
   3 112 test-предсказаний; raw CSV fingerprints совпали с baseline-запуском.

Это набор воспроизводимых проверок, а не формальное доказательство отсутствия
всех возможных утечек. Полный фиксированный цикл проверен на Kaggle; в коде нет
выбора по test predictions или leaderboard.

## Артефакты и backend

- Готовый JSON baseline bundle: `artifacts/ml/baseline_v1/bundle/`.
- Валидный submission: `artifacts/ml/baseline_v1/submission.csv`, 3 112 строк,
  три обязательные колонки, исходный порядок gap-строк, UTF-8, finite prediction.
- Отдельная диагностика: `artifacts/ml/baseline_v1/diagnostics.csv`.
- Обученный CatBoost bundle: `artifacts/ml/trained_gpu_v1/bundle/`.
- Финальный submission первого GPU-запуска:
  `artifacts/ml/trained_gpu_v1/submission.csv`, SHA256
  `12ed0c3b8c4da6c8d512b89cc996f1bdec707cfb83cf2496b9449c3bc36a4637`.
- Его OOF, decisions, source diagnostics, uncertainty и feature importance:
  `artifacts/ml/trained_gpu_v1/`.
- Kaggle GPU-архив: `artifacts/ml/kaggle_ndvi_gpu_v1.zip`, SHA256 рядом; файлы внутри
  дополнительно перечислены с хешами в `kaggle/package_manifest.json`.
- Контракт для backend/DL: `artifacts/ml/CONTRACT_CHANGELOG.md`.

Backend загружает один `load_reconstructor(bundle_path)` на worker и вызывает:

```python
from veg_recovery.contracts import ReconstructionRequest, ReconstructionPayload
from veg_recovery.inference import load_reconstructor

model = load_reconstructor("artifacts/ml/trained_gpu_v1/bundle", trusted=True)
result = model.predict(ReconstructionRequest(frame, gap_mask, "web"))
payload = ReconstructionPayload.from_result(result).model_dump(mode="json")
```

Для полного продуктового ряда:

```python
from veg_recovery.anomalies.baseline import reconstruct_product_series, detect_anomalies
series = reconstruct_product_series(frame, result, model.bundle.config["harmonization"])
anomalies = detect_anomalies(series)  # min 3 previous reference years, >=2 points
```

Observed primary сохраняется, reconstructed primary имеет конкурсную семантику;
`ndvi_harmonized` использует отдельную sensor-wise affine калибровку median/IQR.
Новые gap-точки получают confidence из context/source diagnostics. Отрицательные
events описывают совпадение с погодой/NDWI/согласованностью датчиков, без причинных
утверждений. Порог baseline не объявляется финальным обученным anomaly detector.

**Gate 1 и адаптированный GPU Gate 2 выполнены.** Пользовательским решением
CPU HGB/ExtraTrees заменены на CatBoost GPU; исходный формальный пункт Gate 2 об
их обязательном воспроизведении больше не применяется к текущему пути. Для Gate 3
trained bundle, API/CLI и Pydantic smoke прошли; реальный backend route smoke
выполнить невозможно, поскольку backend в workspace отсутствует. Необходима
интеграция приведённого контракта; ML-логику в route переписывать не нужно.

Открытые риски: сдвиг 2025, деградация CatBoost в temporal C, малый и
неоднородный hard holdout, coverage отдельных folds ниже nominal, исходные
target-выбросы, OOF selection bias и приближённая harmonization. Следующий
независимый шаг — 30-trial GPU tuning с финальной оценкой на тех же A/B/C/D.
