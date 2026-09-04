# Разработчик 1 — Machine Learning

Версия плана: 2026-09-04  
Роль: владелец конкурсной метрики, leakage-safe валидации и общего inference-core  
Главный результат: воспроизводимый submission.csv и модель, которую без переписывания вызывает web-сервис

---

# Часть 1. Что нужно прочитать человеку

## 1. Миссия

Вы отвечаете за максимальный GapScore и доказуемость результата. Сначала создайте правильную локальную проверку, затем улучшайте модель. Не выбирайте метод по одной случайной разбивке и не допускайте, чтобы скрываемое значение попадало в признаки через интерполяцию, агрегаты или климатологию.

Формула оценки:

GapScore = round(30 × max(0, 1 − RMSE / 0.10), 2).

## 2. Что уже установлено исследованием данных

- train_dataset.csv: 99 955 строк, 39 полигонов, 2010–2024.
- test_data.csv: 57 185 строк, 78 полигонов, 3 112 контрольных строк.
- В тесте 39 знакомых и 39 новых полигонов; 464 gap-строки относятся к знакомым, 2 648 — к новым полигонам.
- У gap-строки скрыты не только primary_ndvi, но и все динамические/вычисляемые признаки. Доступны идентификатор полигона, date, crop_type и is_synthetic_gap. year и doy разрешено восстановить из date.
- 2 827 gap-серий имеют длину 1; 136 — длину 2; три — длину 3; одна — длину 4. Для 3 086 из 3 112 точек видны соседи с обеих сторон.
- На всех 48 161 видимых target-значениях train+test подтверждено точное правило:
  Sentinel-2 NDVI, иначе Landsat NDVI, иначе MODIS NDVI.
- Датчики очень разрежены; в отдельных индексах есть физически невозможные выбросы. Нельзя слепо доверять необработанным EVI/NDWI/NDVI.
- Локальный R&D-снимок на корректном маскировании:

| Метод | RMSE | Оценочный GapScore | Вывод |
|---|---:|---:|---|
| Линейная интерполяция primary_ndvi | ~0.0905 | ~2.85 | обязательный baseline |
| Интерполяция с прогнозом источника | ~0.0762 | ~7.14 | источник съёмки очень важен |
| HistGradientBoosting, matched-mask CV | ~0.0600 | ~12.00 | сильный основной кандидат |
| HGB + ExtraTrees, holdout-полигоны | ~0.0614 | ~11.57 | устойчивый стартовый ансамбль |

Эти числа — отправная точка, а не финальный результат. Их надо воспроизвести в репозитории на зафиксированных фолдах.

## 3. Техническое решение

P0:

1. Воспроизводимый генератор псевдопропусков, который копирует реальную маску test.
2. Baseline: ближайшие соседи и линейная интерполяция.
3. Источник primary_ndvi как отдельная классификационная задача.
4. Контекстные признаки без утечки и модели HGB + ExtraTrees.
5. OOF-подбор весов ансамбля и безопасные fallback-правила.
6. Одна команда batch-инференса и строгая проверка submission.

P1:

1. CatBoost и LightGBM на тех же фолдах.
2. Source-aware gating и отдельная логика для редкого контекста.
3. Калиброванные интервалы неопределённости из OOF-остатков.
4. Базовый гармонизированный ряд и robust anomaly baseline для передачи DL/backend.

P2:

1. Whittaker/HANTS как дополнительные признаки или эксперт ансамбля.
2. Осторожные cross-polygon same-date признаки.
3. Conformal/EnbPI только если coverage проверен на temporal/grouped CV.

Не делать основной моделью cubic spline, PCHIP или Transformer без одинакового CV: в первичной проверке PCHIP и Akima были хуже линейной интерполяции.

## 4. Связь с баллами и сроками

| Вклад роли | Максимум в критериях | Что должно стать доказательством |
|---|---:|---|
| GapScore | 30 | валидный submission и честный RMSE |
| Baseline/эксперименты | 10 | воспроизводимые фолды и ablation |
| Anomaly foundation | часть 7 | отдельный harmonized series и baseline events |
| Код/воспроизводимость | 8 | shared package, tests, manifest, CLI |
| Дополнительная идея | часть 5 | source-aware ensemble/uncertainty с измерением |

| Период | Результат |
|---|---|
| Первые 4 часа | data contract, MaskSpec, frozen fold keys, linear baseline |
| 4–12 часов | source classifier, feature builder, HGB/ExtraTrees OOF |
| 12–24 часа | ensemble, model bundle, валидный submission, parity test |
| 24–48 часов | CatBoost/LightGBM ablations, subgroup analysis, uncertainty |
| 48–72+ часов | harmonization/anomaly polish и только подтверждённые P2 |

Ближайший milestone: Gate 1 с валидным submission. Главный blocker: изменение shared schema без согласования. Сообщите Backend contract inference.py сразу после стабилизации P0, не дожидаясь последнего tuning.

## 5. Стек и установка

Базовый стек: Python 3.11, uv, pandas/polars, NumPy, SciPy, scikit-learn, CatBoost, LightGBM, Optuna, MLflow, joblib, Pydantic, PyYAML, pytest, Ruff, mypy, SHAP, MAPIE.

Ожидаемый интерфейс установки:

~~~bash
uv sync --extra ml
uv run pytest -q
~~~

Версии фиксируются в pyproject.toml и uv.lock. Не устанавливайте библиотеки вручную внутрь ноутбука.

## 6. Ваши выходы

- src/veg_recovery/validation/ — фолды и маскирование.
- src/veg_recovery/features/ — единый feature builder.
- src/veg_recovery/models/ — source classifier, регрессоры, ensemble/gating.
- src/veg_recovery/inference.py — публичный интерфейс для CLI и backend.
- src/veg_recovery/anomalies/baseline.py — простой интерпретируемый baseline.
- configs/ml/*.yaml — конфигурации и seed.
- artifacts/ml/ — сериализованный model bundle и manifest.
- reports/experiments.csv — все опыты, включая отрицательные.
- reports/ml_ablation.md — baseline → гипотеза → опыт → решение.
- submission.csv — ровно 3 112 строк с полями anon_polygon_id,date,primary_ndvi_pred.

## 7. Definition of Done

- Один и тот же model bundle используется batch CLI и backend.
- Псевдопропуск скрывает все поля, реально скрытые в test.
- Зафиксированы matched-mask, unseen-polygon, temporal и hard-gap оценки.
- В experiments.csv есть RMSE, GapScore, срезы, runtime, seed, commit и решение.
- submission содержит только gap-ключи, без NaN/inf/дубликатов/лишних строк.
- Команда ниже работает в чистом окружении:

~~~bash
uv run veg-recovery batch \
  --input data/raw/private_features.csv \
  --model artifacts/ml/final_bundle \
  --output submission.csv
~~~

## 8. Координация

- Backend получает от вас стабильный интерфейс inference.py и ModelManifest; не передавайте ему ноутбук.
- DL обязан использовать ваши фолды и feature/data contracts. Он не создаёт альтернативную «удобную» валидацию.
- Beginner делает независимые проверки и отчёты, но финальную проверку submission подписываете вы.
- До готовности P0 не тратьте время на внешнее дообучение или тяжёлый DL.

## 9. Что прочитать

- Case/criteria в корне проекта — источник истины.
- Whittaker: https://algorithm-catalogue.apex.esa.int/apps/whittaker
- HANTS: https://doi.org/10.1016/j.rse.2015.03.018
- BFAST: https://doi.org/10.1016/j.rse.2009.08.014
- CatBoost categories: https://catboost.ai/docs/en/features/categorical-features
- CatBoost missing values: https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing
- LightGBM: https://lightgbm.readthedocs.io/
- Optuna: https://optuna.readthedocs.io/en/stable/
- SHAP TreeExplainer: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
- MAPIE: https://mapie.readthedocs.io/

---

# Часть 2. Техническое задание для кодингового агента

Ниже расположен самодостаточный prompt. Выполняй его как инженерную задачу, а не как предложение вариантов.

## ROLE

Ты senior ML engineer проекта восстановления спутникового NDVI. Твоя зона ответственности — корректная валидация, максимальный GapScore, воспроизводимый batch inference и стабильный Python API для web-сервиса. Работай от тестов и измерений. Не меняй постановку задачи и не внедряй DL.

## SOURCE OF TRUTH

1. case_doc.pdf и criteria.pdf.
2. Реальные схемы train_dataset.csv и test_data.csv.
3. Этот документ.
4. Если документация и данные расходятся, зафиксируй расхождение в reports/data_contract_issues.md и проектируй под фактическую схему, не нарушая правила.

Формат submission:

- только строки test, где is_synthetic_gap == True;
- колонки строго anon_polygon_id,date,primary_ndvi_pred;
- ключ anon_polygon_id+date встречается ровно один раз;
- date в YYYY-MM-DD;
- primary_ndvi_pred — конечное вещественное число;
- UTF-8, разделитель запятая;
- ожидаемое число строк для текущего test_data.csv: 3 112.

## НЕИЗМЕНЯЕМЫЕ ФАКТЫ И ОГРАНИЧЕНИЯ

- На настоящей gap-строке отсутствуют все динамические и вычисляемые признаки. Признак, вычисленный из скрытого значения той же строки, запрещён.
- year/doy/sin(doy)/cos(doy) можно вычислять из date.
- Видимые target в test законно доступны как временной контекст. Используй их transductively, но не допускай self-target leakage при построении псевдообучения.
- primary_ndvi — сырой конкурсный ряд с иерархией S2 → Landsat → MODIS. Не подменяй его гармонизированным продуктовым рядом.
- Для anomaly/UI создай отдельное поле ndvi_harmonized; оно не является submission target.
- Финальные веса, пороги и clip выбираются только по OOF/валидации, не по leaderboard.

## ВЛАДЕНИЕ ПУТЯМИ

Можно создавать/изменять:

- src/veg_recovery/contracts.py
- src/veg_recovery/data/
- src/veg_recovery/validation/
- src/veg_recovery/features/
- src/veg_recovery/models/
- src/veg_recovery/inference.py
- src/veg_recovery/cli/batch.py
- src/veg_recovery/anomalies/baseline.py
- configs/ml/
- tests/ml/
- artifacts/ml/
- reports/experiments.csv
- reports/ml_ablation.md

Не меняй frontend, provider adapters и backend routes. Изменения shared contracts согласуй через Pydantic-модели и changelog.

## ЦЕЛЕВОЙ ПУБЛИЧНЫЙ КОНТРАКТ

Реализуй:

~~~python
@dataclass(frozen=True)
class ReconstructionRequest:
    frame: pd.DataFrame
    gap_mask: pd.Series
    context_mode: Literal["competition", "web"]

@dataclass(frozen=True)
class ReconstructionResult:
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    model_version: str

class NDVIReconstructor(Protocol):
    def predict(self, request: ReconstructionRequest) -> ReconstructionResult: ...
~~~

predictions содержит ключи, primary_ndvi_pred, lower, upper и method. diagnostics содержит source probabilities, distances to context, model disagreement, fallback_reason и quality flags. Для submission CLI берёт только три обязательные колонки.

ModelManifest должен хранить schema_version, created_at, git_commit, train fingerprints, feature_version, model files with SHA256, seeds, CV summary и package versions. Загрузка bundle обязана отклонять несовместимую schema_version.

## ПОРЯДОК РЕАЛИЗАЦИИ

### Шаг 1. Зафиксировать data contract

1. Прочитай CSV с явными dtype, parse_dates и проверками.
2. Уникальность ключа anon_polygon_id+date — обязательна.
3. Не считай строку наблюдением target, если primary_ndvi не finite.
4. Выведи автоматически:
   shapes, polygons, date ranges, crop types, target/gap counts, missingness.
5. Напиши тесты на текущие контрольные числа и отдельные toy fixtures.
6. Исходные CSV никогда не изменяй на месте.

### Шаг 2. Реализовать реальное маскирование

Создай MaskSpec и apply_mask(frame, selected_keys). Для выбранной target-строки выставляй NaN во всех полях, которые NaN на настоящих test gaps:

- primary_ndvi;
- s2_ndvi, s2_evi, s2_ndwi;
- landsat_ndvi, landsat_evi, landsat_ndwi;
- modis_ndvi, modis_evi;
- era5_temp_c, era5_precip_mm;
- year, doy;
- ndvi_climatology_mean, ndvi_climatology_std;
- ndvi_zscore, status, n_reference_years, если они присутствуют.

После маски year и doy пересчитывай только из date внутри feature builder. Никогда не оставляй исходные климатологию/status/z-score в validation row.

Скопируй распределение реального test:

- длины последовательных серий gaps;
- положение внутри сезона;
- доли знакомых/новых полигонов;
- расстояния до ближайшего видимого target;
- приближённое распределение истинного источника датчика.

Сохраняй фолды как ключи, а не номера строк: configs/ml/folds_v1.parquet либо CSV.

### Шаг 3. Создать четыре режима CV

CV-A matched-mask:

- 5 повторов с разными seed;
- псевдопропуски на известных target;
- распределение максимально похоже на test;
- этот режим основной для leaderboard-like выбора.

CV-B unseen-polygon:

- GroupKFold по anon_polygon_id;
- полностью исключай validation-полигоны из fit и из обучающих агрегатов;
- видимая история внутри held-out полигона остаётся разрешённым inference context.

CV-C temporal:

- holdout позднего года/сезона внутри полигонов;
- любые target-derived priors строятся только на прошлом fit-периоде, если они имитируют forecasting;
- дополнительно можно показать interpolation setting с видимыми обеими сторонами, но маркируй отдельно.

CV-D hard:

- gap-run длиной 2–4;
- один доступный сосед;
- расстояние до соседа > 15 дней;
- редкий crop/source;
- новые полигоны.

Для каждой оценки выводи overall RMSE/GapScore, known/unseen, crop, source, gap length, left/right distance bins, year, confidence bin. Composite decision score не скрывай: по умолчанию 50% CV-A, 25% CV-B, 15% CV-C, 10% CV-D; поменять веса можно только с письменным обоснованием.

### Шаг 4. Baselines

Реализуй единый Baseline API:

1. nearest left/right;
2. mean two neighbors — официальный ориентир;
3. time-weighted linear interpolation;
4. seasonal crop climatology;
5. optional PCHIP/Akima;
6. Whittaker/HANTS только как отдельный эксперимент.

Fallback для края сезона:

1. один сосед + crop/polygon seasonal prior;
2. polygon×crop robust climatology;
3. crop robust climatology;
4. global DOY climatology.

Никакой fallback не может вернуть NaN. Сохрани baseline predictions и ошибки на каждом OOF-ключе.

### Шаг 5. Восстановить и проверить источник primary_ndvi

На каждой видимой строке присвой source_label:

1. s2, если finite s2_ndvi и оно равно primary_ndvi в численном tolerance;
2. иначе landsat;
3. иначе modis;
4. иначе unknown.

Сформируй отчёт равенства и несовпадений. Затем обучи source classifier без использования значений скрываемой строки.

Разрешённые признаки:

- date, year, month, ISO week, doy cyclic;
- polygon/crop category;
- исторические календари доступности каждого датчика;
- same-date counts/flags других видимых строк;
- days since/until observation каждого датчика;
- региональные/кластерные календари, если построены без validation label.

При создании same-date признаков исключай текущую masked row. В grouped CV исключай fit-labels held-out полигона, но разрешай реальные видимые соседние строки как inference context. Сохраняй p_s2, p_landsat, p_modis и калибровку/accuracy по source.

### Шаг 6. Feature builder

Все признаки строятся одной чистой функцией build_features(frame, gap_keys, fitted_state). Для каждой gap-точки создай:

- календарь: year, month, doy, sin/cos doy;
- k=1..4 предыдущих и следующих видимых primary_ndvi;
- time delta к каждому соседу;
- linear interpolation, local level, slope, curvature;
- количество наблюдений и плотность в окнах 7/15/30 дней;
- положение в gap-run, длина run, edge flags;
- source probabilities;
- отдельную интерполяцию s2_ndvi, landsat_ndvi, modis_ndvi;
- соседние EVI/NDWI и их robust summaries;
- weather reconstruction из доступных соседних/ same-date данных с availability flags;
- polygon/crop/year robust climatology, рассчитанную только по разрешённым target;
- same-date cross-polygon median/IQR/count с leave-one-polygon-out логикой;
- seen_polygon, crop frequency, context quality.

Для каждого числового исходника храни raw, cleaned и invalid_flag. Не clip target автоматически. Физические проверки используй как признаки/фильтры источника. Финальный clip применяй только если OOF показывает улучшение и фиксируй его границы в config.

Напиши leakage sentinel test: добавь validation target огромную константу до маскирования; итоговые признаки не должны измениться.

### Шаг 7. Модели и tuning

P0-модели:

- HistGradientBoostingRegressor;
- ExtraTreesRegressor.

P1:

- CatBoostRegressor с нативными категориями;
- LightGBM.

Не one-hot кодируй высококардинальный polygon_id для CatBoost. Unknown categories должны обрабатываться явно. Оптимизируй Huber/RMSE варианты только по одинаковым folds. Optuna: сначала 30–60 разумных trials, pruning, один и тот же seed set, запрет подбирать по test predictions.

Сохраняй для каждого model:

- OOF prediction per key;
- fold metrics;
- feature set hash;
- hyperparameters;
- runtime and peak memory;
- feature importance/SHAP summary;
- decision: keep/reject and reason.

### Шаг 8. Ensemble и gating

1. Подбери non-negative weights, sum=1, только на OOF.
2. Сравни global blend с простым 50/50 HGB+ExtraTrees.
3. Проверь gates:
   - короткий двусторонний контекст;
   - длинный/односторонний gap;
   - source confidence;
   - seen/unseen polygon;
   - model disagreement.
4. Gate принимается только если улучшает не один fold, а медиану повторов и не портит unseen RMSE более чем на 0.003.
5. При высоком disagreement или низком context quality переходи к conservative blend/fallback, а не возвращай экстремум.

### Шаг 9. Неопределённость

P0: lower/upper из эмпирических OOF-остатков по context/source bins.

P1: MAPIE split/cross conformal или EnbPI только на корректной calibration схеме. Измерь coverage 80% и 95%, среднюю ширину и coverage по unseen/hard. Не называй интервалы калиброванными, если subgroup coverage провален.

### Шаг 10. Два ряда и anomaly baseline

Верни:

- primary_ndvi_reconstructed — ровно конкурсная семантика;
- ndvi_harmonized — продуктовый ряд после sensor-wise robust affine/quantile calibration к выбранному reference sensor.

Не сглаживай submission target так, чтобы потерять иерархию датчиков.

Для baseline anomaly:

1. по ndvi_harmonized построить polygon/crop robust DOY climatology;
2. median + MAD либо quantile bands с минимальным числом reference years;
3. residual и robust_z;
4. только отрицательные события;
5. объединение соседних точек в event с persistence;
6. severity = функция magnitude × duration × confidence;
7. reason codes на основании precipitation/temp/NDWI/multi-sensor agreement;
8. формулировки только «совпадает с / согласуется с», без причинного утверждения.

Передай контракт DL и backend; продвинутый детектор принадлежит DL.

### Шаг 11. Batch inference

Команда должна:

1. прочитать CSV и проверить schema;
2. вычислить gap mask;
3. загрузить bundle один раз;
4. построить признаки тем же кодом, что в CV;
5. предсказать;
6. применить зафиксированные fallback/clip;
7. сохранить submission в исходном порядке gap-строк;
8. выполнить строгую post-validation;
9. сохранить diagnostics отдельно, не добавляя их в submission.

Exit code != 0 при несовпадении ключей, NaN, inf, duplicate, лишней колонке, неправильной кодировке или нечисловом prediction.

### Шаг 12. Эксперименты и отчёт

reports/experiments.csv имеет минимум:

experiment_id, timestamp, git_commit, data_fingerprint, fold_version,
mask_version, feature_version, model, params_json, seed, split,
rmse, gap_score, known_rmse, unseen_rmse, hard_rmse, runtime_sec,
peak_memory_mb, artifact_uri, conclusion.

Каждая строка в ml_ablation.md:

Наблюдение → гипотеза → одинаковый тест → результат → решение.

Не удаляй отрицательные эксперименты.

## ОБЯЗАТЕЛЬНЫЕ ТЕСТЫ

- schema and key uniqueness;
- exact synthetic masking;
- target/source hierarchy;
- no target leakage sentinel;
- grouped aggregate exclusion;
- feature parity train/inference;
- deterministic fold generation;
- deterministic prediction within tolerance;
- model bundle manifest/hash;
- no-NaN fallback;
- strict submission schema and exact key set;
- toy series: interior, edge, length-4 gap, all-neighbors-missing;
- end-to-end CLI on a small fixture.

## ACCEPTANCE GATES

Gate 1 — baseline:

- reproducible folds;
- official neighbor baseline;
- end-to-end valid submission.

Gate 2 — strong ML:

- HGB and ExtraTrees reproduced;
- OOF ensemble no worse than best member on composite score;
- full subgroup table exists.

Gate 3 — production:

- bundle loads via inference.py;
- backend smoke-call passes;
- batch and API prediction for the same fixture are equal.

Target для P0: повторить RMSE около 0.06 на matched-mask CV и не деградировать критично на unseen polygons. Это ориентир, не разрешение подгонять folds.

## ЗАПРЕТЫ

- Нельзя использовать ndvi_zscore/status как validation feature после маскирования.
- Нельзя строить climatology/rolling/same-date aggregate с текущим target.
- Нельзя выбирать модель по одной random split или leaderboard.
- Нельзя заменять реальный missingness нулём без отдельного missing flag.
- Нельзя переписывать ML-логику в API.
- Нельзя сохранять pickle из непроверенного источника или без manifest/hash.
- Нельзя выдавать возможную причину anomaly как доказанную причинность.

## ФОРМАТ ФИНАЛЬНОГО ОТЧЁТА АГЕНТА

В конце работы сообщи:

1. изменённые файлы;
2. команды запуска;
3. тесты и их результат;
4. таблицу CV-A/B/C/D;
5. baseline и финальный ансамбль;
6. принятые/отклонённые гипотезы;
7. путь к model bundle и submission;
8. открытые риски;
9. что именно должен подключить backend;
10. доказательство отсутствия leakage.
