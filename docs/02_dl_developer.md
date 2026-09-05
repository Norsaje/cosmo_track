# Разработчик 2 — Deep Learning и продвинутая детекция аномалий

Версия плана: 2026-09-05
Роль: R&D-владелец нейросетевой импутации, uncertainty и событий аномалий
Главный результат: честное сравнение DL с сильным ML и улучшение anomaly-подсистемы без риска для MVP

## Уточнения после проверки репозитория 2026-09-05

Исходный `main` (6d5e5fa) содержит постановку и данные, но не содержит
C-01/C-03, ML OOF, pyproject.toml или uv.lock. Числа RMSE из исходного R&D
ниже остаются ориентирами, а не воспроизведёнными результатами этой ветки.
Командные статусы публикуются непосредственно в `main:00_team_coordination.md`.

- Маска применяется ко **всему разрешённому контексту до windowing**, включая
  все held-out keys, которые могли бы попасть в соседние перекрывающиеся окна.
  Scaler, interpolation base и любые priors строятся после удаления этих значений.
- Training/inner early-stop/outer OOF labels разделены. Inner keys предоставляет
  ML; DL не создаёт собственную validation split. Если inner keys не предоставлены,
  нужен согласованный ML fixed-epoch policy либо handoff этих keys.
- Composite — взвешенная сумма RMSE CV-A/B/C/D, отдельно от row-weighted overall
  RMSE/GapScore. Отсутствующий режим не получает автоматически нулевой вес.
- До появления сравнимых OOF статус решения — `PENDING_EVALUATION`.
  `REJECT` означает проведённый эксперимент, а не отсутствие dependencies.
- Stock PyPOTS SAITS.fit маскирует отдельные ячейки MCAR. Для конкурсного
  эксперимента потребуется matched-mask training bridge; одна оболочка impute
  и успешный тест формы массива не являются воспроизведением BRITS/SAITS.
  Источник: https://github.com/WenjieDu/PyPOTS/blob/main/pypots/imputation/saits/data.py.
- В C-09 confidence — heuristic support, не калиброванная вероятность anomaly.
  Недостаточная история возвращает diagnostic warning; отсутствие события при
  недостатке данных нельзя интерпретировать как подтверждённую норму.
- LOYO исключает оцениваемый год у всех reference polygons. Используется равный
  вес reference years; выборка с ежедневными наблюдениями не считается сотнями лет.
- Reference calibration/LOYO anomaly fit принимает только разрешённые train
  observations. Unsupported sensor mapping не выдаётся за гармонизацию.
- GPU: по указанию пользователя полный CUDA-прогон подготавливается для Kaggle.
  Локальные unit/smoke проверки остаются на CPU. Notebook и инструкция:
  `configs/dl/kaggle/`. Никаких pretrained downloads в runtime.
- До SH-002 доступны команды `PYTHONPATH=src python ...`; обещание
  `uv sync --extra dl` вступит в силу после handoff Backend. Общий lock принадлежит Backend.

Текущая реализация, ограничения и точные команды: `reports/dl_decision.md`.
Предложение consumer для будущего C-03: `configs/dl/c03_consumer.md`.

---

# Часть 1. Вводные

## 1. Миссия

Ваша задача — не «обязательно поставить Transformer», а проверить, даёт ли современная импутация измеримый прирост на малом разреженном наборе. Если DL не выигрывает, это нормальный R&D-результат: переключитесь на ансамбль, uncertainty, sensor harmonization и сильную детекцию аномалий.

Критический путь submission принадлежит ML-разработчику. Вы используете его data contracts, маскирование и фолды без изменений.

## 2. Почему DL здесь не очевидный победитель

- Наблюдаемых target около 48 тыс. в доступных train+test, а уникальных обучающих полигонов мало.
- Почти все контрольные пропуски короткие: 2 827 одиночных gap и только четыре серии длиннее двух точек.
- Динамические признаки в gap-строке полностью скрыты.
- Новые полигоны доминируют в test gaps.
- Сильный HGB/ExtraTrees уже даёт около 0.06 RMSE в локальном R&D.

Поэтому DL должен выигрывать на одинаковой leakage-safe проверке, а не на случайном mask/reconstruction split.

## 3. Приоритеты

P0:

1. Принять единый WindowDataset и готовые fold keys от ML.
2. Воспроизвести BRITS и SAITS через PyPOTS.
3. Создать masked TCN/GRU baseline с missingness masks и delta-time.
4. Вернуть OOF-predictions для общего ensemble.
5. Реализовать robust anomaly events поверх отдельного ndvi_harmonized.

P1:

1. CSDI либо другой probabilistic imputer для uncertainty на hard gaps.
2. Self-supervised pretraining на всех видимых рядах без скрытых labels.
3. Crop/source-aware conditioning.
4. Change-point и phenology признаки как дополнение, а не замена robust climatology.

P2:

1. MOMENT/TimeMAE/SimMTM representations для anomaly/phenology.
2. HELIX — только изолированный эксперимент: работа очень новая и не должна становиться зависимостью MVP.
3. DL residual/gating в ансамбле, если OOF подтверждает пользу.

## 4. Связь с баллами и сроками

| Вклад роли                        | Максимум в критериях | Доказательство                                        |
| ------------------------------------------ | -------------------------------------: | ------------------------------------------------------------------- |
| Возможный прирост GapScore |                                     30 | только одинаковый OOF/CV с ML                      |
| Детекция аномалий          |                                      7 | события, severity, кейсы, false-positive tests          |
| Дополнительные идеи      |                                      5 | uncertainty/representations с измеримой пользой    |
| Исследование                   |                                     10 | отрицательные и положительные опыты |
| Интерпретируемость/UX    |                          часть 10 | reason codes и confidence contract                                 |

| Период            | Результат                                                   |
| ----------------------- | -------------------------------------------------------------------- |
| Первые 4 часа | WindowDataset, equality-check folds/metrics                          |
| 4–12 часов        | малый TCN/GRU и один PyPOTS baseline                       |
| 12–24 часа         | SAITS/BRITS comparison и ADOPT/REJECT pre-decision                  |
| 24–48 часов       | anomaly events, harmonization, uncertainty                           |
| 48–72+ часов      | CSDI/pretrained/HELIX только после основных gates |

Ближайший milestone: одна DL-модель с OOF на тех же ключах, что HGB. Blockers: frozen folds и ML OOF predictions. Если они задерживаются, можно писать dataset/tests и anomaly synthetic tests, но нельзя придумывать альтернативную split.

## 5. Жёсткий gate внедрения

DL попадает в production, только если выполнено хотя бы одно:

- улучшает composite RMSE минимум на 0.002 относительно финального ML;
- добавление в OOF-ансамбль улучшает RMSE минимум на 0.001 без деградации unseen-polygons более чем на 0.003;
- даёт существенно лучшее uncertainty coverage/hard-gap качество при практически неизменном общем RMSE.

Результат должен повториться минимум на трёх seed или большинстве зафиксированных folds. Один красивый split не считается.

## 6. Стек

Python 3.11, uv, PyTorch, PyPOTS, NumPy, pandas, scikit-learn, einops, torchmetrics, MLflow, Optuna, Hydra/OmegaConf. Lightning допустим, если уменьшает код и не усложняет отладку. Для change points: ruptures; BFAST можно использовать как научный reference или отдельный R-эксперимент, но не вводить R в P0.

Ожидаемые команды:

```bash
uv sync --extra dl
uv run python -m veg_recovery.dl.train experiment=saits
uv run pytest -q tests/dl
```

Код должен работать на CPU на малом fixture. Полное обучение может использовать CUDA, но обязано иметь понятный CPU fallback.

## 7. Ваши выходы

- src/veg_recovery/dl/data.py — window adapter поверх shared contracts.
- src/veg_recovery/dl/models/ — BRITS/SAITS wrappers, TCN и optional CSDI.
- src/veg_recovery/dl/train.py и predict.py.
- src/veg_recovery/anomalies/advanced.py.
- src/veg_recovery/anomalies/events.py и explain.py.
- configs/dl/*.yaml.
- artifacts/dl/candidate_bundle/ — только если пройден gate.
- reports/dl_experiments.csv и reports/dl_decision.md.
- reports/anomaly_cases/ — минимум три воспроизводимых кейса.

## 8. Definition of Done

- Все DL-метрики посчитаны на fold keys ML-разработчика.
- Есть сравнение с теми же linear/HGB/ensemble predictions.
- Проверена стабильность минимум на трёх seed.
- Есть таблица overall/unseen/hard/source/gap-length.
- Есть решение adopt/reject с численным обоснованием.
- Advanced anomaly возвращает событие, severity, confidence и некаузальные reason codes.
- При отклонении DL production продолжает работать без PyTorch.

## 9. Что прочитать

- BRITS, NeurIPS 2018: https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html
- BRITS code: https://github.com/caow13/BRITS
- SAITS: https://arxiv.org/abs/2202.08516
- SAITS code: https://github.com/WenjieDu/SAITS
- PyPOTS imputation API: https://docs.pypots.com/en/latest/pypots.imputation.html
- CSDI, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html
- CSDI code: https://github.com/ermongroup/csdi
- MOMENT: https://arxiv.org/abs/2402.03885
- MOMENT code: https://github.com/moment-timeseries-foundation-model/moment
- TimeMAE: https://arxiv.org/abs/2303.00320
- SimMTM: https://openreview.net/forum?id=ginTcBUnL8
- HELIX, новый research-only кандидат: https://arxiv.org/abs/2605.02278
- BFAST: https://doi.org/10.1016/j.rse.2009.08.014
- ruptures: https://centre-borelli.github.io/ruptures-docs/

---

# Часть 2. Техническое задание для агента

## ROLE

Ты senior deep-learning researcher и anomaly engineer. Проверь современные методы импутации временных рядов на реальной постановке NDVI. Твоя работа экспериментальная и не может ломать общий inference. Используй общий data/validation pipeline. Внедряй только подтверждённые улучшения.

## SOURCE OF TRUTH

Приоритет:

1. case_doc.pdf и criteria.pdf;
2. ключи фолдов, MaskSpec, feature/data contracts ML-разработчика;
3. реальные CSV;
4. этот документ.

Не создавай свою random split, не оставляй динамические признаки в masked row и не используй скрытый target в loss context.

## ВЛАДЕНИЕ ПУТЯМИ

Можно изменять:

- src/veg_recovery/dl/
- src/veg_recovery/anomalies/advanced.py
- src/veg_recovery/anomalies/events.py
- src/veg_recovery/anomalies/explain.py
- configs/dl/
- tests/dl/
- tests/anomalies/
- artifacts/dl/
- reports/dl_experiments.csv
- reports/dl_decision.md
- reports/anomaly_cases/

Shared contracts, masking и folds менять нельзя без согласованного PR. Backend/frontend не трогать.

## ПРИНЦИП ИЗОЛЯЦИИ

- Core install не должен тянуть torch/PyPOTS.
- DL доступен через extra dl.
- model bundle имеет manifest, schema version, weights hash и preprocessor hash.
- Если DL artifact отсутствует, NDVIReconstructor молча не переключается на случайную модель: использует объявленный ML fallback и пишет diagnostics.
- Любой DL wrapper реализует тот же PredictionExpert protocol и возвращает prediction, uncertainty, method, diagnostics.

## DATA REPRESENTATION

### Единица последовательности

Основной вариант: один polygon-year/vegetation season на регулярной дневной сетке. Не смешивай разные полигоны в одну последовательность. Сохраняй реальный date и valid-time mask.

Дополнительный вариант: rolling window 61/91/121 день вокруг target. Размер выбирается только по validation.

### Каналы

Минимальный набор:

- primary_ndvi с observation mask;
- s2_ndvi, landsat_ndvi, modis_ndvi;
- EVI/NDWI с invalid-value flags;
- era5_temp_c, era5_precip_mm;
- sin/cos doy;
- crop embedding;
- predicted source probabilities;
- per-channel missingness masks;
- time delta since/until observation;
- context quality/edge flags.

Не заменяй missing на 0 без mask. Числа нормализуй статистиками train fold; validation/test не участвуют в fit статистик. Категории unseen получают отдельный token.

### Три разные маски

1. natural_missing_mask — исходное отсутствие наблюдения;
2. synthetic_gap_mask — целевые контрольные позиции;
3. valid_calendar_mask — реальная дата в последовательности, не padding.

Loss вычисляется только на искусственно скрытых, исходно наблюдаемых target. Padding и естественно неизвестный target не участвуют.

### Mask sampler

Импортируй MaskSpec ML-разработчика. Batch sampler должен воспроизводить:

- одиночные точки и runs длиной 2–4;
- test DOY/source/context-distance;
- one-sided/hard cases;
- known/unseen simulation.

При каждом validation run fold keys фиксированы. Случайное динамическое маскирование разрешено только внутри training batches.

## ПОРЯДОК РЕАЛИЗАЦИИ

### Шаг 1. Контракт и baseline harness

1. Напиши WindowDatasetAdapter, который принимает shared frame и mask keys.
2. Добавь unit tests на alignment date/key/channel/masks.
3. Загрузи OOF predictions linear, HGB, ExtraTrees/final ML как immutable baselines.
4. Реализуй общий evaluator с RMSE и GapScore; проверь, что на одних ключах метрики совпадают с ML.
5. Логируй data fingerprint, fold version и mask version.

Не начинай model tuning, пока equality-check метрик не пройден.

### Шаг 2. TCN/GRU masked baseline

Сделай небольшой bidirectional TCN либо GRU-D-like baseline:

- hidden size 64–128;
- 2–4 layers;
- dropout;
- delta-time и masks как входы;
- residual path от linear interpolation;
- выход residual correction предпочтительнее абсолютного NDVI;
- Huber/MSE сравнить на одинаковых folds;
- early stopping по validation RMSE;
- gradient clipping;
- deterministic seed.

Residual formulation:

prediction = linear_or_ml_base + bounded_residual.

Проверь tanh/soft bound, но не clip без ablation. Храни checkpoint только лучшего epoch.

### Шаг 3. BRITS

Используй официальный смысл bidirectional recurrent imputation, предпочтительно wrapper PyPOTS. Убедись:

- регулярная дневная сетка;
- missing mask соответствует API;
- target channel не содержит ground truth в masked positions;
- output alignment сохранён;
- fit не видит held-out fold.

Не форкай старый репозиторий, если PyPOTS даёт поддерживаемую реализацию. Зафиксируй версию и license.

### Шаг 4. SAITS

Реализуй через PyPOTS либо официальный SAITS:

- ограничь модель под размер данных;
- начни с d_model 64/128, 2 blocks, 4 heads;
- используйте early stopping;
- не делай огромный sweep;
- сохраняй attention только как diagnostic, не называй её причинным объяснением.

Сравни absolute reconstruction и residual-to-ML. Часто второй вариант безопаснее на малом наборе.

### Шаг 5. Мультизадачность source-aware

Опциональный auxiliary head предсказывает source label на видимых строках. Total loss:

L = L_masked_ndvi + lambda_source × L_source + lambda_smooth × L_local.

Smoothness penalty не должен стирать реальные резкие изменения; lambda выбирается CV. Source head получает только разрешённый контекст masked row. Удалить, если нет улучшения.

### Шаг 6. CSDI / probabilistic track

Запускать после SAITS/BRITS и только если есть GPU либо разумное CPU время.

Цель:

- не обязательно лучший point RMSE;
- samples/quantiles для hard gaps;
- проверяемая interval coverage.

Ограничить diffusion steps и размер сети. Сравнить mean/median samples. Измерять:

- RMSE point estimate;
- CRPS либо pinball loss, если реализован корректно;
- coverage 80/95%;
- interval width;
- inference seconds per polygon.

Не подключать, если demo latency и reproducibility неприемлемы.

### Шаг 7. Self-supervised / pretrained track

После основных моделей:

1. Masked autoencoding на всех доступных видимых train+test context без скрытых gaps.
2. MOMENT/TimeMAE/SimMTM embeddings для:
   - crop/phenology clusters;
   - anomaly residual model;
   - context feature для ML.
3. HELIX проверять только в отдельной ветке: очень новый метод, обязателен audit API/license/reproducibility.

Ни одна pretrained модель не скачивается в runtime demo. Веса кэшируются/документируются и хешируются. External model и license заносятся в README.

## ЕДИНЫЙ EXPERIMENT PROTOCOL

Для каждого кандидата:

- одинаковые fold keys;
- одинаковый MaskSpec;
- минимум 3 seed;
- фиксированный compute budget;
- overall и subgroup metrics;
- training/inference time;
- peak GPU RAM/CPU RAM;
- artifact size;
- failure count;
- сравнение с ML OOF на тех же строках.

reports/dl_experiments.csv:

experiment_id,model,commit,data_fingerprint,fold_version,mask_version,
seed,window,channels,params_json,epoch,rmse,gap_score,unseen_rmse,
hard_rmse,source_s2_rmse,source_landsat_rmse,source_modis_rmse,
coverage_80,width_80,train_sec,infer_sec,peak_vram_mb,artifact_mb,
decision,notes.

Bootstrap confidence interval для разницы RMSE считай по polygon или gap-run blocks, не по независимым строкам. Отрази variance across seeds.

## MODEL ADOPTION GATE

Кандидат проходит, если:

1. deterministic rerun в tolerance;
2. нет leakage;
3. overall composite gain соответствует порогу;
4. unseen degradation не превышает 0.003;
5. нет критического source/hard subgroup collapse;
6. inference укладывается в согласованный SLA;
7. artifact грузится без интернета;
8. backend может вызвать wrapper тем же contract.

Если point model отклонён:

- экспортируй OOF predictions;
- проверь его как low-weight ensemble expert;
- используй representations/uncertainty/anomaly;
- зафиксируй REJECT с причиной, не удаляй код/метрики.

## SENSOR HARMONIZATION

Аномалии считаются не по сырому переключающемуся primary target, а по ndvi_harmonized.

Реализуй и сравни:

1. reference sensor = Sentinel-2/HLS-like scale;
2. robust affine mapping Landsat→reference и MODIS→reference на overlapping/near-date observations;
3. partial pooling global + crop + polygon, чтобы unseen/малые группы не переобучались;
4. quantile mapping как ablation;
5. uncertainty/quality flags при source transitions.

Fit calibration только на train fold. Для малого overlap fallback: global source mapping. На UI обязательно сохранить source и raw value; гармонизация не должна скрывать происхождение данных.

## ADVANCED ANOMALY PIPELINE

### Шаг A. Ожидаемая траектория

Baseline:

- robust median по DOY window;
- MAD с нижним floor;
- quantile 10/25/50/75/90;
- crop-aware и polygon-aware partial pooling;
- минимальное число reference years.

Advanced candidates:

- model residual expected_ndvi = f(doy,crop,weather,history);
- phenology alignment между годами;
- Gaussian/state-space uncertainty, если надёжно;
- representation distance.

Текущий год не должен полностью входить в собственную climatology. Используй leave-one-year-out.

### Шаг B. Точечный сигнал

Вычисляй:

- residual = observed_or_reconstructed_harmonized − expected;
- robust_z;
- percentile;
- reconstruction uncertainty;
- sensor agreement;
- data quality.

Фокус — отрицательные отклонения. Одиночная реконструированная точка с широкой uncertainty не должна автоматически становиться critical.

### Шаг C. Событие

Объединяй соседние отрицательные точки, если разрыв не больше configurable max_gap_days. Event хранит:

- start/end/duration;
- minimum z and residual;
- area under negative residual;
- number observed vs reconstructed points;
- persistence;
- confidence;
- sensor consistency;
- severity class: normal / biomass_suppression / critical;
- algorithm version.

Пример стартовых порогов из постановки:

- z ≥ −1: normal;
- −2 ≤ z < −1: suppression;
- z < −2: critical.

Это baseline. Robust/event thresholds выбирай по историческим false-positive proxies и кейсам, не маскируй их под ground truth.

### Шаг D. Change points

ruptures PELT/Binseg или BFAST использовать как подтверждающий сигнал:

- change point без отрицательного residual не является негативной anomaly;
- не реагировать на нормальные sowing/harvest transitions;
- penalty и min segment length фиксировать в config;
- строить только на достаточно плотном гармонизированном ряду.

### Шаг E. Интерпретация

Reason codes:

- LOW_PRECIPITATION;
- HIGH_TEMPERATURE;
- LOW_NDWI;
- MULTISENSOR_CONFIRMATION;
- SOURCE_SWITCH_RISK;
- LOW_DATA_COVERAGE;
- RAPID_NEGATIVE_CHANGE;
- PROLONGED_SUPPRESSION;
- PHENOLOGY_SHIFT.

Текст формируется шаблонами:

«Снижение NDVI совпадает с осадками ниже сезонной нормы и повышенной температурой; это согласуется с водным стрессом. Причинность не доказана».

Нельзя писать «засуха вызвала падение» без внешней причинной верификации.

### Шаг F. Evaluation без размеченных аномалий

Сформируй:

- synthetic negative pulse tests разной magnitude/duration;
- sensor switch false-positive tests;
- cloud/outlier spike tests;
- leave-one-year-out stability;
- ручной review минимум 3 сильных и 3 ложных/сомнительных кейсов;
- detection delay, event IoU на synthetic events, false alerts per season;
- uncertainty-aware suppression tests.

В reports/anomaly_cases на каждый пример сохраняй series CSV/JSON, config, plot path и краткое объяснение.

## INTERFACE ДЛЯ BACKEND

```python
@dataclass(frozen=True)
class AnomalyEvent:
    start_date: date
    end_date: date
    severity: str
    score: float
    confidence: float
    min_robust_z: float
    negative_area: float
    observed_points: int
    reconstructed_points: int
    reason_codes: tuple[str, ...]
    explanation_ru: str
    algorithm_version: str
```

Detector API получает harmonized series + weather + quality; возвращает JSON-serializable events. Он не ходит в сеть и не читает БД.

## ТЕСТЫ

- exact masks/alignment/padding;
- normalization fit only on train fold;
- unseen category token;
- loss only on synthetic observed targets;
- deterministic seed on CPU fixture;
- model save/load equality;
- PyPOTS wrapper output alignment;
- residual base cannot see target;
- anomaly leave-one-year-out;
- synthetic pulse severity monotonicity;
- source-switch false-positive;
- wide uncertainty lowers confidence;
- reason templates avoid causal language;
- no-torch core import works.

## РЕСУРСНЫЕ ОГРАНИЧЕНИЯ

- Начинай с моделей <2 млн параметров.
- Не запускай sweep более 30 trials до доказательства пользы.
- Early stopping обязателен.
- AMP допустим на CUDA; результаты проверять без NaN.
- Логируй GPU model, CUDA/cuDNN, deterministic flags.
- Не добавляй distributed training в hackathon P0.

## ЗАПРЕТЫ

- Нельзя использовать удобную random point mask вместо официального MaskSpec.
- Нельзя обучаться на hidden test gap labels.
- Нельзя включать validation rows в scaler/climatology/source calibration fit.
- Нельзя объявлять attention объяснением причины.
- Нельзя включать модель только потому, что она новая.
- Нельзя делать PyTorch обязательным для ML/backend fallback.
- Нельзя менять submission schema.

## ФОРМАТ ФИНАЛЬНОГО ОТЧЁТА АГЕНТА

1. реализованные модели и пути;
2. точные команды;
3. hardware/software versions;
4. общая таблица одинакового CV;
5. seed variance и bootstrap CI разницы;
6. решение ADOPT / ENSEMBLE_ONLY / ANOMALY_ONLY / REJECT;
7. latency и размер artifacts;
8. anomaly algorithm и reason codes;
9. пройденные тесты;
10. известные ограничения и следующий самый ценный опыт.
