# DL R&D — 2026-09-05

Статус: DL-импутация впервые измерена на **реальных фолдах ML**; решение
об внедрении остаётся **PENDING_EVALUATION** по причине, которая не зависит
от качества DL: у ML не опубликованы построчные OOF сильной модели, поэтому
порог gate «улучшение composite на 0.002 относительно финального ML» не имеет
вычислимого знаменателя (B-DL-004). Production candidate не экспортирован.
C-06 поднят до 0.2, C-09 остаётся в REVIEW. ML submission от DL не зависит.

Что изменилось с прошлой публикации:

1. `veg_recovery.dl.c03_bridge` исполняет producer-код ML read-only и строит
   training manifest из его же outer keys, MaskSpec, censoring и baseline OOF.
   Свои folds DL по-прежнему не создаёт; статус входа —
   `derived_from_producer_artifacts`, не `accepted`.
2. **Исправлена утечка в нашем runner.** Inference-контекст брался срезом сырого
   кадра, поэтому producer censoring терялось: CV-C увидел бы наблюдения 2024
   года, CV-D — все точки полигона, кроме одного разрешённого соседа. Метрики
   C и D без этого исправления были бы недействительны.
3. Обучение идёт блоками: каждый блок маскируется отдельно, поэтому плотность
   искусственных пропусков в обучении близка к тестовой, а не в разы выше.
4. Добавлен `SeasonalPrior` и `base_mode=anchored`. В forecasting-режиме окно
   polygon-year пусто, и prior — единственный источник сезонной формы.
5. Epoch не выбирается по outer OOF: `--epoch-policy fixed`, бюджет закреплён
   заранее по inner-кривой пилота.
6. DL доступен за общим интерфейсом C-02 (`ResidualTCNExpert`), поэтому пункт 8
   adoption gate закрыт технически, а не декларативно.

<!--CV_RESULTS_PLACEHOLDER-->

## Проверенная постановка и данные

Изучены все 19 страниц `docs/case_doc.pdf`, `docs/criteria.pdf`, задания ML/DL/
Backend/Beginner, исходный R&D prompt и `main:00_team_coordination.md`.
PDF допускает статистические/ML/DL методы, не требует нейросеть. Задача DL —
подтверждённый прирост восстановления и полезные anomaly events для продукта.

Фактические CSV подтверждены read-only:

| Набор | Строки | Полигонов | Видимых primary_ndvi |
|---|---:|---:|---:|
| train_dataset.csv | 99 955 | 39 | 30 520 |
| test_data.csv | 57 185 | 78 | 17 641 |

Test gaps: 3 112; unseen: 2 648, known: 464. Все динамические и производные
поля gap-строк скрыты. В PDF test называется `private_features.csv`, фактический
файл — `test_data.csv`. Формулировка PDF о train с известным target не означает,
что все 99 955 строк имеют labels: конечных значений только 30 520.
Синхронный train overlap S2/Landsat: 2 308 строк; S2/MODIS: 657.
Точное равенство target и первого доступного S2 → Landsat → MODIS подтверждено
на всех 48 161 видимых target. Gap runs: 2 827 длины 1, 136 длины 2,
3 длины 3 и 1 длины 4. В старом R&D prompt буквы CV-B/C перепутаны относительно
ролевого ML задания; consumer использует смысловые имена и mapping из C-03.

Доступный Python: 3.11.0; NumPy 2.4.6, pandas 2.2.3, torch 2.5.1+cu121,
pytest 9.0.2. CUDA runtime в torch есть, доступного GPU нет. Полный GPU запуск
предусмотрен на Kaggle по указанию пользователя. Общий pyproject/lock ещё не создан
Backend; текущие проверки используют существующий `D:/ml_env` без его изменения.
Временный PDF renderer установлен в локальное `.dl-venv`, не в core dependencies.

## Что реализовано

| Задача | Артефакты | Статус и ограничения |
|---|---|---|
| DL-001/002, C-06 0.2 | `dl/data.py` (`WindowDatasetAdapter`, `SeasonalPrior`), `tests/dl/test_data.py`, `tests/dl/test_c03_bridge.py` | REVIEW: адаптер обучается и предсказывает на реальном `split_fold` во всех 16 фолдах |
| C-03 consumer 0.2 | `dl/c03_bridge.py`, `configs/dl/c03_consumer.md`, `artifacts/dl/c03_derived/` | Producer-код ML исполняется read-only; censoring проверяется round-trip; `review_status=derived_from_producer_artifacts` |
| DL-003 | `models/tcn.py`, `training.py`, `train.py`, `artifacts/dl/cv_tcn_v1/` | Полная CV 16 фолдов × 3 seed; epoch фиксирован заранее, не по outer OOF |
| Отчётность | `dl/report.py`, `reports/dl_experiments.csv` | Три строки на seed: TCN, только base, ML baseline; вес бленда — leave-one-fold-out |
| Интеграция C-02 | `dl/expert.py`, `tests/dl/test_expert.py` | `ResidualTCNExpert.predict(ReconstructionRequest)`; интервалы объявлены некалиброванными, source не классифицируется |
| DL-004 preparation | `models/pypots.py`, tests | Только public impute adapter и X/X_ori alignment; BRITS/SAITS не обучены, библиотека не установлена |
| DL-007/008/009/010, C-09 | `anomalies/{events,advanced,explain}.py` | REVIEW, чистый CPU API; Backend acknowledgement pending |
| Anomaly evidence | `reports/anomaly_cases/{synthetic_v1,real_2024}/` | 7 synthetic + 39 real polygon diagnostics; экспертный review pending |
| Kaggle | `configs/dl/kaggle/{package_dataset.py,run_tcn.ipynb,README.md}`, `artifacts/dl/kaggle_bundle.zip` | Автономный bundle 7.54 МБ; preflight на 16 фолдов проходит из распакованного архива; на GPU не запускался |

TCN: три bidirectional dilated Conv1d блока, missing/invalid masks, календарь,
delta since/until, crop token с unknown=0, нормированные base/prior и уровень
поддержки prior. `input_features` 61 (было 58). Начальное предсказание равно
residual base; поправка ограничена tanh, итоговый NDVI автоматически не clip.
Loss — только первоначально наблюдённый искусственно скрытый target в центре
окна; NaN labels индексируются до вычисления loss. Dropout, clipping, CPU
determinism, SHA256 и `weights_only=True` при загрузке.

### Три вещи, которые делают сравнение честным

1. **Censoring переносится, а не воспроизводится по памяти.** Ключи выводятся
   разностью сырого кадра и `split_fold(...).context_frame`, и сборка падает,
   если `apply_mask(train, censored)` не воспроизвёл producer context.
2. **Обучающая маска похожа на тестовую.** Шесть блоков внутри `fit_frame`,
   каждый маскируется отдельно; длины серий взяты из распределения реального
   test (2827 одиночных, 136 двойных, 3 тройных, 1 четверная).
3. **Epoch не выбирается по оцениваемым строкам.** `--epoch-policy fixed`,
   бюджет 16 закреплён по inner-кривой пилота: плато на 12–17 эпохе в matched,
   unseen и temporal.

CSDI/pretraining/HELIX не запускались: P1/P2 gates и bundle сравнения не готовы.
Тяжёлые framework dependencies без измеримой потребности не добавлялись.

## Почему план уточнён

1. Полная маска контекста до windowing закрывает утечку через соседние окна.
2. **Producer censoring переносится явными ключами.** Срез сырого кадра его
   терял, поэтому CV-C и CV-D увидели бы скрытые ML значения. Это был дефект
   DL runner, а не артефактов ML.
3. Epoch не выбирается ни по outer OOF, ни по DL-inner: бюджет фиксирован
   заранее. Пока ML не передал inner keys, `early_stop` — только проверка
   чувствительности, не основная конфигурация.
4. Обучение блоками, потому что маскировать все обучающие цели сразу означает
   учить модель на контексте заметно реже тестового.
5. Composite и row-weighted RMSE публикуются отдельно; incomplete CV не даёт ADOPT.
6. Отсутствие данных для сравнения — PENDING_EVALUATION, а не отрицательный опыт.
   Сейчас это ровно тот случай: у ML не опубликованы построчные OOF сильной модели.
7. Stock SAITS `fit` выполняет cell-wise MCAR внутри DatasetForSAITS; это отдельная
   training distribution. Блочный маскировщик C-03 даёт готовый bridge, но
   эксперимент пока не проведён и не заявляется.
8. Сезонный prior — статистика обучающего фолда, а не сервинга: в CV он берётся
   из fit-контекста, в `ResidualTCNExpert` — из явного reference-контекста.
9. Для anomaly критичность требует persistence/observed support, а confidence явно
   обозначает поддержку данными. При недостатке истории возвращается warning.

Проверенные первичные источники: [SAITS paper](https://arxiv.org/abs/2202.08516),
[BRITS paper](https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html),
[PyPOTS API](https://docs.pypots.com/en/latest/pypots.imputation.html),
[PyPOTS SAITS masking](https://github.com/WenjieDu/PyPOTS/blob/main/pypots/imputation/saits/data.py),
[PyPOTS license BSD-3-Clause](https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE).
Версия PyPOTS будет закреплена после проверки реальной установки и training bridge;
moving `main`/`latest` не используются как зафиксированная зависимость.

## Anomaly contract для Backend

```python
from veg_recovery.anomalies.advanced import AdvancedAnomalyDetector

detector = AdvancedAnomalyDetector().fit(reference_train_frame)
payload = detector.detect(one_polygon_harmonized_frame).to_dict()
```

Обязательные поля обоих frames: `anon_polygon_id,date,crop_type,ndvi_harmonized,
is_observed`. Один polygon на detect. `is_observed=False` означает реконструированную
точку только при конечном ndvi_harmonized. Optional: quality [0,1], selected_source
(`s2/landsat/modis`), calibration_supported, uncertainty_std либо lower/upper
(уровень интервала задаётся config, default 80%), source_switch_risk,
sensor_negative_count, precip_30d_ratio, temp_anomaly_c, ndwi_robust_z.

Weather reason codes используют **отклонение от нормы**, а не абсолютную
температуру или один сухой день. Эти входы должен передать вызывающий pipeline;
при их отсутствии причинные предположения не придумываются. Неоднозначная
спектральная формула NDWI не переопределяется детектором.

Reference: явные observed train data, quality filtering, LOYO, median/MAD floor,
квантили 10/25/50/75/90, pooling global/crop/polygon. Raw primary не используется
для anomaly и не изменяется гармонизатором. SensorHarmonizer обучает robust affine
S2 mapping на синхронном overlap; near-date и quantile mapping пока ablations.

Events: start/end, duration, min robust z, trapezoidal negative area (NDVI-days),
observed/reconstructed counts, severity, heuristic confidence, reason codes,
explanation_ru, schema/algorithm version. `score = negative_area * confidence`.
Нормальные надёжные точки разрывают event. Missing gaps объединяются только
до max_gap_days; разные годы не склеиваются. Нет сети/БД/torch в anomaly import.

Ограничения: thresholds не калиброваны по реальным размеченным событиям;
confidence не вероятность; p10-p90 reference не prediction interval; detector
ретроспективный. Широкий reconstruction interval ослабляет сигнал. Без допустимой
истории возвращается `INSUFFICIENT_REFERENCE_YEARS`, не заключение «поле нормально».
Change-point, phenology alignment и pooling tuning ещё не выполнены. Визуальный
R&D review 3 сильных/3 сомнительных реальных случаев выполнен агентом в
`reports/anomaly_cases/real_2024/review.md`; экспертное принятие ML/Backend pending.

## Дополнительный результат реального anomaly аудита

Reference/calibration обучены на 2010–2023; диагностический query — 2024, все
39 train polygons. 91 candidate event, включая 12 algorithmic critical.
Это не число подтверждённых повреждений. В графиках видны риски фенофазы,
севооборота/типа культуры и межсенсорного расхождения. На синхронных парах 2024
sensor alignment MAE относительно S2 снизился с 0.05338 до 0.02655 для Landsat
(319 пар) и с 0.09949 до 0.06634 для MODIS (139 пар). Это **не primary NDVI OOF**.

Версия anomaly 0.1.1 исправляет false source switches на календарных NaN и
ложные uncertainty warnings для естественных пропусков. Кэш LOYO summaries
сбрасывается при каждом fit. `analyze(frame)` возвращает scored points и events
за один проход. Реальный скоринг 39 polygons занял 36.81 s CPU; первый вызов
включает cache warmup, параллельно работали тесты, поэтому это не SLA benchmark.

## Команды

PowerShell из корня репозитория. Worktree ML — read-only источник C-01/C-03.

```powershell
git worktree add --detach tmp/dl-review-ml origin/ML
$env:PYTHONPATH='src;tmp/dl-review-ml/src'

# 1. Производный C-03 manifest: outer keys, censoring и baseline OOF от ML
python -m veg_recovery.dl.c03_bridge --ml-root tmp/dl-review-ml --output artifacts/dl/c03_derived

# 2. Проверка входов без torch и без обучения
python -m veg_recovery.dl.train --fold-manifest artifacts/dl/c03_derived/dl_c03.json --preflight-only

# 3. Полная CV (CPU; на Kaggle то же самое с --device cuda)
python -m veg_recovery.dl.train --fold-manifest artifacts/dl/c03_derived/dl_c03.json `
  --seeds 17 42 73 --epochs 16 --epoch-policy fixed --base-mode anchored `
  --window 61 --device cpu --cpu-threads 8 --output artifacts/dl/cv_tcn_v1

# 4. Таблицы отчёта
python -m veg_recovery.dl.report --run artifacts/dl/cv_tcn_v1 --experiments reports/dl_experiments.csv

# 5. Архив для Kaggle Dataset
python configs/dl/kaggle/package_dataset.py --ml-root tmp/dl-review-ml
```

Offline-проверки без кода ML:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests/dl tests/anomalies -p no:cacheprovider
python -m ruff check src tests --no-cache
python -m veg_recovery.dl.train --smoke --seeds 17 42 73 --epochs 8 --window 15 `
  --hidden-size 16 --layers 2 --output artifacts/dl/new_cpu_smoke
python -m veg_recovery.dl.anomaly_cases --output reports/anomaly_cases/synthetic_v1
python -m veg_recovery.dl.real_anomaly_cases --year 2024 --output reports/anomaly_cases/real_2024
```

Сборка входов детерминирована: gzip пишется с `mtime=0`, поэтому SHA256 в
manifest воспроизводятся. `artifacts/dl/.gitattributes` помечает `*.json`,
`*.csv`, `*.gz`, `*.zip`, `*.pt` как `-text`, чтобы Git не переписал EOL уже
захешированным файлам — тот же класс дефекта, что B-DL-003 у bundle ML.

Для повторного smoke используйте новый output path: существующий checkpoint
намеренно не перезаписывается. Runtime import требует numpy/pandas; TCN —
дополнительно torch; графики — matplotlib; тесты — pytest. Shared extra `dl`
и uv.lock принимает Backend по SH-002; чужие dependency files не менялись.

CPU smoke evidence: `artifacts/dl/cpu_smoke_v3/`, 3 seed, fixture-модель
3 817 параметров, 19 183 байта весов на seed, максимальная ошибка после
save/load ровно 0. `cpu_smoke_v2` сохранён как срез C-06 0.1 (58 признаков) и
загружается, но его нельзя питать текущим адаптером: тот отдаёт 61 признак.
Fixture RMSE — **не конкурсный результат и не adoption evidence**.

Для Kaggle: `configs/dl/kaggle/README.md`, пошагово от сборки архива до
скачивания результатов. `--preflight-only` проверяет fingerprint, ключи и
равенство метрик ML до импорта torch.

## Следующий наиболее ценный опыт

По убыванию ценности на текущую дату.

1. **Построчные OOF сильной модели от ML** (B-DL-004). Без них порог gate не
   вычисляется, и любой вывод об ADOPT опирался бы на число из чужого отчёта,
   а не на одинаковые строки. Достаточно двух колонок в существующем OOF-файле.
2. **Ансамбль с сильной моделью ML**, а не замена её. Индуктивные смещения
   разные: дерево берёт табличный контекст, TCN — форму окна и сезонный prior.
   Вес выбирается leave-one-fold-out, уже реализовано в `dl.report`.
3. **Калиброванная неопределённость.** Это единственная ветка gate, которую не
   закрывает точечный RMSE, и она нужна продукту: `lower/upper` сегодня честно
   помечены как некалиброванные. Квантильная голова на том же runner дешевле и
   воспроизводимее, чем CSDI; диффузию запускать только если квантили не хватит.
4. **Проверка чувствительности к политике epoch и к base.** `--epoch-policy
   early_stop` и `--base-mode linear` на тех же фолдах: если решение не меняется,
   возражение о выборе бюджета снимается количественно.
5. **SAITS/BRITS на matched-mask bridge.** Блочный маскировщик C-03 уже даёт
   нужное распределение пропусков, поэтому эксперимент стал дешевле. Но приоритет
   ниже пунктов 1–3: на 39 обучающих полигонах ожидание выигрыша низкое, и
   отрицательный результат тоже нужно уметь опубликовать честно.

Параллельно Backend может интегрировать C-09 draft и подтвердить JSON smoke,
а ML — рассмотреть D-DL-011. До review ни одна задача не помечается DONE и
процент checkpoint не увеличивается.
