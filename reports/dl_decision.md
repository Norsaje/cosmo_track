# DL R&D — 2026-09-05

Статус: **PENDING_EVALUATION** для DL-импутации; **REVIEW** для C-06/C-09 draft.
Production candidate не экспортирован. Решения ADOPT/ENSEMBLE_ONLY/REJECT пока
невозможны: ещё нет DL OOF по согласованной training policy и доступного final ML
ensemble OOF для сравнения. Числа ~0.06 из задания не считаются воспроизведённой
baseline. ML submission не зависит от DL.

**Обновление после нового fetch:** ML@39a8f73 уже публикует C-01/MaskSpec/folds/baseline
OOF, Backend@350faec — SH-002. Они ещё не интегрированы в main. Выполнен реальный
consumer review всех 16 folds/13 317 OOF keys и равенства baseline metrics.
См. [integration review](dl_integration_review.md): остаются DL inner/train mask policy,
адаптация training runner к точному контексту C/D и доступ к final ML ensemble OOF.
Найден воспроизводимый EOL/SHA256 дефект загрузки ML bundle на Windows.

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
| DL-001/002, C-06 | `src/veg_recovery/dl/data.py`, `tests/dl/test_data.py` | REVIEW, CSV fallback; нужен alignment с реальным C-01/C-03 |
| DL-003 | `models/tcn.py`, `training.py`, `train.py`, `predict.py`, `artifacts.py` | CPU fixture проверяется; реальные метрики ждут C-03 |
| DL-004 preparation | `models/pypots.py`, tests | Только public impute adapter и X/X_ori alignment; сами BRITS/SAITS ещё не обучены и библиотека не установлена |
| OOF harness | `evaluation.py`, `ml_handoff.py`, `configs/dl/c03_consumer.md` | Проверены реальные ML folds, keys/labels и baseline metric equality; training policy/consumer ещё требуют ML review |
| DL-007/008/009/010, C-09 | `anomalies/events.py`, `advanced.py`, `explain.py` | REVIEW, чистый CPU API; Backend acknowledgement pending |
| Anomaly evidence | `reports/anomaly_cases/synthetic_v1/`, `real_2024/` | 7 synthetic cases + 39 real polygon diagnostics; 3 сильных/3 сомнительных PNG просмотрены, экспертный review pending |
| Kaggle | `configs/dl/kaggle/run_tcn.ipynb` | Подготовлен, на Kaggle не запускался; требует C-03 и проверку версии runtime |

TCN: три bidirectional dilated Conv1d блока, missing/invalid masks, календарь,
delta since/until, crop token с unknown=0. Начальное предсказание равно linear
base; поправка ограничена tanh, итоговый NDVI автоматически не clip. Loss —
только первоначально наблюдённый искусственно скрытый target в центре каждого
окна. NaN labels индексируются до вычисления loss. Dropout, clipping, CPU
determinism, inner early stopping, SHA256 и `weights_only=True` при загрузке.

CSDI/pretraining/HELIX не запускались: P1/P2 gates и необходимый baseline handoff
не готовы. Не добавлены тяжёлые framework dependencies без измеримой потребности.

## Почему план уточнён

1. Полная маска контекста до windowing закрывает утечку через соседние окна.
2. Inner early stopping исключает выбор epoch по оценочным outer OOF labels.
3. Composite и overall публикуются отдельно; incomplete CV не даёт ADOPT.
4. Отсутствие данных для сравнения — PENDING_EVALUATION, а не отрицательный опыт.
5. Stock SAITS `fit` выполняет cell-wise MCAR внутри DatasetForSAITS; это
   отдельная training distribution. Нужен C-03 training bridge прежде чем
   называть опыт воспроизведением на matched-mask протоколе.
6. Для anomaly критичность требует persistence/observed support, а confidence
   явно обозначает поддержку данными. При недостатке истории возвращается warning.

Проверенные первичные источники: [SAITS paper](https://arxiv.org/abs/2202.08516),
[BRITS paper](https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html),
[PyPOTS API](https://docs.pypots.com/en/latest/pypots.imputation.html),
[PyPOTS SAITS masking](https://github.com/WenjieDu/PyPOTS/blob/main/pypots/imputation/saits/data.py),
[PyPOTS license BSD-3-Clause](https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE).
Версия PyPOTS будет закреплена после проверки реальной установки и training bridge;
moving `main`/`latest` не используются как зафиксированная экспериментальная зависимость.

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

Финальная локальная проверка: **60 tests passed in 25.02s**, Ruff passed.
Дополнительно отдельно выполнен ML consumer audit всех 16 реальных folds;
это не добавлено к числу unit tests и не является DL обучением.

PowerShell из корня репозитория:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests/dl tests/anomalies -p no:cacheprovider
python -m ruff check src/veg_recovery/dl src/veg_recovery/anomalies tests/dl tests/anomalies --no-cache
python -m veg_recovery.dl.train --smoke --seeds 17 42 73 --epochs 3 --window 15 --hidden-size 16 --layers 2 --output artifacts/dl/new_cpu_smoke
python -m veg_recovery.dl.anomaly_cases --output reports/anomaly_cases/synthetic_v1
python -m veg_recovery.dl.real_anomaly_cases --year 2024 --output reports/anomaly_cases/real_2024
```

Для повторного smoke используйте новый output path: существующий checkpoint
намеренно не перезаписывается. Standalone runtime import требует numpy/pandas;
TCN — дополнительно torch; графики — matplotlib; tests — pytest. Shared extra `dl`
и uv.lock должен принять Backend по SH-002; чужие dependency files не менялись.

Для Kaggle: инструкция и notebook в `configs/dl/kaggle/`.
`--preflight-only` с C-03 проверяет fingerprint/keys/metrics до импорта torch.

Опубликованный CPU smoke evidence: `artifacts/dl/cpu_smoke_v2/`, source commit
`a4f2563` plus per-source SHA256. 3 seed, 3 epochs, fixture model 3 769 parameters,
weights 18 991 bytes per seed, save/load max abs error 0. Fixture inner RMSE
0.002503/0.002598/0.001633 — **не конкурсный результат и не DL adoption evidence**.
`reports/dl_experiments.csv` пока содержит только schema: реальные CV experiments
ещё не выполнены; synthetic smoke хранится отдельно во избежание смешения метрик.

## Следующий наиболее ценный опыт

ML подтверждает уже опубликованные C-01/apply_mask/outer folds, предоставляет
inner/train-target policy и отсутствующие final ensemble OOF с исходными метриками.
В DL runner переносим проверенное producer context censoring C/D. Затем одна
TCN-конфигурация на 3 seed CPU или Kaggle;
Huber/MSE, window 61/91/121 — только после equality gate на тех же folds.
Параллельно Backend может интегрировать C-09 draft и подтвердить JSON smoke.
До review ни одна задача не помечается DONE и процент CP не увеличивается.
