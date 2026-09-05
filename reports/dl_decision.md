# DL R&D — 2026-09-05

Статус: **PENDING_EVALUATION** для DL-импутации; **REVIEW** для C-06/C-09 draft.
Production candidate не экспортирован. Решения ADOPT/ENSEMBLE_ONLY/REJECT пока
невозможны: в исходном main нет C-01, C-03 и ML-008 OOF. Числа ~0.06 из задания
не считаются воспроизведённой baseline. ML submission не зависит от DL.

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
| OOF harness | `evaluation.py`, `configs/dl/c03_consumer.md` | Strict keys/labels, equality, subgroup metrics, polygon bootstrap, fail-closed gate; consumer draft требует ML review |
| DL-007/008/009/010, C-09 | `anomalies/events.py`, `advanced.py`, `explain.py` | REVIEW, чистый CPU API; Backend acknowledgement pending |
| Anomaly evidence | `reports/anomaly_cases/synthetic_v1/` | 3 события и 4 negative controls, только synthetic proxies |
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
Change-point, phenology alignment, pooling tuning и ручной review реальных
3 сильных/3 сомнительных случаев ещё не выполнены.

## Команды

PowerShell из корня репозитория:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests/dl tests/anomalies -p no:cacheprovider
python -m ruff check src/veg_recovery/dl src/veg_recovery/anomalies tests/dl tests/anomalies --no-cache
python -m veg_recovery.dl.train --smoke --seeds 17 42 73 --epochs 3 --window 15 --hidden-size 16 --layers 2 --output artifacts/dl/cpu_smoke_v1
python -m veg_recovery.dl.anomaly_cases --output reports/anomaly_cases/synthetic_v1
```

Для повторного smoke используйте новый output path: существующий checkpoint
намеренно не перезаписывается. Standalone runtime import требует numpy/pandas;
TCN — дополнительно torch; графики — matplotlib; tests — pytest. Shared extra `dl`
и uv.lock должен принять Backend по SH-002; чужие dependency files не менялись.

Для Kaggle: инструкция и notebook в `configs/dl/kaggle/`.
`--preflight-only` с C-03 проверяет fingerprint/keys/metrics до импорта torch.

## Следующий наиболее ценный опыт

ML передаёт C-01, apply_mask/MaskSpec, frozen fit/inner/outer/context keys,
linear/HGB/final ensemble OOF с исходными метриками. Принимаем реальный формат
на границе DL consumer. Затем одна TCN-конфигурация на 3 seed CPU или Kaggle;
Huber/MSE, window 61/91/121 — только после equality gate на тех же folds.
Параллельно Backend может интегрировать C-09 draft и подтвердить JSON smoke.
До review ни одна задача не помечается DONE и процент CP не увеличивается.
