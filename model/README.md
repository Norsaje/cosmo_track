# NDVI: рабочее ML/DL-решение для обновлённого test

Здесь находятся код, обученный резервный ансамбль, проверенный `submission.csv`, Kaggle notebook и описание экспериментов. **Новый test: 49 190 строк, 20 новых полигонов, 2 323 контрольных пропуска, 2010–2024.** Старый test не используется.

Начните с `RESULTS.md`: там фактически измеренное качество, выбранный запуск и ограничения. Полное объяснение признаков, обучения, архитектуры и исследования внешних моделей находится в `RESEARCH.md`.

## Самое быстрое действие

Готовый к проверке организатором файл: `runs/local/submission.csv`. Он содержит только `anon_polygon_id,date,primary_ndvi_pred`, ровно один ответ на каждый из 2 323 контрольных ключей. Правильные скрытые ответы нам неизвестны; официальный балл ещё не измерен.

## Запуск на Kaggle

1. Создайте private Kaggle Dataset из `ndvi_best_solution.zip` и добавьте его к notebook.
2. Импортируйте `notebooks/NDVI_Kaggle.ipynb`. Выберите доступный GPU T4/P100, включите Internet.
3. В notebook задайте фактически оставшийся бюджет и выполните Run All. По умолчанию установлен предел 300 минут; установка пакетов идёт до начала этого таймера.
4. Скачайте `/kaggle/working/submission.csv` и `ndvi_kaggle_results.zip`. Сохраните их до завершения сессии.

Notebook сначала проверяет и обучает деревья, затем проверяет увеличенный TabICLv2. Если DL не улучшает development OOF, его вес равен нулю. Успевшие завершиться результаты сохраняются. При ошибке GPU-эксперимента доступен `submission_cpu_reference.csv`.

Kaggle исполняет полный цикл самостоятельно; запуск в вашем аккаунте из этой сессии не выполнялся. Инструкция по включению GPU/Internet: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

## Локальное воспроизведение

Проверенный CPU-стек зафиксирован в `requirements.txt`; версия Python локального запуска — 3.12.13. Kaggle использует совместимые ограничения `requirements-kaggle.txt` и сохраняет установленный CUDA PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m ndvi.inference --run runs/local --output submission_reproduced.csv
```

Команда обучения с исходной конфигурацией полного сравнения:

```bash
python -m ndvi.pipeline --out runs/reproduce --stage all \
  --device CPU --rounds 1 --iterations 1400 --threads 4 --budget-minutes 90
```

Усиленный вариант с тремя кругами масок:

```bash
python -m ndvi.pipeline --out runs/augmented_reproduce --stage all \
  --device CPU --rounds 3 --iterations 1800 --threads 4 \
  --models lgb,lgb_nodonor,source_experts --budget-minutes 120
```

Для проверки TabICL нужен PyTorch, соответствующий вашей платформе, затем `requirements-foundation.txt`. После tree CV:

```bash
python -m pip install -r requirements-foundation.txt
python -m ndvi.pipeline --out runs/reproduce --stage foundation \
  --device CPU --rounds 1 --iterations 1400 --threads 2 \
  --tabicl-context 4000 --tabicl-estimators 2 --budget-minutes 30
python -m ndvi.pipeline --out runs/reproduce --stage final \
  --device CPU --rounds 1 --iterations 1400 --threads 4
python -m ndvi.diagnostics --run runs/reproduce
```

Параметры, входные хеши и версия признаков должны совпадать при продолжении run. Для новых параметров нужен новый `--out`. Чтобы проверить действительно новую версию организаторских данных, сознательно используйте `--allow-new-data` и новый run; предварительно пересмотрите маски и схему.

## Структура

| Путь | Содержание |
|---|---|
| `data/train.csv`, `data/test_features.csv` | Единственные используемые исходные файлы |
| `ndvi/data.py` | Схема, хеши, правила скрытия и подготовка ключей |
| `ndvi/features.py` | 324 признака из доступного контекста |
| `ndvi/models.py` | LightGBM, CatBoost и смесь спутниковых экспертов |
| `ndvi/foundation.py` | TabICLv2; отбор признаков и контекста внутри fold |
| `ndvi/pipeline.py` | Групповая CV, выбор весов, refit, проверки submission |
| `ndvi/inference.py` | Пакетное применение обученной модели |
| `ndvi/diagnostics.py` | Ошибки по срезам, интервалы, кандидаты аномальных периодов |
| `tests/test_integrity.py` | Проверки утечек и формата результата |
| `kaggle_run.py` | Общий бюджет, логи, контрольные сохранения |
| `runs/local/` | Рекомендованный обученный CPU-кандидат и доказательства проверки |
| `reports/` | Аудит данных и дополнительные материалы исследования |

`runs/local` в доставленном архиве — выбранный CPU-кандидат; исходная конфигурация указана в его `artifacts/run_config.json` и `RESULTS.md`. Пересоздаваемые feature caches и неиспользуемые CV-веса нейросети в архив не включены. Это не мешает inference и воспроизведению обучения.

## Научные и инженерные ограничения

- Метки новых тестовых пропусков недоступны. Локальная CV не гарантирует такой же официальный RMSE.
- Валидируем новые полигоны внутри предоставленного набора; независимые регионы не известны.
- Снимки USGS не добавлялись: нужны геометрии AOI и сведения о расчёте исходных индексов.
- Аномалии — проверяемые кандидаты, причины без разметки не установлены.
- Внешняя модель TabICL описана в `RESEARCH.md`, версия checkpoint и хеш фиксируются при запуске. TabPFN-3, Prithvi, TimesFM-3 и Chronos-2 исследованы, но не заявляются использованными моделями.
- Сервер, UI и геосервис в этот пакет не входят.

Не меняйте CSV редактором, который заменяет окончания строк, если нужна точная проверка SHA256. ZIP сохраняет байты. Для Git задан `.gitattributes`.
