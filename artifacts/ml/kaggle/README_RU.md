# Kaggle: воспроизводимое обучение NDVI

Локально выполнены только аудит, baselines, статистики и проверка полученных
артефактов. Фиксированный CatBoost GPU, source classifier и ML-ансамбль успешно
обучены на Kaggle 2026-09-05. Результат интегрирован в
`artifacts/ml/trained_gpu_v1`. Tuning ещё не запускался.

1. Создайте приватный Kaggle Dataset из `kaggle_ndvi_gpu_v1.zip`.
   В portable-архив намеренно не включены baseline OOF `*.csv.gz`: Kaggle может
   изменить вложенные gzip-файлы при создании Dataset, а для обучения они не нужны.
2. Импортируйте `train_ndvi.ipynb` как Notebook, подключите Dataset.
3. Выберите GPU accelerator. Internet нужен для установки CatBoost и остальных
   зависимостей; исходные данные и сохранённые folds уже находятся в архиве.
   Основной CatBoost обучается на CUDA GPU. Подготовка признаков и небольшой
   source classifier выполняются на CPU.
4. Выполните setup и тесты. Следующая ячейка явно запускает обучение с
   `--allow-training --device gpu --models catboost --trials 0`, чтобы
   воспроизвести фиксированную P0-модель.
5. Для второго запуска импортируйте отдельный `tune_ndvi.ipynb`. Он не повторяет
   фиксированное обучение и сразу выполняет 30 Optuna trials CatBoost GPU с
   pruning, теми же ключами фолдов и seeds. После tuning выполняются полный OOF,
   permutation importance, final fit и строгая проверка submission.
6. Скачайте `ndvi_tuned_artifacts.zip` и передайте его для сравнения с первым запуском.
   Сохраняются bundle, OOF, subgroup metrics, source reliability, coverage,
   trials, параметры, feature importance, решения и submission.

Эквивалентная команда внутри распакованного проекта:

```bash
python -m pip install -r configs/ml/requirements-training.txt
PYTHONPATH=src python -m unittest discover -s tests/ml -v
PYTHONPATH=src python -m veg_recovery.models.training --allow-training --device gpu --models catboost --trials 0 --jobs 2 --output artifacts/ml/trained_gpu_v1
# Второй запуск, использующий тот же протокол (tune_ndvi.ipynb):
PYTHONPATH=src python -m veg_recovery.models.training --allow-training --device gpu --models catboost --trials 30 --jobs 2 --output artifacts/ml/tuned_gpu_v1
```

`--resume-cache` возобновляет подготовленный полный cache после проверки SHA256
и fingerprints. Частично подготовленный cache повторно строится; готовый
bundle никогда не перезаписывается. Отрицательные Optuna trials сохраняются в
SQLite/CSV. При запуске без Kaggle GPU скрипт завершается до подготовки фолдов
с понятной ошибкой. Для CatBoost polygon_id остаётся нативной категорией;
неизвестные категории получают явный маркер. GPU-операции CatBoost могут иметь
небольшую недетерминированность при одинаковом seed.

В train features собственная строка и её dynamic/derived поля скрыты. Priors
строятся без pseudo-target ключей; в grouped outer CV — без всего held-out
полигона. Source probabilities для fit регрессоров получаются через GroupKFold.
CV-C использует только прошлые fit/context данные, CV-D — один далёкий сосед.
Test predictions не участвуют в выборе параметров, весов или clip.

OOF используется для выбора модели/ансамбля, поэтому итоговая OOF-оценка имеет
selection bias. Интервалы называются эмпирическими; независимая гарантия coverage
не заявляется. Baseline CV-D содержит 117 точек и чувствителен к выбросам.
Регионы/геометрия отсутствуют в CSV, поэтому географические календари не строятся.

API после получения доверенного собственного обученного bundle:

```python
from veg_recovery.inference import load_reconstructor
from veg_recovery.contracts import ReconstructionRequest, ReconstructionPayload
model = load_reconstructor("artifacts/ml/trained_gpu_v1/bundle", trusted=True)  # один раз
result = model.predict(ReconstructionRequest(frame, gap_mask, "web"))
payload = ReconstructionPayload.from_result(result).model_dump(mode="json")
```

SHA256 проверяет целостность; для joblib дополнительно требуется происхождение
из собственного доверенного задания. Baseline bundle состоит только из JSON и
загружается без `trusted=True`. Сам DataFrame может содержать видимую историю
новых полигонов. Запрос скрыть ключ, уже попавший в сохранённые target priors,
отклоняется, чтобы исключить утечку; нужен bundle с исключением этого ключа.
