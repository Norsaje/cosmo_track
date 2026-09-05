# NDVI model — пакет для Backend

Пакет содержит обученную модель `p0-catboost-gpu-v1`, точный runtime-код,
зафиксированные зависимости, полный тестовый fixture и известные корректные
результаты. Для инференса используется CPU; GPU нужен только для обучения.

## Содержимое

- `bundle/` — каталог C-04, который нужно смонтировать в Backend;
- `runtime/src/` — Python-пакет `veg_recovery` для инференса;
- `requirements-runtime.txt` — точные версии библиотек обучающего окружения;
- `examples/test_data.csv` — полный проверочный вход на 57 185 строк;
- `examples/expected_submission.csv` — 3 112 ожидаемых предсказаний;
- `examples/expected_diagnostics.csv` — диагностика этих предсказаний;
- `evidence/` — OOF, метрики, решения, importance и сведения о запуске;
- `handoff.json` — версия контракта, fingerprints и главные метрики;
- `MANIFEST.sha256` — SHA256 каждого файла внутри пакета;
- `verify_handoff.py` — проверка целостности, загрузки и полного инференса.

Исходный `train_dataset.csv` не нужен Backend и не включён: обученные priors уже
находятся в `bundle/feature_state.json`. В пакет входит полный test fixture, чтобы
потребитель мог независимо проверить интеграцию.

## Проверка после распаковки

Требуется Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.txt
PYTHONPATH=runtime/src python verify_handoff.py
PYTHONPATH=runtime/src python verify_handoff.py --full-inference
```

Первая команда проверяет все SHA256 и загрузку модели. Вторая дополнительно
пересчитывает 3 112 значений и сравнивает их с `expected_submission.csv` с
допуском `1e-10`.

## Подключение к Backend

Backend ожидает каталог с `manifest.json`. Передайте путь непосредственно на
`bundle/`:

```bash
MODEL_BUNDLE_HOST_PATH=/absolute/path/ndvi_backend_handoff_v1/bundle docker compose up --build
```

Если Backend использует каталог репозитория, содержимое `bundle/` можно поместить
в его локальный `artifacts/ml/final_bundle/`. Ветка `models` уже содержит
проверенную копию модельных файлов; дублировать их в рабочей ветке Backend не
требуется.

Вызов из Python:

```python
import pandas as pd
from veg_recovery.contracts import ReconstructionRequest, ReconstructionPayload
from veg_recovery.inference import load_reconstructor

frame = pd.DataFrame(...)  # суточный ряд одного или нескольких полигонов
mask = frame["primary_ndvi"].isna()
model = load_reconstructor("/path/to/bundle", trusted=True)
result = model.predict(ReconstructionRequest(frame, mask, "web"))
payload = ReconstructionPayload.from_result(result)
```

`trusted=True` допустим только после проверки внешнего SHA256 архива и
`MANIFEST.sha256`: `estimators.joblib` следует загружать как доверенный внутренний
артефакт.

Минимальные входные колонки: `anon_polygon_id`, `date`, `crop_type`,
`primary_ndvi`. Для качества предсказаний следует передавать доступные
`s2_*`, `landsat_*`, `modis_*`, `era5_temp_c`, `era5_precip_mm`. Даты должны
быть календарными днями без timezone; ключ `(anon_polygon_id, date)` уникален;
`gap_mask` обязан иметь тот же индекс, что и frame.

Выход содержит `primary_ndvi_pred`, `lower`, `upper`, `method`,
`primary_ndvi_reconstructed`, `ndvi_harmonized` и отдельную таблицу diagnostics.
Для Backend `interval_level=0.95`, но `interval_status` указывает, что интервалы
эмпирические и не сертифицированы как формально калиброванные.

Главное ограничение модели — перенос на будущий сезон: в temporal CV ансамбль
хуже простого baseline. Backend должен сохранять diagnostics и показывать
`quality_flags`, особенно для новых полигонов и слабого контекста.
