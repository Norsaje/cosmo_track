# Demo Checklist — cosmo_track

> **Статус:** стартовый шаблон от Developer 4. Финальный чек-лист заполняется Backend-ревьюером.
> Все `[TODO]`-пункты и issue-notes требуют подтверждения owner.

## За день до demo

- [ ] clean build (`uv sync` чисто завершается, без warnings о пропавших колёсах)
- [ ] tests — все зелёные: `uv run pytest -q` (131 passed на момент Sprint 3)
- [ ] migrations — TODO: уточнить у backend owner
- [ ] model artifact/hash — TODO: уточнить у ML owner
- [ ] provider credentials check — TODO: уточнить у backend owner
- [ ] demo cache — TODO: какие endpoints/snapshots кэшируются?
- [ ] three polygons — TODO: какие именно? (предположительно AOI-0032, AOI-0013, AOI-0067 — top-3 по gap count; уточнить у ML)
- [ ] browser/device check
- [ ] backup video/screenshots
- [ ] submission validator — passes на latest submission.csv (`uv run python scripts/validate_submission.py ...`)

## За 15 минут

- [ ] health/live/ready endpoints — TODO: URL и ожидаемые коды
- [ ] worker/Redis/PostGIS — TODO: health-check команды
- [ ] map tiles — TODO: проверить, что тайлы грузятся в offline-кэше
- [ ] model loaded — TODO: какой эндпоинт проверяет?
- [ ] one prepared analysis — TODO: какой именно кейс показывать?
- [ ] network fallback — план на случай отключения сети

## Основной сценарий

> TODO: нумерованные действия пользователя и ожидаемый результат на каждом шаге.
> Заполняется backend после согласования с ML (какие сценарии покрываются).

## Fallback

> TODO: что делать при failure:
> - GEE (Google Earth Engine) → ?
> - CDSE (Copernicus Data Space Ecosystem) → ?
> - ERA5-Land → ?
> - Map tiles → ?
> - Model provider → ?
>
> Не скрывать, что результат cached.

## После demo

- [ ] logs сохранены (`*.log` → `artifacts/demo_<date>/logs/`)
- [ ] metrics сохранены (RMSE, gap_score по полигон-кейсам)
- [ ] screenshots сохранены (см. `docs/screenshots/README.md`)
- [ ] версии зафиксированы: model, schema_version, script_version, manifest

---

## Issue-notes от Developer 4

### Требует подтверждения (низкий риск, фиксируется для прозрачности)

- **Пути в источнике истины.** В `docs/ed_part.md` пути указаны как `data/raw/...`,
  но реально данные лежат в `data/...` (нет подпапки `raw/`). Скрипты Developer 4
  используют реальный путь `data/`. Не блокер, но требует решения:
  либо переименовать `data/` → `data/raw/`, либо обновить ed_part.md.
- **ML design assumptions.** `tests/fixtures/manifest.json` содержит
  `design_assumptions_requires_ml_confirmation: true`. Все решения по выбору
  полигонов, fallback-стратегиям, synthetic TOY-* — задокументированы, но
  ожидают ревью от ML-владельца. Owner: ML.
- **Hierarchy check 100% match.** Data quality report показывает, что
  `primary_ndvi` в данных полностью воспроизводится из S2 → Landsat → MODIS
  иерархии (`match_rate=1.0`, `max_abs_mismatch=0.0`). Это означает, что в
  ground truth нет других источников — стоит подтвердить, что это ожидаемо
  и не утечка. Owner: ML.
- **85% gaps на unseen полигонах.** Из 3112 gap-дат 464 на known (39 train
  polygons в test) и 2648 на unseen (39 новых). Если модель обучена только
  на known polygons, метрика на test может быть смещена. Owner: ML.

### Требует подтверждения (средний риск)

- **Все `submission_for_test_tiny.csv` / `train_tiny.csv` / `test_tiny.csv`** —
  сгенерированы скриптом `make_test_fixtures.py` с seed=42 и минимальными
  предположениями. Если ML нужен другой seed или больше полигонов — скажите,
  поменяю defaults. Owner: ML.
- **`status` колонка в train (object dtype)** — содержит строки типа
  "active"/"inactive". Нужно проверить, что семантика ясна для всех членов
  команды. Owner: ML.

### Известные ограничения (не требуют решения, документируем)

- В реальных данных есть пропуски в `year`, `doy`, `ndvi_climatology_*` —
  это особенность данных, не bug. См. `reports/data_quality.json`,
  раздел "missingness".
- `crop_type` содержит русскоязычные значения (зерновые, озимая пшеница,
  пастбища/зерновые, подсолнечник). Для англоязычных демо может потребоваться
  маппинг.

---

## Где искать артефакты

| Что | Путь |
|---|---|
| Data quality report | `reports/data_quality.{md,json}` |
| Submission validator | `scripts/validate_submission.py` |
| Test fixtures manifest | `tests/fixtures/manifest.json` |
| Submission schema | `docs/ed_part.md §"Часть 1, раздел 4"` |
| Screenshots index | `docs/screenshots/README.md` (TODO) |
| Experiments table | `reports/experiments.csv` (TODO: P1, `check_experiment_table.py`) |

## Где узнать версии

| Компонент | Версия |
|---|---|
| report_schema_version | `1.0.0` (см. `reports/data_quality.json`) |
| script_version (data_quality) | `1.0.0` |
| script_version (validate_submission) | `1.0.0` |
| script_version (make_test_fixtures) | `1.0.0` |
| manifest_schema_version | `1.0.0` (см. `tests/fixtures/manifest.json`) |