# Разработчик 3 — Backend, geospatial и интеграция web-сервиса

Версия плана: 2026-09-04  
Роль: владелец end-to-end продукта, автоматического сбора данных и demo reliability  
Главный результат: пользователь выбирает/рисует поле и получает ряд, реконструкцию, аномалии и объяснение без ручной загрузки CSV

---

# Часть 1. Что нужно прочитать человеку

## 1. Миссия

Вы строите один надёжный пользовательский путь:

1. открыть карту;
2. найти готовые поля в регионе или нарисовать полигон;
3. запустить анализ;
4. автоматически собрать спутниковые и погодные данные;
5. показать наблюдаемый и восстановленный NDVI;
6. подсветить негативные события и объяснение.

Не переписывайте модель. Batch и web вызывают общий пакет src/veg_recovery и один model bundle ML-разработчика.

## 2. Рекомендуемая архитектура

~~~mermaid
flowchart TD
    UI["React + MapLibre"] --> API["FastAPI API"]
    API --> DB["PostgreSQL + PostGIS"]
    API --> Q["Redis + worker"]
    Q --> P["Data-provider adapters"]
    P --> CORE["Shared NDVI core"]
    CORE --> DB
    API --> UI
~~~

FastAPI BackgroundTasks не используйте для долгих задач сбора/растра: нужен durable worker с повторными попытками. Для хакатона достаточно одного worker в Docker Compose.

## 3. Источники и fallback

Приоритетный продуктовый путь:

- Sentinel-2/Landsat: NASA HLS v2 как гармонизированный 30-метровый источник.
- Прямой Sentinel-2 fallback: Copernicus Data Space STAC.
- MODIS: MOD13A1/MOD13Q1 с quality flags.
- Погода: ERA5-Land.
- Контуры: Fields of the World → OSM/Overpass → WorldCereal → ручной полигон.

Самый быстрый P0 допустимо сделать через Google Earth Engine, если сервисный аккаунт уже работает. Но demo не должен зависеть от единственной внешней авторизации: нужен кэшированный сценарий и как минимум один открытый adapter path.

## 4. Приоритеты

P0:

1. Карта, draw/select/delete polygon.
2. Async analysis job и progress.
3. Один реально работающий provider path.
4. Shared model inference.
5. График observed/reconstructed, anomaly bands и explanation.
6. PostGIS, кэш, Docker Compose, health checks.
7. Предзагруженный demo polygon как честный fallback при сбое API.

P1:

1. HLS/CDSE/ERA5 отдельными заменяемыми adapters.
2. Автоматический поиск контуров Fields of the World/OSM/WorldCereal.
3. Coverage/QA/provenance в UI.
4. Сравнение сезонов и экспорт JSON/CSV.

P2:

1. SAMGeo как необязательный GPU fallback для выделения поля.
2. Сравнение полигонов, отчёт, расширенное объяснение.
3. Масштабирование workers; не нужно до рабочего MVP.

## 5. Связь с баллами и сроками

| Вклад роли | Максимум в критериях | Доказательство |
|---|---:|---|
| Полигоны + сбор + регионы | 15 | live select/draw, provider jobs, второй регион |
| Детекция/интерпретация | 7 | интегрированные events и explanation |
| Дополнительные идеи | 5 | field discovery, provenance, uncertainty |
| UX/ценность | 10 | короткий judge flow и ясные графики |
| Demo MVP | 5 | стабильный основной и fallback сценарий |
| Код/документация | 8 | Compose, tests, README/runbook |

| Период | Результат |
|---|---|
| Первые 4 часа | skeleton API/UI/Compose, DB schema, model mock contract |
| 4–12 часов | polygon CRUD/draw, async job, fixture end-to-end |
| 12–24 часа | один live provider, shared model, chart/anomaly panel, cache |
| 24–48 часов | field search, second adapter/region, provenance, error states |
| 48–72+ часов | product polish и P2 без риска для demo |

Ближайший milestone: offline fixture проходит весь путь draw → job → graph. Blocker модели обходится mock-реализацией того же contract; после получения bundle mock удаляется из production config. Blocker внешнего API обходится явно помеченным cache, не ручным CSV-path в UI.

## 6. Стек

Backend: Python 3.11, uv, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL/PostGIS, Redis, Celery или RQ, httpx, tenacity, structlog.

Geospatial: PySTAC Client, stackstac, xarray, Dask, rioxarray, Rasterio, GeoPandas, Shapely 2, PyProj, exactextract.

Frontend: React, TypeScript, Vite, MapLibre GL JS, Terra Draw либо Mapbox GL Draw-compatible plugin, Plotly/ECharts, TanStack Query. Package manager — pnpm.

Запуск:

~~~bash
docker compose up --build
~~~

## 7. Ваши выходы

- apps/api/ — REST API.
- apps/worker/ — durable tasks.
- apps/web/ — карта и графики.
- src/veg_recovery/providers/ — adapters.
- src/veg_recovery/geospatial/ — QA, indices, zonal aggregation.
- migrations/ — PostGIS schema.
- docker-compose.yml, Dockerfiles, .env.example.
- docs/api.md, docs/data_provenance.md, docs/demo_runbook.md.
- tests/backend/, tests/providers/, tests/e2e/.

## 8. Definition of Done

- Чистый запуск по одной инструкции.
- Новый GeoJSON валидируется, сохраняется и анализируется.
- Готовые контуры находятся хотя бы одним автоматическим источником.
- Сбой/лимит внешнего API не ломает UI: есть retry, понятная ошибка и cached demo.
- API показывает job progress.
- На графике визуально различимы observed, reconstructed, uncertainty и anomaly.
- Batch/API дают одинаковый prediction на одном fixture.
- Секретов нет в git; источники, коллекции и preprocessing зафиксированы.

## 9. Что прочитать

- NASA HLS: https://www.earthdata.nasa.gov/data/projects/hls
- HLS algorithms: https://hls.gsfc.nasa.gov/algorithms/
- HLS v2 overview: https://www.usgs.gov/publications/harmonized-landsat-and-sentinel-2-version-20-surface-reflectance-dataset
- Copernicus Data Space STAC: https://documentation.dataspace.copernicus.eu/APIs/STAC.html
- ERA5-Land: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- MOD13A1 v061: https://lpdaac.usgs.gov/products/mod13a1v061/
- Fields of the World: https://fieldsofthe.world/
- Fields of the World data: https://source.coop/ftw/global-data
- WorldCereal: https://esa-worldcereal.org/en
- OSM farmland: https://wiki.openstreetmap.org/wiki/Tag%3Alanduse%3Dfarmland
- Overpass API: https://wiki.openstreetmap.org/wiki/Overpass_API
- Cloud Score+: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED
- PySTAC Client: https://pystac-client.readthedocs.io/
- stackstac: https://stackstac.readthedocs.io/
- exactextract: https://isciences.github.io/exactextract/
- PostGIS indexes: https://postgis.net/workshops/postgis-intro/indexing.html
- FastAPI BackgroundTasks caveat: https://fastapi.tiangolo.com/tutorial/background-tasks/
- MapLibre drawing example: https://www.maplibre.org/maplibre-gl-js/docs/examples/draw-polygon-with-mapbox-gl-draw/

---

# Часть 2. Техническое задание для кодингового агента

## ROLE

Ты senior backend/full-stack geospatial engineer. Создай воспроизводимый web-сервис мониторинга растительности. Оптимизируй надёжность демонстрации и баллы критериев: автоматический сбор, полигоны, разные регионы, anomaly UX, документация. ML-алгоритм не переписывай.

## SOURCE OF TRUTH

1. case_doc.pdf и criteria.pdf.
2. shared contracts и model bundle ML-разработчика.
3. AnomalyEvent contract DL-разработчика.
4. этот документ.

Если внешний источник недоступен, сохраняй interface и используй документированный fallback. Не имитируй автоматический сбор заранее подготовленным CSV без явной маркировки demo cache.

## ВЛАДЕНИЕ ПУТЯМИ

Разрешено:

- apps/api/
- apps/worker/
- apps/web/
- src/veg_recovery/providers/
- src/veg_recovery/geospatial/
- src/veg_recovery/service/
- migrations/
- infra/
- tests/backend/
- tests/providers/
- tests/e2e/
- Dockerfile*
- docker-compose.yml
- .env.example
- docs/api.md
- docs/data_provenance.md
- docs/demo_runbook.md

Не меняй implementation модели/фолдов/features. Используй src/veg_recovery/inference.py.

## NON-FUNCTIONAL REQUIREMENTS

- Python 3.11 и pinned uv.lock.
- API stateless; состояние в Postgres/Redis/object cache.
- Все даты UTC/ISO; геометрия на границе API — GeoJSON EPSG:4326.
- Площадь/буферы считать в подходящей projected CRS, не в градусах.
- Таймауты, retries с exponential backoff и rate limits для каждого provider.
- Structured logs с request_id/job_id/provider, без secrets.
- Идемпотентность анализа по geometry_hash + date_range + pipeline_version.
- Не держать HTTP request открытым во время спутникового анализа.
- Backend и worker используют один контейнерный image/version.

## РЕПОЗИТОРИЙ

Целевая структура:

~~~text
apps/
  api/main.py
  api/routes/{polygons,jobs,analyses,health}.py
  worker/celery_app.py
  worker/tasks.py
  web/
src/veg_recovery/
  contracts.py
  inference.py
  service/orchestrator.py
  providers/base.py
  providers/{gee,hls,cdse,modis,era5,fields_world,osm,worldcereal}.py
  geospatial/{geometry,qa,indices,aggregate,harmonize}.py
migrations/
configs/providers/
tests/
infra/
~~~

Не создавай «utils.py» как свалку. Каждый модуль имеет узкую ответственность.

## DATABASE SCHEMA

PostgreSQL + PostGIS:

### polygons

- id UUID PK;
- name text;
- source enum manual/fields_world/osm/worldcereal/demo;
- source_id text nullable;
- geometry geometry(MultiPolygon,4326);
- geometry_hash text unique where appropriate;
- area_ha double precision;
- crop_type text nullable;
- properties jsonb;
- created_at/updated_at.

GIST index по geometry, btree по source/source_id и geometry_hash.

### analysis_jobs

- id UUID;
- polygon_id;
- date_from/date_to;
- status enum;
- progress 0..100;
- stage;
- error_code/error_message_safe;
- retry_count;
- pipeline_version/model_version;
- created_at/started_at/finished_at.

### observations

- polygon_id, date, source, collection_id;
- ndvi/evi/ndwi;
- temp_c/precip_mm;
- valid_pixel_fraction;
- pixel_count;
- qa_flags jsonb;
- asset_ids jsonb;
- processing_version;
- unique key polygon/date/source/processing_version.

### reconstructions

- polygon_id, date;
- primary_ndvi_raw;
- primary_ndvi_reconstructed;
- ndvi_harmonized;
- is_observed/is_reconstructed;
- lower/upper;
- selected_source/method;
- quality_flags;
- model_version.

### anomaly_events

- analysis_id, start/end;
- severity, score, confidence;
- min_robust_z, negative_area;
- reason_codes jsonb;
- explanation_ru;
- algorithm_version.

### provider_cache/provenance

Храни provider, request fingerprint, collection/version, parameters, queried_at, expires_at, status, response metadata и blob/object reference. Не сохраняй access tokens.

Alembic migrations обязательны; create_all в production path запрещён.

## API CONTRACT

Prefix /api/v1.

### Polygon

- GET /polygons?bbox=minx,miny,maxx,maxy&source=...
- POST /polygons с GeoJSON Polygon/MultiPolygon.
- GET /polygons/{id}
- DELETE /polygons/{id}
- POST /field-search с bbox/point и limit.

Валидация:

- valid geometry;
- make_valid только с diagnostic;
- без self-intersection после нормализации;
- area configurable, например 0.1–50 000 ha;
- max vertices;
- antimeridian handling;
- координаты в допустимом диапазоне;
- simplify только для preview, не для анализа без контроля ошибки.

### Analysis

- POST /analyses с polygon_id, date_from, date_to, provider preference.
- GET /jobs/{job_id}
- GET /analyses/{analysis_id}
- GET /analyses/{analysis_id}/series
- GET /analyses/{analysis_id}/anomalies
- GET /analyses/{analysis_id}/provenance
- GET /analyses/{analysis_id}/export.csv

POST возвращает 202 + job_id. Повтор того же idempotency key не создаёт дубликат.

### Health

- /health/live — процесс жив;
- /health/ready — DB, Redis и model bundle доступны;
- /health/providers — отдельная диагностическая готовность providers, не блокирует cached demo.

Не включай secrets/stack traces в public response.

## JOB STATE MACHINE

~~~mermaid
stateDiagram-v2
    [*] --> QUEUED
    QUEUED --> FETCHING
    FETCHING --> PREPROCESSING
    PREPROCESSING --> RECONSTRUCTING
    RECONSTRUCTING --> ANALYZING
    ANALYZING --> COMPLETED
    FETCHING --> PARTIAL
    PARTIAL --> PREPROCESSING
    QUEUED --> FAILED
    FETCHING --> FAILED
    PREPROCESSING --> FAILED
    RECONSTRUCTING --> FAILED
    ANALYZING --> FAILED
~~~

Worker task должен быть идемпотентным по stage. Для transient provider errors — retry; для invalid geometry/auth missing — fail fast с понятным кодом. PARTIAL означает, что один источник недоступен, но минимальный ряд построить можно.

## PROVIDER INTERFACES

~~~python
class FieldBoundaryProvider(Protocol):
    async def search(self, bbox: BBox, limit: int) -> list[FieldCandidate]: ...

class OpticalProvider(Protocol):
    async def fetch(self, polygon: GeoJSON, period: DateRange) -> ObservationCube: ...

class WeatherProvider(Protocol):
    async def fetch(self, polygon: GeoJSON, period: DateRange) -> WeatherSeries: ...
~~~

Каждый результат содержит provider, collection_id/version, item/asset IDs, acquisition time, CRS/resolution, QA definition, processing parameters и license/source URL.

Adapters не знают о FastAPI/DB. Orchestrator управляет ими и преобразует в shared DailyFrame.

## ПОРЯДОК DATA PROVIDERS

### P0 fast path

Если GEE credentials готовы:

- Sentinel-2 SR Harmonized;
- Landsat 8/9 Level-2;
- MODIS/061 vegetation indices;
- ERA5-Land;
- WorldCereal/WorldCover.

Агрегацию делай server-side по polygon. Коллекции и параметры фиксируй. GEE auth через service account env/secret mount. Обязателен startup credential check и cached demo.

### P1 open/cloud-native path

1. HLS v2 для harmonized Landsat/Sentinel-2.
2. CDSE STAC для Sentinel-2 direct fallback.
3. MODIS LP DAAC product с QA.
4. ERA5-Land через CDS.

Используй PySTAC Client для поиска и stackstac/xarray/Dask для ленивой загрузки. Ограничивай spatial/temporal subset до чтения пикселей. Не загружай целые сцены.

Проверь auth/licensing в docs/data_provenance.md. Tokens только в env:

- GEE_SERVICE_ACCOUNT / GEE_CREDENTIALS_FILE;
- EARTHDATA_USERNAME / EARTHDATA_PASSWORD либо bearer token;
- CDSE_CLIENT_ID / CDSE_CLIENT_SECRET;
- CDSAPI_URL / CDSAPI_KEY.

Фактические имена уточнить по официальной документации выбранного adapter; .env.example содержит только placeholders.

## ПОЛИГОНЫ

Порядок автоматического поиска:

1. Fields of the World current GeoParquet/data distribution — основной современный глобальный candidate.
2. OSM/Overpass landuse=farmland и связанные agricultural tags.
3. WorldCereal cropland markers/raster polygonization.
4. Ручной draw.

Алгоритм field-search:

- нормализовать bbox;
- запросить providers параллельно с отдельными timeout;
- validate/make_valid;
- убрать слишком малые/большие;
- deduplicate по IoU/geometry hash;
- вернуть source, confidence/quality, area и preview geometry;
- не объединять соседние поля без явной причины.

Fields of the World велик: не скачивать глобальный слой в контейнер. Использовать spatially partitioned remote query/предзагруженные региональные parquet partitions. Если надёжный bbox query не реализован, P0 переключается на OSM и честно показывает источник.

SAMGeo допускается только как GPU P2. Оно не заменяет manual draw и не должно блокировать CPU demo.

## SATELLITE PREPROCESSING

### QA

- HLS: Fmask bits.
- Sentinel-2: SCL/QA + Cloud Score+ при GEE path; порог конфигурируемый.
- Landsat: QA_PIXEL и saturation flags.
- MODIS: SummaryQA/DetailedQA.

Всегда сохраняй valid_pixel_fraction и количество пикселей. Не усредняй cloudy/nodata как нули.

### Индексы

Считать из surface reflectance после scale/offset:

- NDVI = (NIR − RED) / (NIR + RED);
- EVI с документированными коэффициентами;
- NDWI — явно указать используемую формулу и bands.

Защита от деления на ноль. Не clip до QA без diagnostic. Сохранять raw quantiles и invalid fraction.

### Zonal aggregation

- reproject polygon в raster CRS;
- exact coverage-weighted extraction;
- median как устойчивый центральный показатель;
- p25/p75, mean, std, valid pixel count/fraction;
- минимальное coverage threshold;
- маленькое поле меньше нескольких пикселей помечать low_support.

exactextract предпочтителен для частично покрытых pixels. Для больших задач Dask chunks должны быть ограничены; benchmark на demo polygon.

### Daily table

Shared DailyFrame совместим с training schema. Для competition-like raw primary:

S2 NDVI → Landsat NDVI → MODIS NDVI.

Для продукта дополнительно HLS/harmonized representation. Не смешивай без source column. Погоду агрегируй daily; осадки — sum, температуру — mean/min/max по согласованной семантике.

## SHARED ML INTEGRATION

1. Model bundle монтируется read-only.
2. Загружается один раз на worker startup.
3. Проверяются manifest, hash, feature/schema versions.
4. Orchestrator формирует ReconstructionRequest.
5. Результат сохраняется без преобразования model logic.
6. Для одинакового fixture CLI и worker predictions равны в tolerance.

При несовместимой версии job завершается MODEL_SCHEMA_MISMATCH. Нельзя silently пересчитать признаки иначе.

## FRONTEND

### Экран

- MapLibre карта;
- field boundary layer с легендой источников;
- draw/edit/delete polygon;
- кнопка «Анализировать»;
- progress stages;
- time-series panel;
- anomaly list/detail;
- provenance/quality collapsible panel.

### График

Обязательные визуальные семантики:

- observed NDVI — точки/сплошная линия;
- reconstructed — пунктир или другой явный стиль;
- uncertainty — полупрозрачная лента;
- negative anomaly — красная вертикальная область;
- climatology/expected — нейтральная линия/диапазон;
- source в tooltip;
- weather/NDWI включаются отдельно, не перегружают основной график.

Цвет нельзя использовать как единственный сигнал: добавь line style/icon/text. Tooltip показывает date, raw/harmonized, source, observed/reconstructed, confidence, QA.

### Anomaly panel

Показывает severity, период, magnitude/duration/confidence, reason codes, weather context и фразу об ограничении причинности. Не показывай «точность 95%», если это interval coverage, а не вероятность события.

### Error/empty states

- нет полей в bbox → предложить нарисовать;
- нет credentials → cached demo/инструкция;
- мало valid pixels → quality warning;
- внешний provider временно недоступен → retry/partial;
- анализ выполняется → stage + progress;
- нет аномалий → нейтральный вывод, не ошибка.

## CACHE И DEMO RELIABILITY

Cache key:

provider + collection_version + geometry_hash + date_range + bands +
qa_config_hash + aggregation_version.

Сделай:

- Redis short-lived metadata/job cache;
- persistent object/local volume cache для provider results;
- DB provenance;
- configurable TTL;
- cache invalidation при version change.

Подготовь 2–3 demo polygon:

- один нормальный сезон;
- один сильный отрицательный event;
- один partial-data case.

Кэш создаётся документированной командой до demo. UI помечает cached result и дату извлечения. Нельзя выдавать его за live request.

## DOCKER COMPOSE

Сервисы:

- web;
- api;
- worker;
- postgres с PostGIS;
- redis;
- optional MLflow profile, не в default demo.

Обязательны healthchecks, dependency conditions, non-root containers, mounted model/cache volumes, resource limits where practical. Миграции запускаются отдельной командой/entrypoint один раз. Secrets не bake в image.

## OBSERVABILITY

Structured log поля:

timestamp, level, service, request_id, job_id, polygon_id, stage,
provider, attempt, duration_ms, result_count, cache_hit, error_code,
pipeline_version, model_version.

Метрики минимум:

- job duration by stage;
- provider failure/retry;
- cache hit;
- valid pixel coverage;
- analyses success/partial/fail;
- inference duration.

Для MVP достаточно Prometheus-format endpoint или structured metrics logs; не внедряй тяжёлый monitoring stack до рабочего demo.

## SECURITY И ЛИМИТЫ

- CORS allowlist;
- request size limit;
- polygon vertex/area/date-range limits;
- rate limit analysis creation;
- provider URLs конфигурируются сервером, пользователь не задаёт произвольный URL;
- validate uploaded GeoJSON;
- no secrets/client credentials in frontend;
- safe error messages;
- dependency lock and basic image scan if time allows.

## ТЕСТЫ

Unit:

- geometry validation/reprojection/area;
- QA bit masks;
- NDVI/EVI/NDWI formulas and nodata;
- zonal aggregation;
- provider result normalization;
- cache key stability;
- job state transitions;
- API Pydantic schemas.

Contract:

- model manifest compatibility;
- CLI vs worker prediction parity;
- AnomalyEvent JSON.

Integration:

- PostGIS migrations;
- Redis/worker job;
- mocked STAC/GEE/CDS responses;
- retry/timeout/partial provider;
- analysis persists observations/reconstruction/events.

E2E:

- create polygon → analysis → poll → chart data;
- field search → select → analysis;
- cached demo without network;
- delete polygon;
- invalid polygon.

External live tests отделить marker=live и не запускать по умолчанию. Основной CI полностью offline/deterministic.

## DOCUMENTATION

README/backend sections и docs должны содержать:

- architecture;
- exact startup;
- env variables;
- migrations;
- provider collections/API/auth/licenses;
- QA and aggregation parameters;
- model bundle mount;
- demo cache creation;
- live vs cached behavior;
- common failures;
- batch/API parity command.

docs/demo_runbook.md:

1. preflight command;
2. required services;
3. prepared polygon IDs;
4. основной click path;
5. fallback if provider/auth/network fails;
6. reset instructions;
7. ожидаемое время каждого шага.

## ACCEPTANCE GATES

Gate A:

- draw/save polygon;
- one analysis job on fixture;
- shared model result and graph.

Gate B:

- one live provider path;
- QA/aggregation/provenance;
- retry and cached fallback.

Gate C:

- automatic field search;
- anomaly panel;
- Docker clean start and E2E.

Gate D:

- second provider/region;
- quality/uncertainty polish;
- optional stretch.

Не переходи к P2, пока Gate C не пройден.

## ЗАПРЕТЫ

- Нельзя выполнять raster fetch в HTTP request.
- Нельзя использовать FastAPI BackgroundTasks как durable queue.
- Нельзя дублировать feature/model code.
- Нельзя хранить geometry только как bbox.
- Нельзя считать площадь в EPSG:4326.
- Нельзя игнорировать cloud/quality flags.
- Нельзя скачивать глобальный raster/vector слой при каждом запросе.
- Нельзя требовать внешнюю сеть для основного offline CI.
- Нельзя коммитить credentials или обученные артефакты без manifest.
- Нельзя скрывать cached demo под видом live.

## ФОРМАТ ФИНАЛЬНОГО ОТЧЁТА АГЕНТА

1. архитектура и изменённые файлы;
2. одна команда запуска;
3. URL/route основного сценария;
4. реализованные providers и collections;
5. auth/fallback;
6. job stages и средняя latency;
7. QA/aggregation;
8. model/anomaly contract parity;
9. тесты;
10. demo runbook;
11. известные риски и следующий P1.
