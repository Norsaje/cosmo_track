# API cosmo_track — контракт C-07 v0.1

Владелец: Разработчик 3 (Backend). Потребители: `apps/web/` и offline smoke-тесты
Разработчика 4. Версия контракта объявлена в `apps/api/schemas/jobs.py`
(`CONTRACT_ID`, `CONTRACT_VERSION`).

Статус: **draft 0.1**. По правилу версионирования (§4 координации) до `1.0`
интерфейс может меняться, но каждый breaking change записывается заранее.
Добавление optional-поля совместимо; переименование, удаление и смена типа — нет.
Изменение состава `JobState` требует уведомления Разработчика 4: на нём завязаны
его smoke- и demo-проверки.

## Запуск

```bash
uv sync --extra web --extra core --extra dev
uv run uvicorn apps.api.main:app --reload      # http://127.0.0.1:8000/docs
uv run pytest -q                               # offline, без сети
```

Весь стенд:

```bash
cp .env.example .env        # заполнить значения; POSTGRES_PASSWORD обязателен
docker compose up --build   # web :8080, api :8000
```

Переменные самого приложения читаются с префиксом **`COSMO_`** (`COSMO_DATABASE_URL`,
`COSMO_REDIS_URL`, `COSMO_MODEL_BUNDLE_PATH`, `COSMO_CORS_ORIGINS`). Префикс не
косметика: без него конфигурация сервиса подхватывалась бы из окружения CI-раннера
и прогон переставал быть детерминированным.

## Эндпоинты

| Метод | Путь | Статус в BE-001 | Приходит в |
|---|---|---|---|
| GET | `/health/live` | **работает**, 200 | — |
| GET | `/health/ready` | **работает**, 200 (db/redis — `not_configured`) | BE-002, BE-006 |
| GET | `/health/providers` | **работает**, 200 (пустой список) | BE-007, BE-009 |
| GET | `/api/v1/polygons` | **работает**: 200 с пустым списком; `bbox` валидируется, мусор → 422 | BE-004 |
| POST | `/api/v1/polygons` | 501 (успех объявлен как **201**) | BE-004 |
| GET | `/api/v1/polygons/{id}` | 501 (успех — 200) | BE-004 |
| DELETE | `/api/v1/polygons/{id}` | 501 (успех — **204**) | BE-004 |
| POST | `/api/v1/field-search` | 501; принимает `bbox` **или** `point`, ровно одно | BE-010 |
| POST | `/api/v1/analyses` | 501; успех объявлен как **202 + job_id** | BE-006, BE-008 |
| GET | `/api/v1/jobs/{job_id}` | 501 (успех — 200) | BE-006 |
| GET | `/api/v1/analyses/{id}` | 501 (успех — 200) | BE-008 |
| GET | `/api/v1/analyses/{id}/series` | 501 (успех — 200) | BE-008, BE-013 |
| GET | `/api/v1/analyses/{id}/anomalies` | 501 (успех — 200) | BE-012 |
| GET | `/api/v1/analyses/{id}/provenance` | 501 (успех — 200) | BE-009 |
| GET | `/api/v1/analyses/{id}/export.csv` | 501; тип содержимого объявлен как `text/csv` | BE-016 |

Health-роуты живут вне префикса `/api/v1`: их дёргают healthcheck-и Compose,
и версия API не должна их двигать.

## Формат ошибки — единый

Любой не-2xx ответ приходит в одном конверте, включая 404 на несуществующий путь,
422 на невалидное тело и 501 на нереализованный роут:

```json
{"error_code": "NOT_IMPLEMENTED", "message": "Реализуется в BE-006 …", "request_id": "8a9abc…"}
```

`error_code` машиночитаем (`NOT_FOUND`, `VALIDATION_ERROR`, `NOT_IMPLEMENTED`,
`INTERNAL_ERROR`, …), `message` безопасен, `request_id` совпадает с заголовком
`x-request-id` и попадает в структурный лог. Ни секретов, ни stack traces, ни эха
присланного тела наружу не уходит. `ErrorResponse` объявлен в OpenAPI, так что
сгенерированный клиент видит его как схему ответа.

## Состояния джобы

```
QUEUED → FETCHING → PREPROCESSING → RECONSTRUCTING → ANALYZING → COMPLETED
             ↓
          PARTIAL → PREPROCESSING
```

`FAILED` достижим из любой рабочей стадии. `PARTIAL` **не терминальна**: провайдер
отдал неполный набор наблюдений, анализ продолжается, результат честно помечается
как частичный — флаг `partial` есть и в `JobStatus`, и в `AnalysisOut`, потому что
фронт опрашивает только `/jobs/{id}`. Терминальны только `COMPLETED` и `FAILED`.

Таблица разрешённых переходов — `ALLOWED_TRANSITIONS` в `apps/api/schemas/jobs.py`.
И воркер (BE-006), и contract-тест обязаны читать её оттуда; тест сверяет таблицу
с эталоном целиком, а не только состав ключей.

## Правила, зафиксированные схемами

- `POST /analyses` отвечает **202 + job_id**. Растровый fetch внутри HTTP-запроса
  запрещён; повторный запрос с тем же `geometry_hash + date_range + pipeline_version`
  обязан вернуть тот же `analysis_id` и не создавать дубликат джобы.
- Тела запросов объявлены с `extra="forbid"`: опечатка вроде `dateFrom` даёт 422,
  а не молчаливый анализ за период по умолчанию. Диапазон дат проверяется на
  перевёрнутость и на максимум в 5 лет.
- Геометрия на границе API — GeoJSON **EPSG:4326**. `area_ha` считает сервер
  в projected CRS; клиент площадь не присылает.
- `reason_codes` в `AnomalyOut` — **открытый** список строк. `Literal`/`Enum` в схеме
  и `CHECK`/enum в БД запрещены: перечень кодов в ТЗ DL нигде не объявлен
  исчерпывающим, а C-09 ещё `0.1`. Фактический детектор из ветки `DL` ограничивает
  себя девятью кодами — это ограничение производителя, а не контракта.
- `AnomaliesResponse` несёт **`warnings` отдельно от событий**, плюс `schema_version`,
  `algorithm_version` и `confidence_semantics`. Без этого пустой `items` означал бы
  одновременно «поле в норме» и «данных не было», а ТЗ DL прямо запрещает трактовать
  отсутствие события при недостатке данных как подтверждённую норму.
- Сырой `primary_ndvi` и `ndvi_harmonized` — **разные поля** ряда (решение D-003).
- `SeriesPoint` несёт качество (`valid_pixel_fraction` **и** `pixel_count`, `qa_flags`),
  климатологию (`ndvi_climatology_mean/std`) и погодный контекст (`temp_c`, `precip_mm`,
  `evi`, `ndwi`): без них tooltip §8.9, нейтральная линия графика и подтверждение
  reason codes нечем наполнить.
- `interval_coverage` объявляет номинальный уровень ленты `lower`/`upper`. Без него
  UI не имеет права подписывать её процентом — §8.9 запрещает «точность 95 %».
- Метка кэша (`cached`, `data_retrieved_at`) стоит и в `SeriesResponse`, и в
  `AnalysisOut`, а не только в `/provenance`: выдавать кэш за live запрещено,
  а требовать ради этого отдельный запрос — значит гарантировать, что его забудут.
- Пустой список — валидное состояние, а не ошибка: «контуров в bbox не найдено»,
  «аномалий нет». UI обязан показывать их отдельными состояниями.

## Чего в BE-001 намеренно нет

- **ModelStub (BE-003)** — требует `veg_recovery/inference.py` и `contracts.py`
  от Разработчика 1; их пока не существует, а писать их самим — вход в чужую зону.
- **Console-scripts `veg-recovery` / `batch`** — точка входа `veg_recovery/cli/batch.py`
  принадлежит ML. Регистрация скрипта на несуществующий модуль сломала бы
  `veg-recovery --help` в smoke-тестах Разработчика 4.
- **Alembic-миграции и подключение к БД** — BE-002.
- **Вендоринг MapLibre и офлайн-тайлы** — BE-014. Сейчас библиотека и тайлы тянутся
  из сети; без неё страница не умирает — панель состояния сервиса запрашивается
  первой и в отдельном блоке, карта деградирует с явным сообщением.
- **Bundle модели** — монтируется из `MODEL_BUNDLE_HOST_PATH` (по умолчанию заглушка
  `infra/model_bundle`). Каталог `artifacts/ml/final_bundle/` создаёт Разработчик 1
  при передаче C-04. Готовым (`ok`) bundle объявляется только при наличии
  `manifest.json`: пустая точка монтирования даёт `degraded`, отсутствующая —
  `not_configured`.
