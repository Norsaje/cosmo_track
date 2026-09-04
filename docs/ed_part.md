# Разработчик 4 — начинающий программист

Версия плана: 2026-09-04  
Роль: безопасные вспомогательные проверки, fixtures, документация и подготовка demo  
Главный результат: снизить риск ошибок команды, не становясь владельцем критического компонента

---

# Часть 1. Что нужно прочитать человеку

## 1. Миссия

Вам назначены небольшие задачи с чёткими входами, выходами и тестами. Вы не меняете модель, геообработку, базу или внешние API. Каждая работа должна быть проверена ML- или backend-разработчиком.

Основной принцип: сначала маленький рабочий скрипт и тест, затем аккуратная документация.

## 2. Ваши задачи

P0:

1. Отчёт о качестве train/test.
2. Независимый строгий валидатор submission.csv.
3. Малые безопасные fixtures для тестов и demo.
4. Smoke-check команд проекта.
5. Чек-лист демонстрации и журнал источников.

P1:

1. Линтер таблицы экспериментов.
2. README-разделы по данным и проверкам.
3. Индекс скриншотов и примеров для будущей презентации.

Ничего из P2 не назначается до завершения P0.

## 3. Связь с баллами и сроками

| Вклад роли | Критерий | Доказательство |
|---|---|---|
| Воспроизводимость | код/документация | validators, fixtures, smoke tests |
| Исследование | baseline/эксперименты | data report и table checker |
| Demo reliability | презентация/UX | checklist и evidence index |

| Период | Результат |
|---|---|
| Первые 4 часа | data report + тест |
| 4–8 часов | submission validator + bad fixtures |
| 8–16 часов | tiny fixtures + smoke tests |
| После review | data dictionary, source registry, demo checklist |

## 4. Что известно о данных

Ожидаемые контрольные значения:

| Проверка | Значение |
|---|---:|
| train rows | 99 955 |
| train columns | 21 |
| train polygons | 39 |
| train finite primary_ndvi | 30 520 |
| test rows | 57 185 |
| test columns | 20 |
| test polygons | 78 |
| test finite primary_ndvi | 17 641 |
| test gaps | 3 112 |

Ключ строки: anon_polygon_id + date. В submission должно быть ровно 3 112 gap-ключей и три колонки:

anon_polygon_id,date,primary_ndvi_pred.

## 5. Стек

Python 3.11, uv, pandas, Pydantic, pytest, Ruff. Не устанавливайте PyTorch, GDAL, PostGIS или frontend-зависимости для своих задач.

~~~bash
uv sync --extra dev
uv run pytest -q tests/smoke tests/tools
~~~

## 6. Ваши выходы

- scripts/data_quality_report.py
- scripts/validate_submission.py
- scripts/make_test_fixtures.py
- scripts/check_experiment_table.py
- tests/tools/
- tests/smoke/
- tests/fixtures/
- reports/data_quality.md и reports/data_quality.json
- docs/data_dictionary.md
- docs/research_sources.csv
- docs/demo_checklist.md
- docs/screenshots/README.md

## 7. Definition of Done

- Каждый скрипт имеет --help, понятные ошибки и exit code.
- Скрипты не изменяют исходные train/test.
- Валидатор ловит missing/extra/duplicate/NaN/inf/неверные колонки.
- Fixtures детерминированы и малы.
- Тесты работают без сети, GPU и базы.
- ML-разработчик проверил валидатор; backend проверил demo checklist.

## 8. Первые действия

1. Запустите существующие тесты, ничего не меняя.
2. Сделайте data_quality_report.py и тест.
3. Покажите отчёт ML-разработчику.
4. Сделайте validate_submission.py и набор намеренно плохих примеров.
5. Только после review переходите к fixtures и документации.

---

# Часть 2. Техническое задание для кодингового агента

## ROLE

Ты coding agent, помогающий начинающему разработчику. Выполняй только небольшие вспомогательные задачи из этого документа. Не рефактори чужой код «заодно», не меняй модель, API, БД, Docker и frontend. Делай маленькие коммиты/изменения, добавляй тесты и объясняй результат простым русским языком.

## SOURCE OF TRUTH

1. train_dataset.csv и test_data.csv;
2. case_doc.pdf для submission schema;
3. существующие project contracts;
4. этот документ.

Если контрольные числа не совпали, не «исправляй» данные. Заверши скрипт с понятной ошибкой или warning и сообщи reviewer.

## ВЛАДЕНИЕ ПУТЯМИ

Можно изменять только:

- scripts/data_quality_report.py
- scripts/validate_submission.py
- scripts/make_test_fixtures.py
- scripts/check_experiment_table.py
- tests/tools/
- tests/smoke/
- tests/fixtures/
- reports/data_quality.md
- reports/data_quality.json
- docs/data_dictionary.md
- docs/research_sources.csv
- docs/demo_checklist.md
- docs/screenshots/README.md

README.md можно менять только в отдельном маленьком разделе после review. Нельзя изменять src/veg_recovery/models, features, validation, providers, apps, migrations, infra или dependency lock без явного задания reviewer.

## ОБЩИЕ ПРАВИЛА

- Не редактируй файлы в data/raw и upload.
- Не печатай весь dataset.
- Не логируй secrets/env values.
- Все скрипты имеют argparse и функцию main() -> int.
- Ошибка пользователя выводится кратко в stderr; exit 2 для неверных аргументов/схемы, exit 1 для проваленной проверки, exit 0 для успеха.
- Paths передаются аргументами, никаких абсолютных путей.
- Даты читать строго, invalid date считать ошибкой.
- Числа проверять через isfinite, не только notna.
- Результаты детерминированы.
- Комментарии к неочевидной логике писать по-русски.
- Основные функции должны быть импортируемыми и тестируемыми без запуска subprocess.

## ЗАДАЧА 1. DATA QUALITY REPORT

Создай scripts/data_quality_report.py.

### CLI

~~~bash
uv run python scripts/data_quality_report.py \
  --train data/raw/train_dataset.csv \
  --test data/raw/test_data.csv \
  --out-md reports/data_quality.md \
  --out-json reports/data_quality.json
~~~

### Что посчитать

Для каждого файла:

- rows/columns;
- список колонок и dtype после parsing;
- duplicate key count по anon_polygon_id+date;
- invalid date count;
- date min/max;
- polygon count;
- crop types/counts;
- missing/finite count и rate по каждой колонке;
- finite primary_ndvi count;
- rows per polygon min/median/max;
- years and date range per polygon;
- для test: is_synthetic_gap counts;
- gap-run length distribution по polygon и последовательным датам;
- для gap rows: какие колонки всегда missing;
- known vs unseen test polygons;
- gap count on known/unseen polygons.

Дополнительная важная проверка:

На видимых target-строках определить, совпадает ли primary_ndvi с первым доступным источником по иерархии S2 → Landsat → MODIS. Вывести count/match rate/max absolute mismatch при tolerance 1e-10. Эта проверка только отчётная; она не строит модель.

### Markdown output

Краткие разделы:

1. Summary.
2. Schema.
3. Missingness.
4. Gaps.
5. Polygon overlap.
6. Target-source hierarchy.
7. Warnings.

Не вставляй тысячи строк. Таблицы сортировать понятно. JSON должен содержать те же числа в машиночитаемом виде и report_schema_version.

### Acceptance

- текущие основные counts совпадают с таблицей в части 1;
- duplicate keys = 0;
- test gaps = 3 112;
- одинаковый запуск создаёт побайтно одинаковый JSON, кроме отсутствующего timestamp: timestamp вообще не добавлять;
- unit tests работают на toy fixture и хотя бы один integration test — на реальных CSV, если они доступны.

## ЗАДАЧА 2. STRICT SUBMISSION VALIDATOR

Создай scripts/validate_submission.py. Это независимая страховка, а не замена валидатора ML-разработчика.

### CLI

~~~bash
uv run python scripts/validate_submission.py \
  --test data/raw/test_data.csv \
  --submission submission.csv
~~~

Флаг --json-report optional.

### Проверки

1. Файл декодируется UTF-8.
2. CSV delimiter — comma и header читается однозначно.
3. Колонки строго и в порядке:
   anon_polygon_id,date,primary_ndvi_pred.
4. Нет лишнего index column.
5. Каждая date соответствует YYYY-MM-DD и валидной дате.
6. primary_ndvi_pred парсится в float и finite.
7. Ключи уникальны.
8. Key set точно равен test keys с is_synthetic_gap == True.
9. Нет missing и extra keys.
10. Row count равен числу gaps; для текущего файла 3 112.
11. Опциональный --warn-outside MIN MAX только предупреждает о физически странных значениях; по умолчанию не отклоняй prediction только из-за диапазона.

### Сообщения

При ошибке покажи:

- тип проблемы;
- count;
- первые максимум 10 ключей/примеров;
- как исправить.

Не выводи весь файл.

### Тесты

Создай валидный toy submission и отдельные тесты:

- wrong columns;
- wrong column order;
- duplicate key;
- missing key;
- extra key;
- invalid date;
- NaN;
- inf/-inf;
- string prediction;
- semicolon delimiter;
- BOM допустим только если корректно обрабатывается как UTF-8-sig и не меняет header;
- accidental pandas index.

Acceptance: все bad fixtures возвращают non-zero и правильный error code; valid fixture возвращает 0.

## ЗАДАЧА 3. SAFE FIXTURE BUILDER

Создай scripts/make_test_fixtures.py.

### Цель

Сделать маленькие детерминированные файлы для offline unit/integration тестов, не копируя весь dataset и не раскрывая скрытого ground truth.

### Выходы

- tests/fixtures/train_tiny.csv;
- tests/fixtures/test_tiny.csv;
- tests/fixtures/submission_valid_tiny.csv;
- tests/fixtures/submission_invalid_duplicate.csv;
- tests/fixtures/manifest.json.

### Состав

Выбери детерминированно несколько polygon-year:

- знакомый polygon;
- новый test-only polygon;
- внутренний single gap;
- gap run длиной 2–4, если помещается;
- one-sided context;
- разные crop_type/source cases.

Если редкий case отсутствует в маленьком реальном subset, добавь отдельный полностью synthetic toy fixture с идентификаторами TOY-*, не смешивая его с копией реальных строк.

Manifest хранит:

- source fingerprints;
- selection rule;
- selected keys;
- row counts;
- created_by_script_version.

Не добавляй timestamps, чтобы output был deterministic.

### Privacy/size

Это анонимные данные, но fixture всё равно должен быть минимальным. Не включай лишние сезоны/полигоны. Никогда не придумывай hidden labels для test gap.

## ЗАДАЧА 4. EXPERIMENT TABLE CHECKER

Создай scripts/check_experiment_table.py.

Проверяет reports/experiments.csv, не рассчитывая метрики заново.

Обязательные поля:

experiment_id,timestamp,git_commit,data_fingerprint,fold_version,
mask_version,feature_version,model,params_json,seed,split,rmse,
gap_score,runtime_sec,conclusion.

Проверки:

- experiment_id non-empty and unique;
- rmse finite and >= 0;
- gap_score finite and в [0,30];
- gap_score совпадает с формулой в tolerance 0.01;
- params_json валиден;
- conclusion не пуст;
- для финального кандидата присутствуют matched-mask и unseen-polygon rows.

Скрипт выдаёт компактный summary и non-zero при ошибке. Он не меняет experiments.csv.

## ЗАДАЧА 5. SMOKE CHECKS

Создай offline smoke tests, которые не требуют GPU, сети, PostGIS и model training:

- все четыре scripts запускают --help;
- tiny fixture report строится;
- tiny valid submission проходит;
- invalid submission не проходит;
- experiment table checker проверяет toy CSV;
- import базового пакета не требует torch;
- если CLI уже существует: veg-recovery --help и batch --help.

Не пытайся исправлять чужой CLI при падении. Создай короткий issue note в docs/demo_checklist.md и сообщи owner.

## ЗАДАЧА 6. DATA DICTIONARY

Создай docs/data_dictionary.md.

Для каждой фактической колонки train/test:

- имя;
- смысл;
- тип;
- units, если достоверно известны;
- источник;
- доступна ли на synthetic gap;
- используется ли для submission/контекста/anomaly;
- caveat.

Не выдумывай units/формулу NDWI. Если в документах неоднозначно — напиши «требует подтверждения» и укажи owner.

Отдельно зафиксируй:

- train-only ndvi_zscore/status;
- test-only is_synthetic_gap;
- target output primary_ndvi_pred;
- ключ anon_polygon_id+date.

## ЗАДАЧА 7. RESEARCH SOURCES REGISTRY

Создай docs/research_sources.csv с колонками:

topic,title,url,year,source_type,official_or_primary,license_checked,
used_in_component,decision,owner,notes.

Начальные темы:

- Whittaker/HANTS/BFAST;
- BRITS/SAITS/CSDI/PyPOTS;
- HLS/CDSE/MODIS/ERA5-Land;
- Fields of the World/WorldCereal/OSM;
- PostGIS/PySTAC/stackstac/exactextract;
- SHAP/MAPIE.

Не копируй большие цитаты. URL должны быть прямыми. Поля license_checked по умолчанию false, пока senior не проверил.

## ЗАДАЧА 8. DEMO CHECKLIST И SCREENSHOT INDEX

Создай docs/demo_checklist.md:

### За день до demo

- clean build;
- tests;
- migrations;
- model artifact/hash;
- provider credentials check;
- demo cache;
- three polygons;
- browser/device check;
- backup video/screenshots;
- submission validator.

### За 15 минут

- health/live/ready;
- worker/Redis/PostGIS;
- map tiles;
- model loaded;
- one prepared analysis;
- network fallback.

### Основной сценарий

Нумерованные действия пользователя и ожидаемый результат на каждом шаге.

### Fallback

Что делать при GEE/CDSE/ERA5/map/provider failure. Не скрывать, что результат cached.

### После demo

Сохранить logs, metrics, screenshots и версии.

docs/screenshots/README.md — таблица:

filename,screen,state,polygon,analysis_id,pipeline_version,model_version,
captured_at,owner,approved_for_presentation,notes.

Самих скриншотов пока не создавать, если приложение не готово. Создать правила имён и placeholders.

## REVIEW MATRIX

| Результат | Reviewer | Что reviewer подтверждает |
|---|---|---|
| data quality report | ML | числа и отсутствие leakage |
| submission validator | ML | точная schema/key set |
| fixtures | ML + Backend | репрезентативность и безопасность |
| experiment checker | ML | обязательные поля/формула |
| data dictionary | ML | значения колонок |
| sources registry | каждый owner | использование/license |
| demo checklist | Backend | реальный порядок запуска |

Без review не объявляй задачу принятой.

## ТЕСТОВЫЕ КОМАНДЫ

~~~bash
uv run ruff check scripts tests/tools tests/smoke
uv run pytest -q tests/tools tests/smoke
uv run python scripts/data_quality_report.py --help
uv run python scripts/validate_submission.py --help
uv run python scripts/make_test_fixtures.py --help
uv run python scripts/check_experiment_table.py --help
~~~

Если mypy уже настроен проектом, запусти его только на своих scripts. Не меняй глобальную конфигурацию ради подавления ошибок.

## ЗАПРЕТЫ

- Не обучать модели.
- Не менять CV/folds/features.
- Не редактировать submission predictions.
- Не добавлять «автоисправление» неверного submission; валидатор только проверяет.
- Не менять исходные CSV.
- Не подключаться к внешним API.
- Не создавать DB migrations.
- Не менять Docker/CI/frontend.
- Не устанавливать тяжёлые зависимости.
- Не удалять failing test; передать owner.
- Не включать презентацию в текущий scope, только сохранять evidence/index.

## ФОРМАТ ФИНАЛЬНОГО ОТЧЁТА АГЕНТА

Пиши кратко:

1. что сделано;
2. какие файлы изменены;
3. команды запуска;
4. результаты тестов;
5. какие counts получены;
6. какие плохие submission cases пойманы;
7. что требует review ML;
8. что требует review Backend;
9. найденные проблемы, которых ты не исправлял;
10. следующий маленький безопасный task.
