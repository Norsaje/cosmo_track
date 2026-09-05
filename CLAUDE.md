# CLAUDE.md — Разработчик 3 (Backend / Geospatial / Интеграция)

Проект: `Norsaje/cosmo_track` — веб-сервис мониторинга вегетационной динамики (Космохакатон, кейс «Агропульс»).
Наша роль: **Разработчик 3**. Наша ветка: **`backend`**. Ветка интеграции: **`main`** (только читаем).
Полный ролевой промпт: `used_prompts/backend_dev3_agent.md` — читать при старте задачи.

---

## 0. ЗАКОН №1 — СИНХРОНИЗАЦИЯ ПЕРЕД КАЖДЫМ ЗАПРОСОМ

Teamlead и другие роли пушат изменения без предупреждения (координационный файл, ролевые ТЗ,
shared contracts). **Перед ответом на любой запрос пользователя, до чтения кода и до планирования,
выполни блок синхронизации.** Это не «раз в сессию», а именно каждый запрос.

~~~bash
git fetch --prune -q origin && git status -sb && \
echo "— teamlead запушил в НАШУ ветку:" && git log --oneline HEAD..origin/backend && \
echo "— наши неотправленные коммиты:"   && git log --oneline origin/backend..HEAD && \
echo "— новое в main:"                   && git log --oneline HEAD..origin/main && \
echo "— файлы, разошедшиеся с main:"     && git diff --name-status HEAD...origin/main && \
{ git merge-base --is-ancestor HEAD origin/backend && echo "FF: ok"; } || echo "FF: РАСХОЖДЕНИЕ (возможен force-push)"
bash infra/sync_instructions.sh      # дрейф инструкций → обязательная актуализация промпта (§1)
~~~

Разовая настройка клона (иначе видна только своя ветка):
`git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'`

### Как реагировать на результат

| Что увидел | Действие |
|---|---|
| Пусто везде | Молча продолжай задачу. Не отчитывайся о синхронизации отдельным абзацем. |
| `origin/backend` впереди нас | **Стоп.** Показать коммиты и дельту файлов, предложить `git pull --ff-only`. После подтверждения — pull, затем `graphify . --update`. |
| `FF: РАСХОЖДЕНИЕ` | **Стоп.** Никаких `reset --hard`, `push --force`, `rebase` без явного разрешения. Доложить расхождение и ждать решения. |
| Новое в `main` | Прочитать изменённые файлы **до** планирования. Если задеты shared contracts, `00_team_coordination.md` или ролевые ТЗ — обновить план и сказать, что поменялось для нас. |
| Изменился `00_team_coordination.md` | Перечитать разделы 4 (контракты), 5 (зависимости+уведомления), 6 (ownership), 7 (алгоритм), 8 (checkpoints), 9 (TODO Shared), **10 (TODO ML), 11 (TODO DL)**, 12 (TODO Backend), **13 (TODO Beginner)**, 14 (кто что делает), **15–20 (handoff/blocker/decision/progress/DoR/DoD), 21 (порядок интеграции), 22 (red-team), 23 (дневной статус)**. Чужие TODO читаем обязательно: там лежат статусы артефактов, которых мы ждём. Наш план подчиняется этому файлу. |
| Изменились `src/veg_recovery/contracts.py`, `inference.py`, `anomalies/*` | Сверить с нашими вызовами. Несовместимость → зафиксировать Blocker (§4) и не «чинить» чужой код молча. |
| `sync_instructions.sh` вернул ДРЕЙФ | **Стоп.** Сначала актуализировать промпт по §1, только потом задача. |
| Незакоммиченные локальные правки | Показать список, не затирать. |

### Известное состояние на 2026-09-05

- `backend` = `HEAD` = `origin/backend` = `350faec`, FF ok. Merge-base с `main` — `e1e3ac1`;
  в `main` нет двух наших коммитов: `dc38204` (правила + детектор дрейфа) и `350faec` (скелет BE-001).
- `main` = `7e4381b`. Разошедшиеся с нами файлы — только документы teamlead: `00_team_coordination.md`,
  `docs/01_ml_developer.md`, `docs/02_dl_developer.md`, `docs/ed_part.md`.
- Ветки ролей: `ML` = `39a8f73`, `DL` = `60875b3`, `ED` = `0162b48`, **`models` = `3384de9`**,
  **`integration/dl-ml` = `d90358b`** — собранное DL дерево ML+DL («integrate DL into ML without
  touching ML-owned code»; проверено `git diff` по ML-коду — пусто). Ветка снова нашлась только
  через `gh api .../branches`: урок про `models` повторился буквально. С неё мы взяли C-09
  (мерж `cc54a07`), потому что наложить чужой `src` через PYTHONPATH невозможно — у ролей
  `veg_recovery` namespace-пакет, а у нас обычный, и обычный перекрывает namespace.
  `main` = `d981eb4`.
  **Рабочий код ML и DL уже опубликован в их ветках, но не смержен в `main`.**
- **Ветка `models` — поставка C-04** (`add trained NDVI model`, ветвится от `ML@39a8f73`).
  Она долго оставалась незамеченной: `git branch -r` её не показывал, а детектор дрейфа
  перечислял пять веток жёстко. Урок записан в §1: список веток сверяем с `gh api .../branches`,
  а не с локальными remote-ref'ами, которые отражают лишь то, что мы уже фетчили. Читаем его через `git show origin/<ветка>:<путь>`
  или изолированный `git worktree add --detach tmp/<имя> <sha>`; копий чужого кода в нашу ветку не делаем.
- ML **удалил** `docs/01_ml_developer.md` и `used_prompts/r&d.md` из своей ветки (`39a8f73`).
  Актуальная копия ТЗ ML живёт только в `main`, поэтому все ссылки вида `docs/01_ml_developer.md:NNN`
  в этом файле действительны для `origin/main`, а не для `origin/ML`. Источник истины по
  фактическому поведению ML — теперь **код** в `origin/ML`, а не строки ТЗ.
- Расхождение имён: координационный файл ссылается на `03_backend_developer.md` и
  `04_beginner_developer.md`, фактически это `docs/BACKEND.md` и `docs/ed_part.md`. При правке ссылок — согласовать с teamlead, самим не переименовывать.

---

## 1. ЗАКОН №2 — САМОАКТУАЛИЗАЦИЯ ИНСТРУКЦИЙ И ПРОМПТА

Инструкции команды меняются чаще кода. Промпт, который их не догоняет, — главный источник
рассинхрона. Поэтому появление нового или изменение существующего инструкционного файла
**обязывает переписать промпт до начала работы над задачей**.

### Детектор дрейфа

~~~bash
bash infra/sync_instructions.sh                  # check: NEW / CHANGED / REMOVED, exit 1 при дрейфе
bash infra/sync_instructions.sh diff <path> [ref] # что именно изменилось с момента фиксации
bash infra/sync_instructions.sh accept           # зафиксировать состояние ПОСЛЕ актуализации промпта
~~~

Отслеживаются: `00_team_coordination.md` и любые `*coordination|instruction|prompt|task|spec*.md`
в корне, `NN_*.md`, всё `docs/*.md|*.pdf`, `used_prompts/*.md`, `CLAUDE.md`, `AGENTS.md`, `README.md`,
`.claude/*.md` — в ветках `main`, `backend`, `ML`, `DL`, `ED`, **`models`**. Ветки ролей отслеживаются намеренно:
ТЗ появляется там раньше, чем в `main`, и это ранний сигнал о будущем breaking change.
Состояние зафиксировано в `infra/instructions.lock` (`<ref> <path> <blob-sha>`).

**Список веток — не константа.** `origin/models` существовала и не отслеживалась, потому что
`REFS` был захардкожен, а `git branch -r` показывает только уже зафетченные ветки. Перед
доверием списку выполняем `gh api repos/Norsaje/cosmo_track/branches --jq '.[].name'` либо
`git ls-remote --heads origin`; новая ветка → добавить в `REFS` и заново пройти §1.

### Процедура актуализации

1. `diff` по каждому файлу из отчёта; новый файл — прочитать целиком.
2. Классифицировать изменение:
   - **правило работы для нас** → переписать соответствующий раздел `CLAUDE.md` и/или
     `used_prompts/backend_dev3_agent.md`;
   - **контракт, владение путями, статус задачи, checkpoint** → §4 и §5 этого файла;
   - **новая роль, новый документ, переименование** → SOURCE OF TRUTH промпта и §5;
   - **нас не касается** → строка в журнале «учтено, изменений не требует».
3. Противоречие новой инструкции с промптом решается **в пользу инструкции**.
   Противоречие новой инструкции с уже написанным кодом — это **Blocker** (§4),
   а не молчаливая правка кода.
4. Поднять версию промпта и добавить строку в журнал актуализации (§10).
5. `bash infra/sync_instructions.sh accept`.
6. `graphify . --update` — документы тоже узлы графа.
7. В ответе пользователю: что изменил teamlead, что переписано в промпте, что это меняет в плане.

Актуализация **собственных** инструкционных файлов — единственная правка, разрешённая без
Change Ticket: это не наше решение, а перенос чужого. Любое изменение кода, вытекающее из новой
инструкции, — обычный тикет. Чужие инструкции мы не редактируем никогда, даже ради согласования.

---

## 2. ЗАКОН №3 — ГРАФ ЗНАНИЙ ВСЕГДА АКТУАЛЕН

graphify (v0.9.42, https://github.com/Graphify-Labs/graphify) — рабочий инструмент, а не документация.

~~~bash
graphify . --mode deep --directed     # первый подробный граф (если нет graphify-out/)
graphify hook install                 # авторебилд после каждого git commit
graphify . --update                   # после КАЖДОГО принятого изменения кода/доков
graphify query "<вопрос>"             # обязательный первый шаг любого вопроса об архитектуре
graphify path "<A>" "<B>"             # чем связаны две сущности
graphify explain "<узел>"             # что делает узел
~~~

- Есть `graphify-out/graph.json` → на вопрос об архитектуре, зависимостях или влиянии изменения
  сначала `graphify query`, только потом grep/чтение файлов для проверки гипотезы.
- Задача не считается выполненной без `--update` и проверки: новые routes, стадии FSM, таблицы,
  миграции, adapters, модули geospatial, контракты, компоненты web и тесты **присутствуют в графе**.
- Сущность не попала в граф → это дефект структуры кода (обычно «свалочный» модуль), а не графа.
- Shrink-guard не форсировать: уменьшение графа — сигнал о потере кода или сломанной экстракции.
- После pull чужих изменений — тоже `--update`, иначе граф врёт.

---

## 3. ЗАКОН №4 — СОГЛАСОВАНИЕ КАЖДОГО ИЗМЕНЕНИЯ

Ни одна правка файла не начинается без подтверждённого **Change Ticket**. Чтение, тесты,
запросы к графу и диагностика — свободно.

~~~text
CHANGE TICKET <id>  (task ID из §12 координации, напр. BE-004)
1. Цель + критерий/пункт ТЗ
2. Затрагиваемые файлы и узлы графа
3. Blast radius: входящие/исходящие рёбра (graphify path/query)
4. Контракты: C-xx и версия; breaking change да/нет
5. Чужие зоны: пересечение с ML/DL/Beginner
6. Сверка с main: что расходится
7. Совет личностей: кто высказался, вето
8. План шагов
9. Тесты
10. Откат
11. Риски → Вероятность → Влияние → Митигация → Fallback
~~~

Цикл: **sync (§0) → спросить граф → тикет → ждать подтверждения → реализовать ровно согласованное →
тесты → `graphify --update` → отчёт (§8)**. Расширение объёма — только новым тикетом.

Личности совета (детали в `used_prompts/backend_dev3_agent.md`): L1 Хранитель графа, L2 Архитектор,
L3 Геоинженер, L4 Интегратор провайдеров, L5 Страж контрактов, L6 Надёжность демо, L7 Фронтендер,
L8 QA, L9 Red-team. Кворум: L1 + L2 + профильная + L9; чужая зона или контракт добавляет L5; демо-путь — L6.

---

## 4. ПРАВИЛА КООРДИНАЦИИ (из `00_team_coordination.md`, владелец — teamlead)

Источник истины по статусам — координационный файл, а не наша память. Молча считать чужую задачу
завершённой запрещено. Все номера строк ниже — по версии `origin/main` = `6d5e5fa`.

**Статусы:** NOT_STARTED · READY · IN_PROGRESS · WAITING_DEPENDENCY · BLOCKED · REVIEW ·
CHANGES_REQUESTED · DONE · REJECTED.
**Типы зависимостей:** HARD (без артефакта статус не выше REVIEW) · SOFT (идём с fixture) ·
OPTIONAL (не ждём) · REVIEW (нужна проверка владельца).
**Версионирование (§4):** `0.x` — интерфейс может меняться, но breaking change записывается заранее;
`1.0` — frozen для MVP; добавление optional JSON-поля совместимо; переименование, удаление или
смена типа — несовместимы. Утверждение «все контракты `0.1 planned`/`NOT_STARTED`» устарело:
C-01/C-02 у producer уже `1.0` (team freeze pending), C-09 — `schema 0.1 / algorithm 0.1.1`,
наш C-07 — `0.1 draft`. Актуальный срез — в таблице §4.1.

### 4.1. Контракты: что производим и что потребляем

Колонка «Факт» — состояние на `main` = `7e4381b` (обновлено самим DL, не только teamlead).

| ID | Артефакт | Владелец | Путь | Тип | Факт на 2026-09-05 | Наша роль |
|---|---|---|---|---|---|---|
| C-07 | API schemas и job states | **мы** | `apps/api/schemas/` | HARD | **0.1 draft · REVIEW: branch publication observed, DL не проверял API** (`backend@350faec`) | производим; потребители — фронт и smoke-тесты Beginner |
| C-08 | Shared model integration in worker | **мы** | `service/orchestrator.py` | HARD | 1.0 planned · NOT_STARTED | производим |
| C-13 | Provider-normalized ObservationFrame | **мы** | `providers/base.py` | HARD для live web | NOT_STARTED | производим; **reviewer — ML** |
| C-15 | Demo runbook/checklist | **мы + Beginner** | `docs/demo_runbook.md` | HARD для CP-5 | NOT_STARTED | производим **совместно**, не единолично |
| C-01 | DataContract и canonical `DailyFrame` | ML | `contracts.py`, `data/io.py` | HARD | **producer 1.0, team freeze pending · REVIEW: DL read/window alignment passed** (`ML@39a8f73`) | потребляем; изменение — только после review DL и Backend |
| C-02 | `ReconstructionRequest/Result`, `NDVIReconstructor` | ML | `inference.py` | HARD | **producer 1.0, team freeze pending · REVIEW: Backend integration pending** — ждут именно нас | потребляем |
| C-03 | Frozen folds, MaskSpec, baseline OOF | ML | `configs/ml/folds_v1.csv`, `artifacts/ml/baseline_v1/` | HARD | folds_v1 / real_test_v1 · REVIEW (DL проверил 16 folds) | нас не касается напрямую; источник parity-фикстур |
| C-04 | Final ML bundle + manifest/hash | ML | `artifacts/ml/ndvi_backend_handoff_v1/` в ветке **`models`** | HARD | **ОПУБЛИКОВАН** (`models@3384de9`): `bundle_kind: trained`, `model_version: p0-catboost-gpu-v1`, эталоны на 3 112 строк, `verify_handoff.py` | **больше не потребляем в горячем пути** (BE-011R): веб-путь обслуживает поставка `model/`. Handoff H-003 принят, загрузка проверяется тестом `test_contract_c04.py` через `load_ml_bundle` |
| C-05 | ML diagnostics schema | ML | `contracts.py` | SOFT | фактически реализован как `DiagnosticRow` (16 полей) | потребляем; fallback — пустые optional diagnostics |
| C-06 | WindowDatasetAdapter | DL | `dl/data.py` | INTERNAL | 0.1 draft · REVIEW | не потребляем |
| C-09 | AnomalyEvent schema/detector | DL | `src/veg_recovery/anomalies/` | HARD для CP-4 | **schema 0.1 / algorithm 0.1.1 · ПОДКЛЮЧЁН** (мерж `cc54a07`): H-004 принят, `tests/anomalies` 18 passed, веб-путь считает события `AdvancedAnomalyDetector`, baseline остался fallback | потребляем; public contract согласуется с нами |
| C-12 | Tiny fixtures + manifest | Beginner | `tests/fixtures/` | SOFT | 1.0 planned · NOT_STARTED в координации, фактически в `ED` | потребляем; **мы соreviewer вместе с ML** |
| C-11 | Independent submission validator | Beginner | `scripts/validate_submission.py` | REVIEW | NOT_STARTED | входит в наш demo-preflight |

**Два контракта ждут именно нашего acknowledgement — C-02 (Backend integration pending) и
C-09 (Backend ack pending).** Пока мы их не приняли, ни ML, ни DL не могут закрыть свои задачи.
Принятие по §20 — это запущенный contract/smoke test и заполненный handoff, а не «прочитали».

C-02 **фактически опубликован кодом** (`git show origin/ML:src/veg_recovery/contracts.py`),
и ModelStub обязан повторять именно опубликованное, а не строки ТЗ:

~~~python
SCHEMA_VERSION = "1.0"
KEY_COLUMNS = ["anon_polygon_id", "date"]
SUBMISSION_COLUMNS = KEY_COLUMNS + ["primary_ndvi_pred"]

@dataclass(frozen=True)
class ReconstructionRequest:
    frame: pd.DataFrame
    gap_mask: pd.Series
    context_mode: Literal["competition", "web"]

@dataclass(frozen=True)
class ReconstructionResult:
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    model_version: str

@runtime_checkable
class NDVIReconstructor(Protocol):
    def predict(self, request: ReconstructionRequest) -> ReconstructionResult: ...
~~~

Что добавилось сверх ТЗ и обязано быть учтено в C-07/C-08:

- **`PredictionRow` — 8 полей**, а не четыре: `anon_polygon_id`, `date`, `primary_ndvi_pred`,
  `lower`, `upper`, `method`, **`primary_ndvi_reconstructed`**, **`ndvi_harmonized`**.
  `model_config = ConfigDict(extra="forbid", allow_inf_nan=False)`.
- **`DiagnosticRow` — 16 полей**: `p_s2`/`p_landsat`/`p_modis`/`p_unknown` (сумма = 1 ± 1e-6),
  `left_distance_days`/`right_distance_days` (**`float | None`**, в JSON — `null`, не `inf`),
  `model_disagreement`, `fallback_reason`, `context_quality`, `source_confidence`,
  `quality_flags: list[str]`, `interval_status`, `interval_level`, `harmonization_status`.
  Это и есть фактический C-05 — отдельной схемы ждать не нужно.
- **`ReconstructionPayload`** — «JSON-safe DTO for backend; DataFrames stay inside the Python
  boundary», `schema_version: Literal["1.0"]`, конструируется `ReconstructionPayload.from_result(result)`.
  Это готовая граница между worker и API: **DataFrame через HTTP не отдаём**, свою параллельную
  DTO не изобретаем. Changelog ML прямо просит: «Integrators should approve the DTO schema».
- **`validate_request` отвергает** (это требования к нашему orchestrator, не рекомендации):
  отсутствие колонок `{anon_polygon_id, date, primary_ndvi, crop_type}` — **`crop_type` обязателен
  и на веб-пути**; `gap_mask` не `pd.Series` или с индексом, не равным индексу фрейма; не-bool dtype
  или NaN в маске; неуникальный индекс фрейма; tz-aware или ненормализованные даты; дубликаты ключей;
  пустой `anon_polygon_id`. В `competition` дополнительно требуется bool-колонка `is_synthetic_gap`,
  **точно равная маске**. Фрейм копируется — молчаливого выравнивания индексов не будет.
- `method` принимает значения `mean_neighbors` / `oof_ensemble` / `conservative_blend`,
  `fallback_reason` — включая `nonfinite_model_prediction` и `oof_conservative_gate`.
  В API это `str`, без `Literal`/`Enum` — по той же причине, что и `reason_codes`.
- Веб-путь всегда шлёт `context_mode="web"`. **Единственный протокол предсказателя — `NDVIReconstructor`.**
  `PredictionExpert` в опубликованном коде отсутствует; как контракт не используем.

До C-04 работаем с **ModelStub** того же интерфейса *или* с реально опубликованным baseline-бандлом
(§4.5). Никакой fallback не имеет права вернуть NaN (`docs/01_ml_developer.md:315`), ML-логику в API
переписывать нельзя (`:501`). ModelStub обязан быть выключен в production.

C-09 `AnomalyEvent` — 12 полей, `reason_codes: tuple[str, ...]`. **Список кодов открытый**: ТЗ DL нигде
не объявляет его исчерпывающим. `Literal`/`Enum` в API-схеме и CHECK/enum в БД по `reason_codes` запрещены —
хранить jsonb, неизвестный код рендерить как есть. Переход к закрытому списку — только после заморозки
C-09 1.0 и оформленного Decision.


### 4.2. Наша очередь задач — это граф, а не цепочка

Зависимости дословно из §12 координации. Стрелка «→» в прежней редакции вводила в заблуждение.

| Задача | Depends on | Примечание |
|---|---|---|
| **BE-001** API/worker/web/Compose skeleton | none · **Status: READY** | активная |
| **BE-003** ModelStub по C-02 draft | **BE-001** (не BE-002); C-02 SOFT для старта, HARD для DONE | можно сразу после BE-001 |
| BE-002 PostGIS schema/migrations | BE-001 | |
| BE-004 Polygon CRUD/validation | BE-001, BE-002 | |
| BE-005 MapLibre draw/select/delete | BE-004 **SOFT**, fallback — mocked API | |
| BE-006 Durable job state machine | BE-001, BE-002 | |
| BE-007 Provider interfaces и C-13 | C-01 SOFT/HARD перед parity | **не зависит ни от одной BE**; reviewer — ML |
| BE-008 Offline fixture E2E | BE-003, BE-004, BE-006; C-12 SOFT | **блокирует CP-2** |
| BE-009 Один live provider path | BE-006, BE-007 | выход: QA, indices, zonal aggregate, provenance |
| BE-010 Automatic field search | BE-004 | fallback FTW → OSM → WorldCereal → draw |
| BE-011 Подключить модель | C-02, BE-008 | **выполнено как BE-011R**: подключена поставка `model/` (смесь LGB + спутниковые эксперты, 324 признака), а не C-04. Parity веб-пути с пакетным `submission.csv` проверяется тестом |
| BE-012 Подключить C-09 AnomalyEvent | C-09 draft для UI, 1.0 для DONE | **выполнено, статус REVIEW**: детектор DL подключён, severity/reason_codes/explanation_ru приходят от производителя. DONE по §14 объявить нельзя — C-09 всё ещё `0.1`. Восстановленные точки в оценку не идут (у предсказания нет сенсора), погодные reason codes не выставляются — способ расчёта их колонок DL не специфицировал |
| BE-013 Final chart/anomaly/provenance UI | BE-011, BE-012 | |
| BE-014 Cache/retry/partial/demo fallback | BE-006, BE-009 | |
| BE-015…BE-018 | — | optional: второй provider/region, compare/export, SAMGeo после CP-4, observability |

**Shared-задачи, где мы владелец** (§9): SH-001 «Repository structure и ownership» (reviewer ML) ·
SH-002 «Python 3.11, uv, pyproject extras **core/ml/dl/geo/web/dev**» (reviewers ML, DL) ·
SH-005 «Offline CI: **core, contracts, DL smoke, Backend fixture**» (reviewers — все; depends on SH-001/SH-002).
DL smoke мы гоняем, но `tests/dl/` — зона DL, свои тесты туда не пишем.

**Где мы reviewer, а не исполнитель:** `contracts.py` (любое изменение C-01/C-05 — после review DL и Backend,
§6 координации) · SH-004 «Seed policy, UTC и key/date semantics» (owner ML) · JR-004/C-12 fixtures (с ML) ·
JR-009 demo checklist. `src/veg_recovery/providers/` — наш путь, но **ML проверяет shared schema**.

### 4.3. Checkpoints

CP-0 **DONE** → CP-1 contracts frozen → CP-2 baseline E2E → CP-3 сильный ML + live provider →
CP-4 anomaly MVP → CP-5 hardening/docs/demo. К P2 не переходим, пока не закрыт CP-4/Gate C.

**CP-1, срок «первые 4 часа», требует от нас трёх пунктов одновременно:**
`C-07 API/job draft опубликован` · `ModelStub соответствует C-02` · `Contract tests проходят у ML, DL и Backend`.
Отсюда: BE-003 идёт сразу за BE-001, не третьим номером. Отдельной BE-задачи «contract tests» в §12 нет —
закрываем их через SH-005 и DoD каждой задачи, но §21 п.3 `Contract tests DL/Backend` числится за нами.
Закрытие CP-1 упирается в C-02 от ML (`NOT_STARTED`). CP-1 требует подтверждения ML, DL и Backend.

CP-2 требует `Backend fixture E2E проходит` (BE-008). CP-3 — `Backend вызывает реальный bundle`,
`Один live provider работает`, `QA/cloud/zonal processing проверены`, `Provider provenance сохраняется`.
CP-4 — `Batch/API prediction parity`, `raw primary и harmonized series разделены`, `UI различает
observed/reconstructed/uncertainty/anomaly`, `Полигон можно найти или нарисовать`,
`Provider failure имеет retry/partial/cache fallback`. CP-5 — clean install, `docker compose up --build`,
offline CI без сети/GPU, README с batch/web командами, `C-15 runbook принят`, red-team review.

### 4.5. Текущее состояние команды (`main` = `7e4381b`, `ML` = `39a8f73`, `DL` = `44f2f5b`, `ED` = `0162b48`)

Проект `IN_PROGRESS`. Teamlead переписал шапку прогресса: «Принятая интегрированная реализация
(без draft на review)» — **0 %**; 10 % теперь означают «CP-0 принят, остальные checkpoints ещё
не приняты», и отдельной строкой: «DL draft уже реализован в ветке DL и передан на review;
это не означает готовность общей модели/приложения или прохождение CP-1».
**Наличие файлов в чужой ветке не переводит задачу в DONE — ни чужую, ни нашу.**

| Роль | Статус | Текущая задача | Блокер |
|---|---|---|---|
| ML | REVIEW: публикация обнаружена DL; owner acknowledgement pending | C-01/C-02, folds и baseline OOF в `ML@39a8f73` | B-DL-003: EOL/hash bundle; final GPU OOF не опубликован |
| DL | WAITING_DEPENDENCY / REVIEW | DL-001/002 real ML consumer review; DL-007…010 real anomaly cases | train/inner policy, final ML OOF, SH-002 clean-install |
| **Backend (мы)** | **REVIEW: публикация обнаружена DL; owner acknowledgement pending; 0 % принятых** | **BE-001 skeleton и SH-002 в `backend@350faec`**; следующий handoff — C-07 и H-SH-002 | ModelStub разрешён; shared lock integration/review |
| Beginner | READY | JR-001 Data-quality report | нет |

Наш статус выставил **не teamlead, а DL**, обнаружив нашу ветку через fetch. «REVIEW: branch
publication observed, DL не проверял API» означает ровно одно: draft увиден, но не отревьюен.
Сдвинуть его дальше может только наш собственный handoff по C-07.

#### Главное: ML опубликовал реальные C-01/C-02/C-03 и загружаемый bundle

`origin/ML` = `39a8f73` содержит `src/veg_recovery/{contracts,inference}.py`, `data/io.py`,
`features/builder.py`, `models/{bundle,manifest,estimators,calibration,training,baselines}.py`,
`validation/{folds,masking,metrics}.py`, `cli/batch.py`, `anomalies/baseline.py`,
`configs/ml/folds_v1.csv` и `artifacts/ml/baseline_v1/` (24 файла). Схема разобрана в §4.1.
Практические следствия для нас:

- **`artifacts/ml/CONTRACT_CHANGELOG.md` адресован нам дословно:** «Backend imports
  `ReconstructionRequest`, `ReconstructionResult`, `NDVIReconstructor` from `veg_recovery.contracts`
  and `load_reconstructor` from `veg_recovery.inference`. **Load a bundle once at process startup**,
  then pass an aligned boolean gap mask per call. … `ReconstructionPayload.from_result(result)` is
  the Pydantic 2 JSON boundary. … **Integrators should approve the DTO schema when connecting their
  own routes**». Загрузка один раз на старте воркера — теперь требование producer, а не наша догадка.
- **Опубликованный bundle — не C-04.** Путь `artifacts/ml/baseline_v1/bundle/`
  (`bundle_kind: "baseline"`, `model_version: "baseline-v1-mean_neighbors"`,
  `training_status: "no ML estimators trained; baseline statistics only"`, `schema_version 1.0`,
  `feature_version "ndvi-context-v1"`, seeds `[17,29,43,71,101]`). C-04 `artifacts/ml/final_bundle/`
  по-прежнему `NOT_STARTED`. Значит `BE-011 «подключить C-04»` объявлять сделанным нельзя,
  но **BE-003/BE-008 могут работать против реального baseline-бандла вместо ModelStub** —
  это сильнее заглушки и сразу даёт parity-фикстуру. Отдельный тикет, не самовольство.
- `load_bundle` отказывает при: `schema_version != "1.0"`, `feature_version != FEATURE_VERSION`,
  несовпадении состава файлов с `bundle_kind`, неверном `format`, symlink на manifest или файл,
  файле вне корня бандла, **несовпадении SHA256**. Это уточняет инвариант 6: критериев отказа
  у producer больше одного, и все они наши `MODEL_SCHEMA_MISMATCH`.
- **`trained` bundle требует `trusted=True`** — `load_bundle` иначе бросает
  «Trained bundle requires trusted=True after provenance review». Докстрока producer:
  «SHA256 detects corruption, not authenticity… a joblib file can execute Python code».
  Значит в worker флаг доверия — явная настройка окружения с провенанс-проверкой,
  никогда не значение по умолчанию и никогда не «включить, чтобы заработало».
- **Фактическая сигнатура batch-CLI отличается от ТЗ** — см. §7.
- `api_smoke.json`: `{model_version: "baseline-v1-mean_neighbors", n_predictions: 3112,
  n_diagnostics: 3112, schema_version: "1.0"}` — готовый эталон для нашего smoke-теста.
- `reports/data_contract_issues.md` подтверждает наши расхождения (`private_features.csv` vs
  `data/test_data.csv`) и добавляет факт, важный для схем и UI: **target не клипается**,
  реальный диапазон train до `−2.1303786081`, test до `1.8428536898`. Валидировать NDVI
  диапазоном `[-1, 1]` в API-схемах и БД **запрещено** — физическая нештатность идёт флагом
  (`prediction_outside_physical_range` в `quality_flags`), а не отказом 422.

#### C-04 опубликован в ветке `models` — обученная модель, а не baseline

`origin/models` = `3384de9` ветвится от `ML@39a8f73` и добавляет
`artifacts/ml/ndvi_backend_handoff_v1/` — самодостаточный пакет поставки для нас:

- `bundle/` — **`bundle_kind: "trained"`**, `model_version: `p0-catboost-gpu-v1`,
  `schema_version 1.0`, `feature_version ndvi-context-v1`, `estimators.joblib`.
  **Требует `trusted=True`** — и README ML это оговаривает: доверие допустимо
  «только после проверки внешнего SHA256 архива и `MANIFEST.sha256`».
- `runtime/src/veg_recovery/` — **побайтово тот же код**, что `src/` в `ML@39a8f73`
  (сверено по sha256 для `contracts.py`, `inference.py`, `features/builder.py`,
  `models/bundle.py`). Это копия для автономной поставки, не форк.
- `examples/`: `test_data.csv` (57 185 строк), `expected_submission.csv` (3 112),
  `expected_diagnostics.csv` — готовый parity-эталон с допуском **1e-10**.
- `verify_handoff.py` — проверка целостности по `MANIFEST.sha256` и полного инференса.
- `requirements-runtime.txt` — точные версии: `numpy 2.0.2`, `pandas 2.3.3`, `scipy 1.16.3`,
  `scikit-learn 1.6.1`, `pydantic 2.12.3`, `joblib 1.5.3`, **`catboost 1.2.10`**.
  `catboost` в наших extras отсутствовал — это дефект `pyproject.toml`, закрывается в SH-002.
- `handoff.json`: composite RMSE baseline `0.106699` → catboost `0.099562` →
  ансамбль **`0.097412`**, веса `0.309550 / 0.690450`.

**Прямые указания ML, которые мы обязаны соблюдать** (`README_BACKEND.md`):

- «Ветка `models` уже содержит проверенную копию модельных файлов; **дублировать их
  в рабочей ветке Backend не требуется**» — бандл монтируем по пути, к себе не копируем.
- Минимальные колонки входа: `anon_polygon_id`, `date`, `crop_type`, `primary_ndvi`;
  для качества — `s2_*`, `landsat_*`, `modis_*`, `era5_temp_c`, `era5_precip_mm`.
- «Backend должен сохранять diagnostics и показывать `quality_flags`, особенно для новых
  полигонов и слабого контекста».
- Главное ограничение прямым текстом: **«перенос на будущий сезон: в temporal CV ансамбль
  хуже простого baseline»**. В UI и демо это нельзя замалчивать.
- `interval_level = 0.95`, но `interval_status` говорит, что интервалы эмпирические
  и формально не сертифицированы — процентом уверенности их подписывать нельзя.
- `gate = null` и clip не применяется: значения вне `[-1, 1]` физически возможны
  (train до −2.1304, видимый test до 1.8429) — ещё одно подтверждение инварианта 4.

#### Что изменилось в координации на `main@d981eb4`

DL перешёл в `IN_PROGRESS`: C-06 поднят до **0.2**, runner обучается и предсказывает
на реальных folds ML, заведён **H-DL-003** (`DL@8f19c25`, 72 теста). Появились
`D-DL-008…D-DL-011` и новый блокер **B-DL-004** (owner ML): в опубликованном
`oof_predictions.csv.gz` нет колонок `pred_catboost`/`pred_ensemble`, а
`artifacts/ml/trained_gpu_v1/` исключён `.gitignore` ML — у adoption gate нет
вычислимого знаменателя. Нас это не блокирует.

**`D-DL-011` касается нас прямо:** «Глобальный blend ML ухудшает CV-C относительно
их же baseline» — по числам самого ML: C baseline `0.178772`, CatBoost `0.246606`,
ансамбль `0.220068`. Это независимое подтверждение того, что мы уже закрепили тестом
`test_temporal_transfer_is_worse_than_baseline`: **в UI и демо модель нельзя подавать
как безусловно лучшую**, а перенос на будущий сезон нужно оговаривать.

**Наш статус в dashboard устарел** — там всё ещё «BE-001 skeleton и SH-002 в
`backend@350faec`». Координация не знает про BE-002/003/004/006/008/011, потому что
мы их не пушили. Пока не отправим ветку, для команды этой работы не существует.

#### Блокеры

- **B-DL-002 (owner — мы) переформулирован**, но не закрыт: «`backend@350faec` уже публикует
  pyproject/lock, но DL ещё не выполнил clean-install на нём; Backend/ED shared config collision».
  Из `reports/dl_integration_review.md`: `dl = [torch]` достаточно, **PyPOTS не добавлять**;
  **нужен `matplotlib` в reporting/dev extra** (у нас его в `dev` нет — это конкретный дефект
  нашего `pyproject.toml`); DL локально проверял torch 2.5.1 CPU и «Backend lock с другой версией
  не объявляем проверенным нами»; `acknowledgement=pending` до clean-install и smoke.
- **B-DL-003 (owner ML, CHANGES_REQUESTED)** — новый и опасный для нас: в чистом Windows-checkout
  `load_reconstructor(.../baseline_v1/bundle)` падает `ValueError: SHA256 mismatch: feature_state.json`,
  потому что Git переписал EOL уже захешированных JSON. Наш вывод: любой наш артефакт с байтовым
  хешем (bundle, фикстуры parity, demo cache) обязан быть защищён `.gitattributes`, а
  `MODEL_SCHEMA_MISMATCH`-путь в воркере обязан отличать «несовместимая схема» от «повреждённый
  транспорт» в тексте ошибки. Проверку хешей не отключаем никогда.
- B-DL-001 (owner ML + DL consumer) нас не касается напрямую.

#### Decisions, добавленные DL (§17)

`D-DL-001` mask entire allowed context before windowing · `D-DL-002` PENDING_EVALUATION вместо
REJECT без реального OOF · `D-DL-003` CUDA только на Kaggle · `D-DL-004` stock PyPOTS SAITS.fit
не является matched-mask воспроизведением · `D-DL-005` C-03 берётся из реального `split_fold`.
Прямо на нас влияют два:

- **`D-DL-006`** — «byte-hashed artifacts должны переживать Git checkout без EOL rewrite»
  (см. B-DL-003). DL уже защитил свои `artifacts/dl/.gitattributes` (`-text`).
- **`D-DL-007`** — «реальные anomaly candidates не считаются размеченной точностью»:
  91 кандидат, 12 algorithmic critical, экспертной приёмки нет. **BE-013 не имеет права
  показывать эти события как подтверждённые и заявлять точность детектора.**

#### C-09: algorithm 0.1.1, добавлен `analyze()`

`origin/DL` = `44f2f5b`: `SCHEMA_VERSION = "0.1"` не изменился, `ALGORITHM_VERSION` стал
**`robust-loyo-events-0.1.1`**. Добавлен `analyze(frame) -> (points, DetectionResult)`,
`detect()` сохранён, **имена JSON-полей не переименованы**. Правка меняет source-risk/confidence
для рядов с пустыми календарными строками: «natural calendar NaN больше не дают false sensor
switches/reconstruction warning». Наш `AnomalyOut` от этого не ломается, но `algorithm_version`
обязан храниться и показываться — по нему различаются результаты 0.1.0 и 0.1.1.

Состав контракта прежний и остаётся в силе: 12 полей; `severity` — закрытое множество
`normal` / `biomass_suppression` / `critical` (наш тип `str`, но UI различает ровно эти три);
`reason_codes` — `frozenset` из девяти кодов у производителя (`LOW_PRECIPITATION`,
`HIGH_TEMPERATURE`, `LOW_NDWI`, `MULTISENSOR_CONFIRMATION`, `SOURCE_SWITCH_RISK`,
`LOW_DATA_COVERAGE`, `RAPID_NEGATIVE_CHANGE`, `PROLONGED_SUPPRESSION`, `PHENOLOGY_SHIFT`),
конструктор бросает `ValueError` на неизвестном коде — но это ограничение **производителя**,
а не контракта: `Literal`/`Enum` и CHECK/enum по `reason_codes` у нас по-прежнему запрещены.
Валидация producer: хотя бы одна опорная точка, все числа конечны, `0 <= confidence <= 1`,
`score >= 0`, `negative_area >= 0`. `confidence` — heuristic support, **не** калиброванная
вероятность; «нет событий при недостатке истории» — отдельное состояние с warning, не норма.

**Backend action из `reports/dl_integration_review.md` — дословные требования к BE-012/BE-013:**

- `is_reconstructed` у ML **не равно** `is_observed` у C-09: observed выводится из исходного
  конечного `primary_ndvi` и отсутствия реконструкции; **естественные NaN не являются reconstructed**.
- Продуктовый `confidence` **не** передавать как cloud QA — это разные величины.
- `lower`/`upper` в шкале primary **нельзя** выдавать за uncertainty гармонизированной шкалы
  без соответствующего преобразования.
- Reference и query обязаны иметь одну calibration/version; смешивать ML median/IQR affine
  с DL IRLS/pooling в одном вызове детектора запрещено.

**Фикстуры DL готовы и дублировать их не нужно:** `reports/anomaly_cases/synthetic_v1/`
(семь кейсов, пересобраны под 0.1.1: 3 pulses обнаружены, 0 alerts в 4 negative controls) и
`reports/anomaly_cases/real_2024/` (6 разобранных кейсов с CSV/JSON/PNG, `review.md`,
`sensor_alignment.csv`, `polygon_summary.csv`). Это вход BE-012 и материал BE-008.

#### Коллизия владения: `pyproject.toml` и `uv.lock` в ветке `ED`

`origin/ED` добавляет в корень `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`,
`tests/__init__.py` — артефакты SH-002, наши по §9 и §6. Их редакция несовместима с нашей:
`[dependency-groups] dev` (PEP 735) вместо `[project.optional-dependencies]`, `package = false`,
`dependencies = []`, ни одного из шести extras `core/ml/dl/geo/web/dev`, `testpaths` только
на их каталоги. Конфликтуют минимум пять файлов.

Чужую ветку не трогаем и молча не «чиним». Это повод для **Decision по §17** с участием teamlead:
чей `pyproject.toml` канонический. Наша позиция подкреплена координацией («SH-002 · Owner: Backend»),
самим DL («Общий lock принадлежит Backend», «коллизию решает владелец Backend») и опубликованным
handoff `docs/handoffs/H-SH-002.md`. **Дефект нашего handoff:** в поле Git commit там написано
«не закоммичено, HEAD = e1e3ac1» — DL это заметил; после `350faec` текст неверен и требует правки
отдельным тикетом.

#### C-12 фикстуры опубликованы в `ED`

`tests/fixtures/` — 20 файлов: `train_tiny.csv` (60 строк, 21 колонка, реальный `AOI-0002`),
`test_tiny.csv` (125 строк, 20 колонок, 5 гэпов), `submission_valid_tiny.csv`,
`submission_for_test_tiny.csv`, `submission_valid_bom.csv`, одиннадцать `submission_invalid_*`
и `manifest.json`. Синтетика помечена префиксом `TOY-`. В манифесте
`design_assumptions_requires_ml_confirmation: true` — допущения ещё не подтверждены ML.
**Мы соreviewer вместе с ML**: проверяем репрезентативность (single gap, gap run 2–4,
one-sided context, unseen polygon) и отсутствие выдуманного ground truth. Геометрии в фикстурах
нет — полигон для E2E готовим сами в `tests/e2e/`. Сверяться есть с чем: фактический
`validate_submission_file` у ML отвергает BOM, не-UTF-8, неверный заголовок, ширину строки ≠ 3,
дубли и несовпадение порядка ключей — набор `submission_invalid_*` обязан бить в те же случаи.

#### `docs/demo_checklist.md` опубликован в `ED` и ждёт нас

Файл в зоне Разработчика 4, но его шапка гласит: «стартовый шаблон от Developer 4.
**Финальный чек-лист заполняется Backend-ревьюером**». Практически все `[TODO]` адресованы нам
и составляют содержание нашего C-15 `docs/demo_runbook.md`:

- миграции; проверка provider credentials; что именно кэшируется в demo cache;
- **какие три полигона** показываем (их гипотеза — `AOI-0032`, `AOI-0013`, `AOI-0067`, топ-3 по
  числу гэпов, требует подтверждения ML);
- URL и ожидаемые коды health-эндпоинтов; команды проверки worker/Redis/PostGIS;
  offline-кэш тайлов карты; какой эндпоинт подтверждает загруженный bundle;
  какой подготовленный анализ показываем; план на отключение сети;
- нумерованный основной сценарий с ожидаемым результатом каждого шага;
- fallback по каждому источнику: GEE, CDSE, ERA5-Land, тайлы, model provider —
  с прямым напоминанием «не скрывать, что результат cached».

Конвенция путей из их чек-листа, которую нам соблюдать: логи демо — `artifacts/demo_<date>/logs/`,
индекс скриншотов — `docs/screenshots/README.md`. В `ED@0162b48` рядом появился
`docs/research_sources.csv`.

Полезные факты из их issue-notes:

- **Иерархия сенсоров измерена независимо**: `match_rate = 1.0`, `max_abs_mismatch = 0.0`
  для восстановления `primary_ndvi` из S2 → Landsat → MODIS.
- **Расхождение `data/raw/` против `data/` подтверждено ими же**; их скрипты используют реальный
  `data/`. Для нас `data/*.csv` неизменяемы — решение за teamlead.
- **`crop_type` содержит русскоязычные значения** (зерновые, озимая пшеница, пастбища/зерновые,
  подсолнечник). Влияет на таблицу `polygons` и подписи в UI. Напоминание: `crop_type` —
  обязательная колонка `validate_request`, то есть без неё веб-путь к модели просто не пройдёт.
- Колонка `status` в train — object dtype со значениями вида `active`/`inactive`.
- Их прогон Sprint 3: `uv run pytest -q` → 131 passed. Версии их артефактов — `1.0.0`.

---


### 4.4. Протоколы

**Handoff** (§15, 18 полей) — заполнять полностью: Producer · Consumer · Task ID · Contract ID/version ·
**Status: REVIEW** · Artifact path · Git commit · **Data/model fingerprint** · Breaking change yes/no ·
Input schema · Output schema · Validation command · Test result · Known limitations · **Migration/fallback** ·
Reviewer · Consumer acknowledgement pending/accepted/rejected · **Updated at UTC**.
Принят — когда consumer запустил contract/smoke test и поставил `acknowledgement=accepted`.
Наши слоты в журнале §15: **H-006** Backend → ML (C-13, `NOT_STARTED`), **H-009** Backend/Beginner → Team
(C-15, `NOT_STARTED`) и уже заведённый **H-SH-002** Backend → DL (shared dependencies/lock,
`REVIEW, DL clean-install ack pending`, `backend@350faec`). Входящие, требующие **нашего**
acknowledgement: **H-004** DL → Backend (C-09 `schema 0.1 / algorithm 0.1.1`, `REVIEW`, `DL@44f2f5b`,
проверка `PYTHONPATH=src python -m pytest -q tests/anomalies`) и **H-DL-002** DL → ML/Backend
(`DL@44f2f5b`, 60 тестов, `reports/dl_integration_review.md`). Отдельного handoff-слота под C-07
в §15 нет — есть только колонка «следующий handoff» в dashboard; заводя его, согласуем id с teamlead.

**Порядок изменения shared contract (§6, пять шагов):**
1. владелец создаёт contract decision → 2. contract tests → 3. producer → 4. consumers →
5. удаление старого интерфейса после миграции всех consumers.

**Decision (§17, 10 полей)** заводится при **любом shared change**, а не только breaking:
Date · Proposer · Problem · Options · Evidence · Decision · Consequences · Affected contracts/tasks ·
Reviewers · Status PROPOSED/ACCEPTED/REJECTED/SUPERSEDED. Breaking change дополнительно требует новой
версии и миграции consumers. Действующие решения, обязательные для нас: **D-002** один shared inference
для batch/web · **D-003** разделить raw primary и harmonized · **D-004** durable worker вместо
BackgroundTasks · **D-005** cached demo явно маркируется — все ACCEPTED.

**Blocker (§16, 9 полей):** Reporter/task · Required contract/version · Concrete failure ·
**Why fallback is insufficient** · Impact · **Workaround attempted** · **Required decision/owner** ·
**Due** · Status. BLOCKED разрешён, только если нет HARD-зависимости, fallback и другой доступной задачи.

**Progress update (§18)** — Dashboard обновляется после каждой задачи или минимум раз в 4 часа активной
работы; blocker — сразу. «Нельзя писать только „готово" или „работаю"».

**Уведомления (§5), которые обязаны слать мы:** изменили `ObservationFrame` → уведомить ML
(«возможен train-serving skew»); изменили job states → уведомить Beginner («ломаются smoke/demo checks»);
изменили dependencies → уведомить всю команду. Входящие к нам: ML меняет `DailyFrame` или inference output,
DL меняет `AnomalyEvent`, Beginner меняет fixture.

**Definition of Ready (§19):** указан owner/output/reviewer; известны contract versions; HARD-зависимости
готовы или разрешён mock; acceptance criterion проверяем; путь не конфликтует с ownership.
**Definition of Done (§20):** артефакт по пути, тесты проходят, версия/manifest сохранены, breaking changes
задокументированы, handoff заполнен, reviewer принял, consumers подтвердили, TODO/checkpoint обновлены,
limitations записаны. **Код без теста и handoff — REVIEW, не DONE.**

**Нам запрещено объявлять завершённым (§14):** `Backend real inference` без C-04 · `anomaly MVP` без
C-09 v1.0 · `demo runbook` без фактического E2E и review Backend.

**Red-team checklist (§22)** — наши пункты. Перед CP-3: `ModelStub выключен в production`,
`Submission schema точна`. Перед CP-4: `Batch и worker predictions совпадают`, `Raw/harmonized series
не перепутаны`, `Anomaly explanation не утверждает причинность`, `QA не заменяет missing на ноль`,
`Площадь полигона не считается в градусах`. Перед CP-5: `Clean environment запускается`,
`Секретов нет в git`, `Cached/live результаты различимы`, `Provider/network fallback работает`,
`Другой человек прошёл demo по README`.

---

## 5. ГРАНИЦЫ ВЛАДЕНИЯ

**Наше:** `apps/api/`, `apps/worker/`, `apps/web/`, `src/veg_recovery/providers/`,
`src/veg_recovery/geospatial/`, `src/veg_recovery/service/`, `migrations/`, `infra/`,
`tests/backend/`, `tests/providers/`, `tests/e2e/`, `Dockerfile*`, `docker-compose.yml`,
`.env.example`, `docs/api.md`, `docs/data_provenance.md`, `docs/demo_runbook.md` (**совместно с Beginner**:
`docs/demo*` в §6 координации принадлежит «Backend + Beginner»).
По §6 координации `apps/`, `migrations/`, `infra/` — наши, Beginner их не изменяет; `providers/` наш,
но **ML проверяет shared schema**.

**Чужое (только чтение, изменение — через владельца и contract decision):**
ML — `contracts.py`, `inference.py`, `data/`, `validation/`, `features/`, `models/`, `cli/batch.py`,
`anomalies/baseline.py`, `configs/ml/`, `artifacts/ml/`, `tests/ml/`, `reports/experiments.csv`,
`reports/ml_ablation.md`.
DL — `dl/`, `anomalies/advanced.py|events.py|explain.py`, `configs/dl/`, `tests/dl/`, `tests/anomalies/`, `artifacts/dl/`.
Beginner — `scripts/*.py` (`data_quality_report`, `validate_submission`, `make_test_fixtures`,
`check_experiment_table`), `tests/tools/`, `tests/smoke/`, `tests/fixtures/`,
`reports/data_quality.*`, `docs/data_dictionary.md`, `docs/research_sources.csv`, `docs/demo_checklist.md`.
Teamlead — `00_team_coordination.md`, `docs/0*_*.md`, `docs/ed_part.md`, `docs/BACKEND.md`.

`README.md` — общий, правим только свой раздел после review. `data/*.csv` — неизменяемы.

---

## 6. ТЕХНИЧЕСКИЕ ИНВАРИАНТЫ (нарушение = откат)

1. Растровый fetch внутри HTTP-запроса запрещён; долгие задачи — только durable worker (не BackgroundTasks).
2. Площадь и буферы — в projected CRS; EPSG:4326 только для геометрии на границе API.
3. Cloud/QA-флаги не игнорируются; nodata не усредняется как ноль; хранить `valid_pixel_fraction` и `pixel_count`.
   На уровне полей фрейма — то же правило: «Нельзя заменять реальный missingness нулём без отдельного missing flag»
   (`docs/01_ml_developer.md:500`), для каждого числового источника — `raw`, `cleaned`, `invalid_flag` (`:357`),
   погодные признаки — с availability flags.
4. `primary_ndvi` — сырой конкурсный ряд с иерархией **S2 → Landsat → MODIS**: ML подтвердил правило
   на всех 48 161 видимых target-значениях train+test (`docs/01_ml_developer.md:26`); наши 30 520 — train-подмножество
   (train 30 520 + test 17 641 = 48 161). Третье независимое измерение — data-quality отчёт Разработчика 4:
   `match_rate = 1.0`, `max_abs_mismatch = 0.0` (`docs/demo_checklist.md`, issue-notes). В ТЗ (`docs/case_doc.pdf`) иерархия дословно не записана — это эмпирический
   факт, а не требование кейса. **Target не клипается**: реальный диапазон train до `−2.1303786081`,
   test до `1.8428536898` (`ML@39a8f73:reports/data_contract_issues.md`) — валидировать NDVI диапазоном
   `[-1, 1]` в схемах API и в БД запрещено, нештатное значение помечается флагом, а не отвергается.
   Гармонизированный ряд живёт отдельным полем `ndvi_harmonized`, не является target и **производится
   не нами**: фактически его отдаёт ML прямо в `PredictionRow` (`inference.py` → `anomalies/baseline.py:
   harmonize_values`, плюс диагностика `harmonization_status`), поэтому спор ML↔DL за владение
   для нас закрыт де-факто в пользу ML — мы храним и показываем колонку и не пишем
   `geospatial/harmonize.py`. Формального Decision по ML-014 против раздела SENSOR HARMONIZATION
   у DL по-прежнему нет; вопрос к teamlead остаётся открытым.
5. Схема `submission.csv` (`anon_polygon_id,date,primary_ndvi_pred`, только `is_synthetic_gap=True`) неизменна.
6. Модель не дублируется: batch и web зовут один `NDVIReconstructor`; несовместимая модель →
   `MODEL_SCHEMA_MISMATCH`. **С BE-011R работающая модель — поставка `model/`** (`ndvi.inference.
   predict_queries`, смесь LightGBM + спутниковых экспертов CatBoost, 324 признака); прежний
   bundle C-04 остаётся в дереве принятым handoff, но веб-путь его не зовёт. Признаки строятся
   из всего набора (сезонная норма, синхронные ряды, наблюдения других AOI), поэтому набор и
   смесь грузятся **один раз на старте процесса**, а точка, которой в наборе нет, не
   восстанавливается вовсе — предупреждение `POINTS_OUTSIDE_MODEL_DATASET`, а не выдуманное
   значение. Отказы: нет каталога/запуска → `missing`; не совпал SHA256 CSV (`read_data`
   strict) → `transport`; несовместимая схема признаков → `schema`; pickle без явного
   `COSMO_MODEL_BUNDLE_TRUSTED` → `untrusted`. Проверку хешей не отключаем никогда; наши
   байтово-хешированные артефакты защищаем `.gitattributes` (D-DL-006, у поставки свой есть).
   Bundle загружается **один раз на старте процесса** — это требование producer
   (`artifacts/ml/CONTRACT_CHANGELOG.md`), а не наша догадка. Фактический `load_bundle` отказывает
   не по одному критерию, а по семи: `schema_version != "1.0"`, `feature_version != FEATURE_VERSION`,
   состав файлов не соответствует `bundle_kind`, неверный `format`, symlink на manifest или файл,
   файл вне корня бандла, несовпадение SHA256. Все они у нас → `MODEL_SCHEMA_MISMATCH`, но текст
   ошибки обязан отличать несовместимую схему от повреждённого транспорта (B-DL-003: EOL-переписывание
   роняет именно SHA256). Проверку хешей не отключаем никогда; наши байтово-хешированные артефакты
   защищаем `.gitattributes` (D-DL-006). `bundle_kind == "trained"` требует явного `trusted=True`:
   joblib исполняет код, поэтому доверие — осознанная настройка окружения после провенанс-проверки,
   не значение по умолчанию. Состав manifest для отчёта: `created_at`, `git_commit`, train fingerprints,
   `feature_version`, model files с SHA256, seeds, CV summary, package versions
   (`docs/01_ml_developer.md:227`) плюс **preprocessor hash** (`docs/02_dl_developer.md:174`).
7. Идемпотентность анализа по `geometry_hash + date_range + pipeline_version`; `POST /analyses` → 202 + job_id.
8. FSM: QUEUED → FETCHING → PREPROCESSING → RECONSTRUCTING → ANALYZING → COMPLETED, плюс PARTIAL и FAILED.
9. Провайдеры: таймауты, retry с backoff, rate limit, provenance с collection/version; adapters не знают о FastAPI и БД.
10. Cached demo всегда честно помечен в UI; выдавать кэш за live запрещено.
11. Секреты только в env, `.env.example` — placeholders; ни секретов, ни stack traces в ответах API.
12. Миграции только Alembic; `create_all` в production запрещён.
13. Offline CI без сети и GPU; живые тесты — под маркером `live`.
14. Комментарии к неочевидной логике — на русском (отдельный критерий оценивания).
15. Прямой push в `main`, `reset --hard` и `push --force` — запрещены.

---

## 7. ОКРУЖЕНИЕ

- Рабочая директория: `/root/project/backend`, git-репозиторий на ветке `backend`.
- `gh` авторизован (аккаунт `omazda`, есть push). Репозиторий приватный — `raw.githubusercontent.com` отдаёт 404,
  качать только через `gh api` или git.
- `docs/case_doc.pdf` — скан: текст извлекается через `pdftoppm` + `tesseract -l rus`, не `pdftotext`.
- Данные: `data/train_dataset.csv` (99 955 строк, 21 колонка, 39 полигонов, 30 520 finite `primary_ndvi`),
  `data/test_data.csv` (57 185 строк, 20 колонок, 78 полигонов, 17 641 finite, 3 112 синтетических гэпов).
  Распределение гэпов неравномерно: 39 знакомых и 39 новых полигонов, **464 gap-строки у знакомых и 2 648 у новых**
  (`docs/01_ml_developer.md:23`). Длины серий: 2 827 гэпов длины 1, 136 длины 2, три длины 3, один длины 4;
  у 3 086 из 3 112 точек виден контекст с обеих сторон (`:25`). Фикстуры BE-008 обязаны покрывать этот профиль.
- **Расхождение имён тестового файла:** ТЗ кейса и ТЗ ML называют его `private_features.csv`
  и кладут в `data/raw/`; в репозитории это `data/`+`test_data.csv`, каталога `data/raw/` нет ни в одной ветке.
  Batch-вход обязан принимать имя организаторов. Сами файлы не переименовываем — вопрос к teamlead.
- **Batch-CLI: ТЗ и опубликованный код расходятся.** ТЗ ML (`:121-126`) обещает
  `uv run veg-recovery batch --input <csv> --model artifacts/ml/final_bundle --output submission.csv`.
  Фактически в `ML@39a8f73` это `python -m veg_recovery.cli.batch --input <csv> --bundle <dir>
  --output <csv> --diagnostics <csv> [--trusted-bundle] [--expected-count 3112]`: `--model`
  переименован в `--bundle`, `--diagnostics` **обязателен**, при ошибке возвращается ровно `2`.
  Валидация submission у producer уже реализована (`validate_submission_file`): отвергает BOM,
  не-UTF-8, неверный заголовок, ширину строки ≠ 3, дубли/NaN/inf, bool вместо float и любое
  несовпадение с упорядоченными ключами synthetic-gap; запись атомарная через временный файл.
  Console-scripts `veg-recovery` и `batch` регистрируются в `pyproject.toml` — это наша зона (SH-002),
  и объявлять их можно только после согласования фактических имён флагов с ML.
- Метрика: `GapScore = round(30 * max(0, 1 - RMSE/0.10), 2)`; 100 баллов суммарно по `docs/criteria.pdf`.

---

## 8. КОММИТЫ

- Коммитим только по явной просьбе. Ветка всегда `backend`.
- Сообщение: `<тип>: <что>` в стиле репозитория (`add:`, `fix:`, `docs:`).
- Перед коммитом: тесты затронутой зоны + `graphify . --update`.
- В PR в `main` — заполненный handoff и список breaking changes.

---

## 9. ФОРМАТ ОТЧЁТА ПОСЛЕ ЗАДАЧИ

~~~text
1. Что сделано (по пунктам тикета) + task ID
2. Изменённые файлы
3. Дельта графа: новые узлы/рёбра, выполненная команда update
4. Влияние на соседние сегменты: что проверено
5. Синхронизация: состояние backend/main, расхождения, блокеры
6. Контракты: потреблённые/произведённые, версии, breaking changes
7. Тесты: команды и результат честно, включая падения
8. Демо-путь: работает / деградировал / fallback
9. Совет: кто высказался, было ли вето
10. Риски и следующий шаг (P0/P1/P2)
~~~

---

## 10. ЖУРНАЛ АКТУАЛИЗАЦИИ

Версия промпта: **2.0** · Инструкции зафиксированы в `infra/instructions.lock` (7 веток).

| Дата UTC | Триггер | Что изменилось в инструкциях | Что переписано у нас |
|---|---|---|---|
| 2026-09-04 | старт | — | Созданы `CLAUDE.md` и `used_prompts/backend_dev3_agent.md` (роль Dev 3, личности L1–L9, законы графа и согласования) |
| 2026-09-04 | sync `main` | teamlead добавил `00_team_coordination.md` (853 стр.) и `docs/01_ml_developer.md` | Добавлен §4: статусы, контракты C-01…C-15, очередь BE-001…BE-018, checkpoints CP-0…CP-5, handoff/blocker/decision, DoD; уточнены SOURCE OF TRUTH и §5 |
| 2026-09-04 | запрос пользователя | — | Добавлен §1 (самоактуализация), `infra/sync_instructions.sh`, `infra/instructions.lock`, журнал §10 |
| 2026-09-04 | полное чтение 4 документов `main` (`00_team_coordination.md` 853 стр., `docs/01_ml_developer.md`, `docs/02_dl_developer.md`, `docs/ed_part.md`) + bootstrap графа | Ничего не менялось у teamlead — читали то, что уже лежало в `main` и раньше было усвоено лишь частично | §0: список перечитываемых разделов координации расширен с 7 до 17 (добавлены 9–11, 13, 15–23) — прежний список воспроизводил нашу же дыру. §4 переписан целиком: контракты с путями и владельцами (C-07 → `apps/api/schemas/`, C-08 → `service/orchestrator.py`, C-13 → `providers/base.py`, C-15 — **совместно с Beginner**), дословный C-02 с `gap_mask` и `context_mode`, открытый список `reason_codes`, очередь задач переписана как граф (BE-003 зависит только от BE-001; BE-007 — ни от одной BE), CP-1 с трёхчасовыми обязательствами, полные шаблоны handoff (18 полей) / decision (10) / blocker (9), пятишаговый порядок shared change (первым — contract decision владельца), Decision при **любом** shared change, матрица уведомлений, DoR, red-team checklist, наши review-обязанности (SH-004, C-12, JR-009, `contracts.py`). §5: `docs/demo*` — совместно с Beginner, ML добавлены `tests/ml/` и `reports/ml_ablation.md`. §6: инвариант 3 дополнен требованием отдельного missing flag; инвариант 4 — 48 161 вместо 30 520 и пометка, что в ТЗ иерархия дословно не записана, а `ndvi_harmonized` производим не мы; инвариант 6 — полный состав проверок manifest + preprocessor hash. §7: точный профиль гэпов (464/2 648, длины серий), расхождение имени `private_features.csv` vs `data/test_data.csv`, сигнатура batch-CLI и коды возврата. Промпт: `PredictionExpert` убран из контрактов и из обязательных узлов графа, §7 ML переписан по дословному C-02, `anomalies/baseline.py` возвращён в зону ML, добавлен состав SH-005 с командами Dev 4 и состав C-12 |
| 2026-09-05 | дрейф: `00_team_coordination.md` в `main` (`97a2497` «docs(coordination): start DL adapters and anomaly work») | Статус проекта READY_TO_START → IN_PROGRESS; DL переведён в IN_PROGRESS по DL-001/002/007/008; заведены блокеры **B-DL-002 (owner — Backend: нет `pyproject.toml`/`uv.lock`, не определён extra `dl`)** и B-DL-001 (owner ML); добавлен Progress update DL от 2026-09-05; DL пишет журнал прямо в `main` через worktree | Добавлен §4.5 «Текущее состояние команды»: dashboard ролей, оба блокера с указанием, что B-DL-002 — наш, влияние DL-007/DL-008 на BE-012 и BE-008, требование сверяться по §0 перед каждым запросом из-за второго пишущего в координацию |
| 2026-09-05 | дрейф: `docs/02_dl_developer.md` в ветке `DL` (`a4f2563` «feat(dl): add leakage-safe windows, residual TCN and anomaly events»); движение веток `DL` и `ED` | DL переписал ТЗ под фактическую реализацию: добавлен раздел «Уточнения после проверки репозитория 2026-09-05» (маска до windowing, разделение training/inner/outer, composite CV, `PENDING_EVALUATION` вместо `REJECT` без экспериментов, matched-mask bridge для PyPOTS, **confidence как heuristic support, а не вероятность**, LOYO, запрет выдавать sensor mapping за гармонизацию, CUDA на Kaggle, `PYTHONPATH=src` до нашего SH-002, «общий lock принадлежит Backend»). В ветке `DL` появилась реализация C-09 (`anomalies/events.py`, 9 reason codes, severity из трёх значений, `ALGORITHM_VERSION robust-loyo-events-0.1.0`) и семь anomaly-фикстур DL-008. В ветке `ED` появились C-12 (20 файлов `tests/fixtures/`) и **собственные `pyproject.toml`/`uv.lock`/`.gitignore`/`.python-version`** | §4.5 переписан: подраздел о фактическом C-09 с дословным списком девяти reason codes и трёх severity, требование к BE-013 не выдавать `confidence` за вероятность и не считать «нет событий» подтверждённой нормой, готовые фикстуры DL-008 как вход BE-012/BE-008; зафиксирована **коллизия владения SH-002 с веткой `ED`** (пять конфликтующих файлов) как повод для Decision по §17; описан состав C-12 и наша роль соreviewer |
| 2026-09-05 | новый инструкционный файл `docs/demo_checklist.md` в ветке `ED` (`5fabb1f`) | Разработчик 4 опубликовал стартовый шаблон demo-чек-листа с явной передачей нам: «Финальный чек-лист заполняется Backend-ревьюером». Внутри — незакрытые пункты по миграциям, credentials провайдеров, составу demo cache, трём демо-полигонам, health-эндпоинтам, offline-тайлам, подготовленному анализу, сетевому fallback и основному сценарию; issue-notes с независимым подтверждением иерархии сенсоров (`match_rate=1.0`), расхождением `data/raw/` vs `data/`, русскоязычными значениями `crop_type` и object-dtype колонкой `status` | §4.5 дополнен подразделом о `demo_checklist.md`: перечень адресованных нам TODO как содержание C-15, конвенции путей `artifacts/demo_<date>/logs/` и `docs/screenshots/README.md`, четыре факта из issue-notes; §6 инвариант 4 дополнен третьим независимым измерением иерархии |
| 2026-09-05 | дрейф: `00_team_coordination.md` в `main` (`4f40520`, `7e4381b`), `README.md` в `ML`; ML удалил `docs/01_ml_developer.md` и `used_prompts/r&d.md` из своей ветки; движение `DL` → `44f2f5b`, `ED` → `0162b48` | Teamlead и **DL** (пишет в координацию напрямую) переписали dashboard: «принятая интегрированная реализация — 0 %», CP-0 принят, draft в ветке ≠ CP-1. Контракты переведены в REVIEW с реальными путями: C-01/C-02 `producer 1.0` (`ML@39a8f73`), C-03 `folds_v1/real_test_v1`, C-06 `0.1 draft`, **C-07 наш — `0.1 draft`, REVIEW, DL не проверял API**, C-09 `schema 0.1 / algorithm 0.1.1`. Наш статус — «REVIEW: публикация обнаружена DL; owner acknowledgement pending», 0 % принятых. Заведены **H-SH-002** (наш, REVIEW) и **H-DL-001/H-DL-002**; **H-004** DL → Backend переведён в REVIEW. **B-DL-002 (owner — мы) переформулирован**: файлы есть, нет clean-install и решения по коллизии с `ED`; новый **B-DL-003** (owner ML): SHA256 mismatch бандла из-за EOL. Добавлены D-DL-001…D-DL-007. ML опубликовал рабочий код C-01/C-02/C-03 и загружаемый baseline-бандл; DL — C-09 0.1.1 с `analyze()`, шесть реальных anomaly-кейсов и consumer review с прямым «Backend action» | §0 «Известное состояние» переписано на 2026-09-05 (SHA всех пяти веток, merge-base `e1e3ac1`, правило читать чужой код через `git show`/worktree, предупреждение что ТЗ ML осталось только в `main`). §4 шапка: снято устаревшее «все контракты `0.1 planned`». §4.1 переписана в таблицу с колонкой «Факт», дословный C-02 заменён **опубликованным кодом**; добавлены `PredictionRow` (8 полей), `DiagnosticRow` (16 полей = фактический C-05), `ReconstructionPayload` как обязательная JSON-граница, полный список отказов `validate_request` (включая обязательный `crop_type` и точное совпадение индекса маски), словари `method`/`fallback_reason`; отмечено, что C-02 и C-09 ждут **нашего** acknowledgement. §4.4: журнал handoff приведён к факту (H-SH-002, H-004, H-DL-002 и отсутствие слота под C-07). §4.5 переписан целиком: новый dashboard, подраздел про реальные C-01/C-02/C-03 и baseline-бандл (`baseline_v1/bundle` ≠ C-04), `trusted=True` для trained, оба блокера, D-DL-006/D-DL-007, C-09 0.1.1 с `analyze()`, дословный «Backend action» DL, дефект нашего H-SH-002 («не закоммичено»), `matplotlib` как недостающий пункт нашего `dev` extra. §6: инвариант 4 дополнен запретом валидировать NDVI диапазоном `[-1,1]` (train до −2.13, test до 1.84) и фиксацией, что `ndvi_harmonized` фактически отдаёт ML; инвариант 6 переписан под семь реальных критериев отказа `load_bundle`, загрузку один раз на старте и `trusted=True`. §7: зафиксировано расхождение batch-CLI (`--bundle`, обязательный `--diagnostics`, exit code 2) с ТЗ. Промпт: §7 «От Dev 1 (ML)» переписан под опубликованный код, §7 «От Dev 2 (DL)» — под 0.1.1, `analyze()` и Backend action, факты о данных дополнены |
| 2026-09-05 | пользователь указал на ветку **`models`**, которой не было в нашем списке; `gh api .../branches` подтвердил `models` = `3384de9` | Опубликован **C-04**: `artifacts/ml/ndvi_backend_handoff_v1/` — обученный `p0-catboost-gpu-v1` (`bundle_kind: trained`, `estimators.joblib`, `trusted=True`), побайтовая копия runtime-кода ML, эталоны `expected_submission.csv` (3 112) и `expected_diagnostics.csv` с допуском 1e-10, `verify_handoff.py`, `MANIFEST.sha256`, `requirements-runtime.txt` (включая **`catboost 1.2.10`**), `handoff.json` с composite RMSE 0.097412 и весами 0.309550/0.690450. Новый документ `docs/ml_solution.md` (919 строк). Прямые указания нам: бандл монтировать, а не копировать; сохранять diagnostics и показывать `quality_flags`; не замалчивать, что на temporal CV ансамбль хуже baseline | `infra/sync_instructions.sh`: `REFS` дополнен `origin/models`. §0: добавлена ветка `models` и записан урок — список веток сверять с `gh api`/`git ls-remote`, а не с `git branch -r`, который показывает лишь зафетченное. §1: то же правило вынесено в раздел детектора. §4.1: C-04 переведён из `NOT_STARTED` в **ОПУБЛИКОВАН** с реальным путём в ветке `models`. §4.5: новый подраздел про C-04 — состав пакета, `trusted=True` и условие ML, сверка runtime-копии с `ML@39a8f73` по sha256, отсутствие `catboost` в наших extras как дефект SH-002, дословные требования README_BACKEND и ограничение по temporal CV, `gate = null` и отсутствие clip как третье подтверждение инварианта 4 |
| 2026-09-05 | дрейф: `00_team_coordination.md` в `main` (`d981eb4` «DL runs real CV on ML folds, fixes censoring leak»), `README.md` в `DL`; ветка `DL` → `56c4539` | DL перешёл в IN_PROGRESS: C-06 **0.2** с `SeasonalPrior`, обучение и инференс на реальных folds ML, заведён **H-DL-003** (`DL@8f19c25`, 72 теста, производный C-03). DL починил у себя утечку цензурирования (`D-DL-009`) и снял B-DL-001 воркэраундом `--epoch-policy fixed`. Новый блокер **B-DL-004** (owner ML): в OOF нет `pred_catboost`/`pred_ensemble`, `trained_gpu_v1/` исключён их `.gitignore` — у adoption gate нет знаменателя. Добавлены `D-DL-008…D-DL-011`; **`D-DL-011`** фиксирует, что глобальный blend ML ухудшает CV-C относительно их же baseline. README в ветке DL стал указателем на координацию | §0: обновлены SHA `main` и `DL`. §4.5: добавлен подраздел о состоянии `main@d981eb4` — статус DL, новый блокер B-DL-004, `D-DL-011` как независимое подтверждение нашего требования не подавать модель безусловно лучшей (уже закреплено тестом `test_temporal_transfer_is_worse_than_baseline`), и отдельно зафиксировано, что **наш статус в dashboard устарел**: координация не видит BE-002…BE-011, пока ветка не запушена |
| 2026-09-05 | пользователь передал каталог `model/` — самодостаточную поставку ML/DL с обученной моделью и **новым test** | Поставка: `ndvi/` (324 признака, LightGBM + LGB-nodonor + три спутниковых эксперта CatBoost), `data/train.csv` (побайтно наш `train_dataset.csv`) и `data/test_features.csv` (49 190 строк, **20 полигонов**, 2 323 контрольных пропуска, 2010–2024; старый `test_data.csv` на 78 полигонов не используется), `runs/local/` с обученной смесью и `submission.csv` на 2 323 строки, `PACKAGE_MANIFEST.json` с SHA256 всех файлов. Измеренное качество: development RMSE **0.054305**, audit **0.056145** против **0.091812** у полусуммы соседей; эмпирический интервал ±0.078046 с покрытием 89.25 %; скрытые ответы организаторов недоступны, официальный балл не измерялся | **BE-011R, замена модели.** Новый адаптер `src/veg_recovery/service/ndvi_run.py` реализует C-02 поверх `ndvi.inference.predict_queries`; `build_reconstructor` строит его вместо бандла C-04, прежний загрузчик остался как `load_ml_bundle` для исторического contract-теста. Настройки: `COSMO_MODEL_PACKAGE_PATH`, `COSMO_MODEL_RUN_NAME`, `COSMO_MODEL_DATA_DIR` вместо `COSMO_MODEL_BUNDLE_PATH`; compose монтирует `./model` в `/srv/model:ro`; в `ml` extra добавлен `lightgbm`, в образ — `libgomp1`. Offline-источник и `/reference-polygons` переведены на набор поставки (59 рядов вместо 117). §4.1: C-04 помечен как не потребляемый в горячем пути. §4.2: BE-011 закрыт как BE-011R. §6: инвариант 6 переписан под фактическую модель, её загрузку один раз на старте и четыре причины отказа. В UI заменено ограничение C-04 («перенос на будущий сезон») на реальное ограничение работающей модели («качество измерено локально, официальный балл не измерялся») |
| 2026-09-05 | пользователь заметил, что система работает на одной модели, хотя есть вторая; `gh api .../branches` показал неотслеживаемую ветку **`integration/dl-ml`** | DL собрал дерево ML+DL (`d90358b`, «integrate DL into ML without touching ML-owned code»), внутри — рабочий C-09: `anomalies/advanced.py` (606 строк, только numpy/pandas), `events.py` (схема, 12 полей), `explain.py`. `DL` уехал на `60875b3`. Проверено самостоятельно: diff ML-кода против `ML@39a8f73` пустой, `tests/anomalies` — 18 passed | `infra/sync_instructions.sh`: `REFS` дополнен `origin/integration/dl-ml`. §0: список веток и урок — накладывать чужой `src` через PYTHONPATH нельзя, у ролей `veg_recovery` namespace-пакет, у нас обычный; единственный путь — merge. §4.1: C-09 переведён из «ждут нас» в **ПОДКЛЮЧЁН**. §4.2: BE-012 закрыт как REVIEW с перечнем того, что сознательно не подано в детектор. Заведён `docs/handoffs/H-004-C09-acknowledgement.md` (accepted). Код: `_detect_with_c09` с единой калибровкой reference/query через `SensorHarmonizer` DL, `_selected_source` по сверке значения (S2 36.8 % / Landsat 43.5 % / MODIS 19.7 %, несовпадений ноль), baseline остался объявленным fallback с дословной причиной отката; UI различает три severity и показывает `algorithm_version` и пояснение производителя |
