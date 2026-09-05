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

### Известное состояние на 2026-09-04

- `backend` = `e1e3ac1` — является предком `main`.
- `main` = `6d5e5fa`; в нём есть, а у нас нет: `00_team_coordination.md` (корень),
  `docs/01_ml_developer.md`, `docs/02_dl_developer.md`, `docs/ed_part.md`.
- Живые ветки: `main`, `backend`, `ML`, `DL`, `ED`.
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
`.claude/*.md` — в ветках `main`, `backend`, `ML`, `DL`, `ED`. Ветки ролей отслеживаются намеренно:
ТЗ появляется там раньше, чем в `main`, и это ранний сигнал о будущем breaking change.
Состояние зафиксировано в `infra/instructions.lock` (`<ref> <path> <blob-sha>`).

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
смена типа — несовместимы. Все контракты сейчас `0.1 planned` / `NOT_STARTED`.

### 4.1. Контракты: что производим и что потребляем

| ID | Артефакт | Владелец | Путь по координации | Тип | Наша роль |
|---|---|---|---|---|---|
| C-07 | API schemas и job states | **мы** | `apps/api/schemas/` | HARD | производим; потребители — фронт и smoke-тесты Beginner |
| C-08 | Shared model integration in worker | **мы** | `service/orchestrator.py` | HARD | производим |
| C-13 | Provider-normalized ObservationFrame | **мы** | `providers/base.py` | HARD для live web | производим; **reviewer — ML** |
| C-15 | Demo runbook/checklist | **мы + Beginner** | `docs/demo_runbook.md` | HARD для CP-5 | производим **совместно**, не единолично |
| C-01 | DataContract и canonical `DailyFrame` | ML | `src/veg_recovery/contracts.py` | HARD | потребляем; **изменение только после review DL и Backend** |
| C-02 | `ReconstructionRequest/Result`, `NDVIReconstructor` | ML | `src/veg_recovery/inference.py` | HARD | потребляем |
| C-04 | Final ML bundle + manifest/hash | ML | `artifacts/ml/final_bundle/` | HARD | потребляем |
| C-05 | ML diagnostics schema | ML | `contracts.py` | SOFT | потребляем; fallback — пустые optional diagnostics |
| C-09 | AnomalyEvent schema/detector | DL | `src/veg_recovery/anomalies/` | HARD для CP-4 | потребляем; public contract согласуется с нами |
| C-12 | Tiny fixtures + manifest | Beginner | `tests/fixtures/` | SOFT | потребляем; **мы соreviewer вместе с ML** |
| C-11 | Independent submission validator | Beginner | `scripts/validate_submission.py` | REVIEW | входит в наш demo-preflight |

C-02 дословно (`docs/01_ml_developer.md:209-222`) — **ModelStub обязан повторять именно это**:

~~~python
@dataclass(frozen=True)
class ReconstructionRequest:
    frame: pd.DataFrame
    gap_mask: pd.Series
    context_mode: Literal["competition", "web"]


@dataclass(frozen=True)
class ReconstructionResult:
    predictions: pd.DataFrame  # ключи, primary_ndvi_pred, lower, upper, method
    diagnostics: pd.DataFrame  # source probabilities, distances, disagreement, fallback_reason
    model_version: str


class NDVIReconstructor(Protocol):
    def predict(self, request: ReconstructionRequest) -> ReconstructionResult: ...
~~~

Веб-путь всегда шлёт `context_mode="web"`. **Единственный протокол предсказателя — `NDVIReconstructor`.**
`PredictionExpert` в ТЗ ML отсутствует; он назван один раз в `docs/02_dl_developer.md:176` как формулировка DL
и владельца не имеет — как контракт не используем, вопрос к teamlead открыт.

До C-02/C-04 работаем с **ModelStub** того же интерфейса. Содержимое заглушки — официальный ориентир
ТЗ ML: `mean two neighbors` (`docs/01_ml_developer.md:302`). Никакой fallback не имеет права вернуть NaN
(`:315`), ML-логику в API переписывать нельзя (`:501`). ModelStub обязан быть выключен в production.

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
| BE-011 Подключить C-04 bundle | C-02, C-04, BE-008 | acceptance: batch/worker parity |
| BE-012 Подключить C-09 AnomalyEvent | C-09 draft для UI, 1.0 для DONE | fallback: baseline/mock events |
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

### 4.5. Текущее состояние команды (`main` = `97a2497`, `DL` = `a4f2563`, `ED` = `5fabb1f`, 2026-09-05)

Статус проекта в координации переведён в **IN_PROGRESS**. Dashboard:

| Роль | Статус | Текущая задача | Блокер |
|---|---|---|---|
| ML | READY | ML-001 DataContract | нет |
| DL | **IN_PROGRESS** | DL-001/002 adapter и leakage tests; DL-007/008 anomaly contract/fixtures | C-01/C-03 и ML-008 не опубликованы |
| Backend (мы) | READY | BE-001 skeleton | ModelStub разрешён |
| Beginner | READY | JR-001 Data-quality report | нет |

**На нас заведён блокер — B-DL-002, owner: Backend:**
`pyproject.toml/uv.lock отсутствуют; extra dl ещё не определён`, due «до clean-install gate».
DL прямо пишет: «SH-002 принадлежит Backend: shared dependencies/lock не изменяем; подготовим
точный handoff». То есть менять `pyproject.toml` за нас никто не будет, а их dependency proposal
придёт к нам. Закрытие блокера — часть BE-001/SH-002, и по нему нужен handoff в сторону DL.

Второй блокер, B-DL-001 (owner ML), нас не касается напрямую, но объясняет, почему C-09 придёт
раньше сравнимых DL-метрик.

Что это меняет в нашем плане:

- **DL-007 «C-09 AnomalyEvent draft» уже IN_PROGRESS** и помечен `Blocks: Backend anomaly UI`.
  Draft появится раньше, чем мы дойдём до BE-012 — следим за `docs/02_dl_developer.md` и
  `src/veg_recovery/anomalies/`, схему в нашем API правим только через contract decision.
- **DL-008 synthetic anomaly fixtures IN_PROGRESS**, случаи: negative pulse, source switch, outlier,
  wide uncertainty. Это готовый fallback для BE-012 и материал для BE-008, свои аномальные
  фикстуры дублировать не нужно.
- DL ведёт журнал координации **напрямую в `main` через отдельный worktree**: изменения
  `00_team_coordination.md` теперь приходят не только от teamlead, и sync по §0 обязателен
  перед каждым запросом буквально.

#### C-09 фактически появился в ветке `DL` (не в `main`)

`origin/DL` = `a4f2563` содержит рабочую реализацию: `src/veg_recovery/anomalies/events.py`
(`SCHEMA_VERSION = "0.1"`, `ALGORITHM_VERSION = "robust-loyo-events-0.1.0"`), `advanced.py`,
`explain.py`, плюс `configs/dl/tcn.yaml`, `reports/dl_decision.md`, `reports/dl_experiments.csv`
и `configs/dl/c03_consumer.md`. Дословный состав, который обязана принимать наша схема:

- 12 полей ровно как в §7 промпта; `algorithm_version` имеет значение по умолчанию.
- `severity` — **закрытое множество из трёх значений**: `normal`, `biomass_suppression`, `critical`.
  Наш `AnomalyOut.severity` остаётся `str`, но UI обязан различать ровно эти три.
- `reason_codes` — у производителя `frozenset` из **девяти** кодов, и конструктор
  **бросает `ValueError`** на неизвестном коде: `LOW_PRECIPITATION`, `HIGH_TEMPERATURE`,
  `LOW_NDWI`, `MULTISENSOR_CONFIRMATION`, `SOURCE_SWITCH_RISK`, `LOW_DATA_COVERAGE`,
  `RAPID_NEGATIVE_CHANGE`, `PROLONGED_SUPPRESSION`, `PHENOLOGY_SHIFT`.
  Это ограничение **производителя**, а не контракта: C-09 всё ещё `0.1`, список может вырасти.
  Наш запрет на `Literal`/`Enum` и `CHECK` по `reason_codes` остаётся в силе — иначе
  расширение списка у DL уронит наш ответ.
- Валидация у производителя также требует: событие имеет хотя бы одну опорную точку,
  все числа конечны, `0 <= confidence <= 1`, `score >= 0`, `negative_area >= 0`.

**Семантика, обязательная для BE-013** (`docs/02_dl_developer.md`, версия 2026-09-05):
«В C-09 confidence — heuristic support, не калиброванная вероятность anomaly. Недостаточная
история возвращает diagnostic warning; отсутствие события при недостатке данных нельзя
интерпретировать как подтверждённую норму». Значит в UI: `confidence` не называем вероятностью
и не рисуем как процент уверенности; состояние «аномалий нет при недостатке истории» — отдельное
состояние с предупреждением, а не зелёная норма.

**DL-008 фикстуры готовы** — `reports/anomaly_cases/synthetic_v1/`: семь кейсов
(`normal`, `mild_pulse`, `medium_pulse`, `strong_pulse`, `single_outlier`, `source_switch`,
`wide_uncertainty`), каждый с `.csv`, `.json` и `.png`, плюс `summary.json` и `reference.csv`.
Это готовый вход для BE-012 и материал для BE-008 — своих аномальных фикстур не делаем.

Прочее из обновлённого ТЗ DL, что касается нас: полный CUDA-прогон уносится на Kaggle
(`configs/dl/kaggle/`), «никаких pretrained downloads в runtime»; «unsupported sensor mapping
не выдаётся за гармонизацию»; до нашего handoff DL работает через `PYTHONPATH=src python ...`,
и они прямо пишут: «Общий lock принадлежит Backend».

#### Коллизия владения: `pyproject.toml` и `uv.lock` появились в ветке `ED`

`origin/ED` = `5fabb1f` добавляет в корень **`pyproject.toml`, `uv.lock`, `.python-version`,
`.gitignore`** и `tests/__init__.py` — всё это артефакты SH-002 и наши по §9 и §6 координации.
Их редакция несовместима с нашей: `[dependency-groups] dev` (PEP 735) вместо
`[project.optional-dependencies]`, `package = false`, `dependencies = []`, ни одного из шести
extras `core/ml/dl/geo/web/dev`, `testpaths` только на их каталоги. При мердже обеих веток в
`main` конфликтуют минимум пять файлов: `pyproject.toml`, `uv.lock`, `.gitignore`,
`.python-version`, `tests/__init__.py`.

Чужую ветку не трогаем и молча не «чиним». Это повод для **Decision по §17** с участием
teamlead: чей `pyproject.toml` остаётся каноническим. Наша позиция подкреплена координацией
(«SH-002 · Owner: Backend») и самим DL («Общий lock принадлежит Backend»), а наш handoff
`docs/handoffs/H-SH-002.md` уже опубликован и проверен фактической установкой `--extra dl`.

#### C-12 фикстуры опубликованы в `ED`

`tests/fixtures/` содержит 20 файлов: `train_tiny.csv` (60 строк, 21 колонка, реальный
`AOI-0002`), `test_tiny.csv` (125 строк, 20 колонок, 5 гэпов), `submission_valid_tiny.csv`,
`submission_for_test_tiny.csv`, `submission_valid_bom.csv` и одиннадцать `submission_invalid_*`,
плюс `manifest.json`. Синтетика помечена префиксом `TOY-` (`TOY-RUN-A`, `TOY-CTX-A`).
В манифесте стоит `design_assumptions_requires_ml_confirmation: true` — допущения ещё
не подтверждены ML. **Мы соreviewer C-12 вместе с ML**: проверяем репрезентативность
(single gap, gap run 2–4, one-sided context, unseen polygon) и отсутствие выдуманного
ground truth. Геометрии в фикстурах нет — полигон для E2E готовим сами в `tests/e2e/`.

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
индекс скриншотов — `docs/screenshots/README.md`.

Полезные факты из их issue-notes:

- **Иерархия сенсоров измерена независимо**: их data-quality отчёт даёт `match_rate = 1.0`,
  `max_abs_mismatch = 0.0` для восстановления `primary_ndvi` из S2 → Landsat → MODIS. Это третье
  независимое подтверждение инварианта 4 (после наших 30 520 и 48 161 у ML).
- **Расхождение `data/raw/` против `data/` подтверждено ими же** и вынесено как решение
  «либо переименовать `data/` → `data/raw/`, либо обновить `ed_part.md`». Их скрипты уже
  используют реальный `data/`. Для нас `data/*.csv` неизменяемы — решение за teamlead.
- **`crop_type` содержит русскоязычные значения** (зерновые, озимая пшеница, пастбища/зерновые,
  подсолнечник). Это влияет на нашу таблицу `polygons` и на подписи в UI.
- Колонка `status` в train — object dtype со значениями вида `active`/`inactive`.
- Их прогон Sprint 3: `uv run pytest -q` → 131 passed. Версии всех их артефактов — `1.0.0`
  (`report_schema_version`, три `script_version`, `manifest_schema_version`).

---

### 4.4. Протоколы

**Handoff** (§15, 18 полей) — заполнять полностью: Producer · Consumer · Task ID · Contract ID/version ·
**Status: REVIEW** · Artifact path · Git commit · **Data/model fingerprint** · Breaking change yes/no ·
Input schema · Output schema · Validation command · Test result · Known limitations · **Migration/fallback** ·
Reviewer · Consumer acknowledgement pending/accepted/rejected · **Updated at UTC**.
Принят — когда consumer запустил contract/smoke test и поставил `acknowledgement=accepted`.
Наши handoff в журнале: **H-006** Backend → ML (C-13), **H-009** Backend/Beginner → Team (C-15).

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
   факт, а не требование кейса. Гармонизированный ряд живёт отдельным полем `ndvi_harmonized`, не является target
   и **производится не нами**: ML-014 + `docs/01_ml_developer.md:409` против раздела SENSOR HARMONIZATION у DL —
   конфликт владения ML↔DL, вопрос к teamlead открыт; сами `geospatial/harmonize.py` не пишем до ответа.
5. Схема `submission.csv` (`anon_polygon_id,date,primary_ndvi_pred`, только `is_synthetic_gap=True`) неизменна.
6. Модель не дублируется: batch и web зовут один `NDVIReconstructor`; несовместимый bundle → `MODEL_SCHEMA_MISMATCH`.
   На старте worker проверяем manifest целиком: `schema_version` (единственный критерий отказа), `created_at`,
   `git_commit`, train fingerprints, `feature_version`, model files с SHA256, seeds, CV summary, package versions
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
- Batch-CLI из ТЗ ML (`:121-126`): `uv run veg-recovery batch --input <csv> --model artifacts/ml/final_bundle
  --output submission.csv`; exit code != 0 при несовпадении ключей, NaN, inf, duplicate, лишней колонке,
  неверной кодировке или нечисловом prediction (`:440`); submission сохраняется в исходном порядке gap-строк (`:436`).
  Console-scripts `veg-recovery` и `batch` регистрируются в `pyproject.toml` — это наша зона (SH-002).
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

Версия промпта: **1.5** · Инструкции зафиксированы в `infra/instructions.lock` (28 записей, 5 веток).

| Дата UTC | Триггер | Что изменилось в инструкциях | Что переписано у нас |
|---|---|---|---|
| 2026-09-04 | старт | — | Созданы `CLAUDE.md` и `used_prompts/backend_dev3_agent.md` (роль Dev 3, личности L1–L9, законы графа и согласования) |
| 2026-09-04 | sync `main` | teamlead добавил `00_team_coordination.md` (853 стр.) и `docs/01_ml_developer.md` | Добавлен §4: статусы, контракты C-01…C-15, очередь BE-001…BE-018, checkpoints CP-0…CP-5, handoff/blocker/decision, DoD; уточнены SOURCE OF TRUTH и §5 |
| 2026-09-04 | запрос пользователя | — | Добавлен §1 (самоактуализация), `infra/sync_instructions.sh`, `infra/instructions.lock`, журнал §10 |
| 2026-09-04 | полное чтение 4 документов `main` (`00_team_coordination.md` 853 стр., `docs/01_ml_developer.md`, `docs/02_dl_developer.md`, `docs/ed_part.md`) + bootstrap графа | Ничего не менялось у teamlead — читали то, что уже лежало в `main` и раньше было усвоено лишь частично | §0: список перечитываемых разделов координации расширен с 7 до 17 (добавлены 9–11, 13, 15–23) — прежний список воспроизводил нашу же дыру. §4 переписан целиком: контракты с путями и владельцами (C-07 → `apps/api/schemas/`, C-08 → `service/orchestrator.py`, C-13 → `providers/base.py`, C-15 — **совместно с Beginner**), дословный C-02 с `gap_mask` и `context_mode`, открытый список `reason_codes`, очередь задач переписана как граф (BE-003 зависит только от BE-001; BE-007 — ни от одной BE), CP-1 с трёхчасовыми обязательствами, полные шаблоны handoff (18 полей) / decision (10) / blocker (9), пятишаговый порядок shared change (первым — contract decision владельца), Decision при **любом** shared change, матрица уведомлений, DoR, red-team checklist, наши review-обязанности (SH-004, C-12, JR-009, `contracts.py`). §5: `docs/demo*` — совместно с Beginner, ML добавлены `tests/ml/` и `reports/ml_ablation.md`. §6: инвариант 3 дополнен требованием отдельного missing flag; инвариант 4 — 48 161 вместо 30 520 и пометка, что в ТЗ иерархия дословно не записана, а `ndvi_harmonized` производим не мы; инвариант 6 — полный состав проверок manifest + preprocessor hash. §7: точный профиль гэпов (464/2 648, длины серий), расхождение имени `private_features.csv` vs `data/test_data.csv`, сигнатура batch-CLI и коды возврата. Промпт: `PredictionExpert` убран из контрактов и из обязательных узлов графа, §7 ML переписан по дословному C-02, `anomalies/baseline.py` возвращён в зону ML, добавлен состав SH-005 с командами Dev 4 и состав C-12 |
| 2026-09-05 | дрейф: `00_team_coordination.md` в `main` (`97a2497` «docs(coordination): start DL adapters and anomaly work») | Статус проекта READY_TO_START → IN_PROGRESS; DL переведён в IN_PROGRESS по DL-001/002/007/008; заведены блокеры **B-DL-002 (owner — Backend: нет `pyproject.toml`/`uv.lock`, не определён extra `dl`)** и B-DL-001 (owner ML); добавлен Progress update DL от 2026-09-05; DL пишет журнал прямо в `main` через worktree | Добавлен §4.5 «Текущее состояние команды»: dashboard ролей, оба блокера с указанием, что B-DL-002 — наш, влияние DL-007/DL-008 на BE-012 и BE-008, требование сверяться по §0 перед каждым запросом из-за второго пишущего в координацию |
| 2026-09-05 | дрейф: `docs/02_dl_developer.md` в ветке `DL` (`a4f2563` «feat(dl): add leakage-safe windows, residual TCN and anomaly events»); движение веток `DL` и `ED` | DL переписал ТЗ под фактическую реализацию: добавлен раздел «Уточнения после проверки репозитория 2026-09-05» (маска до windowing, разделение training/inner/outer, composite CV, `PENDING_EVALUATION` вместо `REJECT` без экспериментов, matched-mask bridge для PyPOTS, **confidence как heuristic support, а не вероятность**, LOYO, запрет выдавать sensor mapping за гармонизацию, CUDA на Kaggle, `PYTHONPATH=src` до нашего SH-002, «общий lock принадлежит Backend»). В ветке `DL` появилась реализация C-09 (`anomalies/events.py`, 9 reason codes, severity из трёх значений, `ALGORITHM_VERSION robust-loyo-events-0.1.0`) и семь anomaly-фикстур DL-008. В ветке `ED` появились C-12 (20 файлов `tests/fixtures/`) и **собственные `pyproject.toml`/`uv.lock`/`.gitignore`/`.python-version`** | §4.5 переписан: подраздел о фактическом C-09 с дословным списком девяти reason codes и трёх severity, требование к BE-013 не выдавать `confidence` за вероятность и не считать «нет событий» подтверждённой нормой, готовые фикстуры DL-008 как вход BE-012/BE-008; зафиксирована **коллизия владения SH-002 с веткой `ED`** (пять конфликтующих файлов) как повод для Decision по §17; описан состав C-12 и наша роль соreviewer |
| 2026-09-05 | новый инструкционный файл `docs/demo_checklist.md` в ветке `ED` (`5fabb1f`) | Разработчик 4 опубликовал стартовый шаблон demo-чек-листа с явной передачей нам: «Финальный чек-лист заполняется Backend-ревьюером». Внутри — незакрытые пункты по миграциям, credentials провайдеров, составу demo cache, трём демо-полигонам, health-эндпоинтам, offline-тайлам, подготовленному анализу, сетевому fallback и основному сценарию; issue-notes с независимым подтверждением иерархии сенсоров (`match_rate=1.0`), расхождением `data/raw/` vs `data/`, русскоязычными значениями `crop_type` и object-dtype колонкой `status` | §4.5 дополнен подразделом о `demo_checklist.md`: перечень адресованных нам TODO как содержание C-15, конвенции путей `artifacts/demo_<date>/logs/` и `docs/screenshots/README.md`, четыре факта из issue-notes; §6 инвариант 4 дополнен третьим независимым измерением иерархии |
