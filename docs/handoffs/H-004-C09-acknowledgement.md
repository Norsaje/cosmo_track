# H-004 — acknowledgement Backend: C-09 принят и подключён

Заполнено по шаблону §15 `00_team_coordination.md`. Канонический журнал handoff живёт
в координационном файле — он в зоне teamlead, мы его не редактируем. Этот документ
готов к переносу в §15 и в описание PR в `main`.

Закрывает ожидание, стоявшее **на нас**: `C-09 · REVIEW: Backend ack pending`.
До этого веб-путь считал аномалии baseline-детектором ML и честно помечал это
предупреждением `ANOMALY_SOURCE_IS_BASELINE`. Теперь события считает C-09.

---

- **Producer:** DL (Разработчик 2)
- **Consumer:** Backend (Разработчик 3)
- **Task ID:** BE-012 (подключение C-09), BE-013 (различение severity в UI)
- **Contract ID/version:** C-09 `schema 0.1` / `algorithm robust-loyo-events-0.1.1`
- **Status:** **ACCEPTED** консьюмером. Статус самого контракта остаётся за владельцем:
  мы подтверждаем интеграцию, а не объявляем чужую задачу DONE. C-09 остаётся `0.1`,
  поэтому по §14 «anomaly MVP» завершённым мы не объявляем.
- **Artifact path:** `src/veg_recovery/anomalies/{events,advanced,explain}.py`
- **Git commit:** producer — `integration/dl-ml@d90358b` (содержит `DL@60875b3` и `ML@39a8f73`
  целиком; ML-код не тронут — сверено `git diff`, пусто); consumer — мерж `cc54a07` в `backend`
- **Data/model fingerprint:** `SCHEMA_VERSION = "0.1"`,
  `ALGORITHM_VERSION = "robust-loyo-events-0.1.1"`, `REASON_CODES` — 9 кодов,
  `AnomalyConfig` по умолчанию (`min_reference_years = 3`, `suppression_z = -1.0`,
  `critical_z = -2.0`)
- **Breaking change:** нет. Таблица `anomaly_events` уже была спроектирована под 12 полей C-09;
  миграция не потребовалась.
- **Input schema:** кадр на один полигон с `anon_polygon_id`, `date` (календарный день без
  таймзоны), `crop_type`, `ndvi_harmonized`, `is_observed`. Гармонизацию считает
  `SensorHarmonizer` DL, обученный на reference и применённый к обоим кадрам — одна
  калибровка на reference и query, как требует их consumer review.
- **Output schema:** `AnomalyEvent` — 12 полей, переносятся в БД один к одному без
  переименований; `severity` ∈ {`normal`, `biomass_suppression`, `critical`};
  `reason_codes` — открытый список в `jsonb`, без `CHECK` и без `Literal` в схеме API.
- **Validation command:** `PYTHONPATH=src python -m pytest -q tests/anomalies`
- **Test result:** **18 passed**. Плюс наши контрактные тесты
  `tests/backend/test_contract_c09.py` — 6 passed; весь `tests/backend` — 174 passed.
  Сквозной прогон реального анализа в воркере (`AOI-0019`, 2024-04-01…2024-10-30):
  состояние `COMPLETED`, 4 события, `algorithm_version = robust-loyo-events-0.1.1`,
  `severity = biomass_suppression`, коды `SOURCE_SWITCH_RISK`, `PROLONGED_SUPPRESSION`,
  `RAPID_NEGATIVE_CHANGE`, `LOW_DATA_COVERAGE`.

## Known limitations — что мы сознательно не подали в детектор

1. **Восстановленные моделью точки не участвуют в оценке событий.** `SensorHarmonizer`
   переводит значение в шкалу S2 по полю `selected_source`, а у предсказания сенсора нет.
   Объявить его «шкалой S2» было бы неверно: сырой `primary_ndvi` — это S2 лишь на
   **36.8 %** ряда (43.5 % Landsat, 19.7 % MODIS; измерено на `model/data/train.csv`,
   30 520 видимых значений, несовпадений ноль). Вместо подстановки чужого провенанса
   выдаётся предупреждение `RECONSTRUCTED_POINTS_NOT_HARMONIZED`. Детектор при этом
   сам сообщает `MISSING_HARMONIZED_VALUES`.
2. **Погодные reason codes не выставляются.** `LOW_PRECIPITATION`, `HIGH_TEMPERATURE`
   и `LOW_NDWI` требуют колонок `precip_30d_ratio`, `temp_anomaly_c`, `ndwi_robust_z`.
   Пороги DL задал (`< 0.5`, `> 3`, `< -1`), но способ вычисления самих величин — окно,
   норма, робастная оценка — нигде не специфицирован. Придумывать его молча нельзя:
   ошибочный код выглядел бы как утверждение о причине засухи. **Вопрос к DL.**
3. **`MULTISENSOR_CONFIRMATION` не выставляется:** нужен `sensor_negative_count`,
   а его расчёт — это уже логика детектора, переносить её к себе мы не вправе.
4. **`quality` не передаётся.** Продуктовый `confidence` модели не является cloud QA —
   это прямой запрет из consumer review DL, и подменять одно другим мы не стали.
   Детектор в этом случае берёт `quality = 1.0` по умолчанию.
5. **Точность детектора не заявляется.** `D-DL-007`: 91 реальный кандидат, 12
   algorithmic critical, экспертной приёмки нет. UI показывает события и пояснение
   производителя, но не называет их подтверждёнными и не приводит точность.

## Migration / fallback

Baseline-детектор ML остаётся объявленным fallback BE-012 и вызывается, если C-09
отказал по любой причине; причина отката уходит в предупреждение дословно вместе с
`ANOMALY_SOURCE_IS_BASELINE`, а `severity` в этом случае — `baseline_unranked`, потому
что baseline тяжесть не оценивает. Откат проверяется тестом. При истории короче
3 лет события не считаются вовсе, и это отдельное состояние, а не подтверждённая норма.

- **Reviewer:** DL (владелец C-09)
- **Consumer acknowledgement:** **accepted**
- **Updated at UTC:** 2026-09-05
