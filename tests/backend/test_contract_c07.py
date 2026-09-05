"""Contract-тесты C-07 v0.1.

Проверяют то, на что опираются потребители контракта: фронтенд и offline
smoke-тесты Разработчика 4. Изменение состава `JobState` без уведомления
Разработчика 4 ломает его проверки (матрица уведомлений, §5 координации).

Принцип: тест обязан падать при поломке поведения. Проверка «ключи словаря на
месте» этого не даёт — таблица переходов сравнивается с эталоном целиком.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="нужен extra `web`")

from apps.api.schemas import (
    ALLOWED_TRANSITIONS,
    CONTRACT_ID,
    CONTRACT_VERSION,
    TERMINAL_STATES,
    JobState,
)

#: Состав FSM зафиксирован инвариантом 8 и §8.5 ТЗ.
EXPECTED_STATES = {
    "QUEUED",
    "FETCHING",
    "PREPROCESSING",
    "RECONSTRUCTING",
    "ANALYZING",
    "COMPLETED",
    "PARTIAL",
    "FAILED",
}

#: Эталон переходов продублирован здесь намеренно: без него любая перестановка
#: рёбер (например, QUEUED → COMPLETED в обход реконструкции) проходит незамеченной.
EXPECTED_TRANSITIONS = {
    JobState.QUEUED: {JobState.FETCHING, JobState.FAILED},
    JobState.FETCHING: {JobState.PREPROCESSING, JobState.PARTIAL, JobState.FAILED},
    JobState.PARTIAL: {JobState.PREPROCESSING, JobState.FAILED},
    JobState.PREPROCESSING: {JobState.RECONSTRUCTING, JobState.FAILED},
    JobState.RECONSTRUCTING: {JobState.ANALYZING, JobState.FAILED},
    JobState.ANALYZING: {JobState.COMPLETED, JobState.FAILED},
    JobState.COMPLETED: set(),
    JobState.FAILED: set(),
}

#: Все 15 путей контракта. Список полный, а не выборочный: удаление роута обязано
#: валить тест, иначе `docs/api.md` и код разъезжаются молча.
EXPECTED_PATHS = {
    "/api/v1/auth/email-code",
    "/api/v1/auth/session",
    "/api/v1/auth/me",
    "/health/live",
    "/health/ready",
    "/health/providers",
    "/api/v1/polygons",
    "/api/v1/polygons/{polygon_id}",
    "/api/v1/field-search",
    # Временный роут на время отсутствия живых провайдеров (BE-007/BE-009):
    # перечень рядов, по которым offline-источник может дать данные. Он в списке
    # намеренно — набор путей C-07 фиксируется явно, и появление роута мимо этого
    # списка означало бы, что контракт расширили молча.
    "/api/v1/reference-polygons",
    "/api/v1/analyses",
    "/api/v1/analyses/{analysis_id}",
    "/api/v1/analyses/{analysis_id}/series",
    "/api/v1/analyses/{analysis_id}/anomalies",
    "/api/v1/analyses/{analysis_id}/provenance",
    "/api/v1/analyses/{analysis_id}/export.csv",
    "/api/v1/jobs/{job_id}",
}


@pytest.fixture
def schema(client):
    return client.get("/openapi.json").json()


def test_contract_version_is_declared() -> None:
    assert CONTRACT_ID == "C-07"
    assert CONTRACT_VERSION == "0.1"


def test_job_states_match_the_specification() -> None:
    assert {state.value for state in JobState} == EXPECTED_STATES


def test_transition_table_matches_the_specification() -> None:
    """Сравнение целиком: состав целевых множеств — и есть содержание инварианта 8."""
    assert {k: set(v) for k, v in ALLOWED_TRANSITIONS.items()} == EXPECTED_TRANSITIONS


def test_completed_is_reachable_only_through_analyzing() -> None:
    predecessors = {
        s for s, targets in ALLOWED_TRANSITIONS.items() if JobState.COMPLETED in targets
    }
    assert predecessors == {JobState.ANALYZING}


def test_every_working_stage_is_on_the_main_path() -> None:
    """Каждая рабочая стадия достижима из QUEUED — недостижимых стадий нет."""
    seen, frontier = {JobState.QUEUED}, [JobState.QUEUED]
    while frontier:
        for nxt in ALLOWED_TRANSITIONS[frontier.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert seen == set(JobState)


def test_terminal_states_have_no_outgoing_transitions() -> None:
    assert TERMINAL_STATES == {JobState.COMPLETED, JobState.FAILED}
    for state in TERMINAL_STATES:
        assert ALLOWED_TRANSITIONS[state] == frozenset()


def test_failed_is_reachable_from_every_working_stage() -> None:
    for state in set(JobState) - TERMINAL_STATES:
        assert JobState.FAILED in ALLOWED_TRANSITIONS[state], state


def test_partial_is_not_terminal() -> None:
    assert JobState.PARTIAL not in TERMINAL_STATES
    assert JobState.PREPROCESSING in ALLOWED_TRANSITIONS[JobState.PARTIAL]


def test_job_status_can_express_partial_completion(schema) -> None:
    """Фронт опрашивает только /jobs/{id}; без флага «завершено» и «завершено
    на неполных данных» неразличимы."""
    assert "partial" in schema["components"]["schemas"]["JobStatus"]["properties"]


def test_create_analysis_is_asynchronous(schema) -> None:
    """`POST /analyses` объявлен как 202 и отдаёт job_id: держать HTTP-запрос
    открытым во время спутникового анализа запрещено (инвариант 1)."""
    responses = schema["paths"]["/api/v1/analyses"]["post"]["responses"]
    assert "202" in responses and "200" not in responses
    accepted = schema["components"]["schemas"]["AnalysisAccepted"]["properties"]
    assert {"job_id", "analysis_id"} <= set(accepted)


def test_openapi_exposes_every_documented_path(schema) -> None:
    assert set(schema["paths"]) == EXPECTED_PATHS


def test_reason_codes_stay_an_open_list(schema) -> None:
    """Список открытый: ТЗ DL нигде не объявляет девять кодов исчерпывающими,
    C-09 ещё 0.1. Сравнение целиком, а не `"enum" not in items`: для Enum
    pydantic отдаёт `$ref`, и проверка на отсутствие ключа `enum` его пропускала.
    """
    items = schema["components"]["schemas"]["AnomalyOut"]["properties"]["reason_codes"]["items"]
    assert items == {"type": "string"}


def test_anomaly_warnings_are_separate_from_absence_of_events(schema) -> None:
    """C-09 несёт warnings отдельно от событий: пустой items при EMPTY_SERIES
    означает «данных не было», а не «поле в норме»."""
    props = schema["components"]["schemas"]["AnomaliesResponse"]["properties"]
    expected = {"warnings", "confidence_semantics", "schema_version", "algorithm_version"}
    assert expected <= set(props)
    assert props["warnings"]["items"] == {"type": "string"}


def test_anomaly_out_carries_every_c09_field(schema) -> None:
    props = set(schema["components"]["schemas"]["AnomalyOut"]["properties"])
    assert {
        "start_date",
        "end_date",
        "severity",
        "score",
        "confidence",
        "min_robust_z",
        "negative_area",
        "observed_points",
        "reconstructed_points",
        "reason_codes",
        "explanation_ru",
        "algorithm_version",
        "duration_days",
        "schema_version",
    } <= props


def test_series_point_keeps_raw_and_harmonized_apart(schema) -> None:
    """Решение D-003: сырой конкурсный ряд и продуктовый гармонизированный —
    разные поля, перепутать их нельзя."""
    props = set(schema["components"]["schemas"]["SeriesPoint"]["properties"])
    assert {"primary_ndvi", "primary_ndvi_reconstructed", "ndvi_harmonized"} <= props


def test_series_point_carries_quality_and_context(schema) -> None:
    """Инвариант 3 требует и долю, и число пикселей; §8.9 — QA, климатологию
    и погодный контекст в tooltip и панели аномалии."""
    props = set(schema["components"]["schemas"]["SeriesPoint"]["properties"])
    assert {"valid_pixel_fraction", "pixel_count", "qa_flags"} <= props
    assert {"ndvi_climatology_mean", "ndvi_climatology_std"} <= props
    assert {"temp_c", "precip_mm", "evi", "ndwi"} <= props


def test_cache_is_marked_where_the_data_is_read(schema) -> None:
    """Инвариант 10: метка кэша обязана быть там, где фронт берёт результат,
    а не только в /provenance — иначе её забудут показать."""
    for model in ("SeriesResponse", "AnalysisOut"):
        assert "cached" in schema["components"]["schemas"][model]["properties"], model


def test_field_search_accepts_point(client) -> None:
    """§8.4 требует принимать bbox/point: поиск по клику на карте — основной сценарий."""
    assert (
        client.post("/api/v1/field-search", json={"point": [10, 20], "limit": 5}).status_code == 501
    )
    assert client.post("/api/v1/field-search", json={"limit": 5}).status_code == 422


def test_analysis_rejects_reversed_and_huge_date_range(authed_client) -> None:
    base = {"polygon_id": "p", "date_from": "2024-05-01", "date_to": "2024-01-01"}
    assert authed_client.post("/api/v1/analyses", json=base).status_code == 422
    huge = {"polygon_id": "p", "date_from": "1990-01-01", "date_to": "2024-01-01"}
    assert authed_client.post("/api/v1/analyses", json=huge).status_code == 422


def test_unknown_body_field_is_rejected(authed_client) -> None:
    """Опечатка фронта обязана давать 422, а не молча анализировать другой период."""
    body = {"polygon_id": "p", "date_from": "2024-01-01", "date_to": "2024-02-01", "dateFrom": "x"}
    assert authed_client.post("/api/v1/analyses", json=body).status_code == 422


def test_bbox_is_validated(authed_client) -> None:
    assert authed_client.get("/api/v1/polygons?bbox=nonsense").status_code == 422
    assert authed_client.get("/api/v1/polygons?bbox=1,2,3").status_code == 422
    assert authed_client.get("/api/v1/polygons?bbox=200,2,3,4").status_code == 422
    assert authed_client.get("/api/v1/polygons?bbox=1,2,3,4").status_code == 200


def test_every_error_uses_the_contract_envelope(client, schema) -> None:
    """Единый формат ошибки — часть C-07. Две несовместимые формы (`error_code`
    на 500 и `detail` на остальном) ломают потребителей контракта."""
    assert "ErrorResponse" in schema["components"]["schemas"]
    cases = [
        client.get("/nope"),
        client.get("/api/v1/jobs/abc"),
        client.post("/api/v1/analyses", json={"polygon_id": 1}),
        client.get("/api/v1/polygons?bbox=nonsense"),
    ]
    for response in cases:
        body = response.json()
        assert set(body) == {"error_code", "message", "request_id"}, response.url
        assert body["request_id"] == response.headers["x-request-id"]
        assert "Traceback" not in body["message"]


def test_export_csv_is_not_declared_as_json(schema) -> None:
    csv_path = schema["paths"]["/api/v1/analyses/{analysis_id}/export.csv"]
    content = csv_path["get"]["responses"]["200"]
    assert "text/csv" in content["content"]


def test_polygon_list_envelope_is_consistent(authed_client) -> None:
    """Список полигонов — всегда конверт `{items, total}`, и `total` согласован.

    Прежняя редакция требовала буквально пустого списка: это описывало заглушку
    BE-001, а не контракт. После BE-004 роут ходит в БД, и содержимое зависит от
    состояния стенда — но форма ответа и согласованность `total` обязаны держаться
    при любом содержимом, включая пустое.
    """
    response = authed_client.get("/api/v1/polygons")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"items", "total"}
    assert isinstance(body["items"], list)
    assert body["total"] >= len(body["items"])


def test_polygon_with_an_analysis_is_actually_deleted(authed_client) -> None:
    """Удаление поля с анализами обязано доходить до базы, а не только до ответа.

    Каскад объявлен в схеме, но ORM про него не знает и сначала обнуляла
    `analyses.polygon_id` — колонку NOT NULL. Транзакция падала на коммите, уже
    после отправленного 204: интерфейс показывал «поле удалено», а после
    обновления страницы оно возвращалось. Проверяется именно исчезновение строки,
    а не код ответа: код был правильным и тогда.

    Анализ создаётся прямо в базе, а не через `POST /analyses`: роут ставит
    задачу в Celery, и без брокера тест ждал бы его таймаута минутами, проверяя
    при этом совсем не то.
    """
    from datetime import date

    from apps.db.base import session_scope
    from apps.db.models import Analysis, AnalysisJob

    created = authed_client.post(
        "/api/v1/polygons",
        json={
            "name": "Поле с анализом",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[39.5, 45.5], [39.52, 45.5], [39.52, 45.52], [39.5, 45.52], [39.5, 45.5]]
                ],
            },
            "source": "manual",
            "properties": {"anon_polygon_id": "AOI-0005"},
        },
    )
    assert created.status_code == 201
    polygon_id = created.json()["id"]

    with session_scope() as db:
        analysis = Analysis(
            polygon_id=polygon_id,
            date_from=date(2024, 4, 1),
            date_to=date(2024, 4, 30),
            geometry_hash=created.json()["geometry_hash"],
            pipeline_version="0.1.0",
            state="QUEUED",
        )
        db.add(analysis)
        db.flush()
        db.add(AnalysisJob(analysis_id=analysis.id, state="QUEUED"))

    assert authed_client.delete(f"/api/v1/polygons/{polygon_id}").status_code == 204
    assert authed_client.get(f"/api/v1/polygons/{polygon_id}").status_code == 404
    ids = [item["id"] for item in authed_client.get("/api/v1/polygons").json()["items"]]
    assert polygon_id not in ids


def test_reference_polygons_lists_available_series(client) -> None:
    """Перечень рядов, по которым возможен анализ.

    Без него интерфейс не может предложить источник данных, и пользователь узнаёт
    о невозможности анализа только из упавшей джобы.
    """
    response = client.get("/api/v1/reference-polygons")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"items", "total"}
    assert body["total"] == len(body["items"])
    if body["items"]:
        first = body["items"][0]
        required = {"anon_polygon_id", "observations", "first_date", "last_date", "dataset"}
        assert set(first) >= required
        assert first["observations"] > 0
        # Список отсортирован по длине ряда: демо на длинном ряду осмысленнее.
        assert first["observations"] >= body["items"][-1]["observations"]


def test_analysis_without_data_source_is_refused_before_any_job(authed_client) -> None:
    """Поле без привязки к ряду отвергается сразу, а не падающей джобой.

    Раньше запрос принимался, создавалась запись анализа и джоба, воркер через
    несколько секунд падал `NO_DATA_SOURCE`, а в БД оставалась мёртвая пара строк.
    Пользователь при этом видел ошибку уже после ожидания.
    """
    created = authed_client.post(
        "/api/v1/polygons",
        json={
            "name": "Поле без ряда (тест)",
            "source": "manual",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[43.0, 48.0], [43.011, 48.0], [43.011, 48.008], [43.0, 48.008], [43.0, 48.0]]
                ],
            },
        },
    )
    if created.status_code != 201:
        pytest.skip("БД недоступна: тест требует поднятого PostgreSQL")
    polygon_id = created.json()["id"]
    try:
        response = authed_client.post(
            "/api/v1/analyses",
            json={"polygon_id": polygon_id, "date_from": "2024-04-01", "date_to": "2024-09-30"},
        )
        assert response.status_code == 422
        body = response.json()
        detail = body.get("detail", "") or body.get("message", "")
        # Сообщение обязано называть причину и способ исправления, а не только код.
        assert "NO_DATA_SOURCE" in detail
        assert "reference-polygons" in detail
    finally:
        authed_client.delete(f"/api/v1/polygons/{polygon_id}")


def test_data_requires_login(client) -> None:
    """Без входа данные недоступны.

    Проверяется именно 401, а не «страница не сломалась»: до появления входа
    любой, кто знал адрес, видел и правил чужие поля.
    """
    assert client.get("/api/v1/polygons").status_code == 401
    assert client.get("/api/v1/analyses/whatever/series").status_code == 401
    assert client.post("/api/v1/analyses", json={
        "polygon_id": "x", "date_from": "2026-01-01", "date_to": "2026-02-01"}).status_code == 401


def test_reference_list_stays_public(client) -> None:
    """Справочник рядов остаётся открытым.

    Это перечень доступных источников, а не чьи-то данные: прятать его за входом
    значит не дать понять, о чём вообще сервис, до регистрации.
    """
    assert client.get("/api/v1/reference-polygons").status_code == 200


def test_guest_gets_an_answer_not_an_error(client) -> None:
    """`/auth/me` отвечает и гостю.

    401 здесь смешал бы «не вошёл» и «сервис сломан», а интерфейсу нужно их
    различать: в первом случае показать форму, во втором — сообщение об ошибке.
    """
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 200
    body = response.json()
    assert body["authenticated"] is False
    assert "login_available" in body


def test_malformed_email_never_reaches_the_idp(client) -> None:
    """Явная опечатка отсекается до обращения к IdP.

    У внешнего сервиса лимит на отправку писем, и тратить его на «не-почта»
    значит приближать момент, когда настоящий человек не получит код.
    """
    response = client.post("/api/v1/auth/email-code", json={"email": "не-почта"})
    assert response.status_code == 422


def test_own_fields_only(authed_client) -> None:
    """Пользователь видит только свои поля.

    Тест создаёт поле от одного владельца и проверяет, что список у него
    непустой, а гость не получает ничего, кроме 401.
    """
    created = authed_client.post("/api/v1/polygons", json={
        "name": "Поле теста разделения", "source": "manual",
        "geometry": {"type": "Polygon", "coordinates": [
            [[47.0, 43.0], [47.011, 43.0], [47.011, 43.008], [47.0, 43.008], [47.0, 43.0]]]},
    })
    assert created.status_code == 201
    polygon_id = created.json()["id"]
    try:
        mine = authed_client.get("/api/v1/polygons").json()
        assert any(item["id"] == polygon_id for item in mine["items"])

        # Тот же клиент без cookie — уже посторонний.
        authed_client.cookies.clear()
        assert authed_client.get("/api/v1/polygons").status_code == 401
        assert authed_client.get(f"/api/v1/polygons/{polygon_id}").status_code == 401
    finally:
        from apps.db.base import session_scope
        from apps.db.models import Polygon

        with session_scope() as db:
            row = db.get(Polygon, polygon_id)
            if row is not None:
                db.delete(row)
