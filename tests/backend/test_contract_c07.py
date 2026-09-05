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
    "/health/live",
    "/health/ready",
    "/health/providers",
    "/api/v1/polygons",
    "/api/v1/polygons/{polygon_id}",
    "/api/v1/field-search",
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


def test_analysis_rejects_reversed_and_huge_date_range(client) -> None:
    base = {"polygon_id": "p", "date_from": "2024-05-01", "date_to": "2024-01-01"}
    assert client.post("/api/v1/analyses", json=base).status_code == 422
    huge = {"polygon_id": "p", "date_from": "1990-01-01", "date_to": "2024-01-01"}
    assert client.post("/api/v1/analyses", json=huge).status_code == 422


def test_unknown_body_field_is_rejected(client) -> None:
    """Опечатка фронта обязана давать 422, а не молча анализировать другой период."""
    body = {"polygon_id": "p", "date_from": "2024-01-01", "date_to": "2024-02-01", "dateFrom": "x"}
    assert client.post("/api/v1/analyses", json=body).status_code == 422


def test_bbox_is_validated(client) -> None:
    assert client.get("/api/v1/polygons?bbox=nonsense").status_code == 422
    assert client.get("/api/v1/polygons?bbox=1,2,3").status_code == 422
    assert client.get("/api/v1/polygons?bbox=200,2,3,4").status_code == 422
    assert client.get("/api/v1/polygons?bbox=1,2,3,4").status_code == 200


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


def test_polygon_list_envelope_is_consistent(client) -> None:
    """Список полигонов — всегда конверт `{items, total}`, и `total` согласован.

    Прежняя редакция требовала буквально пустого списка: это описывало заглушку
    BE-001, а не контракт. После BE-004 роут ходит в БД, и содержимое зависит от
    состояния стенда — но форма ответа и согласованность `total` обязаны держаться
    при любом содержимом, включая пустое.
    """
    response = client.get("/api/v1/polygons")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"items", "total"}
    assert isinstance(body["items"], list)
    assert body["total"] >= len(body["items"])
