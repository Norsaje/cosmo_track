"""Тесты health-эндпоинтов. Полностью offline: сети не требуют.

Проверяются значения, а не набор ключей: мутация «инвертировать проверку bundle»
или «вернуть всем компонентам ok» обязана валить тест.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="нужен extra `web`")

from apps.api.routes.health import _bundle_status
from apps.api.schemas import ComponentStatus


def test_live_returns_ok(client) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_every_component(client) -> None:
    """Ответ всегда содержит все четыре поля, каким бы ни было состояние стенда.

    Прежняя редакция требовала `not_configured` для БД и Redis — это описывало
    отсутствие подключений в BE-001. После BE-002/BE-006 подключения настоящие,
    и фиксировать в тесте «их нет» значит фиксировать вчерашний день. Что обязано
    держаться всегда — состав ответа и допустимость значений.
    """
    body = client.get("/health/ready").json()
    assert set(body) == {"status", "database", "redis", "model_bundle"}
    allowed = {status.value for status in ComponentStatus}
    assert {body["database"], body["redis"], body["model_bundle"], body["status"]} <= allowed


def test_empty_connection_string_is_not_configured() -> None:
    """Пустой URL — это «не настроено», а не «упало»: разные причины и разные
    действия оператора. Проверяется без сети и без поднятого стенда."""
    from apps.api.routes.health import _database_status, _redis_status

    assert _database_status("") == ComponentStatus.NOT_CONFIGURED
    assert _redis_status("") == ComponentStatus.NOT_CONFIGURED


def test_unreachable_dependency_is_down_not_ok() -> None:
    """Недоступная зависимость обязана давать `down`.

    Это ровно тот случай, где легко соврать: обернуть проверку в try и вернуть ok,
    чтобы healthcheck не мешал. Тогда demo-preflight зелёный при мёртвой БД.
    """
    from apps.api.routes.health import _database_status, _redis_status

    # Порт 1 гарантированно не слушается ни одним сервисом.
    assert _database_status("postgresql+psycopg://u:p@127.0.0.1:1/none") == ComponentStatus.DOWN
    assert _redis_status("redis://127.0.0.1:1/0") == ComponentStatus.DOWN


def test_aggregate_status_is_the_weakest_component(monkeypatch, client) -> None:
    """Агрегат — самый слабый компонент, а не «в основном работает».

    Заявить `ok` при отсутствующей модели значит пообещать анализ, который
    не выполнится ни разу.
    """
    from apps.api.routes import health as health_module

    monkeypatch.setattr(health_module, "_database_status", lambda _: ComponentStatus.OK)
    monkeypatch.setattr(health_module, "_redis_status", lambda _: ComponentStatus.OK)

    monkeypatch.setattr(health_module, "_bundle_status", lambda _: ComponentStatus.OK)
    assert client.get("/health/ready").json()["status"] == ComponentStatus.OK

    monkeypatch.setattr(health_module, "_bundle_status", lambda _: ComponentStatus.DEGRADED)
    assert client.get("/health/ready").json()["status"] == ComponentStatus.DEGRADED

    monkeypatch.setattr(health_module, "_bundle_status", lambda _: ComponentStatus.DOWN)
    assert client.get("/health/ready").json()["status"] == ComponentStatus.DOWN


def test_bundle_status_distinguishes_missing_empty_and_ready(tmp_path) -> None:
    """Главная защита от лжи о готовности модели.

    Compose монтирует `./infra/model_bundle`, и Docker создаёт отсутствующий путь
    пустым каталогом. Проверка `isdir` в этом случае вернула бы ok при полном
    отсутствии бандла — ровно то, что запрещает red-team-пункт перед CP-3.
    """
    assert _bundle_status(str(tmp_path / "missing")) == ComponentStatus.NOT_CONFIGURED
    assert _bundle_status(str(tmp_path)) == ComponentStatus.DEGRADED
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    assert _bundle_status(str(tmp_path)) == ComponentStatus.OK


def test_ready_uses_bundle_path_from_settings(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COSMO_MODEL_BUNDLE_PATH", str(tmp_path))
    from apps.api.settings import get_settings

    get_settings.cache_clear()
    assert client.get("/health/ready").json()["model_bundle"] == ComponentStatus.DEGRADED


def test_providers_does_not_block_demo(client) -> None:
    """Недоступность провайдеров не даёт ошибку HTTP и не выдаёт статус down:
    cached demo обязано работать без единого живого источника (§8.4 ТЗ)."""
    response = client.get("/health/providers")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == ComponentStatus.NOT_CONFIGURED
    assert body["providers"] == []


def test_request_id_is_generated_and_echoed(client) -> None:
    generated = client.get("/health/live").headers["x-request-id"]
    assert len(generated) >= 16
    echoed = client.get("/health/live", headers={"x-request-id": "trace-42"})
    assert echoed.headers["x-request-id"] == "trace-42"


def test_request_id_is_sanitised(client) -> None:
    """Заголовок уходит в структурные логи, поэтому алфавит и длина ограничены."""
    response = client.get("/health/live", headers={"x-request-id": "<script>x</script>"})
    returned = response.headers["x-request-id"]
    assert "<" not in returned and ">" not in returned
    long_id = client.get("/health/live", headers={"x-request-id": "a" * 500})
    assert len(long_id.headers["x-request-id"]) <= 64


@pytest.mark.live
def test_live_marker_canary() -> None:
    """Канарейка маркера `live`.

    Смысл не в утверждении внутри, а в том, что этот тест обязан быть *отфильтрован*
    в обычном прогоне: в выводе `pytest -q` видно «1 deselected». Вместе с
    `--strict-markers` (опечатка в имени маркера теперь падает, а не проглатывается)
    это и есть проверка механизма из инварианта 13.
    """
    assert True
