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


def test_ready_reports_db_and_redis_as_not_configured(client) -> None:
    """До BE-002/BE-006 подключений нет, и health обязан говорить это прямо,
    а не выдумывать ok — иначе demo-preflight зелёный при неподнятом стенде."""
    body = client.get("/health/ready").json()
    assert set(body) == {"status", "database", "redis", "model_bundle"}
    assert body["database"] == ComponentStatus.NOT_CONFIGURED
    assert body["redis"] == ComponentStatus.NOT_CONFIGURED
    assert body["status"] == ComponentStatus.DEGRADED


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
