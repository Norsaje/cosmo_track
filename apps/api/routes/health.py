"""Health-эндпоинты. Единственные роуты, которые в BE-001 реализованы полностью:
на них опираются healthcheck-и Compose и offline smoke-тесты Разработчика 4."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from apps.api.schemas import (
    ComponentStatus,
    LiveResponse,
    ProvidersResponse,
    ReadyResponse,
)
from apps.api.settings import get_settings

router = APIRouter(tags=["health"])

#: Файл, по которому определяем, что смонтирован настоящий bundle, а не пустой каталог.
#: Именно manifest несёт schema_version — единственный критерий отказа по инварианту 6.
BUNDLE_MANIFEST = "manifest.json"


def _bundle_status(bundle_path: str) -> ComponentStatus:
    """Статус model bundle.

    Проверять `isdir` нельзя: `docker-compose.yml` монтирует `./artifacts/ml/final_bundle`,
    и Docker сам создаёт этот путь пустым каталогом, если его нет. Тогда `isdir` вернул бы
    True и health отрапортовал бы готовность модели при полном её отсутствии — ровно та
    ложь, которую запрещает red-team-пункт перед CP-3.
    """
    root = Path(bundle_path)
    if not root.is_dir():
        return ComponentStatus.NOT_CONFIGURED
    if not (root / BUNDLE_MANIFEST).is_file():
        # Каталог есть, манифеста нет — это не «готово» и не «не настроено»,
        # а именно деградация: смонтировали пустой или неполный bundle.
        return ComponentStatus.DEGRADED
    return ComponentStatus.OK


@router.get("/health/live", response_model=LiveResponse)
async def live() -> LiveResponse:
    """Процесс жив. Никаких внешних проверок — иначе healthcheck начнёт падать
    из-за чужой недоступности и Compose будет бесконечно перезапускать контейнер."""
    return LiveResponse()


def _database_status(database_url: str) -> ComponentStatus:
    """Проверка БД реальным запросом, а не наличием строки подключения.

    `SELECT 1` вместо простого открытия соединения: пул может отдать протухший
    сокет, и «подключились» окажется ложью до первого настоящего запроса.
    Заодно проверяем, что миграции применены — пустая схема это не готовность.
    """
    if not database_url:
        return ComponentStatus.NOT_CONFIGURED
    try:
        from sqlalchemy import text

        from apps.db.base import get_engine

        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
            applied = connection.execute(
                text("SELECT count(*) FROM alembic_version")
            ).scalar_one()
    except Exception:
        # Наружу уходит только статус: ни строки подключения, ни текста драйвера
        # в ответе health быть не должно (инвариант 11).
        return ComponentStatus.DOWN
    return ComponentStatus.OK if applied else ComponentStatus.DEGRADED


def _redis_status(redis_url: str) -> ComponentStatus:
    """Ping брокера. Без него воркер не получит ни одной задачи (BE-006)."""
    if not redis_url:
        return ComponentStatus.NOT_CONFIGURED
    try:
        import redis

        client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        client.close()
    except Exception:
        return ComponentStatus.DOWN
    return ComponentStatus.OK


@router.get("/health/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Готовность к работе: БД, Redis, model bundle.

    Агрегированный статус — самый слабый из компонентов. `ok` выставляется только
    когда готовы все три: заявить готовность при отсутствующей модели значит
    обещать анализ, который не выполнится.
    """
    settings = get_settings()
    database = _database_status(settings.database_url)
    redis_status = _redis_status(settings.redis_url)
    bundle = _bundle_status(settings.model_bundle_path)

    components = (database, redis_status, bundle)
    if all(component is ComponentStatus.OK for component in components):
        overall = ComponentStatus.OK
    elif ComponentStatus.DOWN in components:
        overall = ComponentStatus.DOWN
    else:
        overall = ComponentStatus.DEGRADED

    return ReadyResponse(
        status=overall,
        database=database,
        redis=redis_status,
        model_bundle=bundle,
    )


@router.get("/health/providers", response_model=ProvidersResponse)
async def providers() -> ProvidersResponse:
    """Диагностика внешних источников. Не блокирует cached demo (§8.4 ТЗ):
    даже когда все провайдеры недоступны, ответ 200 со статусом degraded."""
    return ProvidersResponse(status=ComponentStatus.NOT_CONFIGURED, providers=[])
