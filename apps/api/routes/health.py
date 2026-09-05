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


@router.get("/health/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Готовность к работе: БД, Redis, model bundle.

    В BE-001 подключений к БД и Redis ещё нет (это BE-002/BE-006), поэтому они
    честно отвечают not_configured, а не выдуманным ok.
    """
    settings = get_settings()
    return ReadyResponse(
        status=ComponentStatus.DEGRADED,
        database=ComponentStatus.NOT_CONFIGURED,
        redis=ComponentStatus.NOT_CONFIGURED,
        model_bundle=_bundle_status(settings.model_bundle_path),
    )


@router.get("/health/providers", response_model=ProvidersResponse)
async def providers() -> ProvidersResponse:
    """Диагностика внешних источников. Не блокирует cached demo (§8.4 ТЗ):
    даже когда все провайдеры недоступны, ответ 200 со статусом degraded."""
    return ProvidersResponse(status=ComponentStatus.NOT_CONFIGURED, providers=[])
