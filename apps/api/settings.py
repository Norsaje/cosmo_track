"""Конфигурация API. Все значения приходят из окружения; в репозитории лежит
только `.env.example` с placeholder-ами (инвариант 11)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    #: env_prefix обязателен: без него `DATABASE_URL`, `REDIS_URL` и `CORS_ORIGINS`
    #: подхватываются из любого окружения CI-раннера и делают прогон недетерминированным.
    model_config = SettingsConfigDict(env_file=".env", env_prefix="COSMO_", extra="ignore")

    app_name: str = "cosmo-track-api"
    api_prefix: str = "/api/v1"
    pipeline_version: str = "0.1.0"

    #: Пусто по умолчанию: тихий старт с подразумеваемой БД скрывал бы непроброшенную
    #: конфигурацию. Пустое значение честно отражается в /health/ready.
    database_url: str = ""
    redis_url: str = "redis://redis:6379/0"

    #: Каталог с model bundle от Разработчика 1 (ML). Монтируется read-only.
    #: Пока bundle нет, `/health/ready` честно отвечает not_configured.
    model_bundle_path: str = "/srv/model_bundle"

    #: CORS — только allowlist, никаких "*" в проде (§8.12 ТЗ).
    cors_origins: list[str] = ["http://localhost:8080"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
