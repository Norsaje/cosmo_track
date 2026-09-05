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

    #: Каталог поставленной модели (`model/`): внутри `ndvi/`, `data/` и `runs/`.
    #: Монтируется read-only. Пока каталога нет, `/health/ready` честно отвечает
    #: not_configured, а воркер работает заглушкой, если она разрешена.
    model_package_path: str = "/srv/model"

    #: Имя обученного запуска внутри `runs/`. Отдельная настройка, а не константа:
    #: Kaggle-прогон приедет соседним каталогом, и переключение на него не должно
    #: требовать пересборки образа.
    model_run_name: str = "local"

    #: Каталог с рядами наблюдений для offline-источника. По умолчанию — данные
    #: самой поставки: сервис обязан отдавать ровно те ряды, на которых модель
    #: обучалась, иначе выбранный в интерфейсе полигон окажется вне её контекста.
    model_data_dir: str = "/srv/model/data"

    #: Контур развёртывания. Единственное значение, при котором заглушка модели
    #: запрещена, — "production" (red-team checklist перед CP-3).
    environment: str = "development"

    #: Разрешена ли подмена отсутствующего bundle заглушкой ModelStub. Сломанный
    #: bundle заглушкой не подменяется никогда — см. `service.build_reconstructor`.
    allow_model_stub: bool = True

    #: Осознанное доверие к обученной модели. Требуется явно, потому что pickle
    #: исполняет код при загрузке: SHA256 ловит порчу файла, но не подлог автора.
    #: Значение по умолчанию False — доверие включается только после проверки провенанса.
    model_bundle_trusted: bool = False

    #: Адрес Braining ID. Вход по коду на почту выполняет он, своей
    #: аутентификации у сервиса нет.
    braining_id_url: str = "https://id.braining.space"
    #: Учётные данные приложения в Braining ID. Пустые значения означают, что
    #: вход не настроен: сервис честно отвечает 503, а не притворяется рабочим.
    braining_id_client_id: str = ""
    braining_id_client_secret: str = ""

    #: Срок жизни сессии. Тридцать дней — компромисс: реже входить приятнее,
    #: но украденная cookie тоже живёт дольше.
    session_ttl_days: int = 30

    #: Secure-флаг cookie. По умолчанию выключен, потому что стенд поднимается
    #: и по http на localhost; в контуре с TLS включается через окружение.
    session_cookie_secure: bool = False

    #: CORS — только allowlist, никаких "*" в проде (§8.12 ТЗ).
    cors_origins: list[str] = ["http://localhost:8080"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
