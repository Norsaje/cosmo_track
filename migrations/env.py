"""Окружение Alembic.

URL берётся из настроек приложения, а не из alembic.ini: пароль не должен лежать
в репозитории (инвариант 11), а контур должен определяться теми же переменными
COSMO_*, что и сам сервис.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from apps.api.settings import get_settings
from apps.db import Base
from apps.db import models as _models  # noqa: F401  — регистрирует таблицы в метаданных

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Всё, чего нет в наших метаданных, миграции не касаются.

    Образ postgis/postgis ставит расширения postgis, topology и tiger_geocoder —
    это десятки таблиц (`spatial_ref_sys`, `layer`, `zcta5`, `street_type_lookup`…).
    Фильтр по именам их не покрывал, и autogenerate честно предложил их удалить:
    первая же миграция снесла бы справочник систем координат вместе с геокодером.
    Поэтому правило простое — таблица наша только если она есть в `Base.metadata`.
    """
    if getattr(obj, "schema", None) not in (None, "public"):
        return False
    if type_ == "table" and name not in target_metadata.tables:
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
