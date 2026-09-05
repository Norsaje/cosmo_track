# Один образ на api и worker: §8.1 ТЗ требует, чтобы они шли одной версией.
# Раздельные Dockerfile'ы разъезжались бы молча, и расхождение всплыло бы на CP-4
# в виде непройденного parity, уводя диагностику в ML-код вместо инфраструктуры.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /usr/local/bin/uv

WORKDIR /app

# README.md копируется вместе с манифестом: pyproject объявляет `readme = "README.md"`,
# и hatchling валидирует метаданные при сборке пакета. Без него `uv sync` падает
# с `OSError: Readme file does not exist` — то есть стенд не собирается вообще.
COPY pyproject.toml uv.lock README.md ./
# extra ml нужен именно в рантайме: C-04 — trained-бандл, и joblib поднимает
# ColumnTransformer из sklearn и CatBoostRegressor. Без него загрузка модели
# падает уже при старте воркера, а не при первом запросе.
RUN uv sync --frozen --no-install-project --extra web --extra core --extra ml

COPY src ./src
COPY apps ./apps
RUN uv sync --frozen --extra web --extra core --extra ml

# Контейнер не работает под root (§8.11 ТЗ).
RUN useradd --create-home --uid 10001 appuser \
    # Каталог кэша создаётся в образе и сразу отдаётся appuser: пустой named volume
    # Docker инициализирует правами каталога из образа. Иначе том монтируется как
    # root:root 0755 и запись кэша в BE-014 упрётся в PermissionError.
    && mkdir -p /srv/cache /srv/model_bundle \
    && chown -R appuser:appuser /app /opt/venv /srv/cache /srv/model_bundle
USER appuser

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
