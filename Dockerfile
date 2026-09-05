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

# libgomp1 — рантайм OpenMP, без которого не грузится LightGBM: колесо тянет
# libgomp.so.1 динамически, а в slim-образе его нет, и импорт падает уже при
# старте воркера (`OSError: libgomp.so.1: cannot open shared object file`).
# Ставится отдельным слоем до зависимостей: он меняется куда реже питоновских.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
# extra ml нужен именно в рантайме: обученная смесь из `model/` поднимается pickle-ом
# и тянет CatBoostRegressor, LGBMRegressor и scikit-learn. Без него загрузка модели
# падает уже при старте воркера, а не при первом запросе.
# extra geo — shapely и pyproj: валидация контура и площадь в projected CRS
# выполняются на сервере (инвариант 2), значит нужны и API, и воркеру.
RUN uv sync --frozen --no-install-project --extra web --extra core --extra ml --extra geo

COPY src ./src
COPY apps ./apps
RUN uv sync --frozen --extra web --extra core --extra ml --extra geo \
    # catboost тянет с собой jupyter-виджеты и nbextensions — почти гигабайт,
    # который в рантайме сервиса не используется никогда. Инференс их не импортирует,
    # а образ с ними не помещается на диск сборочной машины.
    && rm -rf /opt/venv/share/jupyter /opt/venv/share/nbextensions \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
    && find /opt/venv -name "*.pyc" -delete

# Контейнер не работает под root (§8.11 ТЗ).
RUN useradd --create-home --uid 10001 appuser \
    # Каталог кэша создаётся в образе и сразу отдаётся appuser: пустой named volume
    # Docker инициализирует правами каталога из образа. Иначе том монтируется как
    # root:root 0755 и запись кэша в BE-014 упрётся в PermissionError.
    && mkdir -p /srv/cache /srv/model \
    && chown -R appuser:appuser /app /opt/venv /srv/cache /srv/model
USER appuser

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
