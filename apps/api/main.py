"""Точка входа API. Скелет BE-001: роуты объявлены, схемы зафиксированы (C-07 v0.1),
бизнес-логика приходит в BE-002…BE-014.
"""

from __future__ import annotations

import logging
import time
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Именно starlette-версия: 404 на несопоставленный маршрут поднимает её,
# и обработчик, повешенный на fastapi.HTTPException, до неё не доберётся.
from starlette.exceptions import HTTPException as StarletteHTTPException

from apps.api.routes import analyses, auth, health, jobs, polygons
from apps.api.schemas import CONTRACT_ID, CONTRACT_VERSION, ErrorResponse
from apps.api.settings import get_settings

settings = get_settings()

# Структурные логи с request_id (§8.11 ТЗ). Настраиваем на импорте, а не лениво:
# иначе первые запросы после старта потеряются.
logging.basicConfig(format="%(message)s", level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger(service="cosmo-track-api")

app = FastAPI(
    title="cosmo_track API",
    version=settings.pipeline_version,
    description=f"Контракт {CONTRACT_ID} v{CONTRACT_VERSION}",
)

# CORS только по allowlist: "*" в проде запрещён (§8.12 ТЗ).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    # Без expose_headers браузер не отдаст заголовок JS-коду и трейс потеряется.
    expose_headers=["x-request-id"],
)

#: Ограничение на клиентский трейс: заголовок уходит в логи, и неограниченная
#: строка произвольного алфавита засоряла бы их и мешала грепу.
MAX_REQUEST_ID_LEN = 64


def _safe_request_id(raw: str | None) -> str:
    if not raw:
        return uuid.uuid4().hex
    cleaned = "".join(ch for ch in raw[:MAX_REQUEST_ID_LEN] if ch.isalnum() or ch in "-_")
    return cleaned or uuid.uuid4().hex


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    """Единый конверт ошибки C-07 плюс сохранение трейса в заголовке."""
    request_id = getattr(request.state, "request_id", None)
    response = JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error_code=code, message=message, request_id=request_id).model_dump(),
    )
    if request_id:
        response.headers["x-request-id"] = request_id
    return response


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Каждому запросу — request_id, структурная запись в лог и единый конверт ошибки.

    Необработанное исключение перехватываем здесь, а не через `app.exception_handler`:
    тот обработчик Starlette ставит в самый внешний слой, выше CORS и этого middleware,
    поэтому ответ 500 уходил бы без `x-request-id` и без CORS-заголовков — и браузер
    показывал бы непрозрачную сетевую ошибку вместо безопасного сообщения.
    """
    request_id = _safe_request_id(request.headers.get("x-request-id"))
    request.state.request_id = request_id
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        log.exception(
            "unhandled_error",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return _error_response(
            request,
            500,
            "INTERNAL_ERROR",
            "Внутренняя ошибка сервиса. Повторите запрос позже.",
        )
    response.headers["x-request-id"] = request_id
    log.info(
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round((time.monotonic() - started) * 1000, 1),
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Приводим все 4xx/5xx к формату C-07: `{error_code, message, request_id}`.

    Без этого API отдавал бы две несовместимые формы ошибки — `ErrorResponse` только
    на 500 и дефолтный `{"detail": …}` на всём остальном, — а потребители контракта
    (фронт и smoke-тесты Разработчика 4) читают `error_code`.
    """
    codes = {
        400: "BAD_REQUEST",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        501: "NOT_IMPLEMENTED",
        503: "UNAVAILABLE",
    }
    return _error_response(
        request,
        exc.status_code,
        codes.get(exc.status_code, f"HTTP_{exc.status_code}"),
        str(exc.detail),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 в том же конверте. Присланное значение (`input`) наружу не возвращаем:
    это эхо клиентских данных, которому в публичном ответе не место."""
    fields = ", ".join(".".join(str(p) for p in err["loc"][1:]) or "body" for err in exc.errors())
    return _error_response(
        request,
        422,
        "VALIDATION_ERROR",
        f"Некорректное тело запроса. Проверьте поля: {fields}."
        if fields
        else "Некорректное тело запроса.",
    )


# Health-роуты живут вне версионного префикса: их дёргают healthcheck-и Compose
# и smoke-тесты, и версия API их менять не должна.
app.include_router(health.router)
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(polygons.router, prefix=settings.api_prefix)
app.include_router(jobs.router, prefix=settings.api_prefix)
app.include_router(analyses.router, prefix=settings.api_prefix)
