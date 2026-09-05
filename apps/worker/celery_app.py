"""Celery-приложение. Долгие задачи идут только через durable worker —
FastAPI BackgroundTasks в роли очереди запрещены (инвариант 1, решение D-004).
"""

from __future__ import annotations

import os

from celery import Celery

broker_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")

# include обязателен: без него модуль задач никем не импортируется, воркер стартует
# с пустым реестром, и `.delay()` падает с NotRegistered — проверить живость очереди
# становится нечем.
celery_app = Celery(
    "cosmo_track",
    broker=broker_url,
    backend=broker_url,
    include=["apps.worker.tasks"],
)
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    # Задача забирается воркером по одной: стадии анализа длинные и неравномерные,
    # префетч приводил бы к простою свободных воркеров.
    worker_prefetch_multiplier=1,
    timezone="UTC",
    enable_utc=True,
)
