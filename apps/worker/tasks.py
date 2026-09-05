"""Задачи воркера. В BE-001 — только проверка живости очереди.

Задача анализа появляется в BE-006 (durable FSM) и вызывает общий
`NDVIReconstructor` от Разработчика 1; своей модели здесь не будет никогда
(инвариант 6, решение D-002).
"""

from __future__ import annotations

from apps.worker.celery_app import celery_app


@celery_app.task(name="cosmo_track.ping")
def ping() -> str:
    """Смоук-задача: подтверждает, что брокер доступен и воркер забирает задачи."""
    return "pong"
