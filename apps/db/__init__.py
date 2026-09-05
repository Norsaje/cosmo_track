"""Слой хранения (BE-002).

Модели и сессия живут здесь, а не в `src/veg_recovery/`, намеренно: по §8.6 ТЗ
adapters и оркестратор не должны знать о БД. Воркер связывает БД и чистую
логику `service/`, сама логика остаётся тестируемой без PostgreSQL.
"""

from apps.db.base import Base, get_engine, get_session_factory, session_scope
from apps.db.models import (
    Analysis,
    AnalysisJob,
    AnomalyEvent,
    Observation,
    Polygon,
    Provenance,
    Reconstruction,
)

__all__ = [
    "Analysis",
    "AnalysisJob",
    "AnomalyEvent",
    "Base",
    "Observation",
    "Polygon",
    "Provenance",
    "Reconstruction",
    "get_engine",
    "get_session_factory",
    "session_scope",
]
