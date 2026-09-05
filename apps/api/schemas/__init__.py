"""Публичные схемы API — контракт C-07.

Версия контракта объявлена в `jobs.py` (`CONTRACT_ID`, `CONTRACT_VERSION`).
Правило версионирования (§4 координации): 0.x — интерфейс может меняться,
но breaking change записывается заранее; 1.0 — frozen для MVP. Добавление
optional-поля совместимо; переименование, удаление и смена типа — нет.
"""

from .analyses import (
    AnalysisAccepted,
    AnalysisCreate,
    AnalysisOut,
    AnomaliesResponse,
    AnomalyOut,
    ProvenanceEntry,
    ProvenanceResponse,
    SeriesPoint,
    SeriesResponse,
)
from .auth import CurrentUser, EmailCodeRequest, EmailCodeSent, SessionCreate
from .errors import ErrorResponse
from .health import (
    ComponentStatus,
    LiveResponse,
    ProviderHealth,
    ProvidersResponse,
    ReadyResponse,
)
from .jobs import (
    ALLOWED_TRANSITIONS,
    CONTRACT_ID,
    CONTRACT_VERSION,
    TERMINAL_STATES,
    JobState,
    JobStatus,
)
from .polygons import (
    FieldCandidate,
    FieldSearchRequest,
    FieldSearchResponse,
    GeoJSONGeometry,
    PolygonCreate,
    PolygonList,
    PolygonOut,
    PolygonSource,
    ReferenceSeries,
    ReferenceSeriesList,
)

__all__ = [
    "CurrentUser",
    "EmailCodeRequest",
    "EmailCodeSent",
    "SessionCreate",
    "ALLOWED_TRANSITIONS",
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "TERMINAL_STATES",
    "AnalysisAccepted",
    "AnalysisCreate",
    "AnalysisOut",
    "AnomaliesResponse",
    "AnomalyOut",
    "ComponentStatus",
    "ErrorResponse",
    "FieldCandidate",
    "FieldSearchRequest",
    "FieldSearchResponse",
    "GeoJSONGeometry",
    "JobState",
    "JobStatus",
    "LiveResponse",
    "PolygonCreate",
    "PolygonList",
    "PolygonOut",
    "PolygonSource",
    "ReferenceSeries",
    "ReferenceSeriesList",
    "ProviderHealth",
    "ProvenanceEntry",
    "ProvenanceResponse",
    "ProvidersResponse",
    "ReadyResponse",
    "SeriesPoint",
    "SeriesResponse",
]
