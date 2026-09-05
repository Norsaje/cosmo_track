"""C-09 v0.1: JSON contract без зависимостей от DL, сети и базы данных."""

from dataclasses import asdict, dataclass
from datetime import date
import math

SCHEMA_VERSION = "0.1"
ALGORITHM_VERSION = "robust-loyo-events-0.1.1"
REASON_CODES = frozenset(
    {
        "LOW_PRECIPITATION",
        "HIGH_TEMPERATURE",
        "LOW_NDWI",
        "MULTISENSOR_CONFIRMATION",
        "SOURCE_SWITCH_RISK",
        "LOW_DATA_COVERAGE",
        "RAPID_NEGATIVE_CHANGE",
        "PROLONGED_SUPPRESSION",
        "PHENOLOGY_SHIFT",
    }
)


@dataclass(frozen=True)
class AnomalyEvent:
    start_date: date
    end_date: date
    severity: str
    score: float
    confidence: float
    min_robust_z: float
    negative_area: float
    observed_points: int
    reconstructed_points: int
    reason_codes: tuple[str, ...]
    explanation_ru: str
    algorithm_version: str = ALGORITHM_VERSION

    def __post_init__(self):
        if type(self.start_date) is not date or type(self.end_date) is not date:
            raise ValueError("Event dates must be calendar dates")
        if self.end_date < self.start_date:
            raise ValueError("Event end precedes start")
        if self.severity not in {"normal", "biomass_suppression", "critical"}:
            raise ValueError("Unknown severity")
        numeric = (self.score, self.confidence, self.min_robust_z, self.negative_area)
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError("Event numbers must be finite JSON values")
        if not 0 <= self.confidence <= 1 or self.score < 0 or self.negative_area < 0:
            raise ValueError("Invalid score, confidence or area")
        if any(
            type(x) is not int or x < 0
            for x in (self.observed_points, self.reconstructed_points)
        ):
            raise ValueError("Point counts must be nonnegative integers")
        if self.observed_points + self.reconstructed_points == 0:
            raise ValueError("An event needs supporting points")
        if not set(self.reason_codes).issubset(REASON_CODES):
            raise ValueError("Unknown reason code")
        if not self.explanation_ru.strip() or not self.algorithm_version:
            raise ValueError("Explanation and algorithm version are required")

    @property
    def duration_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
            reason_codes=list(self.reason_codes),
            duration_days=self.duration_days,
            schema_version=SCHEMA_VERSION,
        )
        return result


@dataclass(frozen=True)
class DetectionResult:
    """Один polygon на вызов; quality warnings отдельно от отсутствия аномалий."""

    events: tuple[AnomalyEvent, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "events": [e.to_dict() for e in self.events],
            "warnings": list(self.warnings),
            "confidence_semantics": "heuristic_support_not_calibrated_probability",
        }
