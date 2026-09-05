"""Public ML boundary. See artifacts/ml/CONTRACT_CHANGELOG.md for integration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"
KEY_COLUMNS = ["anon_polygon_id", "date"]
SUBMISSION_COLUMNS = KEY_COLUMNS + ["primary_ndvi_pred"]


@dataclass(frozen=True)
class ReconstructionRequest:
    frame: pd.DataFrame
    gap_mask: pd.Series
    context_mode: Literal["competition", "web"]


@dataclass(frozen=True)
class ReconstructionResult:
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    model_version: str


@runtime_checkable
class NDVIReconstructor(Protocol):
    def predict(self, request: ReconstructionRequest) -> ReconstructionResult: ...


class PredictionRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    anon_polygon_id: str
    date: date
    primary_ndvi_pred: float
    lower: float
    upper: float
    method: str
    primary_ndvi_reconstructed: float
    ndvi_harmonized: float


class DiagnosticRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    anon_polygon_id: str
    date: date
    p_s2: float = Field(ge=0, le=1)
    p_landsat: float = Field(ge=0, le=1)
    p_modis: float = Field(ge=0, le=1)
    p_unknown: float = Field(ge=0, le=1)
    left_distance_days: float | None
    right_distance_days: float | None
    model_disagreement: float = Field(ge=0)
    fallback_reason: str
    context_quality: float = Field(ge=0, le=1)
    source_confidence: float = Field(ge=0, le=1)
    quality_flags: list[str]
    interval_status: str
    interval_level: float
    harmonization_status: str


class ReconstructionPayload(BaseModel):
    """JSON-safe DTO for backend; DataFrames stay inside the Python boundary."""
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    model_version: str
    predictions: list[PredictionRow]
    diagnostics: list[DiagnosticRow]

    @classmethod
    def from_result(cls, result: ReconstructionResult) -> "ReconstructionPayload":
        def records(frame):
            clean = frame.astype(object).where(pd.notna(frame), None)
            return clean.to_dict("records")
        return cls(model_version=result.model_version,
                   predictions=records(result.predictions),
                   diagnostics=records(result.diagnostics))


def validate_request(request: ReconstructionRequest) -> tuple[pd.DataFrame, pd.Series]:
    """Validate without mutating the caller's frame or silently realigning masks."""
    if request.context_mode not in ("competition", "web"):
        raise ValueError("context_mode must be competition or web")
    frame = request.frame.copy(deep=True)
    required = {*KEY_COLUMNS, "primary_ndvi", "crop_type"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    mask = request.gap_mask
    if not isinstance(mask, pd.Series) or not mask.index.equals(frame.index):
        raise ValueError("gap_mask must be a Series with the exact frame index")
    if not pd.api.types.is_bool_dtype(mask) or mask.isna().any():
        raise ValueError("gap_mask must contain only boolean values")
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError("null key")
    if frame["date"].dt.tz is not None or not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("dates must be timezone-naive calendar days")
    frame["anon_polygon_id"] = frame["anon_polygon_id"].astype(str)
    if frame["anon_polygon_id"].str.strip().eq("").any() or frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("empty polygon or duplicate key")
    frame["primary_ndvi"] = pd.to_numeric(frame["primary_ndvi"], errors="raise")
    frame.loc[~np.isfinite(frame["primary_ndvi"].to_numpy(dtype=float, na_value=np.nan)), "primary_ndvi"] = np.nan
    if request.context_mode == "competition":
        if "is_synthetic_gap" not in frame or not pd.api.types.is_bool_dtype(frame["is_synthetic_gap"]):
            raise ValueError("competition requires boolean is_synthetic_gap")
        if frame["is_synthetic_gap"].isna().any() or not np.array_equal(mask, frame["is_synthetic_gap"]):
            raise ValueError("competition gap mask must exactly match is_synthetic_gap")
    return frame, mask.astype(bool).copy()
