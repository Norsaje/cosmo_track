"""Deterministic interpolation baselines sharing the production feature path."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from veg_recovery.features import FeatureState, build_features

BASELINE_METHODS = ("nearest_left", "nearest_right", "mean_neighbors", "linear", "seasonal_crop", "pchip", "akima")


@dataclass
class BaselineModel:
    """Fit-free baseline. Edge estimates average one neighbor with the prior.

    The official ``mean_neighbors`` baseline averages nearest finite raw targets
    on the two calendar sides. Linear interpolation uses time distances. Every
    method falls back to a finite seasonal hierarchy when context is absent.
    ``FeatureState()`` has an explicit, uncalibrated default of 0.5.
    """
    method: str = "mean_neighbors"
    state: FeatureState = field(default_factory=FeatureState)

    def __post_init__(self) -> None:
        if self.method not in BASELINE_METHODS:
            raise ValueError(f"Unknown baseline {self.method!r}; choose from {BASELINE_METHODS}")

    def predict(self, frame: pd.DataFrame, gap_keys: Any) -> pd.DataFrame:
        return self.predict_from_features(build_features(frame, gap_keys, self.state))

    def predict_from_features(self, features: pd.DataFrame) -> pd.DataFrame:
        records = []
        column = {"nearest_left": "target_left_1", "nearest_right": "target_right_1",
                  "mean_neighbors": "mean_neighbors", "linear": "linear_interpolation",
                  "seasonal_crop": "crop_seasonal_prior", "pchip": "pchip_interpolation",
                  "akima": "akima_interpolation"}[self.method]
        for feature in features.to_dict("records"):
            left = float(feature.get("target_left_1", np.nan))
            right = float(feature.get("target_right_1", np.nan))
            prior = float(feature.get("seasonal_prior", self.state.global_median))
            if not np.isfinite(prior):
                prior = float(self.state.global_median)
            if not np.isfinite(prior):
                prior = 0.5
            one_neighbor = np.isfinite(left) ^ np.isfinite(right)
            fallback_reason = ""
            method = self.method
            value = float(feature.get(column, np.nan))
            if self.method != "seasonal_crop" and one_neighbor:
                value = ((left if np.isfinite(left) else right) + prior) / 2
                fallback_reason = "one_neighbor_plus_seasonal_prior"
                method = "edge_neighbor_prior"
            elif not np.isfinite(value):
                linear = float(feature.get("linear_interpolation", np.nan))
                if self.method in ("pchip", "akima") and np.isfinite(linear):
                    value = linear
                    fallback_reason = f"{self.method}_unavailable_linear"
                    method = "linear"
                else:
                    value = prior
                    fallback_reason = str(feature.get("seasonal_prior_level", "global_default"))
                    method = "seasonal_fallback"
            if not np.isfinite(value):
                raise ValueError("Baseline prediction must be finite")
            records.append({"anon_polygon_id": feature["anon_polygon_id"], "date": feature["date"],
                            "primary_ndvi_pred": value, "method": method, "fallback_reason": fallback_reason})
        return pd.DataFrame(records, columns=["anon_polygon_id", "date", "primary_ndvi_pred", "method", "fallback_reason"])


def predict_baseline(frame: pd.DataFrame, gap_keys: Any, fitted_state: FeatureState, method: str = "mean_neighbors") -> pd.DataFrame:
    return BaselineModel(method=method, state=fitted_state).predict(frame, gap_keys)
