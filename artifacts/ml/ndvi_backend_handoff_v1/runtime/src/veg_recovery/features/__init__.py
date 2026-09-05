"""Shared, leakage checked feature construction for training and inference."""

from .builder import (
    FEATURE_VERSION,
    FeatureState,
    build_features,
    fit_feature_state,
    infer_source_labels,
    source_feature_columns,
)

__all__ = [
    "FEATURE_VERSION", "FeatureState", "build_features", "fit_feature_state",
    "infer_source_labels", "source_feature_columns",
]
