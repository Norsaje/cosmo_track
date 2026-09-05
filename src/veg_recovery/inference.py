"""One reconstruction implementation for backend calls and batch inference."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from veg_recovery.contracts import (KEY_COLUMNS, DiagnosticRow, PredictionRow,
                                   ReconstructionRequest, ReconstructionResult, validate_request)
from veg_recovery.features import build_features
from veg_recovery.models.baselines import BaselineModel
from veg_recovery.models.bundle import LoadedBundle, load_bundle
from veg_recovery.anomalies.baseline import harmonize_values


class BundleReconstructor:
    def __init__(self, bundle: LoadedBundle):
        self.bundle = bundle

    def predict(self, request: ReconstructionRequest) -> ReconstructionResult:
        frame, mask = validate_request(request)
        gap_keys = frame.loc[mask, KEY_COLUMNS].reset_index(drop=True)
        if gap_keys.empty:
            return ReconstructionResult(pd.DataFrame(columns=PredictionRow.model_fields),
                                        pd.DataFrame(columns=DiagnosticRow.model_fields),
                                        self.bundle.manifest.model_version)
        features = build_features(frame, gap_keys, self.bundle.state)
        baseline = BaselineModel(method=self.bundle.config.get("method", "mean_neighbors"), state=self.bundle.state)
        fallback = baseline.predict_from_features(features)
        if self.bundle.estimators is None:
            values = fallback["primary_ndvi_pred"].to_numpy(dtype=float, copy=True)
            extra = {}
            method = fallback["method"].astype(str).to_numpy()
        else:
            from veg_recovery.models.estimators import predict_bundle_features
            values, extra = predict_bundle_features(self.bundle, features)
            method = np.full(len(features), "oof_ensemble", dtype=object)
        invalid = ~np.isfinite(values)
        values[invalid] = fallback.loc[invalid, "primary_ndvi_pred"]
        clip = self.bundle.config.get("clip")
        if clip is not None:
            if not self.bundle.config.get("clip_oof_evidence"):
                raise ValueError("clip requires recorded OOF evidence")
            values = np.clip(values, *clip)
        probabilities = pd.DataFrame({"p_" + s: np.asarray(extra.get("p_" + s, features.get("p_" + s, np.zeros(len(features)))), dtype=float)
                                      for s in ["s2", "landsat", "modis", "unknown"]})
        if (not np.isfinite(probabilities).all().all() or (probabilities < 0).any().any()
                or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-6)):
            raise ValueError("source probabilities must be finite, nonnegative and sum to one")
        disagreement = np.asarray(extra.get("model_disagreement", np.zeros(len(features))), dtype=float)
        fallback_reason = fallback["fallback_reason"].fillna("").astype(str).to_numpy(dtype=object, copy=True)
        fallback_reason[invalid] = "nonfinite_model_prediction"
        if 'model_nonfinite_replaced' in extra:
            invalid |= np.asarray(extra['model_nonfinite_replaced'], dtype=bool)
            fallback_reason[invalid] = 'nonfinite_model_prediction'
        if "conservative_gate" in extra:
            gated = np.asarray(extra["conservative_gate"], dtype=bool)
            fallback_reason[gated] = "oof_conservative_gate"
            method[gated] = "conservative_blend"
        lower, upper, status, level = self._intervals(values, features, extra)
        harmonized, harmony_status = harmonize_values(values, probabilities, self.bundle.config.get("harmonization"))
        predictions = gap_keys.copy()
        for name, column in {"primary_ndvi_pred": values, "lower": lower, "upper": upper, "method": method,
                             "primary_ndvi_reconstructed": values.copy(), "ndvi_harmonized": harmonized}.items():
            predictions[name] = column
        diagnostics = gap_keys.copy()
        for column in probabilities: diagnostics[column] = probabilities[column].to_numpy()
        left = features["days_left_1"].to_numpy(dtype=float)
        right = features["days_right_1"].to_numpy(dtype=float)
        diagnostics["left_distance_days"] = np.where(np.isfinite(left), left, np.nan)
        diagnostics["right_distance_days"] = np.where(np.isfinite(right), right, np.nan)
        diagnostics["model_disagreement"] = disagreement
        diagnostics["fallback_reason"] = fallback_reason
        diagnostics["context_quality"] = features["context_quality"].to_numpy(dtype=float)
        diagnostics["source_confidence"] = probabilities.max(axis=1).to_numpy()
        diagnostics["interval_status"] = status
        diagnostics["interval_level"] = level
        diagnostics["harmonization_status"] = harmony_status
        flags = []
        for i in range(len(features)):
            current = []
            if not np.isfinite(left[i]) or not np.isfinite(right[i]): current.append("one_sided_or_no_context")
            if not bool(features.iloc[i]["seen_polygon"]): current.append("unseen_polygon")
            if values[i] < -1 or values[i] > 1: current.append("prediction_outside_physical_range")
            if probabilities.iloc[i]["p_unknown"] > .25: current.append("uncertain_sensor_source")
            if status != "empirical_oof_subgroups_passed": current.append("interval_not_coverage_certified")
            if harmony_status[i] == "partial_or_identity": current.append("harmonization_incomplete")
            if invalid[i]: current.append("model_nonfinite_replaced")
            flags.append(current)
        diagnostics["quality_flags"] = flags
        predictions = predictions[list(PredictionRow.model_fields)]
        diagnostics = diagnostics[list(DiagnosticRow.model_fields)]
        if not np.isfinite(predictions[["primary_ndvi_pred", "lower", "upper", "ndvi_harmonized"]]).all().all():
            raise ValueError("reconstructor produced nonfinite output")
        return ReconstructionResult(predictions, diagnostics, self.bundle.manifest.model_version)

    def _intervals(self, values, features, extra):
        config = self.bundle.config
        uncertainty = config.get("uncertainty")
        level = float(config.get("interval_level", .95))
        if uncertainty:
            from veg_recovery.models.calibration import interval_bounds
            diag = {c: features[c].to_numpy() for c in features if c.startswith("p_")}
            diag.update(extra)
            lower, upper = interval_bounds(values, features, diag, uncertainty, level=level)
            return lower, upper, uncertainty.get("status", "empirical_oof_not_certified"), level
        # Explicit finite fallback only; this is NOT called a calibrated interval.
        width = float(config.get("uncalibrated_interval_half_width", 1.))
        if not np.isfinite(width) or width <= 0: raise ValueError("invalid interval half-width")
        return values - width, values + width, "uncalibrated_default", level


def load_reconstructor(bundle_path: str | Path, *, trusted: bool = False) -> BundleReconstructor:
    return BundleReconstructor(load_bundle(bundle_path, trusted=trusted))
