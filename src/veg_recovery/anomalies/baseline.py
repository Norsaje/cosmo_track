"""Separate product harmonization and conservative, descriptive anomaly baseline."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

SENSORS = ("s2", "landsat", "modis")


def fit_harmonization(reference_frame: pd.DataFrame, reference: str = "s2",
                      min_pairs: int = 30) -> dict:
    """Robust affine calibration on visible paired sensors; never alters targets.

    IQR ratio and median intercept are deterministic summary statistics, not a
    supervised reconstruction model. Caller supplies only permitted references.
    Invalid sensor values are excluded from calibration, not clipped in the target.
    """
    if reference not in SENSORS:
        raise ValueError("unsupported reference sensor")
    transforms = {}
    for source in SENSORS:
        a = pd.to_numeric(reference_frame.get(source + "_ndvi", pd.Series(dtype=float)), errors="coerce")
        b = pd.to_numeric(reference_frame.get(reference + "_ndvi", pd.Series(dtype=float)), errors="coerce")
        pairs = pd.concat([a.rename("source"), b.rename("reference")], axis=1).dropna()
        pairs = pairs[np.isfinite(pairs).all(axis=1) & pairs.abs().le(1).all(axis=1)]
        if source == reference:
            slope, offset, status = 1., 0., "reference"
        elif len(pairs) >= min_pairs:
            iqrs = pairs.quantile(.75) - pairs.quantile(.25)
            if iqrs["source"] > 1e-8 and iqrs["reference"] > 1e-8:
                slope = float(iqrs["reference"] / iqrs["source"])
                offset = float(pairs["reference"].median() - slope * pairs["source"].median())
                status = "robust_affine"
            else:
                slope, offset, status = 1., 0., "identity_degenerate_pairs"
        else:
            slope, offset, status = 1., 0., "identity_insufficient_pairs"
        transforms[source] = dict(slope=slope, offset=offset, n_pairs=len(pairs), status=status)
    return dict(reference=reference, method="paired_median_iqr_affine", transforms=transforms)


def harmonize_values(values, probabilities: pd.DataFrame, calibration: dict | None):
    values = np.asarray(values, dtype=float)
    calibration = calibration or {}
    transforms = calibration.get("transforms", {})
    harmonized = np.zeros(len(values))
    total = np.zeros(len(values))
    missing = np.zeros(len(values), dtype=bool)
    for source in SENSORS:
        p = probabilities.get("p_" + source, pd.Series(np.zeros(len(values)))).to_numpy(dtype=float)
        transform = transforms.get(source, {"slope": 1., "offset": 0., "status": "identity_unavailable"})
        harmonized += p * (values * transform["slope"] + transform["offset"])
        total += p
        missing |= (p > 0) & transform["status"].startswith("identity")
    harmonized += np.maximum(0, 1 - total) * values
    status = np.where(missing | (total < .999999), "partial_or_identity", "source_mixture_affine")
    return harmonized, status


def reconstruct_product_series(frame: pd.DataFrame, result, calibration: dict | None = None) -> pd.DataFrame:
    """Keep observed primary values exactly; add two explicitly distinct series."""
    from veg_recovery.features import infer_source_labels
    out = frame.copy(deep=True)
    out["date"] = pd.to_datetime(out["date"])
    if out.duplicated(["anon_polygon_id", "date"]).any():
        raise ValueError("duplicate product key")
    pred = result.predictions.set_index(["anon_polygon_id", "date"])
    keys = pd.MultiIndex.from_frame(out[["anon_polygon_id", "date"]])
    reconstructed = pd.to_numeric(out["primary_ndvi"], errors="raise").to_numpy(dtype=float, na_value=np.nan, copy=True)
    positions = pred.index.get_indexer(keys)
    selected = positions >= 0
    reconstructed[selected] = pred.iloc[positions[selected]]["primary_ndvi_pred"]
    out["primary_ndvi_reconstructed"] = reconstructed
    labels = infer_source_labels(out)
    probs = pd.DataFrame({"p_" + s: (np.asarray(labels) == s).astype(float) for s in SENSORS})
    harmonized, status = harmonize_values(reconstructed, probs, calibration)
    harmonized[selected] = pred.iloc[positions[selected]]["ndvi_harmonized"]
    out["ndvi_harmonized"] = harmonized
    out["harmonization_status"] = status
    out.loc[selected, "harmonization_status"] = "reconstructed_source_mixture"
    out["is_reconstructed"] = selected
    out['confidence'] = np.where(np.isfinite(reconstructed) & (np.abs(reconstructed) <= 1), 1., 0.)
    diagnostic = result.diagnostics.set_index(['anon_polygon_id','date']).reindex(pred.index)
    gap_confidence = diagnostic['context_quality'].to_numpy() * diagnostic['source_confidence'].to_numpy()
    out.loc[selected,'confidence'] = gap_confidence[positions[selected]]
    return out


@dataclass(frozen=True)
class AnomalyResult:
    points: pd.DataFrame
    events: pd.DataFrame


def detect_anomalies(frame: pd.DataFrame, *, reference_frame: pd.DataFrame | None = None,
                     min_reference_years: int = 3, doy_window: int = 15,
                     z_threshold: float = -2., min_persistence: int = 2,
                     max_event_gap_days: int = 16) -> AnomalyResult:
    """Negative events versus *previous* years; no same-year reference leakage.

    Thresholds are descriptive baseline defaults, not claimed OOF-tuned parameters.
    Weather reason codes describe coincidence only; they never assert causality.
    """
    if min_reference_years < 2 or min_persistence < 2 or z_threshold >= 0:
        raise ValueError("require >=2 reference years/points and a negative threshold")
    required = ["anon_polygon_id", "date", "crop_type", "ndvi_harmonized"]
    for table in [frame, reference_frame if reference_frame is not None else frame]:
        if not set(required).issubset(table):
            raise ValueError("anomaly input requires keys, crop_type and ndvi_harmonized")
    out = frame.copy(deep=True)
    out["date"] = pd.to_datetime(out["date"])
    if out.duplicated(["anon_polygon_id", "date"]).any():
        raise ValueError("duplicate anomaly key")
    reference = (reference_frame if reference_frame is not None else frame).copy(deep=True)
    reference["date"] = pd.to_datetime(reference["date"])
    reference["_year"] = reference["date"].dt.year
    reference["_doy"] = reference["date"].dt.dayofyear
    reference = reference[np.isfinite(pd.to_numeric(reference["ndvi_harmonized"], errors="coerce"))]
    groups = {key: group for key, group in reference.groupby(["anon_polygon_id", "crop_type"], dropna=False)}
    statistics = []
    for row in out.itertuples(index=False):
        group = groups.get((row.anon_polygon_id, row.crop_type), reference.iloc[:0])
        dd = (group["_doy"] - row.date.dayofyear).abs()
        ref = group[(group["_year"] < row.date.year) & (np.minimum(dd, 366 - dd) <= doy_window)]
        years = ref["_year"].nunique()
        values = ref["ndvi_harmonized"].to_numpy(dtype=float)
        enough = years >= min_reference_years
        median = float(np.median(values)) if enough else np.nan
        mad = float(np.median(np.abs(values - median))) if enough else np.nan
        residual = float(row.ndvi_harmonized - median) if enough else np.nan
        robust_z = residual / max(1.4826 * mad, .01) if enough else np.nan
        statistics.append((years, median, mad, residual, robust_z, "ok" if enough else "insufficient_reference_years"))
    columns = ["n_reference_years", "climatology_median", "climatology_mad", "residual", "robust_z", "reference_status"]
    for i, col in enumerate(columns): out[col] = [s[i] for s in statistics]
    confidence = pd.to_numeric(out.get("confidence", pd.Series(1., index=out.index)), errors="coerce").fillna(0).clip(0, 1)
    out["confidence"] = confidence
    out["negative_candidate"] = out["robust_z"].le(z_threshold) & out["residual"].lt(0)
    out["event_id"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    events = []
    for polygon, rows in out.groupby("anon_polygon_id", sort=True):
        rows = rows.sort_values("date")
        runs, run = [], []
        previous = None
        for index, row in rows.iterrows():
            contiguous = previous is not None and (row["date"] - previous).days <= max_event_gap_days
            if not row["negative_candidate"] or (run and not contiguous):
                if run: runs.append(run)
                run = []
            if row["negative_candidate"]: run.append(index)
            previous = row["date"]
        if run: runs.append(run)
        for indices in runs:
            if len(indices) < min_persistence: continue
            part = out.loc[indices]
            start, end = part["date"].min(), part["date"].max()
            duration = max(1, (end - start).days + 1)
            magnitude = float((-part["residual"]).mean())
            certainty = float(part["confidence"].mean())
            codes = []
            if "era5_precip_mm" in part and pd.to_numeric(part["era5_precip_mm"], errors="coerce").median() < 1:
                codes.append("coincides_with_low_precipitation")
            if "era5_temp_c" in part and pd.to_numeric(part["era5_temp_c"], errors="coerce").median() > 30:
                codes.append("coincides_with_high_temperature")
            ndwi = [c for c in ["s2_ndwi", "landsat_ndwi"] if c in part]
            if ndwi and part[ndwi].apply(pd.to_numeric, errors="coerce").median().median() < 0:
                codes.append("consistent_with_low_ndwi")
            sensor_cols = [s + "_ndvi" for s in SENSORS if s + "_ndvi" in part]
            if len(sensor_cols) > 1:
                sensor = part[sensor_cols].apply(pd.to_numeric, errors="coerce")
                agrees = sensor.count(axis=1).ge(2) & (sensor.max(axis=1) - sensor.min(axis=1)).le(.05)
                if agrees.any(): codes.append("coincides_with_multi_sensor_agreement")
            if not codes: codes.append("negative_deviation_without_attributed_cause")
            event_id = f"{polygon}:{start:%Y-%m-%d}:{end:%Y-%m-%d}"
            out.loc[indices, "event_id"] = event_id
            events.append(dict(event_id=event_id, anon_polygon_id=polygon, start_date=start,
                               end_date=end, n_points=len(part), duration_days=duration,
                               magnitude=magnitude, confidence=certainty,
                               severity=magnitude * duration * certainty, reason_codes=codes,
                               interpretation="Отрицательное отклонение совпадает с перечисленными признаками; причина не установлена."))
    event_columns = ["event_id", "anon_polygon_id", "start_date", "end_date", "n_points", "duration_days",
                     "magnitude", "confidence", "severity", "reason_codes", "interpretation"]
    return AnomalyResult(out, pd.DataFrame(events, columns=event_columns))
