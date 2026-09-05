"""One feature path for pseudo gaps and real gaps.

No stored climatology, status, z-score or same-row dynamic value is a feature.
State must be fitted on permitted labels only; overlap with requested keys raises.
Visible observations in ``frame`` are permitted transductive context. Physical
checks flag implausible raw values and filter sensor/weather context; they never
clip or replace finite raw competition targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

FEATURE_VERSION = "ndvi-context-v1"
KEYS = ["anon_polygon_id", "date"]
SENSORS = ("s2", "landsat", "modis")
DYNAMIC = (
    "primary_ndvi", "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi",
    "landsat_evi", "landsat_ndwi", "modis_ndvi", "modis_evi",
    "era5_temp_c", "era5_precip_mm", "year", "doy", "ndvi_climatology_mean",
    "ndvi_climatology_std", "ndvi_zscore", "status", "n_reference_years",
)
CONTEXT_COLUMNS = (
    "primary_ndvi", "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi",
    "landsat_evi", "landsat_ndwi", "modis_ndvi", "modis_evi",
    "era5_temp_c", "era5_precip_mm",
)


def _key(polygon: Any, date: Any) -> str:
    return json.dumps([str(polygon), pd.Timestamp(date).strftime("%Y-%m-%d")], ensure_ascii=False)


def _category(value: Any) -> str:
    return "__unknown__" if pd.isna(value) else str(value)


def _frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(KEYS) - set(frame)
    if missing:
        raise ValueError(f"Missing feature keys: {sorted(missing)}")
    result = frame.copy(deep=True)
    if result["anon_polygon_id"].isna().any():
        raise ValueError("Null polygon keys are forbidden")
    result["anon_polygon_id"] = result["anon_polygon_id"].astype(str)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any() or result.duplicated(KEYS).any():
        raise ValueError("Date keys must be non-null and polygon/date keys unique")
    if (result["date"] != result["date"].dt.normalize()).any():
        raise ValueError("Feature dates must be calendar dates without time")
    if "crop_type" not in result:
        result["crop_type"] = "__unknown__"
    result["crop_type"] = result["crop_type"].map(_category)
    return result.reset_index(drop=True)


def _keys(gap_keys: Any, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if isinstance(gap_keys, pd.Series) and pd.api.types.is_bool_dtype(gap_keys.dtype):
        if frame is None or len(gap_keys) != len(frame):
            raise ValueError("Boolean gap mask must match frame length")
        return _frame(frame.loc[gap_keys.to_numpy(), KEYS])[KEYS]
    if isinstance(gap_keys, pd.MultiIndex):
        gap_keys = gap_keys.to_frame(index=False)
        gap_keys.columns = KEYS
    elif not isinstance(gap_keys, pd.DataFrame):
        gap_keys = pd.DataFrame(list(gap_keys), columns=KEYS)
    return _frame(gap_keys[KEYS])[KEYS]


def _numbers(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), np.nan)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return values


def _invalid(values: np.ndarray, name: str) -> np.ndarray:
    present = ~np.isnan(values)
    invalid = present & ~np.isfinite(values)
    if "ndvi" in name or "ndwi" in name:
        invalid |= np.isfinite(values) & ((values < -1) | (values > 1))
    elif name.endswith("_evi"):
        invalid |= np.isfinite(values) & ((values < -1) | (values > 2))
    elif name == "era5_temp_c":
        invalid |= np.isfinite(values) & ((values < -100) | (values > 70))
    elif name == "era5_precip_mm":
        invalid |= np.isfinite(values) & (values < 0)
    return invalid


def _clean(values: np.ndarray, name: str) -> np.ndarray:
    return np.where(_invalid(values, name), np.nan, values)


def infer_source_labels(frame: pd.DataFrame, tolerance: float = 1e-7) -> pd.Series:
    """Recover the raw competition hierarchy, using absolute numeric tolerance."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("source tolerance must be finite and nonnegative")
    target = _numbers(frame, "primary_ndvi")
    labels = np.full(len(frame), "unknown", dtype=object)
    for sensor in SENSORS:
        values = _numbers(frame, f"{sensor}_ndvi")
        match = ((labels == "unknown") & np.isfinite(values) & np.isfinite(target)
                 & np.isclose(values, target, atol=tolerance, rtol=0))
        labels[match] = sensor
    return pd.Series(labels, index=frame.index, name="source_label")


def _bucket(*values: Any) -> str:
    return json.dumps([str(v) for v in values], ensure_ascii=False)


@dataclass(frozen=True)
class FeatureState:
    feature_version: str = FEATURE_VERSION
    # Robust summaries [median, IQR, count] indexed by conditioning variables.
    priors: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    seen_polygons: tuple[str, ...] = ()
    crop_counts: dict[str, int] = field(default_factory=dict)
    fit_keys: tuple[str, ...] = ()
    source_counts: dict[str, int] = field(default_factory=dict)
    global_median: float = 0.5
    global_iqr: float = 0.0
    fit_count: int = 0
    seasonal_bin_days: int = 15

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version, "priors": self.priors,
            "seen_polygons": list(self.seen_polygons), "crop_counts": self.crop_counts,
            "fit_keys": list(self.fit_keys), "source_counts": self.source_counts,
            "global_median": self.global_median, "global_iqr": self.global_iqr,
            "fit_count": self.fit_count, "seasonal_bin_days": self.seasonal_bin_days,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureState":
        payload = dict(payload)
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Incompatible feature state version")
        payload["seen_polygons"] = tuple(payload.get("seen_polygons", ()))
        payload["fit_keys"] = tuple(payload.get("fit_keys", ()))
        state = cls(**payload)
        if not np.isfinite(state.global_median) or state.seasonal_bin_days < 1:
            raise ValueError("Malformed feature state")
        return state


def fit_feature_state(
    fit_frame: pd.DataFrame,
    excluded_keys: Any = None,
    excluded_polygons: Iterable[str] | None = None,
    max_date: Any = None,
) -> FeatureState:
    """Fit robust priors on explicitly allowed labels; this is no ML training.

    CV-B callers must exclude every held-out polygon. Forecasting callers must
    pass a past-only frame or max_date. Context supplied to build_features may
    still contain visible histories of held-out polygons.
    """
    frame = _frame(fit_frame)
    keep = np.isfinite(_numbers(frame, "primary_ndvi"))
    if "is_synthetic_gap" in frame:
        hidden = frame["is_synthetic_gap"].astype(str).str.lower().isin(["true", "1"])
        keep &= ~hidden.to_numpy()
    if excluded_polygons is not None:
        keep &= ~frame["anon_polygon_id"].isin(set(map(str, excluded_polygons))).to_numpy()
    if max_date is not None:
        keep &= (frame["date"] <= pd.Timestamp(max_date)).to_numpy()
    if excluded_keys is not None:
        excluded = _keys(excluded_keys, frame)
        banned = set(map(tuple, excluded.itertuples(index=False, name=None)))
        keep &= np.array([tuple(row) not in banned for row in frame[KEYS].itertuples(index=False, name=None)])
    frame = frame.loc[keep].copy()
    if frame.empty:
        return FeatureState()
    frame["_target"] = _numbers(frame, "primary_ndvi")
    frame["_season"] = (frame["date"].dt.dayofyear - 1) // 15
    frame["_year"] = frame["date"].dt.year
    dimensions = {
        "polygon_crop_season": ["anon_polygon_id", "crop_type", "_season"],
        "polygon_crop": ["anon_polygon_id", "crop_type"],
        "crop_season": ["crop_type", "_season"],
        "crop": ["crop_type"],
        "global_season": ["_season"],
        "polygon_crop_year": ["anon_polygon_id", "crop_type", "_year"],
        "crop_year": ["crop_type", "_year"],
    }
    priors = {}
    for name, columns in dimensions.items():
        table = {}
        for keys, values in frame.groupby(columns, sort=True, observed=True)["_target"]:
            keys = keys if isinstance(keys, tuple) else (keys,)
            array = values.to_numpy(float)
            table[_bucket(*keys)] = [float(np.median(array)), float(np.subtract(*np.percentile(array, [75, 25]))), float(len(array))]
        priors[name] = table
    values = frame["_target"].to_numpy()
    return FeatureState(
        priors=priors,
        seen_polygons=tuple(sorted(frame["anon_polygon_id"].unique())),
        crop_counts={str(k): int(v) for k, v in frame["crop_type"].value_counts().items()},
        fit_keys=tuple(sorted(_key(p, d) for p, d in frame[KEYS].itertuples(index=False, name=None))),
        source_counts={str(k): int(v) for k, v in infer_source_labels(frame).value_counts().items()},
        global_median=float(np.median(values)),
        global_iqr=float(np.subtract(*np.percentile(values, [75, 25]))),
        fit_count=len(frame),
    )


def _prior_features(state: FeatureState, polygon: str, crop: str, date: pd.Timestamp) -> dict[str, Any]:
    season = (date.dayofyear - 1) // state.seasonal_bin_days
    lookup = {
        "polygon_crop_season": (polygon, crop, season), "polygon_crop": (polygon, crop),
        "crop_season": (crop, season), "crop": (crop,), "global_season": (season,),
        "polygon_crop_year": (polygon, crop, date.year), "crop_year": (crop, date.year),
    }
    result = {}
    for name, values in lookup.items():
        value = state.priors.get(name, {}).get(_bucket(*values), [np.nan, np.nan, 0.0])
        for suffix, number in zip(("median", "iqr", "count"), value):
            result[f"prior_{name}_{suffix}"] = number
    hierarchy = ["polygon_crop_season", "polygon_crop", "crop_season", "crop", "global_season"]
    result["seasonal_prior"] = state.global_median
    result["seasonal_prior_level"] = "global_median" if state.fit_count else "unfitted_default"
    for name in hierarchy:
        value = result[f"prior_{name}_median"]
        if np.isfinite(value):
            result["seasonal_prior"] = value
            result["seasonal_prior_level"] = name
            break
    # The crop baseline deliberately ignores polygon effects.
    result["crop_seasonal_prior"] = state.global_median
    for name in ("crop_season", "crop", "global_season"):
        if np.isfinite(result[f"prior_{name}_median"]):
            result["crop_seasonal_prior"] = result[f"prior_{name}_median"]
            break
    result["prior_global_median"] = state.global_median
    result["prior_global_iqr"] = state.global_iqr
    result["seen_polygon"] = int(polygon in state.seen_polygons)
    result["crop_frequency"] = state.crop_counts.get(crop, 0) / max(1, state.fit_count)
    result["fit_target_count"] = state.fit_count
    return result


def _interpolate(days: np.ndarray, values: np.ndarray, query: float) -> tuple[float, float, float, float, float]:
    good = np.isfinite(values)
    x, y = days[good], values[good]
    left = np.flatnonzero(x < query)
    right = np.flatnonzero(x > query)
    li = left[-1] if len(left) else None
    ri = right[0] if len(right) else None
    lv, ld = (float(y[li]), float(query - x[li])) if li is not None else (np.nan, np.nan)
    rv, rd = (float(y[ri]), float(x[ri] - query)) if ri is not None else (np.nan, np.nan)
    if li is not None and ri is not None:
        interpolated = (lv * rd + rv * ld) / (ld + rd)
    elif li is not None:
        interpolated = lv
    else:
        interpolated = rv
    return float(interpolated), lv, rv, ld, rd


def source_feature_columns(features: pd.DataFrame) -> list[str]:
    """An explicit classifier allowlist with no target or vegetation values."""
    calendar = {"anon_polygon_id", "crop_type", "year", "month", "doy", "iso_week", "sin_doy", "cos_doy", "seen_polygon", "crop_frequency"}
    return [c for c in features if c in calendar or c.startswith(("availability_", "source_calendar_"))]


def build_features(frame: pd.DataFrame, gap_keys: Any, fitted_state: FeatureState | dict[str, Any]) -> pd.DataFrame:
    """Build stable features in gap-key order, masking all requested rows first."""
    state = FeatureState.from_dict(fitted_state) if isinstance(fitted_state, dict) else fitted_state
    if state.feature_version != FEATURE_VERSION:
        raise ValueError("Incompatible feature version")
    data = _frame(frame)
    keys = _keys(gap_keys, data)
    index = pd.MultiIndex.from_frame(data[KEYS])
    positions = index.get_indexer(pd.MultiIndex.from_frame(keys))
    if (positions < 0).any():
        raise ValueError("Every requested gap key must exist in frame")
    requested_keys = {_key(p, d) for p, d in keys.itertuples(index=False, name=None)}
    if requested_keys.intersection(state.fit_keys):
        raise ValueError("Target-prior leakage: requested gap keys occur in fitted_state; exclude them before fit")
    hidden = np.zeros(len(data), dtype=bool)
    hidden[positions] = True
    if "is_synthetic_gap" in data:
        hidden |= data["is_synthetic_gap"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    for column in DYNAMIC:
        if column in data:
            if pd.api.types.is_integer_dtype(data[column]):
                data[column] = data[column].astype(float)
            elif not pd.api.types.is_numeric_dtype(data[column]):
                data[column] = data[column].astype(object)
            data.loc[hidden, column] = np.nan
    data["_hidden"] = hidden
    data["_source"] = infer_source_labels(data).to_numpy()
    # pandas 2/3 may store datetime64 in ns OR us; convert units explicitly.
    data["_days"] = data["date"].to_numpy(dtype="datetime64[D]").astype("int64")
    for column in CONTEXT_COLUMNS:
        data[column] = _numbers(data, column)
    groups = {str(p): g.sort_values("date").reset_index(drop=True) for p, g in data.groupby("anon_polygon_id", sort=False)}
    dates = {d: g for d, g in data.groupby("date", sort=False)}
    output = []
    for polygon, date in keys.itertuples(index=False, name=None):
        group = groups[polygon]
        query_row = group.loc[group["date"] == date].iloc[0]
        crop = query_row["crop_type"]
        query = float(query_row["_days"])
        days = group["_days"].to_numpy(float)
        target = group["primary_ndvi"].to_numpy(float)
        valid = np.isfinite(target)
        left = np.flatnonzero(valid & (days < query))[::-1]
        right = np.flatnonzero(valid & (days > query))
        row: dict[str, Any] = {"anon_polygon_id": polygon, "date": date, "crop_type": crop}
        row.update(year=date.year, month=date.month, doy=date.dayofyear, iso_week=int(date.isocalendar().week),
                   sin_doy=float(np.sin(2 * np.pi * date.dayofyear / 365.25)),
                   cos_doy=float(np.cos(2 * np.pi * date.dayofyear / 365.25)))
        row.update(_prior_features(state, polygon, crop, date))
        for side, indices in (("left", left), ("right", right)):
            for k in range(1, 5):
                value, delta = (float(target[indices[k - 1]]), float(abs(days[indices[k - 1]] - query))) if len(indices) >= k else (np.nan, np.nan)
                invalid = bool(_invalid(np.array([value]), "primary_ndvi")[0])
                row[f"target_{side}_{k}"] = value
                row[f"target_{side}_{k}_raw"] = value
                row[f"target_{side}_{k}_cleaned"] = np.nan if invalid else value
                row[f"target_{side}_{k}_invalid_flag"] = int(invalid)
                row[f"target_{side}_{k}_missing_flag"] = int(not np.isfinite(value))
                row[f"days_{side}_{k}"] = delta
        linear, lv, rv, ld, rd = _interpolate(days, target, query)
        row["linear_interpolation"] = linear
        row["mean_neighbors"] = (lv + rv) / 2 if np.isfinite(lv) and np.isfinite(rv) else linear
        row["local_slope"] = (rv - lv) / (ld + rd) if np.isfinite(ld + rd) else np.nan
        near = np.concatenate((left[:4], right[:4]))
        values = target[near]
        row["local_level"] = float(np.median(values)) if len(values) else np.nan
        row["local_iqr"] = float(np.subtract(*np.percentile(values, [75, 25]))) if len(values) else np.nan
        previous_slope = ((lv - target[left[1]]) / (days[left[0]] - days[left[1]])) if len(left) > 1 else np.nan
        next_slope = ((target[right[1]] - rv) / (days[right[1]] - days[right[0]])) if len(right) > 1 else np.nan
        row["local_curvature"] = float(next_slope - previous_slope) if np.isfinite(next_slope + previous_slope) else np.nan
        for window in (7, 15, 30):
            count = int((valid & (np.abs(days - query) <= window)).sum())
            row[f"target_count_{window}d"] = count
            row[f"target_density_{window}d"] = count / (2 * window)
        # Consecutive potential target slots, ignoring natural missing daily rows.
        masked = group["_hidden"].to_numpy(bool)
        potential = np.flatnonzero((valid | masked) & group["date"].dt.year.eq(date.year).to_numpy())
        qpos = int(np.flatnonzero(days[potential] == query)[0])
        run_left = run_right = qpos
        while run_left > 0 and masked[potential[run_left - 1]]:
            run_left -= 1
        while run_right + 1 < len(potential) and masked[potential[run_right + 1]]:
            run_right += 1
        row.update(gap_run_length=run_right - run_left + 1, gap_run_position=qpos - run_left + 1,
                   left_missing=int(not len(left)), right_missing=int(not len(right)),
                   is_edge=int(not len(left) or not len(right)),
                   context_quality=float((int(bool(len(left))) + int(bool(len(right)))) / 2 / (1 + min(ld if np.isfinite(ld) else np.inf, rd if np.isfinite(rd) else np.inf) / 15)))
        same_date = dates[date]
        other = same_date.loc[same_date["anon_polygon_id"] != polygon]
        cross = _numbers(other, "primary_ndvi")
        cross = cross[np.isfinite(cross)]
        row["same_date_target_count"] = len(cross)
        row["same_date_target_median"] = float(np.median(cross)) if len(cross) else np.nan
        row["same_date_target_iqr"] = float(np.subtract(*np.percentile(cross, [75, 25]))) if len(cross) else np.nan
        for column in CONTEXT_COLUMNS:
            raw = group[column].to_numpy(float)
            cleaned = _clean(raw, column)
            own = float(query_row[column])
            own_bad = bool(_invalid(np.array([own]), column)[0])
            row[f"{column}_raw"] = own
            row[f"{column}_cleaned"] = np.nan if own_bad else own
            row[f"{column}_invalid_flag"] = int(own_bad)
            row[f"{column}_missing_flag"] = int(not np.isfinite(own))
            # Primary context uses unaltered finite competition targets.
            context_values = raw if column == "primary_ndvi" else cleaned
            interp, prev, nxt, since, until = _interpolate(days, context_values, query)
            row[f"{column}_interpolated"] = interp
            row[f"{column}_left"] = prev
            row[f"{column}_right"] = nxt
            row[f"{column}_days_since"] = since
            row[f"{column}_days_until"] = until
            window_mask = (np.abs(days - query) <= 30) & (days != query)
            summary = cleaned[window_mask & np.isfinite(cleaned)]
            row[f"{column}_context_median"] = float(np.median(summary)) if len(summary) else np.nan
            row[f"{column}_context_iqr"] = float(np.subtract(*np.percentile(summary, [75, 25]))) if len(summary) else np.nan
            row[f"{column}_context_count"] = len(summary)
            row[f"{column}_context_invalid_count"] = int((_invalid(raw, column) & window_mask).sum())
            if column.startswith("era5_"):
                regional = _clean(_numbers(other, column), column)
                regional = regional[np.isfinite(regional)]
                same_value = float(np.median(regional)) if len(regional) else np.nan
                row[f"{column}_same_date"] = same_value
                row[f"{column}_same_date_count"] = len(regional)
                row[f"{column}_reconstructed"] = same_value if np.isfinite(same_value) else interp
                row[f"{column}_available"] = int(np.isfinite(row[f"{column}_reconstructed"]))
        context_source = group.loc[valid & (np.abs(days - query) <= 30) & (days != query), "_source"]
        date_source = other.loc[np.isfinite(_numbers(other, "primary_ndvi")), "_source"]
        counts = []
        for sensor in SENSORS:
            observed = np.isfinite(group[f"{sensor}_ndvi"].to_numpy(float))
            _, _, _, since, until = _interpolate(days, np.where(observed, 1.0, np.nan), query)
            row[f"availability_{sensor}_days_since"] = since
            row[f"availability_{sensor}_days_until"] = until
            row[f"availability_{sensor}_same_date_count"] = int(np.isfinite(_numbers(other, f"{sensor}_ndvi")).sum())
            for window in (7, 15, 30):
                row[f"availability_{sensor}_count_{window}d"] = int((observed & (np.abs(days - query) <= window) & (days != query)).sum())
            local_count = int((context_source == sensor).sum())
            date_count = int((date_source == sensor).sum())
            row[f"source_calendar_{sensor}_local_count"] = local_count
            row[f"source_calendar_{sensor}_same_date_count"] = date_count
            row[f"source_calendar_{sensor}_fit_count"] = state.source_counts.get(sensor, 0)
            prior_count = state.source_counts.get(sensor, 0) / max(1, sum(state.source_counts.values()))
            counts.append(local_count + date_count + prior_count + .25)
        counts.append(int((context_source == "unknown").sum()) + int((date_source == "unknown").sum())
                      + state.source_counts.get("unknown", 0) / max(1, sum(state.source_counts.values())) + .25)
        probability = np.array(counts, dtype=float) / sum(counts)
        for sensor, value in zip((*SENSORS, "unknown"), probability):
            row[f"p_{sensor}"] = float(value)
        row["source_confidence"] = float(probability.max())
        row["source_unknown_local_count"] = int((context_source == "unknown").sum())
        row["pchip_interpolation"] = np.nan
        row["akima_interpolation"] = np.nan
        if len(left) and len(right) and len(near) >= 3:
            order = near[np.argsort(days[near])]
            try:
                from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
                row["pchip_interpolation"] = float(PchipInterpolator(days[order], target[order], extrapolate=False)(query))
                if len(near) >= 5:
                    row["akima_interpolation"] = float(Akima1DInterpolator(days[order], target[order], extrapolate=False)(query))
            except (ImportError, ValueError, FloatingPointError):
                pass
        output.append(row)
    if not output:
        return pd.DataFrame(columns=KEYS)
    result = pd.DataFrame(output)
    for side in ("left", "right"):
        for k in range(1, 5): result[f"{side}_days_{k}"] = result[f"days_{side}_{k}"]
    result.attrs["feature_version"] = FEATURE_VERSION
    result.attrs["source_probability_method"] = "visible_context_counts_with_fit_frequency_prior; uncalibrated"
    return result
