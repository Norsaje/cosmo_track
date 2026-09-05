"""CPU anomaly pipeline: robust sensor mapping, LOYO climatology, negative events.

Reference передаётся явно из train fold. Текущий оцениваемый год исключается
целиком, включая другие полигоны. Нет чтения исходной CSV climatology/status.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from statistics import NormalDist

import numpy as np
import pandas as pd

from .events import AnomalyEvent, DetectionResult
from .explain import explain_ru


def _validate(frame: pd.DataFrame, *, harmonized: bool = True) -> pd.DataFrame:
    required = {"anon_polygon_id", "date", "crop_type"}
    if harmonized:
        required |= {"ndvi_harmonized", "is_observed"}
    if not required.issubset(frame):
        raise ValueError(f"Required columns: {sorted(required)}")
    out = frame.copy(deep=True).reset_index(drop=True)
    if out[list(required - {"ndvi_harmonized"})].isna().any().any():
        raise ValueError("Missing keys, categories or observation flags")
    out["date"] = pd.to_datetime(out.date, format="%Y-%m-%d", errors="raise")
    if out.date.dt.tz is not None or not out.date.eq(out.date.dt.normalize()).all():
        raise ValueError("Expected timezone-free calendar days")
    if out.duplicated(["anon_polygon_id", "date"]).any():
        raise ValueError("Duplicate polygon/date")
    if harmonized:
        if not out.is_observed.isin([True, False]).all():
            raise ValueError("is_observed must be boolean")
        out["is_observed"] = out.is_observed.astype(bool)
        out["ndvi_harmonized"] = pd.to_numeric(out.ndvi_harmonized, errors="raise")
        if np.isinf(out.ndvi_harmonized).any():
            raise ValueError("Infinite NDVI")
    return out


@dataclass(frozen=True)
class AnomalyConfig:
    doy_radius: int = 10
    min_reference_years: int = 3
    mad_floor: float = 0.04
    pooling_years: float = 3.0
    suppression_z: float = -1.0
    critical_z: float = -2.0
    max_gap_days: int = 10
    min_event_points: int = 2
    min_event_days: int = 3
    critical_min_observed: int = 2
    critical_min_days: int = 7
    critical_confidence: float = 0.55
    min_quality: float = 0.3
    uncertainty_interval_level: float = 0.8

    def __post_init__(self):
        if (
            not 0 <= self.doy_radius <= 60
            or self.min_reference_years < 2
            or self.mad_floor <= 0
            or self.pooling_years <= 0
            or not self.critical_z < self.suppression_z < 0
            or self.max_gap_days < 1
            or self.min_event_points < 2
            or self.min_event_days < 1
            or self.critical_min_observed < 2
            or self.critical_min_days < self.min_event_days
            or not 0 <= self.min_quality <= 1
            or not 0 <= self.critical_confidence <= 1
            or not 0 < self.uncertainty_interval_level < 1
            or not all(np.isfinite(v) for v in asdict(self).values())
        ):
            raise ValueError("Invalid anomaly configuration")


class SensorHarmonizer:
    """Robust affine mapping на синхронном overlap, с pooling global/crop/polygon.

    Отсутствие overlap не выдаётся за успешную калибровку: raw сохраняется,
    harmonized=NaN и calibration_supported=False для неизвестного источника.
    Near-date и quantile mapping оставлены для OOF ablation, не P0 default.
    """

    def __init__(self, *, min_pairs: int = 20, pooling_pairs: float = 40.0):
        if min_pairs < 3 or pooling_pairs <= 0:
            raise ValueError("Insufficient calibration support configuration")
        self.min_pairs = min_pairs
        self.pooling_pairs = pooling_pairs
        self.mappings: dict[tuple[str, str, str], tuple[float, float, int, float]] = {}
        self.fingerprint: str | None = None

    @staticmethod
    def _affine(x, y):
        a = np.column_stack([x, np.ones(len(x))])
        coef = np.linalg.lstsq(a, y, rcond=None)[0]
        for _ in range(20):
            resid = y - a @ coef
            scale = max(
                float(1.4826 * np.median(np.abs(resid - np.median(resid)))), 0.005
            )
            w = np.minimum(1, 1.345 * scale / np.maximum(np.abs(resid), 1e-8))
            updated = np.linalg.lstsq(
                a * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None
            )[0]
            if np.max(np.abs(updated - coef)) < 1e-8:
                coef = updated
                break
            coef = updated
        if not np.isfinite(coef).all() or not 0.2 <= coef[0] <= 3:
            return None
        return float(coef[0]), float(coef[1]), len(x), scale

    def fit(self, reference: pd.DataFrame):
        ref = _validate(reference, harmonized=False)
        if "s2_ndvi" not in ref:
            raise ValueError("Reference sensor s2_ndvi is required")
        if "is_synthetic_gap" in ref:
            if not ref.is_synthetic_gap.isin([True, False]).all():
                raise ValueError("is_synthetic_gap must be boolean")
            ref = ref.loc[~ref.is_synthetic_gap.astype(bool)]
        self.mappings = {}
        for source in ("landsat", "modis"):
            col = source + "_ndvi"
            if col not in ref:
                continue
            usable = ref.loc[
                np.isfinite(ref.s2_ndvi)
                & np.isfinite(ref[col])
                & ref.s2_ndvi.between(-1, 1)
                & ref[col].between(-1, 1)
            ]
            groups = [("global", "*", usable)]
            groups.extend(("crop", str(k), g) for k, g in usable.groupby("crop_type"))
            groups.extend(
                ("polygon", str(k), g) for k, g in usable.groupby("anon_polygon_id")
            )
            for level, key, group in groups:
                if len(group) < self.min_pairs or group[col].std() < 0.02:
                    continue
                result = self._affine(
                    group[col].to_numpy(float), group.s2_ndvi.to_numpy(float)
                )
                if result is not None:
                    self.mappings[source, level, key] = result
        cols = [
            c
            for c in (
                "anon_polygon_id",
                "date",
                "crop_type",
                "s2_ndvi",
                "landsat_ndvi",
                "modis_ndvi",
            )
            if c in ref
        ]
        self.fingerprint = sha256(
            ref[cols]
            .sort_values(["anon_polygon_id", "date"])
            .to_csv(index=False)
            .encode()
        ).hexdigest()
        return self

    def _mapping(self, source, crop, polygon):
        if source == "s2":
            return 1.0, 0.0, True
        global_fit = self.mappings.get((source, "global", "*"))
        if global_fit is None:
            return 1.0, 0.0, False
        slope, offset = global_fit[:2]
        for level, key in (("crop", str(crop)), ("polygon", str(polygon))):
            local = self.mappings.get((source, level, key))
            if local is not None:
                weight = local[2] / (local[2] + self.pooling_pairs)
                slope = weight * local[0] + (1 - weight) * slope
                offset = weight * local[1] + (1 - weight) * offset
        return slope, offset, True

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.fingerprint is None:
            raise RuntimeError("Fit calibration on train reference first")
        out = _validate(frame, harmonized=False)
        if not {"primary_ndvi", "selected_source"}.issubset(out):
            raise ValueError("Explicit primary_ndvi and selected_source are required")
        mapped, supported, slopes, offsets = [], [], [], []
        for row in out.itertuples():
            slope, offset, ok = self._mapping(
                row.selected_source, row.crop_type, row.anon_polygon_id
            )
            value = float(row.primary_ndvi)
            mapped.append(
                slope * value + offset if ok and np.isfinite(value) else np.nan
            )
            supported.append(ok)
            slopes.append(slope if ok else np.nan)
            offsets.append(offset if ok else np.nan)
        out["primary_ndvi_raw"] = out.primary_ndvi
        out["ndvi_harmonized"] = mapped
        out["calibration_supported"] = supported
        out["calibration_version"] = "robust-affine-0.1"
        out["calibration_fingerprint"] = self.fingerprint
        for field in ("uncertainty_std", "lower", "upper"):
            if field in out:
                raw = pd.to_numeric(out[field], errors="raise")
                out["raw_" + field] = raw
                out[field] = raw * np.asarray(slopes)
                if field != "uncertainty_std":
                    out[field] += np.asarray(offsets)
        return out

    def to_dict(self):
        if self.fingerprint is None:
            raise RuntimeError("Harmonizer is not fitted")
        return {
            "schema_version": "0.1",
            "reference_sensor": "s2",
            "min_pairs": self.min_pairs,
            "pooling_pairs": self.pooling_pairs,
            "fingerprint": self.fingerprint,
            "mappings": [
                {"source": k[0], "level": k[1], "key": k[2], "fit": list(v)}
                for k, v in sorted(self.mappings.items())
            ],
        }

    @classmethod
    def from_dict(cls, state):
        if (
            state.get("schema_version") != "0.1"
            or state.get("reference_sensor") != "s2"
        ):
            raise ValueError("Unsupported sensor calibration schema/reference")
        result = cls(min_pairs=state["min_pairs"], pooling_pairs=state["pooling_pairs"])
        fingerprint = state.get("fingerprint", "")
        if len(fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in fingerprint
        ):
            raise ValueError("Invalid calibration fingerprint")
        for entry in state["mappings"]:
            source, level, key = entry["source"], entry["level"], entry["key"]
            fit = entry["fit"]
            if (
                source not in {"modis", "landsat"}
                or level not in {"global", "crop", "polygon"}
                or not isinstance(key, str)
                or len(fit) != 4
                or not np.isfinite(fit).all()
                or not 0.2 <= fit[0] <= 3
                or fit[2] < result.min_pairs
                or fit[3] <= 0
            ):
                raise ValueError("Invalid sensor calibration mapping")
            if (source, level, key) in result.mappings:
                raise ValueError("Duplicate calibration mapping")
            result.mappings[source, level, key] = tuple(fit)
        result.fingerprint = fingerprint
        return result


def _season_day(dates):
    # Одинаковые month/day имеют одинаковую координату в високосные годы.
    return pd.to_datetime(
        "2000-" + dates.dt.strftime("%m-%d"), format="%Y-%m-%d"
    ).dt.dayofyear


class RobustClimatology:
    def __init__(self, config: AnomalyConfig | None = None):
        self.config = config or AnomalyConfig()
        self.reference: pd.DataFrame | None = None

    def fit(self, reference: pd.DataFrame):
        ref = _validate(reference)
        ref = ref.loc[ref.is_observed & np.isfinite(ref.ndvi_harmonized)].copy()
        if "quality" in ref:
            q = pd.to_numeric(ref.quality, errors="raise")
            ref = ref.loc[np.isfinite(q) & q.between(self.config.min_quality, 1)]
        if "calibration_supported" in ref:
            if not ref.calibration_supported.isin([True, False]).all():
                raise ValueError("calibration_supported must be boolean")
            ref = ref.loc[ref.calibration_supported.astype(bool)]
        ref["_year"] = ref.date.dt.year
        ref["_season_day"] = _season_day(ref.date)
        self.reference = ref
        return self

    def _summary(self, ref):
        # Один median на polygon-year: частые съёмки не увеличивают число независимых лет.
        annual = ref.groupby(["anon_polygon_id", "_year"]).ndvi_harmonized.median()
        years = ref._year.nunique()
        if years < self.config.min_reference_years:
            return None
        # Каждый reference year получает одинаковый вес, независимо от числа полигонов.
        annual = annual.groupby(level="_year").median().to_numpy(float)
        quantiles = np.quantile(annual, [0.1, 0.25, 0.5, 0.75, 0.9])
        scale = max(
            1.4826 * float(np.median(np.abs(annual - np.median(annual)))),
            self.config.mad_floor,
        )
        return quantiles, scale, int(years)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.reference is None:
            raise RuntimeError("Fit climatology on explicit reference first")
        query = _validate(frame)
        rows = []
        cache = {}
        days = _season_day(query.date)
        for row, day in zip(query.itertuples(), days):
            key = row.anon_polygon_id, row.crop_type, row.date.year, int(day)
            if key in cache:
                rows.append(cache[key])
                continue
            ref = self.reference
            distance = np.abs(ref._season_day - day)
            ref = ref.loc[
                (ref._year != row.date.year)
                & (np.minimum(distance, 366 - distance) <= self.config.doy_radius)
            ]
            global_summary = self._summary(ref)
            if global_summary is None:
                result = {
                    "expected_ndvi": np.nan,
                    "climatology_scale": np.nan,
                    "reference_years": 0,
                    "reference_level": "insufficient",
                }
                result.update({f"q{q}": np.nan for q in (10, 25, 50, 75, 90)})
            else:
                quantiles, scale, years = global_summary
                level = "global"
                for name, subset in (
                    ("crop", ref.loc[ref.crop_type == row.crop_type]),
                    (
                        "polygon",
                        ref.loc[
                            (ref.anon_polygon_id == row.anon_polygon_id)
                            & (ref.crop_type == row.crop_type)
                        ],
                    ),
                ):
                    local = self._summary(subset)
                    if local is not None:
                        weight = local[2] / (local[2] + self.config.pooling_years)
                        quantiles = weight * local[0] + (1 - weight) * quantiles
                        scale = weight * local[1] + (1 - weight) * scale
                        years, level = local[2], name
                result = {
                    "expected_ndvi": float(quantiles[2]),
                    "climatology_scale": float(scale),
                    "reference_years": years,
                    "reference_level": level,
                }
                result.update(
                    {f"q{q}": float(v) for q, v in zip((10, 25, 50, 75, 90), quantiles)}
                )
            cache[key] = result
            rows.append(result)
        result = pd.DataFrame(rows, index=query.index)
        # Исходные climatology поля никогда не используются как priors.
        query = query.drop(columns=result.columns.intersection(query.columns))
        return pd.concat([query, result], axis=1)


class AdvancedAnomalyDetector:
    def __init__(self, config: AnomalyConfig | None = None):
        self.config = config or AnomalyConfig()
        self.climatology = RobustClimatology(self.config)

    def fit(self, reference: pd.DataFrame):
        self.climatology.fit(reference)
        return self

    def score_points(self, frame: pd.DataFrame) -> pd.DataFrame:
        points = (
            self.climatology.predict(frame)
            .sort_values(["anon_polygon_id", "date"])
            .reset_index(drop=True)
        )
        if points.empty:
            return points
        quality = pd.to_numeric(
            points.get("quality", pd.Series(1.0, index=points.index)), errors="raise"
        )
        if (~quality.between(0, 1) & quality.notna()).any():
            raise ValueError("quality must be in [0, 1]")
        points["quality"] = quality.fillna(0)
        sigma = pd.Series(np.nan, index=points.index, dtype=float)
        if "uncertainty_std" in points:
            sigma = pd.to_numeric(points.uncertainty_std, errors="raise")
        elif {"lower", "upper"}.issubset(points):
            width = pd.to_numeric(points.upper, errors="raise") - pd.to_numeric(
                points.lower, errors="raise"
            )
            sigma = width / (
                2
                * NormalDist().inv_cdf((1 + self.config.uncertainty_interval_level) / 2)
            )
        if (sigma < 0).any() or np.isinf(sigma).any():
            raise ValueError("Uncertainty must be nonnegative and finite when supplied")
        unknown_uncertainty = ~points.is_observed & sigma.isna()
        sigma = sigma.fillna(0)
        points["residual"] = points.ndvi_harmonized - points.expected_ndvi
        points["robust_z"] = points.residual / points.climatology_scale
        combined = np.sqrt(points.climatology_scale**2 + sigma**2)
        points["effective_z"] = points.residual / combined
        agreement = pd.Series(0.0, index=points.index)
        if "sensor_negative_count" in points:
            count = pd.to_numeric(points.sensor_negative_count, errors="raise")
            if not (count.isna() | ((count >= 0) & (count % 1 == 0))).all():
                raise ValueError("sensor_negative_count must be a nonnegative integer")
            agreement = (count >= 2).astype(float)
        points["multisensor_confirmation"] = agreement.astype(bool)
        switches = pd.Series(False, index=points.index)
        if "selected_source" in points:
            previous = points.groupby("anon_polygon_id").selected_source.shift()
            switches = (
                previous.notna()
                & points.selected_source.notna()
                & points.selected_source.ne(previous)
            )
        if "source_switch_risk" in points:
            if not points.source_switch_risk.isin([True, False]).all():
                raise ValueError("source_switch_risk must be boolean")
            switches |= points.source_switch_risk.astype(bool)
        points["source_switch_risk"] = switches
        reference_weight = points.reference_level.map(
            {"polygon": 1.0, "crop": 0.85, "global": 0.5, "insufficient": 0.0}
        )
        confidence = (
            points.quality
            * reference_weight
            * (points.climatology_scale / combined).fillna(0)
        )
        confidence *= np.where(points.is_observed, 1.0, 0.5)
        confidence *= np.where(switches & ~agreement.astype(bool), 0.6, 1.0)
        confidence = np.where(
            unknown_uncertainty, np.minimum(confidence, 0.2), confidence
        )
        points["confidence"] = np.clip(confidence, 0, 1)
        supported = pd.Series(True, index=points.index)
        if "calibration_supported" in points:
            if not points.calibration_supported.isin([True, False]).all():
                raise ValueError("calibration_supported must be boolean")
            supported = points.calibration_supported.astype(bool)
        points["candidate"] = (
            points.effective_z.lt(self.config.suppression_z)
            & points.quality.ge(self.config.min_quality)
            & supported
            & ~unknown_uncertainty
        )
        return points

    def detect(self, frame: pd.DataFrame) -> DetectionResult:
        if frame.empty:
            return DetectionResult((), ("EMPTY_SERIES",))
        if "anon_polygon_id" not in frame or frame.anon_polygon_id.nunique() != 1:
            raise ValueError(
                "detect accepts exactly one polygon; call separately per polygon"
            )
        points = self.score_points(frame)
        warnings = []
        if (points.reference_years < self.config.min_reference_years).any():
            warnings.append("INSUFFICIENT_REFERENCE_YEARS")
        if points.ndvi_harmonized.isna().any():
            warnings.append("MISSING_HARMONIZED_VALUES")
        if (~points.is_observed & points.confidence.le(0.2)).any():
            warnings.append("LOW_RECONSTRUCTION_SUPPORT")
        groups, current = [], []
        for index, row in points.iterrows():
            if row.candidate:
                if current and (
                    (row.date - points.loc[current[-1], "date"]).days
                    > self.config.max_gap_days
                    or row.date.year != points.loc[current[-1], "date"].year
                ):
                    groups.append(current)
                    current = []
                current.append(index)
            elif (
                np.isfinite(row.effective_z) and row.quality >= self.config.min_quality
            ):
                # Нормальное надёжное наблюдение прерывает событие, даже в пределах max_gap.
                if current:
                    groups.append(current)
                    current = []
        if current:
            groups.append(current)
        events = []
        for ids in groups:
            event = self._event(points.loc[ids])
            if event is not None:
                events.append(event)
        return DetectionResult(tuple(events), tuple(warnings))

    def _event(self, rows):
        cfg = self.config
        start, end = rows.date.iloc[0].date(), rows.date.iloc[-1].date()
        duration = (end - start).days + 1
        if len(rows) < cfg.min_event_points or duration < cfg.min_event_days:
            return None
        observed = int(rows.is_observed.sum())
        reconstructed = len(rows) - observed
        confidence = float(rows.confidence.mean())
        support = min(1.0, len(rows) / 3)
        confidence *= support
        severity = "biomass_suppression"
        # Critical требует устойчивости и реальных наблюдений, а не единственного min z.
        if (
            float(rows.effective_z.median()) < cfg.critical_z
            and observed >= cfg.critical_min_observed
            and duration >= cfg.critical_min_days
            and confidence >= cfg.critical_confidence
        ):
            severity = "critical"
        elapsed = (rows.date - rows.date.iloc[0]).dt.total_seconds().to_numpy() / 86400
        depth = np.maximum(-rows.residual.to_numpy(), 0)
        area = float(np.sum((depth[:-1] + depth[1:]) * np.diff(elapsed) / 2))
        codes = []
        for field, code, predicate in (
            ("precip_30d_ratio", "LOW_PRECIPITATION", lambda x: x < 0.5),
            ("temp_anomaly_c", "HIGH_TEMPERATURE", lambda x: x > 3),
            ("ndwi_robust_z", "LOW_NDWI", lambda x: x < -1),
        ):
            if field in rows:
                values = pd.to_numeric(rows[field], errors="raise")
                finite = values[np.isfinite(values)]
                if len(finite) >= max(2, int(np.ceil(len(rows) * 0.7))) and predicate(
                    float(finite.median())
                ):
                    codes.append(code)
        if rows.multisensor_confirmation.sum() >= 2:
            codes.append("MULTISENSOR_CONFIRMATION")
        if rows.source_switch_risk.any():
            codes.append("SOURCE_SWITCH_RISK")
        if confidence < cfg.critical_confidence or reconstructed > observed:
            codes.append("LOW_DATA_COVERAGE")
        if duration >= 14:
            codes.append("PROLONGED_SUPPRESSION")
        if len(rows) >= 2:
            slope = np.diff(rows.residual) / np.maximum(np.diff(elapsed), 1)
            if np.min(slope) < -0.025:
                codes.append("RAPID_NEGATIVE_CHANGE")
        codes = tuple(codes)
        return AnomalyEvent(
            start,
            end,
            severity,
            area * confidence,
            confidence,
            float(rows.robust_z.min()),
            area,
            observed,
            reconstructed,
            codes,
            explain_ru(codes),
        )

    @property
    def config_fingerprint(self):
        return sha256(
            json.dumps(asdict(self.config), sort_keys=True).encode()
        ).hexdigest()
