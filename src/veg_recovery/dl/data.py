"""Окна C-06 draft. До C-01 используем явно обозначенный CSV fallback.

Маскирование применяется ко всему контексту до построения любых окон.
Labels передаются отдельно: соседнее окно не может раскрыть held-out target.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

KEY = ["anon_polygon_id", "date"]
CHANNELS = (
    "primary_ndvi",
    "s2_ndvi",
    "landsat_ndvi",
    "modis_ndvi",
    "s2_evi",
    "s2_ndwi",
    "landsat_evi",
    "landsat_ndwi",
    "modis_evi",
    "era5_temp_c",
    "era5_precip_mm",
)
STATIC = {*KEY, "crop_type", "is_synthetic_gap"}


def canonical_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Строгие уникальные ключи; порядок входа сохраняется."""
    if not set(KEY).issubset(frame):
        raise ValueError(f"Required keys: {KEY}")
    out = frame[KEY].copy().reset_index(drop=True)
    if (
        out.isna().any().any()
        or not out.anon_polygon_id.map(
            lambda x: isinstance(x, str) and bool(x.strip())
        ).all()
    ):
        raise ValueError("Keys must contain nonempty polygon strings and dates")
    dates = pd.to_datetime(out.date, format="%Y-%m-%d", errors="raise")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be timezone-free calendar days")
    if not pd.api.types.is_datetime64_any_dtype(out.date):
        if not out.date.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError("Dates must use YYYY-MM-DD")
    out["date"] = dates
    if out.duplicated(KEY).any():
        raise ValueError("Duplicate polygon/date keys")
    return out


def key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(canonical_keys(frame))


def _frame(frame: pd.DataFrame) -> pd.DataFrame:
    keys = canonical_keys(frame)
    out = frame.copy().reset_index(drop=True)
    out[KEY] = keys
    if "primary_ndvi" not in out or "crop_type" not in out:
        raise ValueError("primary_ndvi and crop_type are required")
    if out.crop_type.isna().any():
        raise ValueError("crop_type cannot be missing")
    return out


def labels_for(frame: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Вызывается только для разрешённых train/validation labels, не test gaps."""
    original = _frame(frame).set_index(KEY)
    selected = canonical_keys(keys)
    idx = key_index(selected)
    if not idx.isin(original.index).all():
        raise ValueError("Label keys absent from frame")
    values = original.loc[idx, "primary_ndvi"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cannot supervise naturally missing or hidden test targets")
    return selected.assign(primary_ndvi=values)


@dataclass
class MaskedContext:
    frame: pd.DataFrame
    natural_missing: pd.Series
    synthetic_gap: pd.Series
    supervision_eligible: pd.Series
    masking_backend: str


def prepare_context(
    frame: pd.DataFrame,
    hidden_keys: pd.DataFrame,
    *,
    apply_mask: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] | None = None,
) -> MaskedContext:
    """Consumer shared apply_mask(frame, keys); fallback не генерирует фолды."""
    original = _frame(frame)
    idx = key_index(original)
    hidden = canonical_keys(hidden_keys)
    if not key_index(hidden).isin(idx).all():
        raise ValueError("Hidden keys absent from context")
    flag = original.get("is_synthetic_gap", pd.Series(False, index=original.index))
    if flag.isna().any() or not flag.isin([True, False]).all():
        raise ValueError("is_synthetic_gap must be boolean, never string truthiness")
    synthetic = idx.isin(key_index(hidden)) | flag.to_numpy(dtype=bool)
    finite = np.isfinite(pd.to_numeric(original.primary_ndvi, errors="raise"))
    natural = (~finite & ~flag.astype(bool)).copy()
    all_hidden = original.loc[synthetic, KEY]
    if apply_mask is not None:
        masked = _frame(apply_mask(original.copy(deep=True), all_hidden.copy()))
        if not key_index(masked).equals(idx):
            raise ValueError("Shared mask changed context key order")
        if not masked.crop_type.equals(original.crop_type):
            raise ValueError("Shared mask changed static crop_type")
        backend = "shared"
    else:
        masked = original.copy(deep=True)
        # Любое новое динамическое поле также скрывается; allowlist обратного вида.
        for col in masked.columns.difference(list(STATIC)):
            masked.loc[synthetic, col] = np.nan
        backend = "csv_fallback_pending_C01_C03"
    dynamic = masked.columns.difference(list(STATIC))
    if masked.loc[synthetic, dynamic].notna().any().any():
        raise ValueError("Mask leaked dynamic fields at a hidden key")
    masked["is_synthetic_gap"] = synthetic
    return MaskedContext(
        masked, natural, pd.Series(synthetic), finite & ~flag.astype(bool), backend
    )


def _clean(
    frame: pd.DataFrame, channels: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    x = (
        frame.reindex(columns=list(channels))
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(float)
    )
    invalid = np.isinf(x)
    for j, col in enumerate(channels):
        # Конкурсный target не clip: физические проверки относятся к sensor context.
        if col != "primary_ndvi" and col.endswith(("_ndvi", "_ndwi")):
            invalid[:, j] |= np.isfinite(x[:, j]) & (np.abs(x[:, j]) > 1)
        elif col.endswith("_evi"):
            invalid[:, j] |= np.isfinite(x[:, j]) & ((x[:, j] < -1) | (x[:, j] > 2))
        elif col == "era5_precip_mm":
            invalid[:, j] |= np.isfinite(x[:, j]) & (x[:, j] < 0)
    x[invalid] = np.nan
    return x, invalid


@dataclass(frozen=True)
class WindowPreprocessor:
    channels: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    crops: tuple[str, ...]
    fit_keys_hash: str
    schema_version: str = "0.1"

    @classmethod
    def fit(cls, context: MaskedContext, fit_keys: pd.DataFrame, channels=CHANNELS):
        channels = tuple(channels)
        if (
            not channels
            or channels[0] != "primary_ndvi"
            or not set(channels).issubset(CHANNELS)
        ):
            raise ValueError(
                "Only audited raw channels, with primary_ndvi first, are allowed"
            )
        if len(channels) != len(set(channels)):
            raise ValueError("Duplicate channels")
        idx = key_index(fit_keys)
        if len(idx) == 0 or not idx.isin(key_index(context.frame)).all():
            raise ValueError("Explicit nonempty fit keys must belong to context")
        fit = context.frame.set_index(KEY).loc[idx]
        x, _ = _clean(fit, channels)
        centers, scales = [], []
        for values in x.T:
            observed = values[np.isfinite(values)]
            centers.append(float(np.median(observed)) if len(observed) else 0.0)
            scales.append(max(float(np.std(observed)), 0.01) if len(observed) else 1.0)
        if not np.isfinite(x[:, 0]).any():
            raise ValueError(
                "Fit context has no visible primary_ndvi for fallback/scaling"
            )
        payload = canonical_keys(fit_keys).sort_values(KEY).to_csv(index=False)
        return cls(
            channels,
            tuple(centers),
            tuple(scales),
            tuple(sorted(set(fit.crop_type.astype(str)))),
            sha256(payload.encode()).hexdigest(),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "channels": list(self.channels),
            "center": list(self.center),
            "scale": list(self.scale),
            "crops": list(self.crops),
            "fit_keys_hash": self.fit_keys_hash,
        }

    @classmethod
    def from_dict(cls, value: dict):
        if value.get("schema_version") != "0.1":
            raise ValueError("Unsupported preprocessor schema")
        obj = cls(
            tuple(value["channels"]),
            tuple(value["center"]),
            tuple(value["scale"]),
            tuple(value["crops"]),
            value["fit_keys_hash"],
        )
        if (
            not obj.channels
            or obj.channels[0] != "primary_ndvi"
            or not set(obj.channels).issubset(CHANNELS)
            or len(obj.channels) != len(obj.center)
            or len(obj.center) != len(obj.scale)
            or not np.isfinite(obj.center).all()
            or not np.isfinite(obj.scale).all()
            or (np.array(obj.scale) <= 0).any()
        ):
            raise ValueError("Invalid preprocessor state")
        return obj

    @property
    def fingerprint(self) -> str:
        return sha256(
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


def _deltas(observed: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(observed))[:, None]
    left = np.maximum.accumulate(np.where(observed, positions, -cap), axis=0)
    right = np.minimum.accumulate(
        np.where(observed, positions, len(observed) + cap)[::-1], axis=0
    )[::-1]
    return np.minimum(positions - left, cap), np.minimum(right - positions, cap)


class WindowDatasetAdapter:
    """Ленивые дневные окна; один loss target в центре, ключи в исходном порядке.

    Объект не импортирует torch, но совместим с torch DataLoader. Padding за
    границами polygon-year не входит в статистики, календарь или loss.
    """

    def __init__(
        self,
        context: MaskedContext,
        target_keys: pd.DataFrame,
        preprocessor: WindowPreprocessor,
        *,
        window_days: int = 61,
        labels: pd.DataFrame | None = None,
    ):
        if window_days < 3 or window_days % 2 == 0:
            raise ValueError("window_days must be odd and >= 3")
        self.keys = canonical_keys(target_keys)
        self.preprocessor = preprocessor
        self.window_days = window_days
        self.center_index = window_days // 2
        source = context.frame.copy(deep=True)
        source_idx = key_index(source)
        idx = key_index(self.keys)
        if not idx.isin(source_idx).all():
            raise ValueError("Target keys absent from context")
        hidden_idx = source_idx[context.synthetic_gap.to_numpy()]
        if not idx.isin(hidden_idx).all():
            raise ValueError("All target keys must be masked globally before windowing")
        self.y = np.full(len(idx), np.nan, dtype=np.float32)
        if labels is not None:
            label_idx = key_index(labels)
            if len(idx) != len(label_idx) or not idx.isin(label_idx).all():
                raise ValueError("Label keys must exactly match target keys")
            target = labels.copy().reset_index(drop=True)
            target[KEY] = canonical_keys(labels)
            self.y = target.set_index(KEY).loc[idx, "primary_ndvi"].to_numpy(np.float32)
            if not np.isfinite(self.y).all():
                raise ValueError(
                    "Loss labels must be finite originally observed targets"
                )
            eligible_idx = source_idx[context.supervision_eligible.to_numpy()]
            if not idx.isin(eligible_idx).all():
                raise ValueError(
                    "Natural missing or hidden test target cannot enter supervised loss"
                )
        source["_natural"] = context.natural_missing.to_numpy()
        source["_synthetic"] = context.synthetic_gap.to_numpy()
        self._sequences = {}
        self._crop_ids = {name: i + 1 for i, name in enumerate(preprocessor.crops)}
        needed = set(zip(self.keys.anon_polygon_id, self.keys.date.dt.year))
        for (polygon, year), group in source.groupby(
            ["anon_polygon_id", source.date.dt.year], sort=False
        ):
            if (polygon, year) not in needed:
                continue
            group = group.sort_values("date").set_index("date")
            calendar = pd.date_range(group.index.min(), group.index.max(), freq="D")
            grid = group.reindex(calendar)
            x, invalid = _clean(grid, preprocessor.channels)
            observed = np.isfinite(x)
            since, until = _deltas(observed, 366)
            values = np.where(
                observed, (x - preprocessor.center) / preprocessor.scale, 0
            ).astype(np.float32)
            visible = np.flatnonzero(observed[:, 0])
            base = (
                np.interp(np.arange(len(grid)), visible, x[visible, 0])
                if len(visible)
                else np.full(len(grid), preprocessor.center[0])
            ).astype(np.float32)
            # Новая культура в held-out получает token 0; unknown не становится известной.
            crops = (
                grid.crop_type.ffill()
                .bfill()
                .astype(str)
                .map(self._crop_ids)
                .fillna(0)
                .to_numpy(np.int64)
            )
            days = calendar.dayofyear.to_numpy()
            phase = 2 * np.pi * (days - 1) / np.where(calendar.is_leap_year, 366, 365)
            cal = np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)
            synthetic = grid._synthetic.astype("boolean").fillna(False).to_numpy(bool)
            natural = grid._natural.astype("boolean").fillna(True).to_numpy(bool)
            self._sequences[polygon, year] = (
                calendar,
                values,
                observed,
                invalid,
                since,
                until,
                base,
                crops,
                cal,
                synthetic,
                natural,
            )

    def __len__(self):
        return len(self.keys)

    @property
    def input_features(self):
        # x, observation mask, invalid flag, since/until, calendar, valid-time.
        return 5 * len(self.preprocessor.channels) + 3

    def __getitem__(self, index):
        row = self.keys.iloc[index]
        seq = self._sequences[row.anon_polygon_id, row.date.year]
        (
            calendar,
            values,
            observed,
            invalid,
            since,
            until,
            base,
            crops,
            cal,
            synthetic,
            natural,
        ) = seq
        position = (row.date - calendar[0]).days
        offsets = np.arange(self.window_days) + position - self.center_index
        valid = (offsets >= 0) & (offsets < len(calendar))

        def take(array, fill=0):
            result = np.full(
                (self.window_days, *array.shape[1:]), fill, dtype=array.dtype
            )
            result[valid] = array[offsets[valid]]
            return result

        x, obs, inv = take(values), take(observed), take(invalid)
        left, right = take(since), take(until)
        features = np.concatenate(
            [x, obs, inv, np.log1p(left), np.log1p(right), take(cal), valid[:, None]],
            axis=-1,
        ).astype(np.float32)
        loss_mask = np.zeros(self.window_days, dtype=bool)
        loss_mask[self.center_index] = np.isfinite(self.y[index])
        labels = np.full(self.window_days, np.nan, dtype=np.float32)
        labels[self.center_index] = self.y[index]
        dates = np.full(self.window_days, np.datetime64("NaT"), dtype="datetime64[D]")
        dates[valid] = calendar.to_numpy(dtype="datetime64[D]")[offsets[valid]]
        return {
            "features": features,
            "values": x,
            "observation_mask": obs,
            "invalid_mask": inv,
            "delta_since": left,
            "delta_until": right,
            "valid_calendar_mask": valid,
            "natural_missing_mask": take(natural),
            "synthetic_gap_mask": take(synthetic),
            "loss_mask": loss_mask,
            "labels": labels,
            "base": take(base),
            "crop_ids": take(crops),
            "dates": dates,
            "index": index,
        }

    def predictions_frame(self, values) -> pd.DataFrame:
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self),) or not np.isfinite(values).all():
            raise ValueError("One finite prediction per target key is required")
        return self.keys.assign(primary_ndvi_pred=values)
