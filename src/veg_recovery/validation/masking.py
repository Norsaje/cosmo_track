"""Псевдопропуски повторяют отсутствие данных на настоящей gap-строке."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from veg_recovery.data import KEY_COLUMNS

MASKED_COLUMNS = (
    'primary_ndvi', 's2_ndvi', 's2_evi', 's2_ndwi', 'landsat_ndvi', 'landsat_evi',
    'landsat_ndwi', 'modis_ndvi', 'modis_evi', 'era5_temp_c', 'era5_precip_mm',
    'year', 'doy', 'ndvi_climatology_mean', 'ndvi_climatology_std', 'ndvi_zscore',
    'status', 'n_reference_years',
)


@dataclass(frozen=True)
class MaskSpec:
    columns: tuple[str, ...] = MASKED_COLUMNS
    version: str = 'real_test_v1'

    @classmethod
    def from_test(cls, frame: pd.DataFrame) -> 'MaskSpec':
        if 'is_synthetic_gap' not in frame:
            raise ValueError('Mask inference requires is_synthetic_gap')
        gaps = frame.loc[frame.is_synthetic_gap.astype(bool)]
        if gaps.empty:
            raise ValueError('No real synthetic gaps for mask inference')
        # Дополнительные новые поля тоже маскируются, если test их полностью скрывает.
        inferred = [c for c in frame if c not in KEY_COLUMNS + ['crop_type', 'is_synthetic_gap']
                    and gaps[c].isna().all()]
        forbidden_visible = [c for c in MASKED_COLUMNS if c in frame and gaps[c].notna().any()]
        if forbidden_visible:
            raise ValueError(f'Real gaps retain forbidden fields: {forbidden_visible}')
        return cls(tuple(dict.fromkeys(MASKED_COLUMNS + tuple(inferred))))


def normalize_keys(selected_keys: pd.DataFrame | pd.MultiIndex | Iterable) -> pd.DataFrame:
    if isinstance(selected_keys, pd.DataFrame):
        keys = selected_keys[KEY_COLUMNS].copy()
    elif isinstance(selected_keys, pd.MultiIndex):
        keys = selected_keys.to_frame(index=False)
        keys.columns = KEY_COLUMNS
    else:
        keys = pd.DataFrame(list(selected_keys), columns=KEY_COLUMNS)
    keys['date'] = pd.to_datetime(keys['date'], errors='raise')
    keys['anon_polygon_id'] = keys.anon_polygon_id.astype('string')
    if keys.isna().any().any() or keys.duplicated(KEY_COLUMNS).any():
        raise ValueError('Null or duplicate selected keys')
    return keys


def apply_mask(frame: pd.DataFrame, selected_keys: pd.DataFrame | pd.MultiIndex | Iterable,
               spec: MaskSpec | None = None) -> pd.DataFrame:
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError('Duplicate keys in context frame')
    keys = normalize_keys(selected_keys)
    available = pd.MultiIndex.from_frame(frame[KEY_COLUMNS])
    requested = pd.MultiIndex.from_frame(keys)
    if not requested.isin(available).all():
        raise ValueError('Selected masking key absent from frame')
    selected = available.isin(requested)
    out = frame.copy(deep=True)
    for col in (spec or MaskSpec()).columns:
        if col in out:
            # Nullable string / category колонки сохраняют pandas-тип.
            if pd.api.types.is_integer_dtype(out[col].dtype):
                out[col] = out[col].astype(float)
            out.loc[selected, col] = np.nan
    if 'is_synthetic_gap' not in out:
        out['is_synthetic_gap'] = False
    out.loc[selected, 'is_synthetic_gap'] = True
    return out
