"""Контракт фактических файлов конкурса; отсутствующий target не наблюдение."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

KEY_COLUMNS = ['anon_polygon_id', 'date']
NUMERIC_COLUMNS = [
    's2_ndvi', 's2_evi', 's2_ndwi', 'landsat_ndvi', 'landsat_evi',
    'landsat_ndwi', 'modis_ndvi', 'modis_evi', 'era5_temp_c', 'era5_precip_mm',
    'year', 'primary_ndvi', 'doy', 'ndvi_climatology_mean',
    'ndvi_climatology_std', 'ndvi_zscore', 'n_reference_years',
]
TRAIN_COLUMNS = KEY_COLUMNS + NUMERIC_COLUMNS + ['status', 'crop_type']
TEST_COLUMNS = [c for c in TRAIN_COLUMNS if c not in ('ndvi_zscore', 'status')] + ['is_synthetic_gap']
CURRENT_CONTROLS = {
    'train': {'shape': (99955, 21), 'polygons': 39, 'finite_targets': 30520,
              'date_min': '2010-04-01', 'date_max': '2024-10-30'},
    'test': {'shape': (57185, 20), 'polygons': 78, 'finite_targets': 17641,
             'date_min': '2010-04-01', 'date_max': '2025-10-30', 'synthetic_gaps': 3112},
}


def validate_frame(frame: pd.DataFrame, kind: Literal['train', 'test'] | None = None) -> None:
    required = set(KEY_COLUMNS + ['crop_type', 'primary_ndvi'])
    if kind is not None:
        required = set(TRAIN_COLUMNS if kind == 'train' else TEST_COLUMNS)
    missing = required - set(frame)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError('Null key values are forbidden')
    if frame.anon_polygon_id.astype(str).str.strip().eq('').any():
        raise ValueError('Empty polygon key')
    if not pd.api.types.is_datetime64_any_dtype(frame.date):
        raise ValueError('date must be datetime64')
    if frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError('Dates must be timezone-naive calendar dates')
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError('Duplicate anon_polygon_id + date key')
    if frame.crop_type.isna().any():
        raise ValueError('crop_type may not be missing')
    for col in set(NUMERIC_COLUMNS) & set(frame):
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f'{col} must be numeric')
    if 'is_synthetic_gap' in frame:
        if frame.is_synthetic_gap.isna().any() or not pd.api.types.is_bool_dtype(frame.is_synthetic_gap):
            raise ValueError('is_synthetic_gap must contain only booleans')
        gaps = frame.is_synthetic_gap.to_numpy(dtype=bool)
        if np.isfinite(frame.loc[gaps, 'primary_ndvi'].to_numpy(dtype=float)).any():
            raise ValueError('Synthetic gap has visible finite target')
        dynamic = (set(NUMERIC_COLUMNS) | {'status'}) & set(frame)
        if gaps.any() and frame.loc[gaps, sorted(dynamic)].notna().any().any():
            raise ValueError('Synthetic gap retains dynamic or derived fields')


def read_dataset(path: str | Path, kind: Literal['train', 'test'] = 'train',
                 strict_current: bool = False) -> pd.DataFrame:
    if kind not in ('train', 'test'):
        raise ValueError("kind must be 'train' or 'test'")
    # Проверяем заголовок до pandas: автоматическое переименование дублей недопустимо.
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        columns = next(csv.reader(handle), [])
    if len(columns) != len(set(columns)):
        raise ValueError('Duplicate CSV column names')
    expected = TRAIN_COLUMNS if kind == 'train' else TEST_COLUMNS
    if set(columns) != set(expected):
        raise ValueError(f'CSV schema differs: missing={sorted(set(expected)-set(columns))}; '
                         f'extra={sorted(set(columns)-set(expected))}')
    dtypes = {c: 'float64' for c in NUMERIC_COLUMNS if c in columns}
    dtypes.update({c: 'string' for c in ['anon_polygon_id', 'crop_type', 'status'] if c in columns})
    if 'is_synthetic_gap' in columns:
        dtypes['is_synthetic_gap'] = 'boolean'
    frame = pd.read_csv(path, encoding='utf-8', dtype=dtypes, parse_dates=['date'],
                        true_values=['True'], false_values=['False'])
    validate_frame(frame, kind)
    if strict_current:
        actual = summarize_frame(frame)
        for field, expected_value in CURRENT_CONTROLS[kind].items():
            got = tuple(actual[field]) if field == 'shape' else actual[field]
            if got != expected_value:
                raise ValueError(f'Current {kind} control {field}: {got!r} != {expected_value!r}')
    return frame


def summarize_frame(frame: pd.DataFrame) -> dict:
    finite = np.isfinite(frame.primary_ndvi.to_numpy(dtype=float))
    gap = frame.get('is_synthetic_gap', pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return {
        'shape': list(frame.shape), 'polygons': int(frame.anon_polygon_id.nunique()),
        'date_min': frame.date.min().strftime('%Y-%m-%d'),
        'date_max': frame.date.max().strftime('%Y-%m-%d'),
        'finite_targets': int(finite.sum()), 'synthetic_gaps': int(gap.sum()),
        'natural_missing_targets': int((~finite & ~gap.to_numpy()).sum()),
        'crop_types': {str(k): int(v) for k, v in frame.crop_type.value_counts().items()},
        'missing_counts': {k: int(v) for k, v in frame.isna().sum().items()},
        'infinite_counts': {k: int(np.isinf(frame[k].to_numpy(dtype=float)).sum())
                            for k in NUMERIC_COLUMNS if k in frame},
        'duplicate_keys': int(frame.duplicated(KEY_COLUMNS).sum()),
    }


def fingerprint(source: str | Path | pd.DataFrame) -> str:
    digest = hashlib.sha256()
    if isinstance(source, pd.DataFrame):
        digest.update(source.to_csv(index=False, date_format='%Y-%m-%d', lineterminator='\n').encode('utf-8'))
    else:
        with Path(source).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()
