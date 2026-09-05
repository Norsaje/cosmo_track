"""Официальный GapScore, строгая привязка ошибок к ключам и срезы качества."""
from __future__ import annotations

import numpy as np
import pandas as pd

from veg_recovery.data import KEY_COLUMNS
from .masking import normalize_keys

COMPOSITE_WEIGHTS = {'A': 0.50, 'B': 0.25, 'C': 0.15, 'D': 0.10}


def rmse(y_true, y_pred) -> float:
    truth, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError('RMSE needs equally sized nonempty one-dimensional arrays')
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite truth/prediction cannot be scored')
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def gap_score(value: float) -> float:
    if not np.isfinite(value) or value < 0:
        raise ValueError('RMSE must be finite and nonnegative')
    # Формула criteria.pdf, стр. 2 / case_doc.pdf, стр. 14.
    return round(30 * max(0.0, 1.0 - float(value) / 0.10), 2)


def composite_score(mode_rmse: dict[str, float]) -> float:
    if set(mode_rmse) != set(COMPOSITE_WEIGHTS):
        raise ValueError('Composite requires all four CV modes A/B/C/D')
    if any(not np.isfinite(v) or v < 0 for v in mode_rmse.values()):
        raise ValueError('Invalid component RMSE')
    return float(sum(COMPOSITE_WEIGHTS[k] * mode_rmse[k] for k in COMPOSITE_WEIGHTS))


def source_labels(frame: pd.DataFrame, tolerance: float = 1e-7) -> pd.Series:
    target = frame.primary_ndvi.to_numpy(dtype=float)
    labels = np.full(len(frame), 'unknown', dtype=object)
    for name in ('s2', 'landsat', 'modis'):
        if name + '_ndvi' not in frame:
            continue
        vals = frame[name + '_ndvi'].to_numpy(dtype=float)
        match = np.isfinite(vals) & np.isfinite(target) & np.isclose(vals, target, atol=tolerance, rtol=0)
        labels[match & (labels == 'unknown')] = name
    return pd.Series(labels, index=frame.index, name='source_label')


def distance_bin(values) -> pd.Categorical:
    vals = pd.Series(values, dtype=float)
    return pd.Categorical(np.select(
        [vals.isna() | np.isinf(vals), vals <= 7, vals <= 15, vals <= 30],
        ['none', '0-7', '8-15', '16-30'], default='31+'))


def context_diagnostics(frame: pd.DataFrame, gap_keys, known_polygons=()) -> pd.DataFrame:
    keys = normalize_keys(gap_keys).reset_index(drop=True)
    indexed = frame.set_index(KEY_COLUMNS)
    req_index = pd.MultiIndex.from_frame(keys)
    if not req_index.isin(indexed.index).all():
        raise ValueError('Gap keys absent from context')
    out = keys.copy()
    out['crop_type'] = indexed.loc[req_index, 'crop_type'].to_numpy()
    out['year'] = out.date.dt.year
    out['month'] = out.date.dt.month
    out['seen_polygon'] = out.anon_polygon_id.isin(set(known_polygons))
    out['left_days'] = np.nan
    out['right_days'] = np.nan
    out['gap_length'] = 1
    out['gap_position'] = 0
    positions = pd.Series(np.arange(len(out)), index=req_index)
    # Серия определяется по потенциальным target-наблюдениям, а не по дневной сетке.
    for polygon, rows in frame.sort_values(KEY_COLUMNS).groupby('anon_polygon_id', sort=False):
        group_positions = out.index[out.anon_polygon_id.eq(polygon)]
        if not len(group_positions):
            continue
        selected = pd.MultiIndex.from_frame(rows[KEY_COLUMNS]).isin(req_index)
        finite = np.isfinite(rows.primary_ndvi.to_numpy(dtype=float)) & ~selected
        visible_dates = rows.loc[finite, 'date'].to_numpy(dtype='datetime64[D]')
        gap_dates = out.loc[group_positions, 'date'].to_numpy(dtype='datetime64[D]')
        if len(visible_dates):
            insertion = np.searchsorted(visible_dates, gap_dates)
            left = insertion > 0
            right = insertion < len(visible_dates)
            out.loc[group_positions[left], 'left_days'] = (gap_dates[left] - visible_dates[insertion[left]-1]).astype(float)
            out.loc[group_positions[right], 'right_days'] = (visible_dates[insertion[right]] - gap_dates[right]).astype(float)
        potential = rows.loc[finite | selected, KEY_COLUMNS].copy()
        potential['gap'] = selected[finite | selected]
        potential['year'] = potential.date.dt.year
        run_id = (potential.gap.ne(potential.gap.shift()) | potential.year.ne(potential.year.shift())).cumsum()
        for _, run in potential.loc[potential.gap].groupby(run_id):
            ix = positions.loc[pd.MultiIndex.from_frame(run[KEY_COLUMNS])].to_numpy()
            out.loc[ix, 'gap_length'] = len(run)
            out.loc[ix, 'gap_position'] = np.arange(len(run))
    out['left_distance_bin'] = distance_bin(out.left_days)
    out['right_distance_bin'] = distance_bin(out.right_days)
    return out


def test_mask_profile(test: pd.DataFrame, train: pd.DataFrame | None = None) -> dict:
    gaps = test.loc[test.is_synthetic_gap.astype(bool), KEY_COLUMNS]
    known = () if train is None else train.anon_polygon_id.unique()
    detail = context_diagnostics(test, gaps, known)
    sources = source_labels(test)
    visible_counts = sources.loc[np.isfinite(test.primary_ndvi)].value_counts(normalize=True)
    return {
        'rows': len(gaps), 'known_fraction': float(detail.seen_polygon.mean()),
        'gap_length_counts': {str(k): int(v) for k, v in detail.gap_length.value_counts().sort_index().items()},
        'month_counts': {str(k): int(v) for k, v in detail.month.value_counts().sort_index().items()},
        'year_counts': {str(k): int(v) for k, v in detail.year.value_counts().sort_index().items()},
        'left_distance_counts': {str(k): int(v) for k, v in detail.left_distance_bin.value_counts().items()},
        'right_distance_counts': {str(k): int(v) for k, v in detail.right_distance_bin.value_counts().items()},
        'source_proxy_visible_distribution': {str(k): float(v) for k, v in visible_counts.items()},
        'source_proxy_warning': 'True source on real gaps is hidden; visible test hierarchy is only a proxy.',
        'run_definition': 'Consecutive hidden target slots within polygon/year; naturally missing daily rows ignored.',
    }


def metric_table(targets: pd.DataFrame, predictions: pd.DataFrame,
                 diagnostics: pd.DataFrame | None = None) -> pd.DataFrame:
    for name, obj in [('targets', targets), ('predictions', predictions)]:
        if obj.duplicated(KEY_COLUMNS).any():
            raise ValueError(f'Duplicate {name} keys')
    if not pd.MultiIndex.from_frame(targets[KEY_COLUMNS]).isin(pd.MultiIndex.from_frame(predictions[KEY_COLUMNS])).all() or len(targets) != len(predictions):
        raise ValueError('Prediction key set differs from validation targets')
    merged = targets.merge(predictions[KEY_COLUMNS + ['primary_ndvi_pred']], on=KEY_COLUMNS, validate='one_to_one')
    if diagnostics is not None:
        extra = [c for c in diagnostics if c not in merged]
        merged = merged.merge(diagnostics[KEY_COLUMNS + extra], on=KEY_COLUMNS, how='left', validate='one_to_one')
    if 'year' not in merged:
        merged['year'] = pd.to_datetime(merged.date).dt.year
    if 'confidence_bin' not in merged:
        if 'p_s2' in merged and 'p_landsat' in merged and 'p_modis' in merged:
            confidence = merged[['p_s2', 'p_landsat', 'p_modis']].max(axis=1)
            merged['confidence_bin'] = pd.cut(confidence, [-np.inf, .5, .8, np.inf], labels=['low', 'medium', 'high'])
        else:
            merged['confidence_bin'] = 'unavailable'
    records = []
    def add(dimension, subgroup, group):
        value = rmse(group.primary_ndvi, group.primary_ndvi_pred)
        records.append({'dimension': dimension, 'subgroup': str(subgroup), 'n': len(group),
                        'rmse': value, 'gap_score': gap_score(value)})
    add('overall', 'all', merged)
    for dimension in ['seen_polygon', 'crop_type', 'source_label', 'gap_length',
                      'left_distance_bin', 'right_distance_bin', 'year', 'confidence_bin']:
        if dimension not in merged:
            records.append({'dimension': dimension, 'subgroup': 'unavailable', 'n': len(merged),
                            'rmse': np.nan, 'gap_score': np.nan})
            continue
        for key, group in merged.groupby(dimension, observed=True, dropna=False):
            add(dimension, key, group)
    return pd.DataFrame(records)
