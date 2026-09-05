"""Фолды сохраняют ключи и явную политику fit/context, без номеров строк."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from veg_recovery.data import KEY_COLUMNS
from .masking import MaskSpec, apply_mask
from .metrics import context_diagnostics, source_labels

FOLD_VERSION = 'folds_v1'
DEFAULT_SEEDS = (17, 29, 43, 71, 101)


@dataclass(frozen=True)
class FoldSplit:
    fit_frame: pd.DataFrame
    context_frame: pd.DataFrame
    targets: pd.DataFrame
    gap_keys: pd.DataFrame
    metadata: dict


def _templates(test, train):
    if test is None or 'is_synthetic_gap' not in test or not test.is_synthetic_gap.any():
        return pd.DataFrame([{'gap_length': 1, 'month': 7, 'year': 2020,
                              'crop_type': str(train.crop_type.iloc[0]), 'seen_polygon': True,
                              'left_days': 5., 'right_days': 5.}])
    keys = test.loc[test.is_synthetic_gap.astype(bool), KEY_COLUMNS]
    detail = context_diagnostics(test, keys, train.anon_polygon_id.unique())
    result = detail.loc[detail.gap_position.eq(0)].reset_index(drop=True)
    result.attrs['source_proxy'] = source_labels(test.loc[np.isfinite(test.primary_ndvi)]).value_counts().to_dict()
    return result


def _sample_matched(visible, templates, count, excluded, rng):
    """Подбираем серии по сезону/культуре/эпохе/расстояниям без отбора по ошибке."""
    if visible.empty or count <= 0:
        return visible.head(0).copy()
    visible = visible.sort_values(KEY_COLUMNS).reset_index(drop=True).copy()
    polygons = visible.anon_polygon_id.to_numpy()
    years = visible.date.dt.year.to_numpy()
    months = visible.date.dt.month.to_numpy()
    crops = visible.crop_type.astype(str).to_numpy()
    days = visible.date.to_numpy(dtype='datetime64[D]').astype('int64')
    sources = source_labels(visible).to_numpy()
    unseen = visible.anon_polygon_id.isin(excluded).to_numpy()
    occupied = np.zeros(len(visible), dtype=bool)
    selected = []
    # Истинные источники test gaps неизвестны: источники visible служат прокси.
    source_values, source_counts = np.unique(sources, return_counts=True)
    if templates.attrs.get('source_proxy'):
        source_values = np.array(list(templates.attrs['source_proxy']))
        source_counts = np.array(list(templates.attrs['source_proxy'].values()))
    for _ in range(max(count * 3, 30)):
        if len(selected) >= count:
            break
        template = templates.iloc[int(rng.integers(len(templates)))]
        length = min(int(template.gap_length), count - len(selected), 8)
        desired_unseen = not bool(template.seen_polygon)
        candidates = np.flatnonzero((~occupied) & (unseen == desired_unseen))
        if not len(candidates):
            candidates = np.flatnonzero(~occupied)
        if not len(candidates):
            break
        start = rng.choice(candidates, size=min(160, len(candidates)), replace=False)
        end = start + length - 1
        valid = end < len(visible)
        start, end = start[valid], end[valid]
        valid = (polygons[start] == polygons[end]) & (years[start] == years[end])
        for j in range(length):
            valid &= ~occupied[start + j]
        start, end = start[valid], end[valid]
        if not len(start):
            continue
        left = np.full(len(start), 365., dtype=float)
        right = np.full(len(start), 365., dtype=float)
        has_left = (start > 0) & (polygons[np.maximum(start-1, 0)] == polygons[start])
        has_right = (end+1 < len(visible)) & (polygons[np.minimum(end+1, len(visible)-1)] == polygons[end])
        left[has_left] = days[start[has_left]] - days[start[has_left]-1]
        right[has_right] = days[end[has_right]+1] - days[start[has_right]]
        tleft = 365. if pd.isna(template.left_days) else float(template.left_days)
        tright = 365. if pd.isna(template.right_days) else float(template.right_days)
        tsource = rng.choice(source_values, p=source_counts/source_counts.sum())
        mdiff = np.abs(months[start] - int(template.month))
        score = np.minimum(mdiff, 12-mdiff) * 1.2 + (crops[start] != str(template.crop_type)) * 3.0
        # 2015 и 2017 задают смену доступности S2; 2025 не переносим буквально.
        score += (np.digitize(years[start], [2015, 2017]) != np.digitize(int(template.year), [2015, 2017])) * 1.5
        score += np.abs(np.log1p(left)-np.log1p(tleft)) + np.abs(np.log1p(right)-np.log1p(tright))
        score += (sources[start] != tsource) * .7 + rng.random(len(start)) * .25
        chosen = int(start[np.argmin(score)])
        selected.extend(range(chosen, chosen+length))
        occupied[chosen:chosen+length] = True
        # Оставляем хотя бы один видимый слот между независимо выбранными сериями.
        if chosen and polygons[chosen-1] == polygons[chosen]:
            occupied[chosen-1] = True
        if chosen+length < len(visible) and polygons[chosen+length] == polygons[chosen]:
            occupied[chosen+length] = True
    return visible.iloc[selected].copy()


def _records(selected, mode, repeat, fold, seed, excluded=(), context_policy='transductive', cutoff_date=''):
    keys = selected[KEY_COLUMNS].copy()
    keys['mode'], keys['repeat'], keys['fold'], keys['seed'] = mode, repeat, fold, seed
    keys['role'] = 'validation'
    keys['seen_polygon'] = ~keys.anon_polygon_id.isin(excluded)
    keys['fit_excluded_polygons'] = json.dumps(sorted(str(x) for x in excluded), ensure_ascii=False)
    keys['context_policy'] = context_policy
    keys['cutoff_date'] = cutoff_date
    return keys


def generate_folds(train: pd.DataFrame, test: pd.DataFrame | None = None,
                   seeds=DEFAULT_SEEDS, n_splits: int = 5,
                   max_gaps_per_split: int = 1200) -> pd.DataFrame:
    if train.duplicated(KEY_COLUMNS).any():
        raise ValueError('Training keys must be unique')
    if max_gaps_per_split < 1:
        raise ValueError('max_gaps_per_split must be positive')
    visible = train.loc[np.isfinite(train.primary_ndvi.to_numpy(dtype=float))].copy()
    polygons = np.array(sorted(visible.anon_polygon_id.unique()), dtype=object)
    if len(polygons) < 2:
        raise ValueError('At least two observed polygons required for grouped CV')
    templates = _templates(test, train)
    results = []
    for repeat, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        # Половина полигонов остаётся для fit; sampling задаёт долю unseen gaps.
        excluded = set(rng.permutation(polygons)[:max(1, len(polygons)//2)])
        selected = _sample_matched(visible, templates, max_gaps_per_split, excluded, rng)
        results.append(_records(selected, 'A', repeat, 0, seed, excluded))
    # GroupKFold: жадное балансирование числа target, без случайного разделения строк.
    grouped = [[] for _ in range(min(n_splits, len(polygons))) ]
    sizes = np.zeros(len(grouped), dtype=int)
    counts = visible.anon_polygon_id.value_counts()
    for polygon, n in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
        bucket = int(np.argmin(sizes))
        grouped[bucket].append(polygon)
        sizes[bucket] += n
    base_seed = int(seeds[0])
    for fold, group in enumerate(grouped):
        excluded = set(group)
        pool = visible.loc[visible.anon_polygon_id.isin(excluded)]
        rng = np.random.default_rng(base_seed + 1000 + fold)
        selected = _sample_matched(pool, templates, min(max_gaps_per_split, len(pool)//3), excluded, rng)
        results.append(_records(selected, 'B', 0, fold, base_seed+1000+fold, excluded))
    last_year = int(visible.date.dt.year.max())
    cutoff = f'{last_year}-01-01'
    latest = visible.loc[visible.date.dt.year.eq(last_year)]
    rng = np.random.default_rng(base_seed+2000)
    selected = _sample_matched(latest, templates, min(max_gaps_per_split, len(latest)//2), set(), rng)
    # Polygons without any past fit target are explicitly unseen in temporal CV.
    past_polygons = set(visible.loc[visible.date < pd.Timestamp(cutoff), 'anon_polygon_id'])
    excluded = set(polygons) - past_polygons
    results.append(_records(selected, 'C', 0, 0, base_seed+2000, excluded, 'past_only', cutoff))
    for fold, group in enumerate(grouped):
        excluded = set(group)
        pool = visible.loc[visible.anon_polygon_id.isin(excluded)]
        rng = np.random.default_rng(base_seed+3000+fold)
        selected_parts = []
        cutoffs = {}
        # Последняя серия сезона, затем удаляем близкие левые наблюдения.
        # Поздние наблюдения цензурируются явной policy, исключая правую сторону.
        for polygon, rows in pool.sort_values(KEY_COLUMNS).groupby('anon_polygon_id', sort=True):
            year_counts = rows.date.dt.year.value_counts()
            eligible_years = year_counts[year_counts >= 7].index.to_numpy()
            if not len(eligible_years):
                continue
            year = int(rng.choice(eligible_years))
            season = rows.loc[rows.date.dt.year.eq(year)]
            length = int(rng.integers(2, 5))
            tail = season.tail(length)
            cutoff_day = tail.date.min() - pd.Timedelta(days=15)
            if not rows.date.lt(cutoff_day).any():
                continue
            selected_parts.append(tail)
            cutoffs[str(polygon)] = cutoff_day.strftime('%Y-%m-%d')
        if selected_parts:
            selected = pd.concat(selected_parts).head(max_gaps_per_split)
            rec = _records(selected, 'D', 0, fold, base_seed+3000+fold, excluded,
                           'one_sided_hard', '')
            rec['cutoff_date'] = rec.anon_polygon_id.astype(str).map(cutoffs)
            results.append(rec)
    out = pd.concat(results, ignore_index=True)
    out['fold_version'] = FOLD_VERSION
    return out.sort_values(['mode', 'repeat', 'fold'] + KEY_COLUMNS).reset_index(drop=True)


def split_fold(frame: pd.DataFrame, folds: pd.DataFrame, mode: str,
               repeat: int = 0, fold: int = 0, spec: MaskSpec | None = None) -> FoldSplit:
    rows = folds.loc[folds['mode'].eq(mode) & folds['repeat'].eq(repeat) & folds['fold'].eq(fold)].copy()
    if rows.empty:
        raise ValueError(f'Unknown fold {mode}/{repeat}/{fold}')
    validation = rows.loc[rows.role.eq('validation')]
    keys = validation[KEY_COLUMNS].copy()
    if keys.duplicated(KEY_COLUMNS).any():
        raise ValueError('Duplicate fold validation keys')
    excluded = set(json.loads(rows.fit_excluded_polygons.iloc[0]))
    if rows.fit_excluded_polygons.nunique() != 1 or rows.context_policy.nunique() != 1:
        raise ValueError('Inconsistent fold policies')
    index = pd.MultiIndex.from_frame(frame[KEY_COLUMNS])
    requested = pd.MultiIndex.from_frame(keys)
    if not requested.isin(index).all():
        raise ValueError('Saved fold key absent from frame')
    hidden = index.isin(pd.MultiIndex.from_frame(rows[KEY_COLUMNS]))
    policy = rows.context_policy.iloc[0]
    if policy == 'past_only':
        cutoff = pd.Timestamp(rows.cutoff_date.iloc[0])
        hidden |= frame.date.ge(cutoff).to_numpy()
    elif policy == 'one_sided_hard':
        for polygon, group in validation.groupby('anon_polygon_id'):
            cutoff = pd.Timestamp(group.cutoff_date.iloc[0])
            polygon_rows = frame.anon_polygon_id.eq(polygon)
            allowed = frame.loc[polygon_rows & frame.date.lt(cutoff) & np.isfinite(frame.primary_ndvi)]
            if allowed.empty: raise ValueError('Hard fold requires one remote left neighbor')
            neighbor = allowed.date.max()
            # Exactly one observed target in this polygon remains, >15 days away.
            hidden |= (polygon_rows & frame.date.ne(neighbor)).to_numpy()
    elif policy != 'transductive':
        raise ValueError(f'Unknown context policy {policy}')
    fit = frame.loc[~hidden & ~frame.anon_polygon_id.isin(excluded)].copy()
    context = apply_mask(frame, frame.loc[hidden, KEY_COLUMNS], spec)
    # Censoring context does not create additional scored synthetic-gap keys.
    # Preserve the distinction from natural missing daily grid rows in features.
    context['is_synthetic_gap'] = index.isin(requested)
    targets = frame.set_index(KEY_COLUMNS).loc[requested].reset_index()
    if not np.isfinite(targets.primary_ndvi.to_numpy(dtype=float)).all():
        raise ValueError('Validation keys must have finite ground truth')
    targets['source_label'] = source_labels(targets)
    targets['seen_polygon'] = targets.anon_polygon_id.isin(fit.anon_polygon_id.unique())
    targets['year'] = targets.date.dt.year
    metadata = {'mode': mode, 'repeat': repeat, 'fold': fold,
                'seed': int(rows.seed.iloc[0]), 'context_policy': policy,
                'fit_excluded_polygons': sorted(excluded), 'n_context_hidden': int(hidden.sum()),
                'n_validation': len(keys), 'fold_version': FOLD_VERSION}
    return FoldSplit(fit.reset_index(drop=True), context, targets, keys, metadata)


def save_folds(folds: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.parquet':
        folds.to_parquet(path, index=False)
    else:
        folds.to_csv(path, index=False, encoding='utf-8', date_format='%Y-%m-%d')


def load_folds(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    folds = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(
        path, dtype={'anon_polygon_id': 'string', 'mode': 'string'}, parse_dates=['date'], keep_default_na=False)
    if set(folds.fold_version) != {FOLD_VERSION}:
        raise ValueError('Incompatible fold version')
    if folds.duplicated(['mode', 'repeat', 'fold'] + KEY_COLUMNS).any():
        raise ValueError('Duplicate fold keys')
    return folds
