import numpy as np, pandas as pd, time
from .core import Ctx, load_raw, DYNAMIC
from .features import Prep, build_features


def weather_groups(base):
    """AOIs sharing an identical ERA5 cell (exact temp+precip match on overlap)."""
    codes, names = pd.factorize(base['anon_polygon_id'], sort=True)
    A = len(names)
    W = base.pivot_table(index='t', columns='anon_polygon_id', values='era5_temp_c')
    P = base.pivot_table(index='t', columns='anon_polygon_id', values='era5_precip_mm')
    W = W.reindex(columns=names); P = P.reindex(columns=names)
    Wv, Pv = W.values, P.values
    par = list(range(A))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(A):
        for j in range(i + 1, A):
            m = np.isfinite(Wv[:, i]) & np.isfinite(Wv[:, j])
            if m.sum() < 400: continue
            if np.abs(Wv[m, i] - Wv[m, j]).max() < 1e-9 and np.abs(Pv[m, i] - Pv[m, j]).max() < 1e-9:
                ri, rj = find(i), find(j)
                if ri != rj: par[ri] = rj
    g = {}
    for i in range(A): g.setdefault(find(i), []).append(i)
    return {i: g[find(i)] for i in range(A)}, names


def make_blocks(base, obs_rows, nblocks, seed):
    """Partition observed target rows into nblocks stratified by (aoi, year)."""
    rng = np.random.default_rng(seed)
    sub = base.loc[obs_rows, ['anon_polygon_id', 'year']]
    key = sub['anon_polygon_id'].astype(str) + '_' + sub['year'].astype(str)
    out = np.zeros(len(obs_rows), np.int32)
    for k, idx in pd.Series(np.arange(len(obs_rows))).groupby(key.values):
        idx = idx.values
        perm = rng.permutation(len(idx))
        out[idx[perm]] = np.arange(len(idx)) % nblocks
    return out


def featurize(base, hidden_rows, query_rows, wg, prep_cache=None):
    ctx = Ctx(base, hidden_rows)
    prep = Prep(ctx, wg)
    q = base.loc[query_rows]
    qa = ctx.aoi_codes[query_rows]
    X = build_features(ctx, prep, qa, q['t'].values.astype(np.int64),
                       q['doy'].values.astype(np.int64), q['year'].values.astype(np.int64))
    X.insert(0, 'crop_code', pd.factorize(q['crop_type'].values)[0] * 0 +
             pd.Categorical(q['crop_type'].values,
                            categories=['озимая пшеница','зерновые','подсолнечник','пастбища/зерновые']).codes)
    X.index = np.arange(len(X))
    return X


def build_training_set(base, eval_hidden, nblocks, seeds, wg, log=print):
    """Generate (X, y, meta) for supervised rows: observed targets hidden block-wise
    on top of eval_hidden (which is always hidden - it mimics the real test gaps)."""
    obs = np.flatnonzero(np.isfinite(base['primary_ndvi'].values))
    ev = np.zeros(len(base), bool); ev[eval_hidden] = True
    obs = obs[~ev[obs]]
    Xs, ys, ms = [], [], []
    for si, seed in enumerate(seeds):
        blk = make_blocks(base, obs, nblocks, seed)
        for b in range(nblocks):
            rows = obs[blk == b]
            hid = np.concatenate([eval_hidden, rows])
            t0 = time.time()
            X = featurize(base, hid, rows, wg)
            Xs.append(X)
            ys.append(base['primary_ndvi'].values[rows])
            ms.append(base.loc[rows, ['anon_polygon_id', 'date', 'year', 'source', 'split',
                                      's2_ndvi', 'landsat_ndvi', 'modis_ndvi']]
                      .assign(row_id=rows, seed=seed, block=b).reset_index(drop=True))
            log(f'  round {si} block {b}: {len(rows)} rows, {time.time()-t0:.1f}s')
    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    meta = pd.concat(ms, ignore_index=True)
    return X, y, meta
