import warnings, os, sys, time, json, argparse
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from ndvi2.pipeline import weather_groups, featurize, make_blocks, build_training_set

ap = argparse.ArgumentParser()
ap.add_argument('--rounds', type=int, default=1)
ap.add_argument('--nblocks', type=int, default=7)
ap.add_argument('--seed', type=int, default=17)
ap.add_argument('--out', default='work/cv1')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

base = pd.read_pickle('base.pkl')
wg, names = weather_groups(base)
obs = np.flatnonzero(np.isfinite(base['primary_ndvi'].values))
is_train_aoi = (base['split'].values == 'train')

# ---- evaluation pseudo-gaps: 15% of observed rows of the 39 train AOIs ----
cand = obs[is_train_aoi[obs]]
blk = make_blocks(base, cand, a.nblocks, a.seed)
eval_rows = np.sort(cand[blk == 0])
print('eval rows', len(eval_rows), 'share of train-AOI observed %.4f' % (len(eval_rows) / len(cand)))
d = base.loc[eval_rows]
runs = []
for aoi, g in d.groupby('anon_polygon_id'):
    tt = np.sort(g['t'].values); c = 1
    for i in range(1, len(tt)):
        if tt[i] - tt[i - 1] == 1: c += 1
        else: runs.append(c); c = 1
    runs.append(c)
import collections
print('eval gap-run lengths', dict(collections.Counter(runs)))

t0 = time.time()
Xtr, ytr, mtr = build_training_set(base, eval_rows, a.nblocks, list(range(a.seed, a.seed + a.rounds)), wg)
print('train set', Xtr.shape, 'in', round(time.time() - t0, 1), 's')
Xev = featurize(base, eval_rows, eval_rows, wg)
yev = base['primary_ndvi'].values[eval_rows]
mev = base.loc[eval_rows, ['anon_polygon_id', 'date', 'year', 'source',
                           's2_ndvi', 'landsat_ndvi', 'modis_ndvi']].reset_index(drop=True)
np.save(f'{a.out}/yev.npy', yev)
Xtr.to_pickle(f'{a.out}/Xtr.pkl'); np.save(f'{a.out}/ytr.npy', ytr); mtr.to_pickle(f'{a.out}/mtr.pkl')
Xev.to_pickle(f'{a.out}/Xev.pkl'); mev.to_pickle(f'{a.out}/mev.pkl')
np.save(f'{a.out}/eval_rows.npy', eval_rows)
print('saved to', a.out)
